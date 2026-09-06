#!/usr/bin/env python3
"""计算 DDRNet 参数量和端到端推理 FPS。

使用方法：
    # 默认读取红外测试集，使用 test.yaml、640x640、batch size=1
    python tools/get_fps_and_pm.py --device cuda

    # 使用训练后的 checkpoint 测试
    python tools/get_fps_and_pm.py --device cuda \
        --checkpoint output/infrared_images/test/best.pth

    # 指定配置、图片目录和输入尺寸
    python tools/get_fps_and_pm.py \
        --cfg experiments/cityscapes/ddrnet23_slim.yaml \
        --image-dir data/infrared_images/images/test \
        --height 640 --width 640

    # 使用 CUDA FP16 测试
    python tools/get_fps_and_pm.py --device cuda --fp16

计时范围：磁盘读图与解码、缩放、归一化、CPU 到 GPU 传输、模型 forward、
选择主输出、logits 双线性上采样、argmax，以及最终分割图传回 CPU。
不包含配置/模型加载和结果保存。
"""

import argparse
import os
import statistics
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import _init_paths  # noqa: F401
import models
from config import config


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile DDRNet parameters and end-to-end inference FPS')
    parser.add_argument(
        '--cfg', default='experiments/cityscapes/test.yaml',
        help='DDRNet experiment configuration file')
    parser.add_argument('--height', default=None, type=int,
                        help='input height (default: TEST.IMAGE_SIZE height)')
    parser.add_argument('--width', default=None, type=int,
                        help='input width (default: TEST.IMAGE_SIZE width)')
    parser.add_argument(
        '--image-dir', default='data/infrared_images/images/test',
        help='directory containing input images (searched recursively)')
    parser.add_argument(
        '--mean', default=[0.6423561, 0.1127488, 0.4362715],
        type=float, nargs=3, metavar=('R', 'G', 'B'),
        help='RGB normalization mean')
    parser.add_argument(
        '--std', default=[0.2345256, 0.1510579, 0.1994846],
        type=float, nargs=3, metavar=('R', 'G', 'B'),
        help='RGB normalization standard deviation')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='inference batch size')
    parser.add_argument('--warmup', default=50, type=int,
                        help='warmup iterations')
    parser.add_argument('--iterations', default=200, type=int,
                        help='timed batches per repeat')
    parser.add_argument('--repeats', default=3, type=int,
                        help='number of timed repeats')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='benchmark device')
    parser.add_argument('--fp16', action='store_true',
                        help='use FP16 inference (CUDA only)')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='optional .pt/.pth checkpoint')
    parser.add_argument(
        '--output-index', default=None, type=int,
        help='output index for a multi-output model '
             '(default: TEST.OUTPUT_INDEX)')
    return parser.parse_args()


def load_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            'configuration file not found: {}'.format(config_path))
    config.defrost()
    config.merge_from_file(config_path)
    config.freeze()
    return config


def apply_config_defaults(args, cfg):
    if args.height is None:
        args.height = int(cfg.TEST.IMAGE_SIZE[1])
    if args.width is None:
        args.width = int(cfg.TEST.IMAGE_SIZE[0])
    if args.output_index is None:
        args.output_index = int(cfg.TEST.OUTPUT_INDEX)


def validate_args(args):
    for name in ('height', 'width', 'batch_size', 'iterations', 'repeats'):
        if getattr(args, name) <= 0:
            raise ValueError('--{} must be positive'.format(
                name.replace('_', '-')))
    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative')
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(
            'image directory not found: {}'.format(args.image_dir))
    if any(value <= 0 for value in args.std):
        raise ValueError('--std values must be positive')


def resolve_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device(requested)


def build_model(cfg):
    try:
        model_module = getattr(models, cfg.MODEL.NAME)
    except AttributeError as error:
        raise ValueError(
            'unknown model in configuration: {}'.format(
                cfg.MODEL.NAME)) from error

    if torch.__version__.startswith('1'):
        model_module.BatchNorm2d_class = torch.nn.BatchNorm2d
        model_module.BatchNorm2d = torch.nn.BatchNorm2d
    return model_module.get_seg_model(cfg)


def normalize_checkpoint_key(key):
    for prefix in ('module.model.', 'model.', 'module.'):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'checkpoint not found: {}'.format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint does not contain a state dict')

    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    for key, value in checkpoint.items():
        model_key = normalize_checkpoint_key(key)
        if model_key not in model_state:
            continue
        if model_state[model_key].shape != value.shape:
            shape_mismatch.append(model_key)
            continue
        loaded[model_key] = value

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    missing = [key for key in model_state if key not in loaded]
    print('Checkpoint: {}'.format(checkpoint_path))
    print('Loaded state entries: {} / {}'.format(
        len(loaded), len(model_state)))
    if missing:
        print('Missing state entries: {}'.format(len(missing)))
    if shape_mismatch:
        print('Shape-mismatched entries: {}'.format(shape_mismatch))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def find_image_files(image_dir):
    image_paths = []
    for directory, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(directory, filename))
    image_paths.sort()
    if not image_paths:
        raise RuntimeError(
            'no supported images found in {}'.format(image_dir))
    return image_paths


def preprocess_image(image_path, height, width, mean, std):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('failed to read image: {}'.format(image_path))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)[:, :, ::-1] / 255.0
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    return np.ascontiguousarray(image)


def select_logits(outputs, output_index):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (list, tuple)):
        try:
            logits = outputs[output_index]
        except IndexError as error:
            raise IndexError(
                'output index {} is invalid for {} model outputs'.format(
                    output_index, len(outputs))) from error
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                'selected model output is not a tensor: {}'.format(
                    type(logits)))
        return logits
    raise TypeError('unsupported model output type: {}'.format(type(outputs)))


def inference_with_postprocess(model, input_tensor, output_index,
                               align_corners):
    logits = select_logits(model(input_tensor), output_index)
    logits = F.interpolate(
        logits,
        size=input_tensor.shape[-2:],
        mode='bilinear',
        align_corners=align_corners)
    return torch.argmax(logits, dim=1)


def select_batch(image_paths, batch_index, batch_size):
    start = batch_index * batch_size
    return [
        image_paths[(start + offset) % len(image_paths)]
        for offset in range(batch_size)]


def end_to_end_inference(model, batch_paths, height, width, mean, std,
                         dtype, device, output_index, align_corners):
    images = [
        preprocess_image(path, height, width, mean, std)
        for path in batch_paths]
    cpu_batch = np.stack(images, axis=0)
    input_tensor = torch.from_numpy(cpu_batch).to(
        device=device, dtype=dtype, non_blocking=False)
    return inference_with_postprocess(
        model, input_tensor, output_index, align_corners).cpu()


def benchmark(model, image_paths, args, mean, std, dtype, device,
              align_corners):
    with torch.inference_mode():
        for batch_index in range(args.warmup):
            batch_paths = select_batch(
                image_paths, batch_index, args.batch_size)
            end_to_end_inference(
                model, batch_paths, args.height, args.width,
                mean, std, dtype, device, args.output_index, align_corners)
        synchronize(device)

        results = []
        for repeat_index in range(args.repeats):
            synchronize(device)
            start = time.perf_counter()
            for batch_index in range(args.iterations):
                dataset_index = (
                    repeat_index * args.iterations + batch_index)
                batch_paths = select_batch(
                    image_paths, dataset_index, args.batch_size)
                end_to_end_inference(
                    model, batch_paths, args.height, args.width,
                    mean, std, dtype, device, args.output_index,
                    align_corners)
            synchronize(device)
            elapsed = time.perf_counter() - start

            latency_ms = elapsed * 1000.0 / args.iterations
            fps = args.batch_size * args.iterations / elapsed
            results.append((latency_ms, fps))
    return results


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    apply_config_defaults(args, cfg)
    validate_args(args)
    device = resolve_device(args.device)
    if args.fp16 and device.type != 'cuda':
        raise ValueError('--fp16 is supported only with CUDA')

    torch.backends.cudnn.benchmark = device.type == 'cuda'
    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        model.half()
    image_paths = find_image_files(args.image_dir)
    mean = np.asarray(args.mean, dtype=np.float32).reshape((1, 1, 3))
    std = np.asarray(args.std, dtype=np.float32).reshape((1, 1, 3))
    align_corners = bool(cfg.MODEL.ALIGN_CORNERS)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)

    sample_paths = select_batch(image_paths, 0, args.batch_size)
    with torch.inference_mode():
        prediction = end_to_end_inference(
            model, sample_paths, args.height, args.width,
            mean, std, dtype, device, args.output_index, align_corners)
    synchronize(device)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(
            'expected a tensor prediction, got {}'.format(type(prediction)))

    print('Model: {}'.format(cfg.MODEL.NAME))
    print('Config: {}'.format(os.path.abspath(args.cfg)))
    print('Device: {} ({})'.format(device, dtype))
    if device.type == 'cuda':
        print('GPU: {}'.format(torch.cuda.get_device_name(device)))
    print('Image directory: {}'.format(os.path.abspath(args.image_dir)))
    print('Images found: {}'.format(len(image_paths)))
    print('Input shape: {}'.format(
        (args.batch_size, 3, args.height, args.width)))
    print('Prediction shape: {}'.format(tuple(prediction.shape)))
    print('Output index: {}'.format(args.output_index))
    print('Timing scope: read/decode + resize/normalize + H2D + forward '
          '+ output selection + upsample/argmax + D2H')
    print('Parameters: {:,} ({:.6f} M)'.format(
        total_params, total_params / 1e6))
    print('Trainable parameters: {:,} ({:.6f} M)'.format(
        trainable_params, trainable_params / 1e6))
    print('Warmup / iterations / repeats: {} / {} / {}'.format(
        args.warmup, args.iterations, args.repeats))

    results = benchmark(
        model, image_paths, args, mean, std, dtype, device,
        align_corners)
    latencies = [item[0] for item in results]
    fps_values = [item[1] for item in results]
    for index, (latency, fps) in enumerate(results, start=1):
        print('Repeat {}: latency={:.3f} ms/batch, FPS={:.3f} images/s'.format(
            index, latency, fps))

    print('Average latency: {:.3f} ms/batch'.format(
        statistics.mean(latencies)))
    print('Average FPS: {:.3f} images/s'.format(
        statistics.mean(fps_values)))
    if len(results) > 1:
        print('FPS std: {:.3f}'.format(statistics.stdev(fps_values)))


if __name__ == '__main__':
    main()
