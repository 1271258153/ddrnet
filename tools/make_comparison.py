# -*- coding: utf-8 -*-
"""生成对比图：原图 | 真值 | 预测 | 叠加，或仅生成叠加图

用法:
    python -B tools/make_comparison.py \
        --cfg experiments/cityscapes/test.yaml \
        --pred-dir output/infrared_images/evaluation_result/test_results \
        --out-dir output/infrared_images/comparison_images

    # 仅生成预测叠加图
    python -B tools/make_comparison.py \
        --cfg experiments/cityscapes/test.yaml \
        --pred-dir output/infrared_images/evaluation_result/test_results \
        --out-dir output/infrared_images/overlay_images \
        --image example.png \
        --overlay-only
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

import _init_paths
import datasets
from config import config
from config import update_config

# 红外 10 类配色 [R, G, B]，与 infrared_images.py 中 color_list 一致
COLOR_LIST = [
    [0, 0, 0],         # 0 _background_
    [220, 20, 60],     # 1 BL_Device
    [30, 144, 255],    # 2 CC_Server
    [50, 205, 50],     # 3 DP_Server
    [255, 165, 0],     # 4 KDVideo_Device
    [148, 0, 211],     # 5 KVM_Switcher
    [0, 206, 209],     # 6 SP_Cloud
    [255, 215, 0],     # 7 VPN_Gateway
    [255, 105, 180],   # 8 WEB_Firewall
    [128, 128, 128],   # 9 YP_Server
]


def label2color(label):
    color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
    for i, v in enumerate(COLOR_LIST):
        color_map[label == i] = v
    return color_map


def overlay(image, color_mask, alpha=0.5):
    return (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)


def hconcat(imgs, gap=4, gap_color=(255, 255, 255)):
    h = max(im.shape[0] for im in imgs)
    w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    canvas = np.full((h, w, 3), gap_color, dtype=np.uint8)
    x = 0
    for im in imgs:
        canvas[:im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Generate comparison or overlay images')
    parser.add_argument('--cfg', default='experiments/cityscapes/test.yaml', type=str)
    parser.add_argument('--pred-dir', default='output/infrared_images/evaluation_result',
                        type=str, help='彩色预测 mask 目录')
    parser.add_argument('--out-dir', default='output/infrared_images/comparison_images',
                        type=str, help='对比图输出目录')
    parser.add_argument('--alpha', default=0.5, type=float, help='叠加图中预测的权重')
    parser.add_argument('--overlay-only', action='store_true',
                        help='仅保存原图与预测 mask 的叠加图')
    parser.add_argument('--image', type=str,
                        help='仅处理指定图片，支持文件名、无扩展名名称或列表中的相对路径')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    root = config.DATASET.ROOT
    list_path = os.path.join(root, config.DATASET.TEST_SET)

    with open(list_path) as f:
        items = [line.strip().split() for line in f if line.strip()]

    if args.image:
        target = os.path.normpath(args.image)
        target_basename = os.path.basename(target)
        target_stem = os.path.splitext(target_basename)[0]

        def item_names(item):
            paths = [os.path.normpath(path) for path in item[:2]]
            basenames = [os.path.basename(path) for path in paths]
            stems = [os.path.splitext(name)[0] for name in basenames]
            return paths, basenames, stems

        exact = [item for item in items if target in item_names(item)[0]]
        same_basename = [item for item in items
                         if target_basename in item_names(item)[1]]
        same_stem = [item for item in items if target_stem in item_names(item)[2]]
        items = exact or same_basename or same_stem

        if not items:
            parser.error(f'未在测试列表中找到图片: {args.image}')
        if len(items) > 1:
            parser.error(f'图片名称不唯一，请使用列表中的相对路径: {args.image}')

    os.makedirs(args.out_dir, exist_ok=True)

    n_ok, n_miss = 0, 0
    for item in items:
        img_rel, lbl_rel = item[0], item[1]
        name = os.path.splitext(os.path.basename(lbl_rel))[0]
        pred_path = os.path.join(args.pred_dir, name + '.png')
        if not os.path.exists(pred_path):
            n_miss += 1
            continue

        image = cv2.imread(os.path.join(root, 'infrared_images', img_rel), cv2.IMREAD_COLOR)
        pred = cv2.imread(pred_path, cv2.IMREAD_COLOR)

        h, w = image.shape[:2]
        # 预测是 resize 到 crop_size 的，统一缩回原图尺寸
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        pred_color = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ov = overlay(image_rgb, pred_color, alpha=args.alpha)
        if args.overlay_only:
            output = ov
        else:
            label = cv2.imread(os.path.join(root, 'infrared_images', lbl_rel),
                               cv2.IMREAD_GRAYSCALE)
            gt_color = label2color(label)
            output = hconcat([image_rgb, gt_color, pred_color, ov])

        Image.fromarray(output).save(os.path.join(args.out_dir, name + '.png'))
        n_ok += 1

    print(f'done: {n_ok} saved to {args.out_dir}, {n_miss} missing predictions')


if __name__ == '__main__':
    main()
