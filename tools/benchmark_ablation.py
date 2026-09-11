#!/usr/bin/env python3
"""Benchmark DDRNet-23-Slim + EMA under one fixed protocol.

Fixed protocol
--------------
* Input: 1 x 3 x 640 x 640, FP32
* Device: the first visible CUDA device (use CUDA_VISIBLE_DEVICES to select it)
* Warm-up: 100 forward passes
* Timing: 500 forward passes, batch size 1
* Synchronization: torch.cuda.synchronize() immediately before and after every
  timed forward pass
* Complexity: THOP MACs converted with 1 MAC = 2 FLOPs
* Inference graph: augment=False, excluding the training-only auxiliary head
* Output: benchmark_results.csv

By default, the checkpoint at output/infrared_images/test/best.pth is loaded
before the latency/FPS test. Use --checkpoint to select a different file.

Example:
    python tools/benchmark_ablation.py

    python tools/benchmark_ablation.py \
        --checkpoint path/to/ddrnet23_slim_ema.pth
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import _init_paths  # noqa: F401  (adds lib/ to sys.path)
from models.ddrnet_23_slim import BasicBlock, DualResNet


INPUT_SHAPE = (1, 3, 640, 640)
NUM_CLASSES = 10
EMA_FACTOR = 8
WARMUP_RUNS = 100
TIMED_RUNS = 500
CSV_PATH = Path("benchmark_results.csv")
DEFAULT_CHECKPOINT = Path("output/infrared_images/test/best.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DDRNet-23-Slim + EMA. Input size, batch size, warm-up "
            "count, timed runs, precision and CSV path are fixed."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="checkpoint path (default: %(default)s)",
    )
    return parser.parse_args()


def build_model() -> torch.nn.Module:
    """Build the repository's DDRNet-23-Slim + EMA inference graph."""
    return DualResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        planes=32,
        spp_planes=128,
        head_planes=64,
        augment=False,
        use_ema=True,
        ema_factor=EMA_FACTOR,
    )


def unwrap_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping.")

    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = state.get(key)
        if isinstance(value, dict):
            state = value
            break

    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("No valid model state_dict was found in the checkpoint.")
    return state


def key_candidates(key: str) -> List[str]:
    """Return key variants for raw, DataParallel and FullModel checkpoints."""
    candidates = [key]
    current = key
    prefixes = ("module.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                candidates.append(current)
                changed = True
                break
    return candidates


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))

    raw_checkpoint = torch.load(str(path), map_location="cpu")
    raw_state = unwrap_state_dict(raw_checkpoint)
    model_state = model.state_dict()
    matched: Dict[str, torch.Tensor] = {}

    for raw_key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        for candidate in key_candidates(raw_key):
            if candidate in model_state and value.shape == model_state[candidate].shape:
                matched[candidate] = value
                break

    if not matched:
        raise RuntimeError(
            "No checkpoint tensors matched the model for: {}".format(path)
        )

    incompatible = model.load_state_dict(matched, strict=False)
    print(
        "  Loaded checkpoint: {} (matched {}, missing {}, unexpected {})".format(
            path,
            len(matched),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    )


def count_complexity(
    model: torch.nn.Module, input_tensor: torch.Tensor
) -> Tuple[float, float]:
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError(
            "THOP is required for complexity measurement. Install it with: "
            "pip install thop"
        ) from exc

    params = sum(parameter.numel() for parameter in model.parameters()) / 1e6

    # THOP reports multiply-accumulate operations (MACs). Keep the same
    # arithmetic convention as the PIDNet reference benchmark.
    raw_macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    gflops = (2.0 * float(raw_macs)) / 1e9
    return params, gflops


def measure_latency(
    model: torch.nn.Module, input_tensor: torch.Tensor
) -> Tuple[float, float]:
    model.eval()
    elapsed_seconds: List[float] = []

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)
        torch.cuda.synchronize()

        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(input_tensor)
            torch.cuda.synchronize()
            elapsed_seconds.append(time.perf_counter() - start)

    latency_ms = (sum(elapsed_seconds) / TIMED_RUNS) * 1000.0
    fps = 1000.0 / latency_ms
    return latency_ms, fps


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = ("Model", "Params(M)", "GFLOPs", "Latency(ms)", "FPS")
    formatted = [
        (
            str(row["Model"]),
            "{:.4f}".format(row["Params(M)"]),
            "{:.3f}".format(row["GFLOPs"]),
            "{:.3f}".format(row["Latency(ms)"]),
            "{:.2f}".format(row["FPS"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in formatted))
        for index in range(len(headers))
    ]

    def line(values: Tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    print("\n" + line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print(line(row))


def save_csv(rows: List[Dict[str, object]]) -> None:
    fieldnames = ["Model", "Params(M)", "GFLOPs", "Latency(ms)", "FPS"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "Params(M)": "{:.4f}".format(row["Params(M)"]),
                    "GFLOPs": "{:.6f}".format(row["GFLOPs"]),
                    "Latency(ms)": "{:.6f}".format(row["Latency(ms)"]),
                    "FPS": "{:.6f}".format(row["FPS"]),
                }
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for latency and FPS testing.")

    torch.manual_seed(304)
    torch.cuda.manual_seed_all(304)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    input_tensor = torch.randn(INPUT_SHAPE, device=device, dtype=torch.float32)

    print("PyTorch: {}".format(torch.__version__))
    print("CUDA runtime: {}".format(torch.version.cuda))
    print("GPU: {}".format(torch.cuda.get_device_name(device)))
    print("Input: {}, FP32, batch size 1".format(INPUT_SHAPE))
    print("Warm-up: {}; timed runs: {}".format(WARMUP_RUNS, TIMED_RUNS))
    print("GFLOPs: THOP MACs x 2 (1 MAC = 2 FLOPs)")

    print("\nDDRNet-23-Slim")
    model = build_model()
    load_checkpoint(model, args.checkpoint)

    model.eval().to(device)
    params_m, gflops = count_complexity(model, input_tensor)
    latency_ms, fps = measure_latency(model, input_tensor)
    results: List[Dict[str, object]] = [
        {
            "Model": "DDRNet-23-Slim",
            "Params(M)": params_m,
            "GFLOPs": gflops,
            "Latency(ms)": latency_ms,
            "FPS": fps,
        }
    ]

    print_table(results)
    save_csv(results)
    print("\nSaved CSV to: {}".format(CSV_PATH.resolve()))


if __name__ == "__main__":
    main()
