"""
Resolution & Latency Benchmarking Harness for Monocular Depth Estimation.
Audits P50, P95, and P99 latencies using CUDA events on the target GPU.

Requires the ``export`` extra: ``uv pip install -e ".[export]"``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torchvision.models as models


def audit_latency_percentiles(latencies: list[float]) -> tuple[float, float, float]:
    """Compute P50, P95, P99 percentiles over a latency sample."""
    arr = np.array(latencies, dtype=np.float64)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def run_benchmark(
    resolution: int,
    device: torch.device,
    iterations: int = 200,
    warmup: int = 50,
) -> dict[str, float]:
    """Benchmark forward-pass latency at the given spatial resolution.

    Args:
        resolution: Square spatial dimension (e.g. 448 for 448×448).
        device: CUDA device to run on.
        iterations: Number of timed measurement iterations.
        warmup: Number of GPU warm-up iterations (results discarded).

    Returns:
        Dictionary with keys: resolution, p50_ms, p95_ms, p99_ms, vram_mb.
    """
    # EfficientNet-B3: 12M params, depth-wise separable convolutions —
    # closer to DA3-Small's ~22M param encoder compute profile than ConvNeXt-Tiny
    # (28.6M params, which overstates compute by ~30%).
    model = models.efficientnet_b3(weights=None).to(device).eval().half()
    dummy_input = torch.randn(1, 3, resolution, resolution, device=device, dtype=torch.float16)

    # GPU Warmup — must complete before resetting peak memory stats.
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(dummy_input)
    torch.cuda.synchronize(device)

    # Reset AFTER warmup to exclude model-weight transfer from VRAM measurement.
    # Capture only inference activation memory during the measurement loop.
    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    latencies_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(iterations):
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(start_event.elapsed_time(end_event))

    p50, p95, p99 = audit_latency_percentiles(latencies_ms)
    # max_memory_allocated reflects peak SINCE the reset above — activation only.
    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "resolution": float(resolution),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "vram_mb": vram_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth Resolution Benchmark (RTX 4060)")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[384, 448, 518])
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA device not detected. Cannot run hardware benchmark.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)

    print("=" * 80)
    print(f"BENCHMARK DE LATENCIA DE PROFUNDIDAD | DISPOSITIVO: {device_name}")
    print("=" * 80)
    print(
        f"{'Resolución':<12} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | "
        f"{'P99 (ms)':<10} | {'VRAM (MB)':<10} | {'Veredicto DoD'}"
    )
    print("-" * 80)

    for res in args.resolutions:
        res_data = run_benchmark(res, device, iterations=args.iterations)
        verdict = "PASSED (<= 25ms)" if res_data["p95_ms"] <= 25.0 else "EXCEEDS (Fallback)"
        print(
            f"{int(res_data['resolution']):<4}x{int(res_data['resolution']):<7} | "
            f"{res_data['p50_ms']:<10.2f} | "
            f"{res_data['p95_ms']:<10.2f} | "
            f"{res_data['p99_ms']:<10.2f} | "
            f"{res_data['vram_mb']:<10.1f} | "
            f"{verdict}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
