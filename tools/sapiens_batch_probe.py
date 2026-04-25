from __future__ import annotations

import gc
import sys
import time

import torch

from pipeline.pose.sapiens_pose import PoseBackend, PoseEstimator, PoseEstimatorConfig

def _probe_one(est: PoseEstimator, batch: int, warmup: int = 1) -> tuple[float, float]:
    assert est.backend is not None
    device = est.device
    in_h, in_w = est.cfg.input_hw

    x = torch.randn(batch, 3, in_h, in_w, device=device, dtype=torch.float32)
    x = (x - est._mean) / est._std
    x = x.to(est._dtype)

    for _ in range(warmup):
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=est._dtype, enabled=device.type == "cuda",
        ):
            y = est.backend(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        del y

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=est._dtype, enabled=device.type == "cuda",
    ):
        y = est.backend(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del x, y
    gc.collect()
    torch.cuda.empty_cache()
    return peak, dt

def main() -> None:
    print("Loading Sapiens-1B torchscript...")
    est = PoseEstimator(PoseEstimatorConfig(
        backend=PoseBackend.SAPIENS,
        device="cuda",
        dtype="bfloat16",
    ))
    if est.mode != "sapiens":
        print(f"[probe] ERROR: expected sapiens mode, got {est.mode}")
        sys.exit(1)
    assert est.backend is not None

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[probe] device total VRAM: {total_gb:.1f} GB")

    candidates = [1, 2, 4, 8, 16, 32, 64, 80, 88, 96]
    max_ok = 0
    last_peak = 0.0
    last_dt = 0.0
                                                                      
    try:
        _probe_one(est, 8, warmup=2)
    except RuntimeError:
        pass
    for b in candidates:
        try:
            peak, dt = _probe_one(est, b)
        except (RuntimeError,) as e:
            msg = str(e).splitlines()[0]
            if "out of memory" in msg.lower() or isinstance(
                e, getattr(torch.cuda, "OutOfMemoryError", RuntimeError),
            ):
                print(f"[probe] batch={b} OOM: {msg[:120]}")
                gc.collect()
                torch.cuda.empty_cache()
                break
            raise
        throughput = b / dt if dt > 0 else 0.0
        print(
            f"[probe] batch={b:>3d}  peak={peak:6.2f} GB  forward={dt*1000:6.1f} ms  "
            f"throughput={throughput:6.1f} frames/s"
        )
        max_ok = b
        last_peak = peak
        last_dt = dt

    print("")
    if max_ok:
        print(
            f"[probe] max safe batch: {max_ok}  "
            f"(peak {last_peak:.2f} GB, forward {last_dt*1000:.1f} ms)"
        )
    else:
        print("[probe] no batch succeeded (initial load failed?)")

if __name__ == "__main__":
    main()
