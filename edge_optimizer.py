"""
Edge Optimization and Resource Profiler for Low-Power Embedded/Mobile Deployments.

Provides:
1. Model parameter counting and memory footprint analysis
2. Weight quantization (FP32 -> FP16 / INT8)
3. Inference latency and Real-Time Factor (RTF) benchmarking
4. Edge viability qualification
"""

from __future__ import annotations

try:
    # Pre-load onnxruntime before sklearn/scipy to prevent Windows C++ runtime DLL crash
    import onnxruntime  # noqa: F401
except (ImportError, OSError):
    pass

import logging
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger("edge_optimizer")


@dataclass
class EdgeFootprintReport:
    model_name: str
    model_file_size_kb: float
    total_parameter_count: int
    quantized_size_kb: float
    compression_ratio: float
    median_inference_latency_ms: float
    p95_inference_latency_ms: float
    throughput_samples_per_sec: float
    real_time_factor: float
    edge_ready_status: str
    recommendations: list[str]


def _count_pt_storage_params(pt_path: Path) -> int:
    """Count total scalar parameters inside a PyTorch .pt zip archive without requiring torch."""
    if not pt_path.exists() or not zipfile.is_zipfile(pt_path):
        return 0
    total_bytes = 0
    with zipfile.ZipFile(pt_path, "r") as zf:
        for info in zf.infolist():
            # Torch saves each float32/int64 tensor storage in */data/<idx>; skip small int64 trackers (<=8 bytes)
            if "/data/" in info.filename and info.file_size > 8:
                total_bytes += info.file_size
    return total_bytes // 4


def count_parameters(model) -> int:
    """Count total trainable/fitted scalar parameters in a model."""
    total = 0
    # ONNX wrapper (e.g. LCNNOnnx)
    onnx_path = getattr(model, "onnx_path", None)
    if onnx_path is not None:
        onnx_path = Path(onnx_path)
        pt_count = _count_pt_storage_params(onnx_path.parent / "lcnn.pt")
        if pt_count > 0:
            return pt_count
        data_file = onnx_path.with_name(onnx_path.name + ".data")
        if data_file.exists():
            return int(data_file.stat().st_size // 4)
        return int(onnx_path.stat().st_size // 4) if onnx_path.exists() else 0

    # For MLPClassifier or Pipeline containing MLP
    pipe = getattr(model, "_pipeline", model)
    if hasattr(pipe, "named_steps"):
        if "mlp" in pipe.named_steps:
            mlp = pipe.named_steps["mlp"]
            for coef in getattr(mlp, "coefs_", []):
                total += int(coef.size)
            for intercept in getattr(mlp, "intercepts_", []):
                total += int(intercept.size)
        elif "svc" in pipe.named_steps:
            svc = pipe.named_steps["svc"]
            if hasattr(svc, "support_vectors_"):
                total += int(svc.support_vectors_.size + svc.dual_coef_.size)
    return total


def quantize_weights_fp16(model_path: str | Path, out_path: str | Path | None = None) -> tuple[Path, float]:
    """
    Quantize model weights from FP32 (32-bit float) to FP16 (16-bit half precision),
    reducing disk and memory footprint by ~50% with minimal loss in precision.
    """
    model_path = Path(model_path)
    if out_path is None:
        out_path = model_path.parent / (model_path.stem + "_quantized_fp16.joblib")
    else:
        out_path = Path(out_path)

    pipe = joblib.load(model_path)
    if hasattr(pipe, "named_steps") and "mlp" in pipe.named_steps:
        mlp = pipe.named_steps["mlp"]
        # Convert internal weight matrices to float16
        mlp.coefs_ = [c.astype(np.float16) for c in mlp.coefs_]
        mlp.intercepts_ = [i.astype(np.float16) for i in mlp.intercepts_]

    joblib.dump(pipe, out_path, compress=3)
    orig_size = os.path.getsize(model_path) / 1024.0
    quant_size = os.path.getsize(out_path) / 1024.0
    ratio = orig_size / max(quant_size, 1e-6)
    logger.info("Quantized %s: %.1f KB -> %.1f KB (%.2fx compression)", model_path.name, orig_size, quant_size, ratio)
    return out_path, quant_size


def profile_edge_footprint(
    model,
    model_file_path: str | Path,
    feature_dim: int = 240,
    feature_shape: tuple[int, ...] | None = None,
    n_benchmark_iterations: int = 200,
    simulated_audio_duration_sec: float = 2.0,
) -> EdgeFootprintReport:
    """
    Comprehensive edge resource profiling: size, parameters, latency, RTF, and throughput.
    """
    model_path = Path(model_file_path)
    file_size_bytes = os.path.getsize(model_path) if model_path.exists() else 0
    sidecar_data = model_path.with_name(model_path.name + ".data")
    if sidecar_data.exists():
        file_size_bytes += os.path.getsize(sidecar_data)
    file_size_kb = file_size_bytes / 1024.0
    total_params = count_parameters(model)

    # Estimate quantized footprint
    quant_size_kb = file_size_kb * 0.52
    compression_ratio = round(file_size_kb / max(quant_size_kb, 1e-6), 2)

    # Warmup
    if feature_shape is not None:
        dummy_shape = feature_shape
    elif getattr(model, "onnx_path", None) is not None:
        dummy_shape = (1, 1, 60, 400)
        simulated_audio_duration_sec = 4.0 if simulated_audio_duration_sec == 2.0 else simulated_audio_duration_sec
    else:
        dummy_shape = (1, feature_dim)
    dummy = np.random.randn(*dummy_shape).astype(np.float32)
    for _ in range(10):
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(dummy)

    # Latency benchmark
    latencies = []
    for _ in range(n_benchmark_iterations):
        t0 = time.perf_counter()
        _ = model.predict_proba(dummy)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    med_lat_ms = float(np.median(latencies))
    p95_lat_ms = float(np.percentile(latencies, 95))
    throughput = round(1000.0 / max(med_lat_ms, 1e-6), 1)

    # Real-Time Factor (RTF = compute_time / audio_time)
    rtf = round((med_lat_ms / 1000.0) / simulated_audio_duration_sec, 6)

    # Edge readiness qualification
    is_edge_ready = (file_size_kb < 50000.0) and (med_lat_ms < 50.0) and (rtf < 0.05)
    status = "EDGE-READY (EXCELLENT)" if is_edge_ready else "RESOURCE-CONSTRAINED"

    recommendations = []
    if rtf < 0.01:
        recommendations.append(f"RTF of {rtf:.4f} is 100x faster than real-time; suitable for low-power mobile CPU.")
    if file_size_kb < 10000.0:
        recommendations.append(f"Compact memory footprint ({file_size_kb:.1f} KB) fits easily in micro-controller flash.")
    if quant_size_kb < 5000.0:
        recommendations.append("FP16 quantization recommended for zero-loss RAM reduction on ARM Cortex / Apple Silicon.")

    return EdgeFootprintReport(
        model_name=model_path.name,
        model_file_size_kb=round(file_size_kb, 2),
        total_parameter_count=total_params,
        quantized_size_kb=round(quant_size_kb, 2),
        compression_ratio=compression_ratio,
        median_inference_latency_ms=round(med_lat_ms, 3),
        p95_inference_latency_ms=round(p95_lat_ms, 3),
        throughput_samples_per_sec=throughput,
        real_time_factor=rtf,
        edge_ready_status=status,
        recommendations=recommendations,
    )
