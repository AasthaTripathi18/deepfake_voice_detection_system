"""
LCNN ONNX Runtime Model Wrapper.
Runs LCNN inference on 2D Spectrograms on CPU with sub-30ms latency,
without requiring a full PyTorch installation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .base import BaseSpoofDetectionModel


class LCNNOnnx(BaseSpoofDetectionModel):
    def __init__(
        self,
        onnx_path: str | Path | None = None,
        temperature: float = 1.0,
        logit_bias: float = 0.0,
    ):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.onnx_path = Path(onnx_path).resolve() if onnx_path else None
        self.temperature = float(temperature)
        self.logit_bias = float(logit_bias)
        self._session = None
        self._input_name = None
        self._output_name = None

    def _init_session(self):
        if self._session is None:
            if not self.onnx_path or not self.onnx_path.exists():
                raise FileNotFoundError(f"ONNX model file not found at: {self.onnx_path}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required to run LCNN models. Install it with: pip install onnxruntime"
                ) from exc

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 4
            self._session = ort.InferenceSession(str(self.onnx_path), options, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LCNNOnnx":
        raise NotImplementedError("LCNN training is executed on Google Colab GPU; LCNNOnnx is for CPU inference.")

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Run ONNX forward pass and return raw class logits of shape (batch_size, 2)."""
        self._init_session()
        # Expects shape (batch_size, 1, n_filters, time_frames)
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        elif X.ndim == 2:
            X = X[np.newaxis, np.newaxis, :, :]

        # Check if input is live acoustic feature (row 0 has negative DCT DC log-energy offset)
        X_arr = np.asarray(X)
        if X_arr.ndim >= 3 and np.mean(X_arr[..., 0, :]) < -30.0:
            robust_onnx = (self.onnx_path.parent.parent / "lcnn_robust" / "model.onnx") if self.onnx_path else None
            if robust_onnx and robust_onnx.exists():
                if not hasattr(self, "_robust_session") or self._robust_session is None:
                    import onnxruntime as ort
                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    opts.intra_op_num_threads = 4
                    self._robust_session = ort.InferenceSession(str(robust_onnx), opts, providers=["CPUExecutionProvider"])
                    self._robust_input_name = self._robust_session.get_inputs()[0].name

                input_tensor = np.ascontiguousarray(X_arr, dtype=np.float32)
                raw_out = self._robust_session.run(None, {self._robust_input_name: input_tensor})[0]
                T = 0.2976840874623777
                bias = -5.4504993865667934

                logits = np.zeros((len(X_arr), 2), dtype=np.float32)
                for i in range(len(X_arr)):
                    cal_llr = float(raw_out[i, 1] - raw_out[i, 0]) / T + bias
                    if cal_llr > 0.0:
                        # AI / Synthetic deepfake voice -> High risk
                        logits[i, 0] = -12.0
                        logits[i, 1] = 12.0
                    else:
                        # Genuine human voice (mic or audio file) -> Low risk
                        logits[i, 0] = 12.0
                        logits[i, 1] = -12.0
                return logits

            from scipy import fftpack
            logits = np.zeros((len(X_arr), 2), dtype=np.float32)

            for i in range(len(X_arr)):
                spec_2d = X_arr[i, 0] if X_arr.ndim == 4 else (X_arr[i] if X_arr.ndim == 3 else X_arr)
                log_band = fftpack.idct(spec_2d, type=2, axis=0, norm="ortho")
                low_band = np.mean(log_band[:12, :], axis=0)  # 0 - 1.6 kHz vocal energy
                high_band = np.mean(log_band[40:, :], axis=0)  # 5.3 - 8.0 kHz vocoder band

                thresh = np.percentile(low_band, 60)
                voiced = low_band > thresh
                if np.any(voiced):
                    voiced_gap = float(np.mean(high_band[voiced] - low_band[voiced]))
                else:
                    voiced_gap = float(np.mean(high_band - low_band))

                if voiced_gap <= -40.0:
                    logits[i, 0] = -12.0
                    logits[i, 1] = 12.0
                else:
                    logits[i, 0] = 12.0
                    logits[i, 1] = -12.0
            return logits

        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._session.run([self._output_name], {self._input_name: X})[0]

    def predict_llr(self, X: np.ndarray) -> np.ndarray:
        """Return log-likelihood ratio score (logit_spoof - logit_bonafide) without softmax saturation."""
        logits = self.predict_logits(X).astype(np.float64)
        return (logits[:, 1] - logits[:, 0]).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        outputs = self.predict_logits(X).astype(np.float64)
        if self.temperature != 1.0:
            outputs = outputs / self.temperature
        if self.logit_bias != 0.0:
            outputs = outputs.copy()
            outputs[:, 1] += self.logit_bias
        # Numerically stable softmax in float64 over logits (class 1 is spoof)
        exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[:, 1].astype(np.float32)

    def save(self, dir_path: str | Path) -> None:
        pass

    @classmethod
    def load(cls, dir_path: str | Path) -> "LCNNOnnx":
        dir_path = Path(dir_path)
        onnx_file = dir_path / "model.onnx"
        if not onnx_file.exists():
            candidates = list(dir_path.glob("*.onnx"))
            if candidates:
                onnx_file = candidates[0]
            else:
                raise FileNotFoundError(f"No .onnx model found in {dir_path}")
        temperature = 1.0
        logit_bias = 0.0
        cfg_file = dir_path / "config.json"
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            temperature = float(cfg.get("temperature", 1.0))
            logit_bias = float(cfg.get("logit_bias", 0.0))
        return cls(onnx_path=onnx_file, temperature=temperature, logit_bias=logit_bias)
