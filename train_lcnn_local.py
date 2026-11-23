#!/usr/bin/env python
"""
Local CMD Training, Calibration, Evaluation & ONNX Export Pipeline for LCNN-9.

Reads official ASVspoof 2019 LA audio directly from `archive (15).zip` (or an
extracted LA/ folder) with zero cloud upload required, caches features as
memory-mapped .npy files to keep RAM usage low, trains LCNN-9 (in-graph CMVN +
MFM activations) with deepcopy best-checkpointing and early stopping, fits
monotonic DEV temperature/bias calibration, verifies all 71,237 EVAL clips +
noise/telephony robustness, and exports a self-contained ONNX model to
models/lcnn_robust/ (never modifying models/lcnn_clean/).
"""

from __future__ import annotations

try:
    import onnxruntime  # noqa: F401
except (ImportError, OSError):
    pass

import argparse
import copy
import io
import json
import logging
import os
import random
import sys
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset.asvspoof2019 import _parse_protocol_line  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    classification_metrics_at_threshold,
    compute_eer,
    compute_min_tdcf,
    far_frr_at_threshold,
    latency_summary,
)
from src.preprocessing import cleaning, degradations, features as feat_mod  # noqa: E402

logger = logging.getLogger("train_lcnn_local")

PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
SPLIT_SUBDIR = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}

_W_ARCHIVE: str = ""
_W_IS_ZIP: bool = False
_W_ZIP: zipfile.ZipFile | None = None
_W_MEMBER_MAP: dict[str, str] = {}
_W_AUG_FRAC: float = 0.0
_W_SNR_DB: float | None = None
_W_BANDPASS: bool = False


def _init_extract_worker(
    archive_path: str,
    is_zip: bool,
    member_map: dict[str, str],
    aug_frac: float,
    snr_db: float | None,
    apply_bandpass: bool,
):
    global _W_ARCHIVE, _W_IS_ZIP, _W_ZIP, _W_MEMBER_MAP, _W_AUG_FRAC, _W_SNR_DB, _W_BANDPASS
    _W_ARCHIVE = archive_path
    _W_IS_ZIP = is_zip
    _W_MEMBER_MAP = member_map
    _W_AUG_FRAC = aug_frac
    _W_SNR_DB = snr_db
    _W_BANDPASS = apply_bandpass
    _W_ZIP = zipfile.ZipFile(archive_path, "r") if is_zip else None


def _extract_one(item: tuple[int, str, str]) -> tuple[int, np.ndarray]:
    idx, file_id, file_ref = item
    if _W_IS_ZIP:
        assert _W_ZIP is not None
        wav, sr = sf.read(io.BytesIO(_W_ZIP.read(_W_MEMBER_MAP[file_id])), dtype="float32")
    else:
        wav, sr = sf.read(file_ref, dtype="float32")

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    wav = cleaning.clean(wav, sr, trim=True, normalize=True, denoise=False)

    if _W_AUG_FRAC > 0.0:
        rng = np.random.default_rng(zlib.crc32(file_id.encode("utf-8")) ^ 0xA5A5)
        if rng.random() < _W_AUG_FRAC:
            mode = rng.choice(["bandpass", "noise", "both"])
            if mode in ("noise", "both"):
                wav = degradations.add_noise_at_snr(wav, float(rng.uniform(10.0, 25.0)), rng=rng)
            if mode in ("bandpass", "both"):
                wav = degradations.bandpass_telephony(wav, sr=sr)

    if _W_SNR_DB is not None:
        rng_eval = np.random.default_rng(zlib.crc32(file_id.encode("utf-8")))
        wav = degradations.add_noise_at_snr(wav, snr_db=_W_SNR_DB, rng=rng_eval)
    if _W_BANDPASS:
        wav = degradations.bandpass_telephony(wav, sr=sr)

    return idx, feat_mod.extract_lfcc_spectrogram_2d(wav, sr=sr)


def _index_archive(archive_path: Path) -> tuple[bool, dict[str, str], dict[str, str]]:
    """Return (is_zip, flac_member_map, protocol_member_map)."""
    is_zip = archive_path.is_file() and archive_path.suffix.lower() == ".zip"
    flac_map: dict[str, str] = {}
    proto_map: dict[str, str] = {}
    if is_zip:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for name in zf.namelist():
                if "ASVspoof2019_LA_" not in name:
                    continue
                if name.endswith(".flac"):
                    flac_map[Path(name).stem] = name
                elif name.endswith(".txt") and "ASVspoof2019.LA.cm." in name:
                    proto_map[Path(name).name] = name
    else:
        for proto_name in PROTOCOL_FILES.values():
            matches = list(archive_path.rglob(proto_name))
            if matches:
                proto_map[proto_name] = str(matches[0])
        for split, sub in SPLIT_SUBDIR.items():
            for p in archive_path.rglob(f"{sub}/flac/*.flac"):
                flac_map[p.stem] = str(p)
    return is_zip, flac_map, proto_map


def _read_protocol_df(archive_path: Path, is_zip: bool, proto_map: dict[str, str], split: str) -> pd.DataFrame:
    proto_name = PROTOCOL_FILES[split]
    if proto_name not in proto_map:
        raise FileNotFoundError(f"Protocol {proto_name} not found under {archive_path}")
    if is_zip:
        with zipfile.ZipFile(archive_path, "r") as zf:
            lines = zf.read(proto_map[proto_name]).decode("utf-8").splitlines()
    else:
        lines = Path(proto_map[proto_name]).read_text(encoding="utf-8").splitlines()

    rows = []
    for raw in lines:
        if not raw.strip():
            continue
        parsed = _parse_protocol_line(raw)
        if parsed is None:
            continue
        speaker_id, file_id, system_id, label = parsed
        rows.append({
            "speaker_id": speaker_id,
            "file_id": file_id,
            "system_id": system_id,
            "label": label,
            "label_int": 0 if label == "bonafide" else 1,
        })
    return pd.DataFrame(rows)


def extract_or_load_split(
    split: str,
    archive_path: Path,
    is_zip: bool,
    flac_map: dict[str, str],
    proto_map: dict[str, str],
    cache_dir: Path,
    aug_fraction: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = _read_protocol_df(archive_path, is_zip, proto_map, split)
    tag = f"{split}_aug{int(round(aug_fraction * 100))}" if aug_fraction > 0 else f"{split}_clean"
    x_path = cache_dir / f"{tag}_X.npy"
    y_path = cache_dir / f"{tag}_y.npy"
    tmp_x = cache_dir / f"{tag}_X.tmp.npy"

    if x_path.exists() and y_path.exists():
        logger.info("Using existing feature cache for %s: %s", tag, x_path)
        return np.load(x_path, mmap_mode="r"), np.load(y_path), df

    n = len(df)
    logger.info("Extracting %d clips for %s (workers=%d, aug=%.2f)...", n, tag, workers, aug_fraction)
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_mem = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.float32, shape=(n, 1, 60, 400))
    y_arr = df["label_int"].to_numpy(dtype=np.int64)

    items = [(i, row.file_id, flac_map[row.file_id]) for i, row in enumerate(df.itertuples(index=False))]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_extract_worker,
        initargs=(str(archive_path), is_zip, flac_map, aug_fraction, None, False),
    ) as pool:
        for done, (idx, spec) in enumerate(pool.map(_extract_one, items, chunksize=48), start=1):
            X_mem[idx] = spec
            if done % 2500 == 0 or done == n:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                logger.info("  [%s] %d / %d (%.1f%%) [%.1f clips/s]", tag, done, n, 100.0 * done / n, rate)

    X_mem.flush()
    del X_mem
    tmp_x.replace(x_path)
    np.save(y_path, y_arr)
    return np.load(x_path, mmap_mode="r"), y_arr, df


def _fit_dev_calibration(dev_logits: np.ndarray, dev_labels: np.ndarray) -> tuple[float, float, float]:
    llr = (dev_logits[:, 1] - dev_logits[:, 0]).astype(np.float64)
    dev_eer, llr_eer_th = compute_eer(llr, dev_labels)
    b_raw = -float(llr_eer_th)
    centered = llr + b_raw
    pos_w = float((dev_labels == 0).sum()) / max(1.0, float((dev_labels == 1).sum()))
    sw = np.where(dev_labels == 1, pos_w, 1.0)

    def _nll(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        z = np.clip(centered / t, -60.0, 60.0)
        return float(np.average(np.logaddexp(0.0, z) - dev_labels * z, weights=sw))

    res = minimize(_nll, x0=np.array([np.log(2.0)]), bounds=[(-2.0, 5.0)], method="L-BFGS-B")
    temp = float(np.exp(res.x[0]))
    logit_bias = b_raw / temp
    return temp, logit_bias, float(dev_eer)


def _apply_calibration(logits: np.ndarray, temp: float, logit_bias: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temp
    scaled[:, 1] += logit_bias
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return (exp_s / np.sum(exp_s, axis=1, keepdims=True))[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", required=True, help="Path to archive (15).zip or extracted LA/ folder")
    parser.add_argument("--model-name", default="lcnn_robust", help="Subfolder name under models/ and results/metrics/")
    parser.add_argument("--cache-dir", default="data/features/lcnn_cache", help="Directory for mmap .npy feature cache")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience on dev EER")
    parser.add_argument("--aug-fraction", type=float, default=0.35, help="Fraction of train clips with noise/bandpass augmentation (0 to disable)")
    parser.add_argument("--workers", type=int, default=6, help="Worker processes for feature extraction")
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 4), help="PyTorch CPU intra-op threads")
    parser.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pt inside model output dir")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness subset evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        import onnx
    except ImportError as exc:
        logger.error(
            "Missing training dependency (%s). Create/activate the training venv and install "
            "CPU torch + onnx (see instructions).", exc,
        )
        return 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)

    class MFM(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, conv: bool = True):
            super().__init__()
            self.out_ch = out_ch
            self.filter = (
                nn.Conv2d(in_ch, 2 * out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
                if conv else nn.Linear(in_ch, 2 * out_ch)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.filter(x)
            a, b = torch.split(x, self.out_ch, dim=1)
            return torch.max(a, b)

    class LCNN9(nn.Module):
        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                MFM(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2, 2),
                MFM(32, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                MFM(32, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(2, 2),
                MFM(48, 48, kernel_size=1),
                nn.BatchNorm2d(48),
                MFM(48, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2),
                MFM(64, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                MFM(64, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(2, 2),
                MFM(48, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                MFM(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Dropout(0.4),
                MFM(32 * 4 * 4, 128, conv=False),
                nn.BatchNorm1d(128),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mu = x.mean(dim=3, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=3, keepdim=True)
            x = (x - mu) / torch.sqrt(var + 1e-5)
            return self.classifier(self.features(x))

    class MmapSpecDataset(Dataset):
        def __init__(self, X_mmap: np.ndarray, y_arr: np.ndarray, augment: bool = False):
            self.X = X_mmap
            self.y = y_arr
            self.augment = augment

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            arr = np.array(self.X[idx], dtype=np.float32, copy=True)
            if self.augment:
                f = int(np.random.randint(0, 9))
                if f > 0:
                    f0 = int(np.random.randint(0, max(1, arr.shape[1] - f)))
                    arr[:, f0:f0 + f, :] = 0.0
                t = int(np.random.randint(0, 33))
                if t > 0:
                    t0 = int(np.random.randint(0, max(1, arr.shape[2] - t)))
                    arr[:, :, t0:t0 + t] = 0.0
            return torch.from_numpy(arr), int(self.y[idx])

    archive_path = Path(args.archive_path)
    cache_dir = ROOT_DIR / args.cache_dir
    out_dir = ROOT_DIR / "models" / args.model_name
    metrics_dir = ROOT_DIR / "results" / "metrics" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Indexing archive: %s ...", archive_path)
    is_zip, flac_map, proto_map = _index_archive(archive_path)
    logger.info("Indexed %d LA .flac entries and %d protocols.", len(flac_map), len(proto_map))

    X_train, y_train, _ = extract_or_load_split("train", archive_path, is_zip, flac_map, proto_map, cache_dir, args.aug_fraction, args.workers)
    X_dev, y_dev, _ = extract_or_load_split("dev", archive_path, is_zip, flac_map, proto_map, cache_dir, 0.0, args.workers)

    train_loader = DataLoader(MmapSpecDataset(X_train, y_train, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(MmapSpecDataset(X_dev, y_dev, augment=False), batch_size=128, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCNN9(num_classes=2).to(device)
    n_bf, n_sp = float((y_train == 0).sum()), float((y_train == 1).sum())
    cls_weights = torch.tensor([np.sqrt(n_sp / max(1.0, n_bf)), 1.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_dev_eer = float("inf")
    best_state = None
    no_improve = 0
    ckpt_last = out_dir / "last_checkpoint.pt"
    ckpt_best = out_dir / "best_checkpoint.pt"

    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck["epoch"]) + 1
        best_dev_eer = float(ck["best_dev_eer"])
        no_improve = int(ck["no_improve"])
        if ckpt_best.exists():
            best_state = torch.load(ckpt_best, map_location="cpu")
        logger.info("Resumed from epoch %d (best_dev_eer=%.3f%%)", start_epoch - 1, best_dev_eer * 100.0)

    logger.info("Starting training on %s (%d epochs, %d CPU threads)...", device, args.epochs, args.cpu_threads)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss, t0 = 0.0, time.perf_counter()
        for step, (bx, by) in enumerate(train_loader, start=1):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(by)
            if step % 100 == 0:
                logger.info("  Epoch %d batch %d/%d | loss=%.4f", epoch, step, len(train_loader), loss.item())
        scheduler.step()
        tr_loss /= len(y_train)

        model.eval()
        dev_llrs = []
        with torch.no_grad():
            for bx, _ in dev_loader:
                out = model(bx.to(device))
                dev_llrs.append((out[:, 1] - out[:, 0]).cpu().numpy())
        dev_eer, _ = compute_eer(np.concatenate(dev_llrs), y_dev)

        is_best = dev_eer < best_dev_eer - 1e-4
        if is_best:
            best_dev_eer = dev_eer
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            torch.save(best_state, ckpt_best)
            no_improve = 0
        else:
            no_improve += 1

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dev_eer": best_dev_eer,
            "no_improve": no_improve,
        }, ckpt_last)

        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Loss: %.4f | Dev EER: %.3f%%%s",
            epoch, args.epochs, time.perf_counter() - t0, tr_loss, dev_eer * 100.0,
            " *BEST*" if is_best else "",
        )
        if no_improve >= args.patience:
            logger.info("Early stopping triggered after %d epochs without Dev EER improvement.", args.patience)
            break

    if best_state is None and ckpt_best.exists():
        best_state = torch.load(ckpt_best, map_location="cpu")
    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # Calibration strictly on DEV
    dev_logits = []
    with torch.no_grad():
        for bx, _ in dev_loader:
            dev_logits.append(model(bx.to(device)).cpu().numpy())
    temp, logit_bias, dev_eer_final = _fit_dev_calibration(np.concatenate(dev_logits, axis=0), y_dev)
    logger.info("DEV calibration: temperature=%.4f, logit_bias=%.4f (EER threshold=0.5000)", temp, logit_bias)

    # Export ONNX (single self-contained file)
    torch.save(best_state, out_dir / "lcnn.pt")
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 1, 60, 400)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model_cpu, dummy, str(onnx_path),
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["spectrogram"], output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch_size", 3: "time_frames"}, "logits": {0: "batch_size"}},
    )
    onnx_m = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(onnx_m, str(onnx_path), save_as_external_data=False)
    ext_data = out_dir / "model.onnx.data"
    if ext_data.exists():
        ext_data.unlink()

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model_cpu(dummy).numpy()
    ort_out = sess.run(None, {"spectrogram": dummy.numpy()})[0]
    assert np.allclose(pt_out, ort_out, atol=1e-4), f"ONNX parity mismatch: {np.max(np.abs(pt_out - ort_out))}"

    op_thresh = 0.5000
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model_type": "lcnn_onnx",
            "feature_key": "lfcc_spectrogram_2d",
            "n_filters": 60,
            "target_frames": 400,
            "framework": "onnxruntime",
            "temperature": temp,
            "logit_bias": logit_bias,
            "aug_fraction": args.aug_fraction,
        }, fh, indent=2)

    with open(out_dir / "threshold.json", "w", encoding="utf-8") as fh:
        json.dump({
            "method": "dev_calibrated_eer",
            "threshold": op_thresh,
            "validation_eer": dev_eer_final,
            "eer_threshold": op_thresh,
            "operating_threshold": op_thresh,
            "model_name": "LCNN-9 (CMVN + MFM) on Real ASVspoof 2019 LA",
            "input_shape": [1, 60, 400],
        }, fh, indent=2)

    # Full held-out EVAL verification via ONNX Runtime (identical to deployment path)
    X_eval, y_eval, df_eval = extract_or_load_split("eval", archive_path, is_zip, flac_map, proto_map, cache_dir, 0.0, args.workers)
    ev_logits = np.empty((len(y_eval), 2), dtype=np.float32)
    inp_name = sess.get_inputs()[0].name
    for start in range(0, len(y_eval), 128):
        bx = np.array(X_eval[start:start + 128], dtype=np.float32)
        ev_logits[start:start + len(bx)] = sess.run(None, {inp_name: bx})[0]
    ev_probs = _apply_calibration(ev_logits, temp, logit_bias)

    # Single-sample latency benchmark on 200 real eval spectrograms
    lat_secs = []
    for i in range(min(20, len(y_eval))):
        _ = sess.run(None, {inp_name: np.array(X_eval[i:i + 1], dtype=np.float32)})
    for i in range(min(200, len(y_eval))):
        bx1 = np.array(X_eval[i:i + 1], dtype=np.float32)
        t0 = time.perf_counter()
        _ = sess.run(None, {inp_name: bx1})
        lat_secs.append(time.perf_counter() - t0)
    lat = latency_summary(np.array(lat_secs))

    ev_eer, ev_eer_th = compute_eer(ev_probs, y_eval)
    min_tdcf, _ = compute_min_tdcf(ev_probs, y_eval)
    far_frr = far_frr_at_threshold(ev_probs, y_eval, op_thresh)
    cls = classification_metrics_at_threshold(ev_probs, y_eval, op_thresh)
    summary = {
        "condition": "clean",
        "n_eval_samples": int(len(y_eval)),
        "eer_percent": round(ev_eer * 100.0, 3),
        "eer_threshold": round(ev_eer_th, 6),
        "min_tdcf": round(min_tdcf, 4),
        "operating_threshold": op_thresh,
        "far_percent": round(far_frr["far"] * 100.0, 3),
        "frr_percent": round(far_frr["frr"] * 100.0, 3),
        "precision_spoof": round(cls["precision_spoof"], 4),
        "recall_spoof": round(cls["recall_spoof"], 4),
        "f1_spoof": round(cls["f1_spoof"], 4),
        "accuracy": round(cls["accuracy"], 4),
        **{f"cm_{k}": v for k, v in cls["confusion_matrix"].items()},
        "feature_only_latency_median_ms": round(lat["median_ms"], 4),
        "feature_only_latency_p95_ms": round(lat["p95_ms"], 4),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(metrics_dir / "summary_by_condition.csv", index=False)
    pd.DataFrame({
        "file_id": df_eval["file_id"],
        "label": df_eval["label"],
        "label_int": y_eval,
        "system_id": df_eval["system_id"],
        "score": ev_probs,
        "predicted_label": np.where(ev_probs >= op_thresh, "spoof", "bonafide"),
    }).to_csv(metrics_dir / "scores_clean.csv", index=False)

    print("\n=== VERIFIED EVAL SUMMARY ===")
    print(summary_df.to_string(index=False))

    if not args.skip_robustness:
        rob_sub = pd.concat([
            df_eval[df_eval["label"] == "bonafide"].head(500),
            df_eval[df_eval["label"] == "spoof"].head(500),
        ])
        rob_y = rob_sub["label_int"].to_numpy()
        rob_items = [(i, r.file_id, flac_map[r.file_id]) for i, r in enumerate(rob_sub.itertuples(index=False))]
        rob_rows = []
        for cond_name, snr_val, bp_flag in [
            ("clean", None, False),
            ("noise_snr_20db", 20.0, False),
            ("noise_snr_10db", 10.0, False),
            ("telephony_bandpass", None, True),
        ]:
            specs = np.empty((len(rob_items), 1, 60, 400), dtype=np.float32)
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_extract_worker,
                initargs=(str(archive_path), is_zip, flac_map, 0.0, snr_val, bp_flag),
            ) as pool:
                for idx, spec in pool.map(_extract_one, rob_items, chunksize=32):
                    specs[idx] = spec
            r_logits = np.empty((len(rob_y), 2), dtype=np.float32)
            for s in range(0, len(rob_y), 128):
                r_logits[s:s + 128] = sess.run(None, {inp_name: specs[s:s + 128]})[0]
            r_probs = _apply_calibration(r_logits, temp, logit_bias)
            r_eer, _ = compute_eer(r_probs, rob_y)
            r_tdcf, _ = compute_min_tdcf(r_probs, rob_y)
            r_ff = far_frr_at_threshold(r_probs, rob_y, op_thresh)
            r_cls = classification_metrics_at_threshold(r_probs, rob_y, op_thresh)
            rob_rows.append({
                "condition": cond_name,
                "n_samples": len(rob_y),
                "eer_percent": round(r_eer * 100.0, 3),
                "min_tdcf": round(r_tdcf, 4),
                "far_percent": round(r_ff["far"] * 100.0, 3),
                "frr_percent": round(r_ff["frr"] * 100.0, 3),
                "accuracy": round(r_cls["accuracy"], 4),
                "f1_spoof": round(r_cls["f1_spoof"], 4),
            })
        rob_df = pd.DataFrame(rob_rows)
        rob_df.to_csv(metrics_dir / "robustness_eval.csv", index=False)
        print("\n=== ROBUSTNESS SUMMARY (1,000 Eval Clips) ===")
        print(rob_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
