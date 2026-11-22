"""In-RAM caches of the datasets at a fixed resolution (uint8).

BRISC2025: single-channel T1 slices read through the manifest-driven reader in
brisc_dataset.py (official train split -> train/val via the manifest; official
test split kept as test).

Kvasir-SEG: RGB colonoscopy images. No official split exists, so a fixed
permutation (seed 42) of the sorted file names gives train 550 / val 100 /
test 350. The test split is larger than usual so that each calibration half
(175 cases) keeps enough accepted cases for the binomial tests to have power.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

KVASIR_SPLIT = {"train": (0, 550), "val": (550, 650), "test": (650, 1000)}


def _cache_path(cache_dir, dataset, split, size):
    if cache_dir is None:
        return None
    name = f"brisc_{split}_{size}.npz" if dataset == "brisc" else f"{dataset}_{split}_{size}.npz"
    return Path(cache_dir) / name


def _load_brisc(manifest, split, data_root, size):
    from .brisc_dataset import BRISC2025Dataset
    ds = BRISC2025Dataset(manifest, split, data_root=data_root, image_size=size,
                          train=False, imagenet_norm=False)
    n = len(ds)
    x = np.empty((n, size, size), np.uint8)
    y = np.empty((n, size, size), np.uint8)
    meta = {"case_id": [], "tumor_label": [], "plane_label": []}
    for i in range(n):
        img, msk, m = ds[i]
        x[i] = np.clip(img[0].numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
        y[i] = (msk[0].numpy() > 0.5).astype(np.uint8)
        for k in meta:
            meta[k].append(m[k])
    return x, y, meta


def _load_kvasir(split, data_root, size):
    root = Path(data_root)
    names = sorted(p.name for p in (root / "images").glob("*.jpg"))
    if len(names) != 1000:
        raise RuntimeError(f"Kvasir-SEG: expected 1000 images, found {len(names)} in {root}")
    perm = np.random.default_rng(42).permutation(len(names))
    a, b = KVASIR_SPLIT[split]
    sel = [names[i] for i in perm[a:b]]
    x = np.empty((len(sel), size, size, 3), np.uint8)
    y = np.empty((len(sel), size, size), np.uint8)
    for i, nm in enumerate(sel):
        img = cv2.imread(str(root / "images" / nm), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(root / "masks" / nm), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(nm)
        x[i] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size),
                          interpolation=cv2.INTER_AREA)
        y[i] = (cv2.resize(msk, (size, size), interpolation=cv2.INTER_NEAREST) > 127)
    meta = {"case_id": [Path(n).stem for n in sel], "tumor_label": ["polyp"] * len(sel),
            "plane_label": ["na"] * len(sel)}
    return x, y, meta


def load_split(manifest: str | Path | None, split: str, data_root: str | Path, size: int,
               cache_dir: str | Path | None = None, dataset: str = "brisc"):
    cache = _cache_path(cache_dir, dataset, split, size)
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["x"], z["y"], z["meta"].item()
    if dataset == "brisc":
        x, y, meta = _load_brisc(manifest, split, data_root, size)
    elif dataset == "kvasir":
        x, y, meta = _load_kvasir(split, data_root, size)
    else:
        raise ValueError(dataset)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=x, y=y, meta=np.array(meta, dtype=object))
    return x, y, meta


class ArrayDataset(Dataset):
    """x: (N,H,W) grayscale or (N,H,W,3) RGB uint8; y: (N,H,W) {0,1}."""

    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x, self.y, self.augment = x, y, augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        img = self.x[i].astype(np.float32) / 255.0
        msk = self.y[i].astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.25:
                img, msk = img[::-1, :], msk[::-1, :]
            if random.random() < 0.35:
                k = random.choice([1, 2, 3])
                img, msk = np.rot90(img, k), np.rot90(msk, k)
            if random.random() < 0.3:
                img = np.clip(img * random.uniform(0.9, 1.1) + random.uniform(-0.04, 0.04), 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)[None] if img.ndim == 2 else torch.from_numpy(img).permute(2, 0, 1)
        msk = torch.from_numpy(np.ascontiguousarray(msk))[None]
        return img.contiguous(), msk
