"""Build the resized in-RAM caches once (run under a lock before training)."""
import argparse

from src.data import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="brisc", choices=["brisc", "kvasir"])
ap.add_argument("--manifest", default=None)
ap.add_argument("--data-root", required=True)
ap.add_argument("--cache-dir", default="cache")
ap.add_argument("--size", type=int, default=256)
a = ap.parse_args()
for split in ("train", "val", "test"):
    x, y, meta = load_split(a.manifest, split, a.data_root, a.size, a.cache_dir, dataset=a.dataset)
    empty = int((y.reshape(len(y), -1).sum(1) == 0).sum())
    print(f"[cache] {split}: n={len(x)} empty_masks={empty}")
