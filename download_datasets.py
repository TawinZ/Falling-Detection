"""
Download additional fall detection datasets to improve model performance.

Datasets:
  1. GMDCSA-24   (1.1 GB)  — 160 videos incl. sleeping, bending, push-ups → reduces false alarms
  2. CAUCAFall   (~3 GB)   — 5 fall types (forward/backward/side/sitting) in home env
  3. Le2i        (8.95 GB) — JPEG frame sequences in 4 real home environments

Run:
    python3 download_datasets.py
"""

import urllib.request
import zipfile
import tarfile
import shutil
import sys
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)


def progress(block, block_size, total):
    downloaded = block * block_size
    if total > 0:
        pct = min(100, downloaded * 100 / total)
        mb  = downloaded / 1e6
        tot = total / 1e6
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.1f}%  {mb:.1f}/{tot:.1f} MB", end="", flush=True)


def download(url, dest, name):
    dest = Path(dest)
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return True
    print(f"\nDownloading {name}...")
    print(f"  → {url[:80]}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print(f"\n  Saved: {dest}")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_zip(zip_path, out_dir, name):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  Already extracted: {out_dir.name}/")
        return True
    print(f"\nExtracting {name}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            total = len(z.namelist())
            for i, f in enumerate(z.namelist(), 1):
                z.extract(f, out_dir)
                if i % 100 == 0 or i == total:
                    print(f"\r  {i}/{total} files", end="", flush=True)
        print(f"\n  Extracted to: {out_dir}")
        return True
    except Exception as e:
        print(f"\n  Extract failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Fall Detection Dataset Downloader")
    print("=" * 60)

    results = {}

    # ── 1. GMDCSA-24 ──────────────────────────────────────────────
    print("\n[1/3] GMDCSA-24")
    print("  160 videos: falls + sleeping/bending/push-ups/pick objects")
    print("  Key for reducing false alarms in everyday activities")

    url = ("https://zenodo.org/records/12921216/files/ekramalam/"
           "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.0.zip"
           "?download=1")
    zip_path = RAW / "gmdcsa24.zip"
    out_dir  = RAW / "gmdcsa24"

    ok = download(url, zip_path, "GMDCSA-24")
    if ok:
        ok = extract_zip(zip_path, out_dir, "GMDCSA-24")
    results["GMDCSA-24"] = "OK" if ok else "FAILED"

    # ── 2. CAUCAFall ──────────────────────────────────────────────
    print("\n[2/3] CAUCAFall")
    print("  5 fall types: forward, backward, left, right, sitting-fall")
    print("  5 ADL types in real home environment  (23 fps AVI)")

    url = "https://data.mendeley.com/api/datasets/7w7fccy7ky/versions/4/zip"
    zip_path = RAW / "caucafall.zip"
    out_dir  = RAW / "caucafall"

    ok = download(url, zip_path, "CAUCAFall")
    if ok:
        ok = extract_zip(zip_path, out_dir, "CAUCAFall")
    results["CAUCAFall"] = "OK" if ok else "FAILED"

    # ── 3. Le2i ───────────────────────────────────────────────────
    print("\n[3/3] Le2i Fall Detection Dataset")
    print("  191 videos in 4 home environments (home/office/kitchen/lecture room)")
    print("  JPEG frame sequences, 25 fps  (8.95 GB)")
    print("  WARNING: Large file — ~9 GB download")

    ans = input("\n  Download Le2i? (y/N): ").strip().lower()
    if ans == 'y':
        url      = "https://search-data.ubfc.fr/dl_data.php?file=101"
        zip_path = RAW / "le2i.zip"
        out_dir  = RAW / "le2i"
        ok = download(url, zip_path, "Le2i")
        if ok:
            ok = extract_zip(zip_path, out_dir, "Le2i")
        results["Le2i"] = "OK" if ok else "FAILED"
    else:
        print("  Skipped.")
        results["Le2i"] = "SKIPPED"

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Download Summary:")
    for name, status in results.items():
        icon = "✓" if status == "OK" else ("–" if status == "SKIPPED" else "✗")
        print(f"  {icon} {name}: {status}")

    downloaded = [k for k, v in results.items() if v == "OK"]
    if downloaded:
        print(f"\nNext step: run  python3 src/data_preparation/1_extract_keypoints.py")
        print("(pipeline automatically detects and processes the new datasets)")
    print("=" * 60)


if __name__ == "__main__":
    main()
