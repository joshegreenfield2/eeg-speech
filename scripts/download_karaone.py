#!/usr/bin/env python3
"""Download and extract the KaraOne EEG dataset.

Subjects: MM05, MM08-MM12, MM14-MM16, MM18-MM21, P02 (14 total)
Files: {subject}.tar.bz2 → data/raw/{subject}/ (with .set, epoch_inds.mat, kinect_data/)
"""

import os
import shutil
import tarfile
import sys
from pathlib import Path

import requests
from tqdm import tqdm

SUBJECTS = [
    "MM05", "MM08", "MM09", "MM10", "MM11", "MM12",
    "MM14", "MM15", "MM16", "MM18", "MM19", "MM20", "MM21", "P02",
]

BASE_URL = "http://www.cs.toronto.edu/~complingweb/data/karaOne/"
# Extracted files live under this path inside the tar
TAR_INNER_PATH = "p/spoclab/users/szhao/EEG/data/"


def download_subject(subject: str, raw_dir: Path, skip_existing: bool = True) -> Path:
    dest = raw_dir / f"{subject}.tar.bz2"

    if skip_existing and dest.exists():
        url = BASE_URL + f"{subject}.tar.bz2"
        r = requests.head(url, timeout=10)
        remote_size = int(r.headers.get("content-length", 0))
        if dest.stat().st_size >= remote_size:
            print(f"  {subject}: already downloaded, skipping")
            return dest

    url = BASE_URL + f"{subject}.tar.bz2"
    print(f"  {subject}: downloading from {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=subject, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def extract_subject(subject: str, raw_dir: Path, delete_archive: bool = False):
    archive = raw_dir / f"{subject}.tar.bz2"
    subject_dir = raw_dir / subject

    if subject_dir.exists():
        print(f"  {subject}: already extracted, skipping")
        return

    print(f"  {subject}: extracting...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(raw_dir)

    # Move from nested path to raw_dir/subject
    extracted = raw_dir / TAR_INNER_PATH / subject
    if extracted.exists():
        shutil.move(str(extracted), str(raw_dir / subject))
        # Clean up the nested folder structure
        top_level = raw_dir / TAR_INNER_PATH.split("/")[0]
        if top_level.exists():
            shutil.rmtree(top_level)
    else:
        print(f"  WARNING: expected path {extracted} not found after extraction")
        print(f"  Check contents of {archive}")

    if delete_archive:
        archive.unlink()


def main():
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Parse args: --delete-archive to remove .tar.bz2 after extraction (save space)
    args = sys.argv[1:]
    delete_archive = "--delete-archive" in args
    args = [a for a in args if not a.startswith("--")]
    subjects = args if args else SUBJECTS
    print(f"KaraOne download — {len(subjects)} subjects → {raw_dir}")
    if delete_archive:
        print("(archives will be deleted after extraction)")
    print()

    for subject in subjects:
        print(f"[{subject}]")
        try:
            download_subject(subject, raw_dir)
            extract_subject(subject, raw_dir, delete_archive=delete_archive)
            # Verify
            subject_dir = raw_dir / subject
            set_files = list(subject_dir.glob("*.set"))
            has_epoch_inds = (subject_dir / "epoch_inds.mat").exists()
            has_labels = (subject_dir / "kinect_data" / "labels.txt").exists()
            status = "OK" if set_files and has_epoch_inds and has_labels else "INCOMPLETE"
            print(f"  {subject}: {status} (.set={bool(set_files)}, epoch_inds={has_epoch_inds}, labels={has_labels})")
        except Exception as e:
            print(f"  {subject}: ERROR — {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
