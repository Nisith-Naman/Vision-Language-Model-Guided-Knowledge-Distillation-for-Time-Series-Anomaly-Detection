#!/usr/bin/env python
"""Download and extract the Daphnet Freezing of Gait dataset from the UCI ML Repository.

Dataset details:
    - UCI ID: 245
    - Source: https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait
    - URL: https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip
    - Features: 9 accelerometer channels (ankle, thigh, trunk x/y/z)
    - Labels: 1 = normal walking, 2 = freezing of gait, 0 = unmeasured/transitional (to be dropped)
    - Sampling: 64 Hz
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_daphnet")

UCI_DAPHNET_URL = "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
DEFAULT_OUTPUT_DIR = Path("./data/daphnet")


def download_file(url: str, destination: Path, chunk_size: int = 16384) -> None:
    """Download a remote file with streaming progress bar.

    Args:
        url: Remote HTTP/HTTPS URL.
        destination: Local target path.
        chunk_size: Stream buffer size in bytes.

    Raises:
        requests.RequestException: If download fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".tmp")

    logger.info(f"Connecting to: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))

    with open(tmp_path, "wb") as f, tqdm(
        total=total_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading Daphnet",
        ncols=80,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    if tmp_path.exists():
        if destination.exists():
            destination.unlink()
        tmp_path.rename(destination)
    logger.info(f"Saved archive to: {destination.resolve()}")


def extract_zip(archive_path: Path, output_dir: Path) -> List[Path]:
    """Safely extract all files from a ZIP archive.

    Args:
        archive_path: Path to the ZIP file.
        output_dir: Destination folder.

    Returns:
        List of extracted file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: List[Path] = []

    logger.info(f"Extracting archive to: {output_dir.resolve()}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting files", ncols=80):
            target_path = output_dir / member.filename
            # Prevent zip-slip vulnerability
            if not target_path.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"Unsafe path detected in zip: {member.filename}")
            zf.extract(member, output_dir)
            if not member.is_dir():
                extracted_files.append(target_path)

    return extracted_files


def verify_integrity(dataset_dir: Path) -> Tuple[bool, List[str], List[Path]]:
    """Verify that expected recording files exist and are non-empty.

    Args:
        dataset_dir: Directory containing extracted files.

    Returns:
        Tuple of (is_valid, list_of_errors, list_of_recording_files).
    """
    errors: List[str] = []
    # Daphnet files are named like S01R01.txt or similar text files
    all_txt = list(dataset_dir.rglob("*.txt"))
    recordings = [
        p for p in all_txt if not p.name.lower().startswith("readme") and not p.name.startswith(".")
    ]

    if not recordings:
        errors.append(f"No recording text files found in {dataset_dir}")
        return False, errors, []

    for rec in recordings:
        size = rec.stat().st_size
        if size == 0:
            errors.append(f"File {rec.name} is empty (0 bytes)")

    is_valid = len(errors) == 0
    return is_valid, errors, recordings


def main() -> int:
    """CLI entry point for downloading Daphnet dataset."""
    parser = argparse.ArgumentParser(
        description="Download and prepare the Daphnet Freezing of Gait dataset (UCI ID 245)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=UCI_DAPHNET_URL,
        help="Download URL for the dataset archive",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and extract even if files already exist",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Retain the downloaded .zip file after extraction",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    archive_path = out_dir / "daphnet.zip"

    # Check if already downloaded and valid
    if out_dir.exists() and not args.force:
        is_valid, _, recordings = verify_integrity(out_dir)
        if is_valid and len(recordings) > 0:
            logger.info(
                f"Daphnet dataset is already present at {out_dir.resolve()} "
                f"({len(recordings)} recordings found). Use --force to re-download."
            )
            return 0

    try:
        download_file(args.url, archive_path)
        extract_zip(archive_path, out_dir)

        is_valid, errors, recordings = verify_integrity(out_dir)
        if not is_valid:
            logger.error("Dataset integrity check FAILED:")
            for err in errors:
                logger.error(f"  - {err}")
            return 1

        total_size_mb = sum(r.stat().st_size for r in recordings) / (1024 * 1024)
        logger.info(
            f"Successfully verified {len(recordings)} recording files ({total_size_mb:.2f} MB total)."
        )

        if not args.keep_archive and archive_path.exists():
            archive_path.unlink()
            logger.info("Cleaned up temporary zip archive.")

        print("\n" + "=" * 70)
        print("Daphnet Freezing of Gait Dataset Download Complete!")
        print("=" * 70)
        print(f"Location: {out_dir.resolve()}")
        print(f"Recordings: {len(recordings)} files")
        print("\nPipeline Instructions:")
        print("1. Update config.yaml if needed: daphnet.data_dir = './data/daphnet'")
        print("2. The data loader (src/data/datasets/daphnet.py) filters out annotation=0 rows.")
        print("3. Split strategy: split_by = 'subject' (S01-S07 train, S08 val, S09-S10 test).")
        print("4. Window slicing uses gap protection so windows never cross experiment pauses.")
        print("=" * 70 + "\n")
        return 0

    except Exception as exc:
        logger.exception(f"Failed to download or extract Daphnet dataset: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
