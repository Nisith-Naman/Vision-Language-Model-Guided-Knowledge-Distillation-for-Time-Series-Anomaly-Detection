#!/usr/bin/env python
"""Download and extract the MetroPT-3 dataset from the UCI ML Repository.

Dataset details:
    - UCI ID: 791
    - Source: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
    - URL: https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip
    - Target file: MetroPT3(AirCompressor).csv (extracted into data/)
    - Features: 7 analog signals + 8 digital signals from an urban train air compressor
    - Anomaly labels: Derived from 4 documented maintenance failure windows
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_metropt")

UCI_METROPT_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
DEFAULT_OUTPUT_DIR = Path("./data")
TARGET_FILENAME = "MetroPT3(AirCompressor).csv"


def download_file(url: str, destination: Path, chunk_size: int = 32768) -> None:
    """Download a remote file with streaming progress bar.

    Args:
        url: Remote URL.
        destination: Local target path.
        chunk_size: Buffer size for chunks.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".tmp")

    logger.info(f"Connecting to: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))

    with open(tmp_path, "wb") as f, tqdm(
        total=total_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading MetroPT-3",
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


def extract_metropt_csv(archive_path: Path, output_file: Path) -> Path:
    """Extract the MetroPT-3 CSV file from the downloaded archive.

    Inspects archive members to find the primary CSV data file.

    Args:
        archive_path: Path to the downloaded .zip file.
        output_file: Destination path for MetroPT3(AirCompressor).csv.

    Returns:
        Path to the extracted CSV file.

    Raises:
        FileNotFoundError: If no suitable CSV file is found inside the archive.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        target_member = None
        for name in zf.namelist():
            base_lower = Path(name).name.lower()
            if "metropt" in base_lower and base_lower.endswith(".csv"):
                target_member = name
                break

        if target_member is None:
            # Fallback: largest .csv file in archive
            csv_members = [
                info for info in zf.infolist() if info.filename.lower().endswith(".csv")
            ]
            if csv_members:
                largest = max(csv_members, key=lambda info: info.file_size)
                target_member = largest.filename
            else:
                raise FileNotFoundError(
                    f"No CSV file found in MetroPT archive. Archive members: {zf.namelist()}"
                )

        logger.info(f"Extracting member '{target_member}' -> '{output_file.resolve()}'")
        with zf.open(target_member) as source, open(output_file, "wb") as target:
            shutil.copyfileobj(source, target)

    return output_file


def count_csv_rows_and_size(csv_path: Path) -> Tuple[int, float]:
    """Count rows and compute size in megabytes of a CSV file.

    Args:
        csv_path: Path to the target CSV.

    Returns:
        Tuple of (row_count, size_in_mb).
    """
    if not csv_path.exists():
        return 0, 0.0

    size_mb = csv_path.stat().st_size / (1024 * 1024)

    row_count = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        # Fast line count
        for _ in f:
            row_count += 1

    return row_count, size_mb


def main() -> int:
    """CLI entry point for downloading MetroPT-3 dataset."""
    parser = argparse.ArgumentParser(
        description="Download and extract the MetroPT-3 dataset (UCI ID 791)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=TARGET_FILENAME,
        help=f"Target CSV filename (default: {TARGET_FILENAME})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=UCI_METROPT_URL,
        help="Download URL for the dataset archive",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite even if file already exists",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Retain the downloaded .zip file after extraction",
    )
    args = parser.parse_args()

    out_file: Path = args.output_dir / args.filename
    archive_path = args.output_dir / "metropt3.zip"

    # Check if already downloaded and valid
    if out_file.exists() and not args.force:
        rows, size_mb = count_csv_rows_and_size(out_file)
        if rows > 1000:
            logger.info(
                f"MetroPT-3 dataset is already present at {out_file.resolve()} "
                f"({rows:,d} rows, {size_mb:.2f} MB). Use --force to re-download."
            )
            return 0

    try:
        download_file(args.url, archive_path)
        extract_metropt_csv(archive_path, out_file)

        rows, size_mb = count_csv_rows_and_size(out_file)
        if rows == 0:
            logger.error(f"Extracted CSV is empty: {out_file.resolve()}")
            return 1

        logger.info(
            f"Successfully verified {out_file.name}: {rows:,d} rows, {size_mb:.2f} MB."
        )

        if not args.keep_archive and archive_path.exists():
            archive_path.unlink()
            logger.info("Cleaned up temporary zip archive.")

        print("\n" + "=" * 70)
        print("MetroPT-3 Dataset Download Complete!")
        print("=" * 70)
        print(f"File Path: {out_file.resolve()}")
        print(f"Row count: {rows:,d}")
        print(f"File size: {size_mb:.2f} MB")
        print("\nPipeline & Domain Notes:")
        print("1. Sensor channels: 7 analog features + 8 digital control signals.")
        print("2. Raw data sampling frequency varies; data loader resamples to 1-minute")
        print("   resolution using median (analog) and max (digital).")
        print("3. Ground-truth labels are created from 4 documented failure events:")
        print("   - Air Leak 1: 2020-04-18 (24 hours)")
        print("   - Air Leak 2: 2020-05-29 to 2020-05-30")
        print("   - Air Leak 3: 2020-06-05 to 2020-06-07")
        print("   - Air Leak 4: 2020-07-15")
        print("4. High variance expected due to small number of failure events (n=4).")
        print("=" * 70 + "\n")
        return 0

    except Exception as exc:
        logger.exception(f"Failed to download or extract MetroPT-3 dataset: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
