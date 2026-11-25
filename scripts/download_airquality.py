#!/usr/bin/env python
"""Download and extract the UCI Air Quality dataset.

Dataset details:
    - UCI ID: 360
    - Source: https://archive.ics.uci.edu/dataset/360/air+quality
    - URL: https://archive.ics.uci.edu/static/public/360/air+quality.zip
    - Target file: AirQualityUCI.csv (extracted into data/)
    - Format: Semicolon-delimited CSV, with comma decimals
    - Missing code: -200
"""

from __future__ import annotations

import argparse
import csv
import io
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
logger = logging.getLogger("download_airquality")

UCI_AIRQUALITY_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
DEFAULT_OUTPUT_DIR = Path("./data")
TARGET_FILENAME = "AirQualityUCI.csv"


def download_file(url: str, destination: Path, chunk_size: int = 16384) -> None:
    """Stream download remote file with progress bar.

    Args:
        url: Remote URL to download.
        destination: Destination file path.
        chunk_size: Streaming chunk size.
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
        desc="Downloading Air Quality",
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


def extract_airquality_csv(archive_path: Path, output_file: Path) -> Path:
    """Extract AirQualityUCI.csv from the downloaded ZIP archive.

    Handles zip archives with nested folders or case variations.

    Args:
        archive_path: Path to the downloaded .zip archive.
        output_file: Destination file path for AirQualityUCI.csv.

    Returns:
        Path to the extracted CSV file.

    Raises:
        FileNotFoundError: If AirQualityUCI.csv is not found inside archive.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        # Locate candidate member
        target_member = None
        for name in zf.namelist():
            base_name = Path(name).name.lower()
            if base_name in ("airqualityuci.csv", "airquality.csv"):
                target_member = name
                break

        if target_member is None:
            # Fallback: search for any .csv in archive
            csv_candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if csv_candidates:
                target_member = csv_candidates[0]
            else:
                available = zf.namelist()
                raise FileNotFoundError(
                    f"Could not locate {TARGET_FILENAME} inside zip. Archive contents: {available}"
                )

        logger.info(f"Extracting '{target_member}' -> '{output_file.resolve()}'")
        with zf.open(target_member) as source, open(output_file, "wb") as target:
            shutil.copyfileobj(source, target)

    return output_file


def verify_csv_file(csv_path: Path) -> Tuple[bool, int, int, str]:
    """Verify that the extracted CSV file is readable and non-empty.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Tuple of (is_valid, row_count, col_count, separator).
    """
    if not csv_path.exists():
        return False, 0, 0, ""

    size = csv_path.stat().st_size
    if size == 0:
        return False, 0, 0, ""

    # Read sample to detect separator and inspect rows
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        sample_lines = [f.readline() for _ in range(10)]

    sample_text = "".join(sample_lines)
    sep = ";" if sample_text.count(";") > sample_text.count(",") else ","

    row_count = 0
    col_count = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=sep)
        for i, row in enumerate(reader):
            # AirQualityUCI often has empty trailing columns / blank lines at end
            non_empty_cells = [c.strip() for c in row if c.strip()]
            if non_empty_cells:
                row_count += 1
                if col_count == 0:
                    col_count = len(non_empty_cells)

    is_valid = row_count > 100
    return is_valid, row_count, col_count, sep


def main() -> int:
    """CLI entry point for downloading UCI Air Quality dataset."""
    parser = argparse.ArgumentParser(
        description="Download and extract the UCI Air Quality dataset (UCI ID 360)."
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
        default=UCI_AIRQUALITY_URL,
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
    archive_path = args.output_dir / "airquality.zip"

    # Check if already downloaded and valid
    if out_file.exists() and not args.force:
        is_valid, rows, cols, sep = verify_csv_file(out_file)
        if is_valid:
            logger.info(
                f"Air Quality dataset is already present at {out_file.resolve()} "
                f"({rows:,d} rows, ~{cols} cols, separator='{sep}'). Use --force to re-download."
            )
            return 0

    try:
        download_file(args.url, archive_path)
        extract_airquality_csv(archive_path, out_file)

        is_valid, rows, cols, sep = verify_csv_file(out_file)
        if not is_valid:
            logger.error(f"Extracted CSV verification failed for {out_file.resolve()}")
            return 1

        file_size_mb = out_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Successfully verified {out_file.name} ({rows:,d} rows, {cols} columns, "
            f"{file_size_mb:.2f} MB, delimiter='{sep}')."
        )

        if not args.keep_archive and archive_path.exists():
            archive_path.unlink()
            logger.info("Cleaned up temporary zip archive.")

        print("\n" + "=" * 70)
        print("UCI Air Quality Dataset Download Complete!")
        print("=" * 70)
        print(f"File Path: {out_file.resolve()}")
        print(f"Rows: {rows:,d} | Columns: {cols} | Size: {file_size_mb:.2f} MB")
        print("\nImportant Dataset Notes:")
        print("1. Original data has NO ground-truth anomaly labels.")
        print("2. The pipeline injects synthetic anomalies (point spikes, level shifts,")
        print("   drifts, noise bursts) into validation and test sets only.")
        print("3. Missing values (-200) are replaced with NaN and linearly interpolated.")
        print("4. Column 'NMHC(GT)' has >50% missing and is automatically dropped.")
        print("5. Configuration: see config.yaml -> datasets.airquality")
        print("=" * 70 + "\n")
        return 0

    except Exception as exc:
        logger.exception(f"Failed to download or extract Air Quality dataset: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
