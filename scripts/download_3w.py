#!/usr/bin/env python
"""Download assistant and instructions for the Petrobras 3W Oil Well Dataset.

Dataset details:
    - Repository: https://github.com/ricardovvargas/3w_dataset
    - Reference: Vargas et al., Journal of Petroleum Science and Engineering (2019)
    - Target directory: ./data/3w
    - Structure: 9 class folders (0 through 8), containing 1,984 CSV files
    - Official repository stores four 7z volumes (~314 MB compressed, ~5.1 GB extracted)
    - Features: P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, T-JUS-CKGL, QGL
    - Label column: 'class' (0 = normal, 1-8 = specific undesirable oil well events)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_3w")

REPO_URL = "https://github.com/ricardovvargas/3w_dataset.git"
DEFAULT_TARGET_DIR = Path("./data/3w")

EVENT_CLASS_NAMES: Dict[int, str] = {
    0: "Normal operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in CKP",
    7: "Scaling in CKP",
    8: "Hydrate Formation in Production Line",
}


def check_git_available() -> bool:
    """Check whether git executable is available on the system PATH."""
    return shutil.which("git") is not None


def inspect_existing_dataset(target_dir: Path) -> Dict[str, int]:
    """Inspect existing 3W directory and count CSV instances per class.

    Args:
        target_dir: Path to the 3W dataset directory.

    Returns:
        Dictionary mapping class folder name to count of CSV files.
    """
    counts: Dict[str, int] = {}
    if not target_dir.exists():
        return counts

    for class_id in range(9):
        class_folders = [
            folder for folder in target_dir.rglob(str(class_id))
            if folder.is_dir() and ".git" not in folder.parts
        ]
        count = sum(len(list(folder.glob("*.csv"))) for folder in class_folders)
        if count:
            counts[str(class_id)] = count

    return counts


def clone_repository(repo_url: str, target_dir: Path) -> bool:
    """Clone the 3W dataset git repository into target_dir.

    Args:
        repo_url: Git clone URL.
        target_dir: Destination path.

    Returns:
        True if clone succeeded, False otherwise.
    """
    logger.info(f"Cloning {repo_url} into {target_dir.resolve()}...")
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Use shallow clone to minimize download size
        cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir.resolve())]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info("Git clone completed successfully.")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"Git clone failed with return code {exc.returncode}: {exc.stderr}")
        return False
    except Exception as exc:
        logger.exception(f"Unexpected error running git clone: {exc}")
        return False


def print_instructions(target_dir: Path, git_available: bool) -> None:
    """Print detailed manual download and setup instructions for the 3W dataset."""
    print("\n" + "=" * 76)
    print("3W OIL WELL DATASET SETUP INSTRUCTIONS")
    print("=" * 76)
    print(f"Target Directory: {target_dir.resolve()}\n")

    print("The 3W dataset is hosted on GitHub by Petrobras and Ricardo Vargas:")
    print("  Official Repository: https://github.com/ricardovvargas/3w_dataset")
    print("  The data is split across four 7z volumes (~314 MB compressed; ~5.1 GB extracted).\n")

    print("METHOD 1: Automated Git Clone (Recommended)")
    if git_available:
        print(f"  git is detected on your system. Run this script with '--clone':")
        print(f"    python scripts/download_3w.py --clone --target-dir {target_dir}")
    else:
        print("  git is NOT detected on your system PATH.")
        print("  Install git (https://git-scm.com/) or use Method 2 below.")

    print("\nMETHOD 2: Manual Terminal Clone")
    print("  Run in your terminal:")
    print(f"    git clone --depth 1 https://github.com/ricardovvargas/3w_dataset.git \"{target_dir.resolve()}\"")

    print("\nMETHOD 3: Download Zip from GitHub")
    print("  1. Clone the repository or download its four data.7z.001-.004 volumes.")
    print("  2. Extract data.7z.001 with 7-Zip; it will read the remaining volumes automatically.")
    print(f"  3. Put the extracted class folders 0-8 under: {target_dir.resolve()}")

    print("\nExpected Directory Structure After Download:")
    print("  data/3w/")
    print("  ├── 0/   (Normal operation instances, e.g. WELL-00001_20140124110000.csv)")
    print("  ├── 1/   (Abrupt Increase of BSW)")
    print("  ├── 2/   (Spurious Closure of DHSV)")
    print("  ├── 3/   (Severe Slugging)")
    print("  ├── 4/   (Flow Instability)")
    print("  ├── 5/   (Rapid Productivity Loss)")
    print("  ├── 6/   (Quick Restriction in CKP)")
    print("  ├── 7/   (Scaling in CKP)")
    print("  ├── 8/   (Hydrate Formation in Production Line)")
    print("  └── dataset/ (some versions nest classes 0-8 inside a 'dataset/' folder)")

    print("\nConfiguration Check (config.yaml):")
    print("  datasets:")
    print("    oil3w:")
    print(f"      data_dir: \"{target_dir}\"")
    print("      feature_cols: [\"P-PDG\", \"P-TPT\", \"T-TPT\", \"P-MON-CKP\",")
    print("                     \"T-JUS-CKP\", \"P-JUS-CKGL\", \"T-JUS-CKGL\", \"QGL\"]")
    print("      time_col: \"timestamp\"")
    print("      label_col: \"class\"")
    print("      transient_is_anomaly: true  # classes 101-108 treated as anomalous")
    print("=" * 76 + "\n")


def main() -> int:
    """CLI entry point for 3W dataset download helper."""
    parser = argparse.ArgumentParser(
        description="Download helper and setup instructions for the Petrobras 3W dataset."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Target directory for 3W dataset (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--clone",
        action="store_true",
        help="Attempt automatic git clone if git is available",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=REPO_URL,
        help=f"Git repository URL (default: {REPO_URL})",
    )
    args = parser.parse_args()

    target_dir: Path = args.target_dir
    git_available = check_git_available()

    # Inspect current status
    counts = inspect_existing_dataset(target_dir)
    total_csvs = sum(counts.values())

    if total_csvs > 0:
        print("\n" + "=" * 70)
        print("3W Dataset Already Detected!")
        print("=" * 70)
        print(f"Location: {target_dir.resolve()}")
        print(f"Total instances found: {total_csvs:,d} CSV files across classes:")
        for c_id in sorted(counts.keys(), key=lambda x: int(x)):
            c_int = int(c_id)
            desc = EVENT_CLASS_NAMES.get(c_int, "Unknown event")
            print(f"  Class {c_id} ({desc:<38}): {counts[c_id]:>4d} instances")
        print("=" * 70 + "\n")
        return 0

    if args.clone:
        if not git_available:
            logger.error(
                "Git is not available on your PATH. Cannot clone repository automatically. "
                "Please follow the manual instructions below."
            )
            print_instructions(target_dir, git_available=False)
            return 1

        success = clone_repository(args.repo_url, target_dir)
        if success:
            counts = inspect_existing_dataset(target_dir)
            total = sum(counts.values())
            if total == 0:
                archive_candidates = sorted(target_dir.rglob("data.7z.001"))
                print("\nRepository cloned, but it does not yet contain extracted CSV instances.")
                if archive_candidates:
                    archive_path = archive_candidates[0]
                    print("The repository stores its instances in a four-part 7z archive.")
                    print("If 7-Zip is installed, extract it with:")
                    print(f'  7z x "{archive_path}" -o"{target_dir.resolve()}"')
                else:
                    print_instructions(target_dir, git_available=True)
                return 1
            print("\n" + "=" * 70)
            print("Successfully Cloned 3W Dataset!")
            print("=" * 70)
            print(f"Location: {target_dir.resolve()}")
            print(f"Total CSV instances: {total:,d}")
            print("=" * 70 + "\n")
            return 0
        else:
            print_instructions(target_dir, git_available=True)
            return 1

    # Default action: print instructions
    print_instructions(target_dir, git_available)
    return 0


if __name__ == "__main__":
    sys.exit(main())
