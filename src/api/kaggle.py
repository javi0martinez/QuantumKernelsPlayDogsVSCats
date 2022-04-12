"""
Module to download the official Kaggle competition dataset "dogs-vs-cats".
Uses: kaggle competitions download -c dogs-vs-cats
"""

import subprocess
import sys
from pathlib import Path

# Project-relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
DATA_DIR = PROJECT_ROOT / "data" / "DogsAndCats"

# Kaggle competition identifier
KAGGLE_COMPETITION = "dogs-vs-cats"


def download_competition(
    competition: str = KAGGLE_COMPETITION,
    destination: Path = DATA_DIR,
    force: bool = False
) -> None:
    """
    Downloads and unzips the Kaggle competition dataset into the destination folder.

    Args:
        competition (str): Kaggle competition name (e.g., dogs-vs-cats).
        destination (Path): Folder where files will be stored.
        force (bool): If True, forces re-download even if files already exist.

    Raises:
        RuntimeError: If authentication or download fails.
    """
    destination.mkdir(parents=True, exist_ok=True)

    zip_path = destination / f"{competition}.zip"

    if zip_path.exists() and not force:
        print(f"Competition zip already exists at {zip_path}. Not re-downloading.")
        return

    print(f"Destination: {destination}")

    try:
        # kaggle competitions download -c <competition> -p <path> [--force]
        cmd = [
            "kaggle", "competitions", "download",
            "-c", competition,
            "-p", str(destination)
        ]
        if force:
            cmd.append("--force")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("Download completed.")
        print(result.stdout)

        # Unzip the downloaded file
        if zip_path.exists():
            print(f"Unzipping {zip_path}...")
            subprocess.run(
                ["powershell", "-Command",
                 f"Expand-Archive -Path '{zip_path}' -DestinationPath '{destination}' -Force"],
                check=True,
                capture_output=True,
                text=True
            )
            print("Unzip completed.")

    except subprocess.CalledProcessError as e:
        print("Error downloading or unzipping the competition.")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print("Error: 'kaggle' is not detected.")
        sys.exit(1)
