"""Download the six classical ADBench archives used by the project."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from src.adbench import DATASET_FILES, DEFAULT_DATA_DIR, load_dataset


DATASET_URL = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/"
    "adbench/datasets/Classical/"
)


def download_datasets(data_dir: Path) -> None:
    """Download missing archives and validate files before publishing them."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in DATASET_FILES:
        destination = data_dir / filename
        if destination.exists():
            load_dataset(filename, data_dir)
            print(f"Already present and valid: {destination}")
            continue

        with TemporaryDirectory(dir=data_dir) as temporary_dir:
            temporary_path = Path(temporary_dir) / filename
            with urlopen(DATASET_URL + filename, timeout=30) as response:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            load_dataset(filename, temporary_dir)
            try:
                destination.hardlink_to(temporary_path)
            except FileExistsError:
                load_dataset(filename, data_dir)
                print(f"Already present and valid: {destination}")
            else:
                print(f"Downloaded: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    try:
        download_datasets(args.data_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
