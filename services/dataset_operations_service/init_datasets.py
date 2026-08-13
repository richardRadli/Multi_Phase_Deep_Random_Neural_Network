import os
import shutil
import subprocess
import logging
import requests
import gdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetDownloader")

EXPECTED_DATASETS = [
    "connect4", "isolete", "letter", "mnist", "mnist_fashion",
    "musk2", "optdigits", "page_blocks", "satimages", "segment",
    "shuttle", "spambase", "usps",
]


def validate_dataset_url(url_or_id: str) -> bool:
    """Checks if the dataset URL or Drive ID is non-empty and reachable."""
    if not url_or_id or not isinstance(url_or_id, str):
        logger.error("Dataset URL/ID is missing or invalid in environment variables.")
        return False

    url_or_id = url_or_id.strip()

    if url_or_id.startswith(("http://", "https://")):
        try:
            response = requests.head(url_or_id, allow_redirects=True, timeout=5)
            if response.status_code not in [200, 301, 302]:
                logger.error(f"Dataset URL returned invalid status code: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Failed to reach dataset URL: {e}")
            return False
    elif len(url_or_id) < 10:
        logger.error(f"Provided Google Drive File ID appears too short: '{url_or_id}'")
        return False

    return True


def check_and_download_datasets():
    """Checks for existing datasets and downloads them from Google Drive if missing."""
    dataset_root = os.getenv("DATASET_ROOT", "/app/datasets")
    gdrive_url = os.getenv("GDRIVE_DATASET_URL", "")

    os.makedirs(dataset_root, exist_ok=True)

    # 1. Ellenőrzés
    all_exist = all(
        os.path.exists(os.path.join(dataset_root, ds)) and len(os.listdir(os.path.join(dataset_root, ds))) > 0
        for ds in EXPECTED_DATASETS
    )

    if all_exist:
        logger.info("All datasets are present in DATASET_ROOT. Skipping download.")
        return

    # 2. Link érvényesítés
    logger.info("Missing datasets detected. Validating download URL...")
    if not validate_dataset_url(gdrive_url):
        logger.error("Skipping dataset download due to invalid URL/ID configuration.")
        return

    # 3. Letöltés és kicsomagolás
    rar_path = os.path.join(dataset_root, "datasets.rar")
    try:
        logger.info("Downloading datasets archive from Google Drive...")
        gdown.download(gdrive_url, rar_path, quiet=False)

        if not os.path.exists(rar_path):
            raise FileNotFoundError("Download completed but 'datasets.rar' was not created.")

        unar_bin = shutil.which("unar") or "/usr/bin/unar"

        logger.info(f"Extracting RAR archive using '{unar_bin}'...")
        cmd = [unar_bin, "-o", dataset_root, "-f", rar_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Extraction complete. Checking for nested 'datasets' folder...")

            # AUTOMATIKUS SIMÍTÁS: Ha létrejött egy belső datasets/datasets mappa
            nested_dir = os.path.join(dataset_root, "datasets")
            if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
                logger.info("Nested 'datasets' directory found. Flattening directory structure...")
                for item in os.listdir(nested_dir):
                    src_path = os.path.join(nested_dir, item)
                    dst_path = os.path.join(dataset_root, item)
                    if os.path.exists(dst_path):
                        if os.path.isdir(dst_path):
                            shutil.rmtree(dst_path)
                        else:
                            os.remove(dst_path)
                    shutil.move(src_path, dst_path)
                os.rmdir(nested_dir)
                logger.info("Directory structure successfully flattened!")

            logger.info("Successfully setup all datasets!")
        else:
            logger.error(f"Extraction failed: {result.stderr}")

    except Exception as e:
        logger.error(f"Error during dataset downloading/extraction: {e}")
    finally:
        if os.path.exists(rar_path):
            os.remove(rar_path)


if __name__ == "__main__":
    check_and_download_datasets()