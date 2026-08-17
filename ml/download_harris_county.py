"""Download the official Harris County raw appraisal ZIP files into the project data folder.

This script is intentionally strict because HCAD file URLs are not always stable
across years. If the source exposes a direct download base URL, provide it via
--base-url or HCAD_BASE_URL. If the files are already on a public URL, use the
individual --account-url/--building-url options.

Example usage:
    python ml/download_harris_county.py --year 2025 --base-url "https://example.org/hcad/2025"
    python ml/download_harris_county.py --year 2025 --account-url "https://.../Real_acct_owner.zip" --building-url "https://.../Real_building_land.zip"
"""

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "raw" / "harris-county"
DEFAULT_ACCOUNT_FILE = "Real_acct_owner.zip"
DEFAULT_BUILDING_FILE = "Real_building_land.zip"


def resolve_download_url(base_url, filename):
    if not base_url:
        return None
    base = base_url.rstrip("/")
    return f"{base}/{filename}"


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path):
    ensure_parent(destination)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as upstream, destination.open("wb") as handle:
        shutil.copyfileobj(upstream, handle)
    print(f"Saved to {destination}")


def main():
    parser = argparse.ArgumentParser(description="Download HCAD property-data ZIP files into the project data folder.")
    parser.add_argument("--year", type=int, required=True, help="HCAD release year to download, such as 2025.")
    parser.add_argument("--base-url", default=os.environ.get("HCAD_BASE_URL"), help="Base URL containing the raw ZIP files. Example: https://example.org/hcad/2025")
    parser.add_argument("--account-file", default=os.environ.get("HCAD_ACCOUNT_FILE", DEFAULT_ACCOUNT_FILE), help="Filename for the account-owner ZIP file.")
    parser.add_argument("--building-file", default=os.environ.get("HCAD_BUILDING_FILE", DEFAULT_BUILDING_FILE), help="Filename for the building-land ZIP file.")
    parser.add_argument("--account-url", default=os.environ.get("HCAD_ACCOUNT_URL"), help="Complete direct download URL for the account ZIP file.")
    parser.add_argument("--building-url", default=os.environ.get("HCAD_BUILDING_URL"), help="Complete direct download URL for the building ZIP file.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override. Defaults to data/raw/harris-county/<year>.")
    args = parser.parse_args()

    if args.account_url is None and args.building_url is None:
        if not args.base_url:
            raise SystemExit(
                "No download source supplied. Provide --base-url or HCAD_BASE_URL, or give exact --account-url and --building-url values."
            )

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / str(args.year)
    output_dir.mkdir(parents=True, exist_ok=True)

    account_url = args.account_url or resolve_download_url(args.base_url, args.account_file)
    building_url = args.building_url or resolve_download_url(args.base_url, args.building_file)

    if not account_url or not building_url:
        raise SystemExit("Unable to resolve both ZIP download URLs. Please provide --account-url and --building-url explicitly.")

    account_path = output_dir / args.account_file
    building_path = output_dir / args.building_file

    if account_path.exists() and building_path.exists():
        print(f"Both HCAD files already exist in {output_dir}")
        print("Skipping download.")
        return

    download_file(account_url, account_path)
    download_file(building_url, building_path)

    print("\nDownloaded raw HCAD files.")
    print(f"Folder: {output_dir}")
    print("Next step:")
    print(f"  python ml/ingest_harris_county.py --year {args.year}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:  # pragma: no cover - CLI-friendly failure handling
        print(f"Download failed: {error}", file=sys.stderr)
        sys.exit(1)
