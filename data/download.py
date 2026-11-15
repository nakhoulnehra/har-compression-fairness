"""Download HARTH v1.2 and HAR70+ from the UCI ML Repository into data/raw/.

Usage (from the repo root):
    python data/download.py            # download + extract + verify both datasets
    python data/download.py --force    # re-download even if files already exist

Resulting layout:
    data/raw/archives/harth.zip, har70.zip
    data/raw/harth/S006.csv ... S029.csv      (22 subjects)
    data/raw/har70plus/501.csv ... 518.csv    (18 subjects)

Stdlib only - no extra packages required.
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from verify_checksums import DATA_DIR, RAW_DIR, load_manifest, sha256, verify

ARCHIVE_DIR = RAW_DIR / "archives"

DATASETS = {
    # name: (url, archive filename, extracted folder)
    "HARTH v1.2": ("https://archive.ics.uci.edu/static/public/779/harth.zip", "harth.zip", "harth"),
    "HAR70+": ("https://archive.ics.uci.edu/static/public/780/har70.zip", "har70.zip", "har70plus"),
}


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        _download_urllib(url, tmp)
    except urllib.error.URLError as e:
        # Some Python builds (notably on Windows) pick up stale CA certificates and fail TLS
        # verification against UCI. curl uses the OS-native TLS stack, so fall back to it.
        # Verification stays ON either way, and the SHA-256 check guards the content.
        if not isinstance(e.reason, ssl.SSLError) or not shutil.which("curl"):
            raise
        print(f"  urllib TLS error ({e.reason.__class__.__name__}); retrying with curl")
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(tmp), url], check=True)
    tmp.replace(dest)


def _download_urllib(url: str, tmp: Path) -> None:
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing any member that would land outside `target`."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            out = (target / member.filename).resolve()
            if target not in out.parents and out != target:
                raise RuntimeError(f"Unsafe path in {archive.name}: {member.filename}")
        zf.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download and re-extract everything")
    args = parser.parse_args()

    manifest = load_manifest()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for name, (url, zip_name, folder) in DATASETS.items():
        print(f"[{name}]")
        archive = ARCHIVE_DIR / zip_name
        expected = manifest[f"archives/{zip_name}"]

        if args.force or not archive.exists() or sha256(archive) != expected:
            print(f"  downloading {url}")
            download(url, archive)
        else:
            print("  archive already present")

        actual = sha256(archive)
        if actual != expected:
            print(f"  ERROR: checksum mismatch for {zip_name}\n    expected {expected}\n    got      {actual}")
            return 1
        print("  archive checksum OK")

        if args.force and (RAW_DIR / folder).exists():
            shutil.rmtree(RAW_DIR / folder)
        print(f"  extracting -> {(RAW_DIR / folder).relative_to(DATA_DIR.parent)}")
        safe_extract(archive, RAW_DIR)

    print("\nVerifying extracted files...")
    return 0 if verify(manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
