"""Download HARTH v1.2, HAR70+ and Phishing Websites from the UCI ML Repository into data/raw/.

Usage (from the repo root):
    python data/download.py            # download + extract + verify all datasets
    python data/download.py --force    # re-download even if files already exist

Resulting layout:
    data/raw/archives/harth.zip, har70.zip
    data/raw/harth/S006.csv ... S029.csv      (22 subjects)
    data/raw/har70plus/501.csv ... 518.csv    (18 subjects)
    data/raw/phishing/features.csv, targets.csv  (UCI ID 327, 11055 rows, raw)

HARTH and HAR70+ use the stdlib only. Phishing is fetched with `ucimlrepo` (pip install
ucimlrepo), exactly as the LCI paper's authors do: fetch_ucirepo(id=327).
"""

from __future__ import annotations

import argparse
import os
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


# Phishing Websites (UCI ID 327, NOT 967). Fetched via ucimlrepo, saved raw and unmodified.
PHISHING_UCI_ID = 327
PHISHING_DIR = RAW_DIR / "phishing"
PHISHING_FILES = ("features.csv", "targets.csv")


def fetch_phishing() -> None:
    """fetch_ucirepo(id=327) and save features/targets unmodified. Raises on any failure;
    there is no synthetic-data fallback."""
    from ucimlrepo import fetch_ucirepo  # lazy: the HAR datasets need no extra packages

    try:
        ds = fetch_ucirepo(id=PHISHING_UCI_ID)
    except Exception as e:
        # ucimlrepo verifies its API call against certifi's CA bundle, but lets pandas fetch the
        # CSV with Python's default store, which can be stale on Windows. Retry with the default
        # store pointed at certifi's bundle (verification stays ON). Anything else is re-raised.
        if not _caused_by_ssl(e) or os.environ.get("SSL_CERT_FILE"):
            raise
        import certifi

        print("  TLS error with the default CA store; retrying with certifi's CA bundle")
        os.environ["SSL_CERT_FILE"] = certifi.where()
        ds = fetch_ucirepo(id=PHISHING_UCI_ID)
    features, targets = ds.data.features, ds.data.targets
    check_phishing(features, targets)
    PHISHING_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PHISHING_DIR / "features.csv", index=False)
    targets.to_csv(PHISHING_DIR / "targets.csv", index=False)


def _caused_by_ssl(e: BaseException | None) -> bool:
    while e is not None:
        if isinstance(e, ssl.SSLError) or (isinstance(e, urllib.error.URLError) and isinstance(e.reason, ssl.SSLError)):
            return True
        e = e.__cause__ or e.__context__
    return False


def check_phishing(features, targets) -> None:
    """Shape / label sanity checks from the spec, plus class counts (paper: 44:56)."""
    assert features.shape == (11055, 30), f"unexpected features shape {features.shape}"
    assert targets.shape[0] == 11055, f"unexpected targets rows {targets.shape[0]}"
    counts = targets.iloc[:, 0].value_counts().sort_index()
    assert len(counts) == 2, f"expected 2 classes, got {counts.to_dict()}"
    pct = (100 * counts / counts.sum()).round(1)
    print("  class counts (-1 = phishing, 1 = legitimate): "
          + ", ".join(f"{k}: {v} ({pct[k]}%)" for k, v in counts.items()))


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

    print("[Phishing Websites (UCI 327)]")
    files = [PHISHING_DIR / f for f in PHISHING_FILES]
    keys = [f"phishing/{f}" for f in PHISHING_FILES]
    pinned = all(k in manifest for k in keys)
    if args.force or not all(p.exists() for p in files) or (
        pinned and any(sha256(p) != manifest[k] for p, k in zip(files, keys))
    ):
        print(f"  fetching via ucimlrepo: fetch_ucirepo(id={PHISHING_UCI_ID})")
        fetch_phishing()
    else:
        import pandas as pd

        print("  files already present")
        check_phishing(pd.read_csv(files[0]), pd.read_csv(files[1]))
    print(f"  saved -> {PHISHING_DIR.relative_to(DATA_DIR.parent)}")

    print("\nVerifying extracted files...")
    return 0 if verify(manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
