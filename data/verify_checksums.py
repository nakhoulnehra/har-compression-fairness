"""Verify data/raw/ against the SHA-256 manifest in data/checksums.sha256.

Usage (from the repo root):
    python data/verify_checksums.py           # check every file in the manifest
    python data/verify_checksums.py --update  # regenerate the manifest from data/raw/
                                              # (only if you intentionally change dataset versions)

The manifest uses the standard `sha256sum` format, so on Linux/macOS you can also run:
    cd data/raw && sha256sum -c ../checksums.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
MANIFEST = DATA_DIR / "checksums.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, str]:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, rel = line.split(maxsplit=1)
            entries[rel.lstrip("*")] = digest
    return entries


def verify(manifest: dict[str, str] | None = None) -> bool:
    manifest = manifest or load_manifest()
    missing, bad = [], []
    for rel, expected in manifest.items():
        path = RAW_DIR / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != expected:
            bad.append(rel)

    for rel in missing:
        print(f"  MISSING   {rel}")
    for rel in bad:
        print(f"  MISMATCH  {rel}")
    ok = not missing and not bad
    print(f"{len(manifest) - len(missing) - len(bad)}/{len(manifest)} files OK" + ("" if ok else " - run `python data/download.py` to repair"))
    return ok


def update() -> None:
    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file() and not p.name.endswith(".part"))
    lines = ["# SHA-256 of data/raw/ (paths relative to data/raw). Regenerate: python data/verify_checksums.py --update"]
    lines += [f"{sha256(p)}  {p.relative_to(RAW_DIR).as_posix()}" for p in files]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} entries to {MANIFEST.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="regenerate the manifest from data/raw/")
    args = parser.parse_args()
    if args.update:
        update()
    else:
        sys.exit(0 if verify() else 1)
