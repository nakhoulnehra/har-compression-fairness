# Requirements

Technical requirements for the project so far (data download and verification stage).

## Network
- HTTPS access to `archive.ics.uci.edu` (one-time download, about 355 MB):
  - HARTH v1.2: `https://archive.ics.uci.edu/static/public/779/harth.zip` (311 MB)
  - HAR70+: `https://archive.ics.uci.edu/static/public/780/har70.zip` (44 MB)

## Disk
- About 1.4 GB free for `data/raw/` (355 MB of zip archives plus the extracted CSVs).

## Data integrity
- All 42 files (2 archives plus 40 subject CSVs: 22 HARTH, 18 HAR70+) must match the SHA-256 hashes pinned in `data/checksums.sha256`.
- Check with `python data/verify_checksums.py`.
