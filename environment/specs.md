# Device Specs

The machines the experiments and timings in this repo are run on. Every entry in [time.md](time.md) names the machine it was measured on. All runs are CPU only, except model-choice training on Nakhoul's RTX 3050 Ti (torch 2.7.1+cu118, from 2026-11-19). Aya's GPU is not used.

## Nakhoul's machine

Recorded 2026-11-15. Ran the LCI Phishing reproduction in `results/lci-phishing-reproduction/`.

| Component | Spec |
|---|---|
| Model | Acer Swift X (SFX14-41G) laptop |
| CPU | AMD Ryzen 7 5800U, 8 cores / 16 threads, 1.9 GHz base |
| RAM | 16 GB (15.4 GB usable) |
| GPU (discrete) | NVIDIA GeForce RTX 3050 Ti Laptop, 4 GB VRAM, driver 591.59, CUDA 13.1; used for model-choice training with torch 2.7.1+cu118 (machine name `nakhoul-3050`) |
| GPU (integrated) | AMD Radeon Graphics |
| Storage | 512 GB NVMe SSD (WD PC SN530) |
| OS | Windows 11 Home, 64-bit, build 26200 |

## Aya's machine

Recorded 2026-11-18.

| Component | Spec |
|---|---|
| Model | Lenovo ThinkPad E14 Gen 5 (21JLS20R00) laptop |
| CPU | Intel Core i7-1355U (13th gen), 10 cores (2 P + 8 E) / 12 threads, 1.7 GHz base |
| RAM | 16 GB (15.7 GB usable) |
| GPU (discrete) | NVIDIA GeForce MX550, 2 GB VRAM, driver 595.95, CUDA 13.2 (not used) |
| GPU (integrated) | Intel Iris Xe Graphics |
| Storage | 512 GB NVMe SSD (WD PC SN740) |
| OS | Windows 11 Home Single Language, 64-bit, build 26200 |
