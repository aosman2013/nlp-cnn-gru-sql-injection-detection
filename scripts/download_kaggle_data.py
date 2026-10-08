#!/usr/bin/env python3
"""Download the public SQL injection dataset using the Kaggle CLI.

Prerequisite: configure Kaggle credentials outside the repository.
See https://www.kaggle.com/docs/api for credential setup.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import zipfile

DATASET = "gambleryu/biggest-sql-injection-dataset"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/raw")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "biggest-sql-injection-dataset.zip"

    subprocess.run([
        "kaggle", "datasets", "download", "-d", DATASET,
        "-p", str(out), "--force"
    ], check=True)

    if not archive.exists():
        matches = list(out.glob("*.zip"))
        if not matches:
            raise FileNotFoundError("Kaggle archive was not found after download.")
        archive = matches[0]

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out)
    print(f"Dataset extracted to: {out.resolve()}")


if __name__ == "__main__":
    main()
