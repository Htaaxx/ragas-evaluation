#!/usr/bin/env python
"""Build MaDeTai_MaNguon.zip for thesis source submission.

Usage:
    python scripts/package_submission.py
    python scripts/package_submission.py --topic-code CKTxx

Creates dist/MaDeTai_MaNguon.zip (or {topic}_MaNguon.zip) with:
    MaDeTai/
      src/
      configs/
      scripts/
      notebooks/
      data/          (labeled_merged*.csv only)
      tests/
      docs/
      requirements.txt
      README.md
      START_HERE.md
      HuongDanCaiDat.txt
      HuongDanSuDung.txt
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = [
    "src",
    "configs",
    "scripts",
    "notebooks",
    "tests",
    "docs",
]
INCLUDE_FILES = [
    "requirements.txt",
    "README.md",
    "START_HERE.md",
    "HuongDanCaiDat.txt",
    "HuongDanSuDung.txt",
]
DATA_FILES = [
    "data/labeled_merged.csv",
    "data/labeled_merged_test.csv",
]
SKIP_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
    "venv",
    ".venv",
    "rag_output",
    "kaggle",
}


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def build(topic_code: str = "MaDeTai") -> Path:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    staging = dist / "staging"
    if staging.exists():
        shutil.rmtree(staging)

    package_root = staging / topic_code
    package_root.mkdir(parents=True)

    for name in INCLUDE_DIRS:
        _copy_tree(ROOT / name, package_root / name)

    for name in INCLUDE_FILES:
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, package_root / name)

    data_dst = package_root / "data"
    data_dst.mkdir(exist_ok=True)
    for rel in DATA_FILES:
        src = ROOT / rel
        if src.is_file():
            shutil.copy2(src, data_dst / src.name)
        else:
            print(f"WARNING: missing {rel}")

    # Drop packaging script from the submitted scripts/ (optional clutter)
    pack_script = package_root / "scripts" / "package_submission.py"
    if pack_script.exists():
        pack_script.unlink()

    zip_name = f"{topic_code}_MaNguon.zip"
    zip_path = dist / zip_name
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in package_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path} ({size_mb:.1f} MB)")
    if size_mb > 1024:
        print("WARNING: package > 1 GB — upload to Google Drive per submission rules.")
    print(f"Staging folder kept at: {package_root}")
    print(
        "Rename topic folder/zip to your official MaDeTai code before submit "
        "if you used the default name."
    )
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package thesis source zip")
    parser.add_argument(
        "--topic-code",
        default="MaDeTai",
        help="Folder + zip prefix (official topic code)",
    )
    args = parser.parse_args()
    build(args.topic_code)


if __name__ == "__main__":
    main()
