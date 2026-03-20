from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManifestRow:
    path: str
    class_name: str
    label_index: int


def load_manifest_rows(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ManifestRow(
                    path=row["path"],
                    class_name=row["class_name"],
                    label_index=int(row["label_index"]),
                )
            )
    return rows


def load_class_to_index(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifests(manifest_dir: Path) -> dict[str, list[ManifestRow]]:
    return {
        "train": load_manifest_rows(manifest_dir / "train.csv"),
        "val": load_manifest_rows(manifest_dir / "val.csv"),
        "test": load_manifest_rows(manifest_dir / "test.csv"),
    }
