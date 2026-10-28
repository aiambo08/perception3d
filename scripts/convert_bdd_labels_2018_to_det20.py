"""Convert BDD100K 2018 per-image JSONs → Detection-2020-style ``det_*.json``.

The official ``det_20`` ZIP is currently offline / mis-hosted on the Berkeley
mirror (see bdd100k#371). The still-available ``bdd100k_labels.zip`` (2018)
has one JSON per image under ``bdd100k/labels/100k/{train,val,test}/``.

This script:
* maps ``person→pedestrian``, ``bike→bicycle``, ``motor→motorcycle``;
* keeps only detection classes with ``box2d``;
* writes a single Scalabel list JSON compatible with ``eval/bdd.py``.

Example::

    uv run python scripts/convert_bdd_labels_2018_to_det20.py \\
        --root ~/datasets/bdd100k --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAME_MAP = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}
DET_CLASSES = {
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
}


def _objects_from_2018(doc: dict) -> list[dict]:
    frames = doc.get("frames") or ()
    if not frames:
        return []
    objects = frames[0].get("objects") or ()
    out: list[dict] = []
    for obj in objects:
        cat = NAME_MAP.get(str(obj.get("category") or ""), str(obj.get("category") or ""))
        if cat not in DET_CLASSES or "box2d" not in obj:
            continue
        out.append(
            {
                "id": str(obj.get("id")),
                "category": cat,
                "attributes": obj.get("attributes") or {},
                "box2d": obj["box2d"],
            }
        )
    return out


def convert_split(split_dir: Path, out_file: Path) -> tuple[int, int]:
    frames: list[dict] = []
    n_boxes = 0
    for path in sorted(split_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        name = str(doc.get("name") or path.stem)
        if not name.endswith((".jpg", ".jpeg", ".png")):
            name = f"{name}.jpg"
        labels = _objects_from_2018(doc)
        n_boxes += len(labels)
        frames.append(
            {
                "name": name,
                "attributes": doc.get("attributes") or {},
                "labels": labels,
            }
        )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(frames), encoding="utf-8")
    return len(frames), n_boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="bdd100k dataset root")
    ap.add_argument("--split", choices=("train", "val", "test"), default="val")
    ap.add_argument(
        "--labels-2018",
        type=Path,
        help="default: <root>/bdd100k/labels/100k/<split> or <root>/labels/100k/<split>",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="default: <root>/labels/det_20/det_<split>.json",
    )
    args = ap.parse_args()

    candidates = [
        args.labels_2018,
        args.root / "bdd100k" / "labels" / "100k" / args.split,
        args.root / "labels" / "100k" / args.split,
    ]
    split_dir = next((p for p in candidates if p is not None and p.is_dir()), None)
    if split_dir is None:
        raise SystemExit(
            f"2018 labels dir not found under {args.root}; "
            "unzip bdd100k_labels.zip first"
        )
    out = args.out or (args.root / "labels" / "det_20" / f"det_{args.split}.json")
    n_img, n_box = convert_split(split_dir, out)
    print(f"{args.split}: {n_img} images, {n_box} boxes → {out}")


if __name__ == "__main__":
    main()
