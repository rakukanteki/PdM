"""DCASE directory discovery, recording splits, fold isolation, and datasets."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np
from torch.utils.data import Dataset

from .features import normalized_record
from .utils import read_csv, sha256, write_csv, write_json

FIELDS = ("recording_id", "path", "category", "machine_id", "split", "label", "audio_sha256")
SPLITS = {"fit", "validation", "calibration", "test"}


def make_manifest(root, output, seed=42, validation_fraction=0.10, calibration_fraction=0.10):
    """A newly chosen split, not a reconstruction of the manuscript's lost split."""
    root = Path(root).resolve()
    if min(validation_fraction, calibration_fraction) <= 0 or validation_fraction + calibration_fraction >= 1:
        raise ValueError("Validation and calibration fractions must be positive and sum to <1")
    rows, seen_hashes = [], set()
    for path in sorted(root.rglob("*.wav")):
        rel = path.relative_to(root)
        indexes = [i for i, part in enumerate(rel.parts[:-1]) if part.lower() in {"train", "test"}]
        if not indexes:
            continue
        i = indexes[-1]
        if i == 0:
            raise ValueError("Expected ROOT/category/{train,test}/...wav")
        category, group = rel.parts[i - 1], rel.parts[i].lower()
        match = re.search(r"(?:^|_)id_(\d+)(?:_|\.)", path.name, flags=re.IGNORECASE)
        label_match = re.match(r"(normal|anomaly)_", path.name, flags=re.IGNORECASE)
        if not match or not label_match:
            raise ValueError(f"Expected DCASE normal_id_XX/anomaly_id_XX naming: {path}")
        label = int(label_match[1].lower() == "anomaly")
        if group == "train" and label:
            raise ValueError(f"Anomaly in normal-only training folder: {path}")
        digest = sha256(path)
        if digest in seen_hashes:
            raise ValueError(f"Byte-identical duplicate audio detected: {path}")
        seen_hashes.add(digest)
        rows.append(dict(recording_id=hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24],
                         path=rel.as_posix(), category=category, machine_id=match[1],
                         split="pending" if group == "train" else "test",
                         label=label, audio_sha256=digest))
    if not rows:
        raise ValueError(f"No DCASE recordings found under {root}")
    pools = defaultdict(list)
    for row in rows:
        if row["split"] == "pending":
            pools[(row["category"], row["machine_id"])].append(row)
    rng = np.random.default_rng(seed)
    for key, pool in sorted(pools.items()):
        order = rng.permutation(len(pool))
        nv = max(1, int(len(pool) * validation_fraction))
        nc = max(2, int(len(pool) * calibration_fraction))
        if len(pool) - nv - nc < 2:
            raise ValueError(f"Too few recordings for fit/validation/calibration: {key}")
        for position, idx in enumerate(order):
            pool[idx]["split"] = "validation" if position < nv else (
                "calibration" if position < nv + nc else "fit")
    validate_manifest(rows)
    write_csv(output, rows, FIELDS)
    counts = Counter((r["category"], r["machine_id"], r["split"], int(r["label"])) for r in rows)
    write_json(Path(output).with_suffix(".splits.json"), {
        "implementation_choice": "New seeded recording-level split; not the historical paper split",
        "seed": seed, "validation_fraction": validation_fraction,
        "calibration_fraction": calibration_fraction,
        "counts": [dict(category=k[0], machine_id=k[1], split=k[2], label=k[3], count=v)
                   for k, v in sorted(counts.items())]})
    return rows


def validate_manifest(rows):
    if not rows:
        raise ValueError("Empty manifest")
    ids, paths, hashes = set(), set(), set()
    for row in rows:
        if not set(FIELDS[:-1]).issubset(row):
            raise ValueError(f"Manifest requires columns {FIELDS}")
        if row["split"] not in SPLITS or int(row["label"]) not in (0, 1):
            raise ValueError(f"Invalid split or label: {row}")
        if row["split"] != "test" and int(row["label"]) != 0:
            raise ValueError("Fitting, validation, and calibration must be normal-only")
        if row["recording_id"] in ids or row["path"] in paths:
            raise ValueError("A recording/path occurs in multiple manifest rows")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["recording_id"]):
            raise ValueError("recording_id must contain only letters, digits, underscores and hyphens")
        digest = row.get("audio_sha256")
        if digest and digest in hashes:
            raise ValueError("Byte-identical recordings occur in multiple manifest rows")
        ids.add(row["recording_id"])
        paths.add(row["path"])
        if digest:
            hashes.add(digest)


def load_manifest(path):
    rows = read_csv(path)
    validate_manifest(rows)
    return rows


def fold_rows(rows, category, protocol, heldout=None):
    if protocol not in {"known", "unseen"}:
        raise ValueError("protocol must be known or unseen")
    if protocol == "unseen" and heldout is None:
        raise ValueError("unseen protocol requires a held-out machine ID")
    if protocol == "known" and heldout is not None:
        raise ValueError("known protocol cannot have a held-out ID")
    rows = [r for r in rows if r["category"] == category]
    result = {split: [] for split in sorted(SPLITS)}
    for row in rows:
        if row["split"] == "test":
            if protocol == "known" or row["machine_id"] == heldout:
                result["test"].append(row)
        elif protocol == "known" or row["machine_id"] != heldout:
            result[row["split"]].append(row)
    seen = sorted({r["machine_id"] for r in result["fit"]})
    if len(seen) < 2 or any(not result[s] for s in SPLITS):
        raise ValueError("Every fold needs >=2 fitting IDs and nonempty fit/validation/calibration/test splits")
    for split in ("fit", "validation", "calibration"):
        counts = Counter(r["machine_id"] for r in result[split])
        if set(counts) != set(seen):
            raise ValueError(f"Seen IDs missing or extraneous in {split}")
        if split != "validation" and min(counts.values()) < 2:
            raise ValueError(f"Need >=2 normal recordings per seen ID in {split}")
    if protocol == "known" and not {r["machine_id"] for r in result["test"]} <= set(seen):
        raise ValueError("Known-ID test contains an unseen ID")
    return result, {name: i for i, name in enumerate(seen)}


class PatchDataset(Dataset):
    def __init__(self, rows, store, stats, id_map, fitting=False):
        self.rows, self.store, self.stats = rows, store, stats
        self.id_map, self.fitting = id_map, fitting
        self.index = [(i, j) for i, r in enumerate(rows)
                      for j in range(len(store.patches(r, fitting)))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, j = self.index[index]
        patches, desc = normalized_record(self.rows[i], self.store, self.stats, self.fitting)
        return patches[j], desc, self.id_map[self.rows[i]["machine_id"]]


def sample_auxiliary(rows, store, stats, id_map, cfg, rng):
    """Full bags from distinct recordings; >=2 IDs so within-ID pair availability is explicit."""
    by_id = defaultdict(list)
    for i, row in enumerate(rows):
        by_id[row["machine_id"]].append(i)
    names = rng.choice(sorted(by_id), size=min(cfg.auxiliary_ids, len(by_id)), replace=False)
    patches, descriptors, groups, labels = [], [], [], []
    for name in names:
        indexes = rng.choice(by_id[name], size=min(cfg.auxiliary_recordings_per_id, len(by_id[name])),
                             replace=False)
        for index in indexes:
            p, d = normalized_record(rows[index], store, stats, fitting=True)
            groups.extend([len(descriptors)] * len(p))
            patches.append(p)
            descriptors.append(d)
            labels.append(id_map[name])
    return (np.concatenate(patches), np.asarray(descriptors),
            np.asarray(groups), np.asarray(labels))
