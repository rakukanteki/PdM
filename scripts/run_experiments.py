"""Sequential experiment driver; invoke from the repository root after installation."""
import argparse
from pathlib import Path
import subprocess
import sys

from dr_aad.data import load_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--categories", nargs="+")
    p.add_argument("--variants", nargs="+", choices=("cnn", "fusion", "relational", "full"),
                   default=["cnn", "fusion", "relational", "full"])
    p.add_argument("--protocols", nargs="+", choices=("known", "unseen"), default=["known", "unseen"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1337])
    p.add_argument("--epochs", type=int)
    args = p.parse_args()
    rows = load_manifest(args.manifest)
    categories = args.categories or sorted({r["category"] for r in rows})
    evaluations = []
    for category in categories:
        ids = sorted({r["machine_id"] for r in rows if r["category"] == category and r["split"] == "fit"})
        if not ids:
            raise ValueError(f"No fitting IDs for category {category}")
        for protocol in args.protocols:
            for heldout in [None] if protocol == "known" else ids:
                for variant in args.variants:
                    tag = "known" if heldout is None else "unseen_" + heldout
                    base = Path(args.output) / category / tag / variant
                    train, evaluate = base / "run", base / "evaluation"
                    command = [sys.executable, "-m", "dr_aad", "train", "--root", args.root,
                        "--manifest", args.manifest, "--cache", args.cache, "--config", args.config,
                        "--category", category, "--protocol", protocol, "--variant", variant,
                        "--output", str(train), "--device", args.device,
                        "--seeds", *map(str, args.seeds)]
                    if heldout is not None:
                        command.extend(["--heldout", heldout])
                    if args.epochs is not None:
                        command.extend(["--epochs", str(args.epochs)])
                    subprocess.run(command, check=True)
                    subprocess.run([sys.executable, "-m", "dr_aad", "evaluate", "--run", str(train),
                        "--root", args.root, "--cache", args.cache, "--output", str(evaluate),
                        "--device", args.device], check=True)
                    evaluations.append(str(evaluate / "predictions.csv"))
    subprocess.run([sys.executable, "-m", "dr_aad", "summarize", "--predictions", *evaluations,
                    "--output", str(Path(args.output) / "summary.json")], check=True)


if __name__ == "__main__":
    main()
