"""Command-line entry points. Run `python -m dr_aad --help`."""
import argparse
from pathlib import Path

from .audio import CONDITIONS
from .config import Config
from .utils import read_csv, write_json


def parser():
    p = argparse.ArgumentParser(description="DR-AAD manuscript-based implementation")
    commands = p.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("make-manifest", help="Create a new recording-level DCASE split")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--validation-fraction", type=float, default=0.10)
    manifest.add_argument("--calibration-fraction", type=float, default=0.10)
    config = commands.add_parser("write-config", help="Write the full default configuration")
    config.add_argument("--output", required=True)
    for name in ("train", "baseline"):
        c = commands.add_parser(name)
        c.add_argument("--manifest", required=True)
        c.add_argument("--root", required=True)
        c.add_argument("--cache", default="cache")
        c.add_argument("--output", required=True)
        c.add_argument("--config")
        c.add_argument("--category", required=True)
        c.add_argument("--protocol", choices=("known", "unseen"), default="known")
        c.add_argument("--heldout")
        if name == "train":
            c.add_argument("--variant", choices=("cnn", "fusion", "relational", "full"), default="full")
            c.add_argument("--device", default="cpu")
            c.add_argument("--seeds", nargs="+", type=int)
            c.add_argument("--epochs", type=int)
        else:
            c.add_argument("--method", choices=("iforest", "ocsvm", "knn"), default="iforest")
            c.add_argument("--seed", type=int, default=42)
            c.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["clean", "noise20", "noise10"])
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--root", required=True)
    evaluate.add_argument("--cache", default="cache")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--seeds", type=int, nargs="+")
    evaluate.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["clean", "noise20", "noise10"])
    predict = commands.add_parser("predict", help="Score a WAV using saved clean calibration")
    predict.add_argument("--run", required=True)
    predict.add_argument("--calibration", required=True)
    predict.add_argument("--audio", required=True)
    predict.add_argument("--machine-id")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True)
    review = commands.add_parser("review")
    review.add_argument("--predictions", required=True)
    review.add_argument("--condition", default="clean", choices=CONDITIONS)
    review.add_argument("--coverage", type=float, default=0.90)
    review.add_argument("--pool", choices=("global", "id"), default="global")
    review.add_argument("--seed", type=int, default=42)
    review.add_argument("--output", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--predictions", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--max-fpr", type=float, default=0.10)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--run", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--repetitions", type=int, default=100)
    benchmark.add_argument("--batch-size", type=int, default=32)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "make-manifest":
        from .data import make_manifest
        if Path(args.output).exists():
            raise FileExistsError(f"Manifest already exists: {args.output}")
        rows = make_manifest(args.root, args.output, args.seed, args.validation_fraction,
                             args.calibration_fraction)
        print(f"Created manifest for {len(rows)} recordings")
    elif args.command == "write-config":
        write_json(args.output, Config().to_dict())
    elif args.command in {"train", "baseline"}:
        from .data import load_manifest
        cfg = Config.load(args.config)
        if args.command == "train":
            from .training import train_run
            if args.seeds is not None:
                cfg.training.seeds = args.seeds
            if args.epochs is not None:
                cfg.training.epochs = args.epochs
            train_run(load_manifest(args.manifest), cfg, args.root, args.cache, args.output,
                      args.category, args.protocol, args.heldout, args.variant, args.device)
        else:
            from .baselines import run_baseline
            run_baseline(load_manifest(args.manifest), cfg, args.root, args.cache, args.output,
                         args.category, args.method, args.protocol, args.heldout, args.seed, args.conditions)
    elif args.command == "evaluate":
        from .evaluation import evaluate_run
        evaluate_run(args.run, args.root, args.cache, args.output, args.conditions, args.device, args.seeds)
    elif args.command == "predict":
        from .inference import load_detector
        detector = load_detector(args.run, args.calibration, args.device)
        write_json(args.output, detector.predict_file(args.audio, args.machine_id))
    elif args.command == "review":
        from .metrics import review_metrics
        rows = [r for r in read_csv(args.predictions) if r["condition"] == args.condition]
        if not rows:
            raise ValueError("No predictions for requested condition")
        experiment_keys = {(r["protocol"], r["variant"], r["seeds"]) for r in rows}
        if len(experiment_keys) != 1:
            raise ValueError("Review one model/protocol/seed selection at a time")
        write_json(args.output, review_metrics(rows, args.coverage, args.pool, args.seed))
    elif args.command == "summarize":
        from .evaluation import summarize_predictions
        summarize_predictions(args.predictions, args.output, args.max_fpr)
    elif args.command == "benchmark":
        from .benchmark import benchmark_run
        benchmark_run(args.run, args.output, args.device, args.warmup, args.repetitions, args.batch_size)
