"""Select scene cutoffs on development data, then evaluate saved controls on test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_scene_head import fingerprint


def select_threshold(metrics):
    candidates = metrics.get("threshold_sweep", [])
    if not candidates:
        raise ValueError("The development report must contain a threshold sweep")
    return max(candidates, key=lambda entry: (
        entry["f1"], -abs(entry["threshold"] - 0.5), -entry["threshold"]))


def prompt_keys(path):
    with Path(path).open(encoding="utf-8") as source:
        return {"".join(json.loads(line)["messages"][1]["content"].split())
                for line in source if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    if len({run.name for run in args.runs}) != len(args.runs):
        parser.error("Run directory names must be distinct")
    test_keys = prompt_keys(args.test)
    selections = []
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        arguments = manifest["arguments"]
        if arguments["mode"] != "train":
            raise ValueError(f"{run}: expected a training run")
        for name, hash_key in (("train", "train_sha256"), ("validation", "validation_sha256")):
            source = Path(arguments[name])
            if fingerprint(source) != manifest[hash_key]:
                raise ValueError(f"{source}: data changed since training")
            if test_keys & prompt_keys(source):
                raise ValueError(f"{source}: test prompts overlap training or development")
        kind = manifest.get("adapter_kind", "semantic")
        variant = {"none": "head_only", "local": "local_adapter",
                   "rosa": "rosa_adapter", "semantic": "multi_hop"}[kind]
        development = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
        selected = select_threshold(development[variant])
        selections.append(dict(run=str(run), variant=variant, seed=arguments["seed"],
                               threshold=selected["threshold"], development=selected,
                               development_sha256=manifest["validation_sha256"],
                               development_report_sha256=fingerprint(run / "evaluation.json"),
                               checkpoint_sha256=fingerprint(run / "adapter.pt")))
    args.output.mkdir(parents=True, exist_ok=True)
    selection = dict(test_sha256=fingerprint(args.test), selections=selections,
                     selection_rule="Maximum development F1; ties closest to 0.5, then lower cutoff",
                     split_limitation="Prompt disjointness only; source novel IDs unavailable")
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    results = []
    for selected in selections:
        run = Path(selected["run"])
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        destination = args.output / run.name
        command = [sys.executable, "-u", str(Path(__file__).with_name("train_scene_head.py")),
                   "evaluate", "--checkpoint", str(run / "adapter.pt"),
                   "--validation", str(args.test), "--output", str(destination),
                   "--device", args.device, "--vocab", manifest["arguments"]["vocab"],
                   "--max-length", str(manifest["arguments"]["max_length"]),
                   "--threshold", str(selected["threshold"])]
        print(f"Evaluating {run.name} at development cutoff {selected['threshold']}", flush=True)
        subprocess.run(command, check=True)
        report = json.loads((destination / "evaluation.json").read_text(encoding="utf-8"))
        results.append(dict(**selected, test=report[selected["variant"]]))
        (args.output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
