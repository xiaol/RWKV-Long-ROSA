"""Prepare deterministic, task-balanced JSONL splits for semantic-memory pilots.

This tool does not claim to recover the publisher's official split. It is intended
for local smoke tests when only JSONL blobs are available. Exact duplicate user
prompts are deduplicated before assignment and never cross the resulting split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def task_for(system):
    if "归因" in system:
        return "attribution"
    if "叙事单元" in system:
        return "unit"
    if "scene" in system.lower() or "边界" in system:
        return "scene"
    raise ValueError(f"Unrecognized task system prompt: {system[:100]!r}")


def normalized_prompt(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Each row must contain exactly three messages")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Messages must be system, user, assistant")
    if not all(isinstance(message.get("content"), str) for message in messages):
        raise ValueError("Message content must be strings")
    json.loads(messages[2]["content"])
    return "".join(messages[1]["content"].split())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be strictly between zero and one")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    return args


def main():
    args = parse_args()
    rows = []
    seen = set()
    sources = []
    duplicate_count = 0
    for path in sorted(args.input):
        if not path.is_file():
            raise ValueError(f"Input is not a file: {path}")
        source_count = Counter()
        source_duplicates = 0
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    key = normalized_prompt(row)
                    task = task_for(row["messages"][0]["content"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}:{line_number}: invalid example: {error}") from error
                identity = (task, key)
                if identity in seen:
                    duplicate_count += 1
                    source_duplicates += 1
                    continue
                seen.add(identity)
                rows.append((task, row))
                source_count[task] += 1
        sources.append(dict(path=str(path), sha256=digest(path), rows=sum(source_count.values()),
                            tasks=dict(source_count), duplicates=source_duplicates))

    train = []
    validation = []
    for task, row in rows:
        key = normalized_prompt(row)
        score = int(hashlib.sha256(f"{args.seed}\0{task}\0{key}".encode()).hexdigest()[:16], 16) / 2**64
        (validation if score < args.validation_fraction else train).append(row)

    train.sort(key=lambda row: (task_for(row["messages"][0]["content"]), normalized_prompt(row)))
    validation.sort(key=lambda row: (task_for(row["messages"][0]["content"]), normalized_prompt(row)))
    train_keys = {(task_for(row["messages"][0]["content"]), normalized_prompt(row)) for row in train}
    validation_keys = {(task_for(row["messages"][0]["content"]), normalized_prompt(row)) for row in validation}
    overlap = train_keys & validation_keys
    if overlap:
        raise RuntimeError(f"Split leakage detected: {len(overlap)} duplicate prompts")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, selected in (("train.jsonl", train), ("validation.jsonl", validation)):
        with (args.output / name).open("w", encoding="utf-8") as destination:
            for row in selected:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(format="novel_agent_pilot_split_v1", seed=args.seed,
                    validation_fraction=args.validation_fraction, input_sources=sources,
                    exact_duplicates_removed=duplicate_count, total_rows=len(rows),
                    train_rows=len(train), validation_rows=len(validation),
                    train_tasks=dict(Counter(task_for(row["messages"][0]["content"]) for row in train)),
                    validation_tasks=dict(Counter(task_for(row["messages"][0]["content"]) for row in validation)))
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
