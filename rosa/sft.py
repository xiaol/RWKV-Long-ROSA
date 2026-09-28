"""Single-response JSONL preparation and task metrics for novel-agent experiments."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch.nn import functional as F


def load_examples(path, tokenizer, max_length):
    examples = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{line_number}: expected system/user/assistant")
            prompt = (messages[0]["content"] + "\n\nUser: " + messages[1]["content"]
                      + "\n\nAssistant:")
            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(" " + messages[2]["content"] + "\n\n")
            if not prompt_ids or not answer_ids:
                raise ValueError(f"{path}:{line_number}: empty token sequence")
            if len(prompt_ids) + len(answer_ids) > max_length:
                raise ValueError(f"{path}:{line_number}: exceeds max_length; no silent truncation")
            examples.append(dict(prompt_ids=prompt_ids, answer_ids=answer_ids,
                                 prompt=prompt, user=messages[1]["content"],
                                 target=json.loads(messages[2]["content"]), row=line_number))
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples


def make_batch(example, device, disable_memory=False):
    prefix = example["prompt_ids"]
    tokens = prefix + example["answer_ids"]
    inputs = torch.tensor([tokens[:-1]], device=device)
    labels = torch.tensor([tokens[1:]], device=device)
    labels[:, :len(prefix) - 1] = -100
    memory_mask = torch.arange(inputs.shape[1], device=device)[None] < len(prefix)
    if disable_memory:
        memory_mask.zero_()
    return inputs, labels, {"memory_mask": memory_mask}


def answer_loss(logits, labels):
    selected = labels != -100
    if not selected.any():
        raise ValueError("No supervised assistant tokens")
    return F.cross_entropy(logits[selected].float(), labels[selected])


def load_scene_examples(path, tokenizer, max_length):
    examples = []
    marker_pattern = re.compile(r"\[P(\d+)\]")
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{line_number}: expected system/user/assistant")
            system, user, answer = (message["content"] for message in messages)
            target = json.loads(answer)
            if not isinstance(target, dict) or not isinstance(target.get("boundaries"), list):
                raise ValueError(f"{path}:{line_number}: expected boundaries list")
            markers = list(marker_pattern.finditer(user))
            if not markers or len({int(marker.group(1)) for marker in markers}) != len(markers):
                raise ValueError(f"{path}:{line_number}: paragraph markers must be unique")
            prompt = system + "\n\nUser: " + user + "\n\nAssistant:"
            user_offset = len(system + "\n\nUser: ")
            prompt_ids = tokenizer.encode(prompt)
            positions = []
            paragraph_ids = []
            for index, marker in enumerate(markers):
                end = markers[index + 1].start() if index + 1 < len(markers) else len(user)
                while end > marker.end() and user[end - 1].isspace():
                    end -= 1
                prefix = prompt[:user_offset + end]
                position = len(tokenizer.encode(prefix)) - 1
                if position < 0 or position >= len(prompt_ids):
                    raise ValueError(f"{path}:{line_number}: invalid paragraph token position")
                paragraph_ids.append(int(marker.group(1)))
                positions.append(position)
            if len(prompt_ids) > max_length:
                raise ValueError(f"{path}:{line_number}: exceeds max_length; no silent truncation")
            boundaries = {int(value) for value in target["boundaries"]}
            unknown = boundaries.difference(paragraph_ids)
            if unknown:
                raise ValueError(f"{path}:{line_number}: unknown boundary IDs {sorted(unknown)}")
            examples.append(dict(prompt_ids=prompt_ids, user=user, paragraph_ids=paragraph_ids,
                                 paragraph_positions=positions,
                                 boundary_labels=[int(value in boundaries) for value in paragraph_ids],
                                 target=target, row=line_number))
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples


def boundary_metrics(gold_rows, predicted_rows):
    if len(gold_rows) != len(predicted_rows):
        raise ValueError("Expected one boundary prediction per example")
    true_positive = predicted = gold = exact = 0
    for example, prediction in zip(gold_rows, predicted_rows):
        actual = {paragraph_id for paragraph_id, label in zip(
            example["paragraph_ids"], example["boundary_labels"]) if label}
        predicted_set = {paragraph_id for paragraph_id, label in zip(
            example["paragraph_ids"], prediction) if label}
        true_positive += len(actual & predicted_set)
        predicted += len(predicted_set)
        gold += len(actual)
        exact += int(actual == predicted_set)
    precision = true_positive / max(1, predicted)
    recall = true_positive / max(1, gold)
    return dict(rows=len(gold_rows), true_positive=true_positive, predicted=predicted,
                gold=gold, precision=precision, recall=recall,
                f1=2 * true_positive / max(1, predicted + gold),
                exact_match=exact / max(1, len(gold_rows)))


def score_predictions(examples, texts):
    if len(examples) != len(texts):
        raise ValueError("Expected one prediction per example")
    groups = {}
    for example, text in zip(examples, texts):
        target = example["target"]
        task = "scene" if "boundaries" in target else "unit" if "labels" in target else "attribution"
        stats = groups.setdefault(task, dict(rows=0, json_valid=0, schema_valid=0,
                                            exact=0, true_positive=0, predicted=0,
                                            gold=0, correct=0, total=0, uncertain_correct=0))
        stats["rows"] += 1
        try:
            prediction = json.loads(text.strip())
            stats["json_valid"] += 1
        except (ValueError, TypeError):
            prediction = None
        valid = isinstance(prediction, dict) and prediction.keys() == target.keys()
        if task == "scene":
            paragraph_ids = {int(value) for value in re.findall(r"\[P(\d+)\]", example["user"])}
            valid = valid and isinstance(prediction.get("boundaries"), list)
            if valid:
                boundaries = prediction["boundaries"]
                valid = all(type(value) is int and value in paragraph_ids for value in boundaries)
                valid = valid and len(set(boundaries)) == len(boundaries)
            gold = set(target["boundaries"])
            predicted = set(prediction["boundaries"]) if valid else set()
            stats["true_positive"] += len(gold & predicted)
            stats["predicted"] += len(predicted)
            stats["gold"] += len(gold)
            stats["exact"] += int(valid and gold == predicted)
        elif task == "unit":
            gold = {label["unit_id"]: label["type"] for label in target["labels"]}
            valid = valid and isinstance(prediction.get("labels"), list)
            predicted = {}
            if valid:
                for label in prediction["labels"]:
                    if (not isinstance(label, dict) or set(label) != {"unit_id", "type"}
                            or not isinstance(label["unit_id"], str)
                            or label["unit_id"] in predicted
                            or label["type"] not in ("dialogue", "narration", "thought", "action", "scene_description")):
                        valid = False
                        break
                    predicted[label["unit_id"]] = label["type"]
                valid = valid and predicted.keys() == gold.keys()
            stats["total"] += len(gold)
            stats["correct"] += sum(predicted.get(key) == value for key, value in gold.items()) if valid else 0
            stats["exact"] += int(valid and gold == predicted)
        else:
            candidates = set(re.findall(r"^[-*]\s+(.+)$", example["user"], re.M))
            valid = (valid and isinstance(prediction.get("best_candidate"), str)
                     and type(prediction.get("uncertain")) is bool
                     and prediction["best_candidate"] in candidates)
            stats["correct"] += int(valid and prediction["best_candidate"] == target["best_candidate"])
            stats["uncertain_correct"] += int(valid and prediction["uncertain"] == target["uncertain"])
            stats["total"] += 1
            stats["exact"] += int(valid and prediction == target)
        stats["schema_valid"] += int(valid)
    for task, stats in groups.items():
        stats["json_valid_rate"] = stats["json_valid"] / stats["rows"]
        stats["schema_valid_rate"] = stats["schema_valid"] / stats["rows"]
        stats["exact_match"] = stats["exact"] / stats["rows"]
        if task == "scene":
            stats["boundary_precision"] = stats["true_positive"] / max(1, stats["predicted"])
            stats["boundary_recall"] = stats["true_positive"] / max(1, stats["gold"])
            stats["boundary_f1"] = 2 * stats["true_positive"] / max(1, stats["predicted"] + stats["gold"])
        else:
            stats["accuracy"] = stats["correct"] / max(1, stats["total"])
            if task == "attribution":
                stats["uncertain_accuracy"] = stats["uncertain_correct"] / stats["rows"]
    return groups
