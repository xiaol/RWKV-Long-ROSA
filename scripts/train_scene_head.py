"""Train a paragraph-level scene-boundary head with frozen RWKV semantic memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rosa.gpu import pick_gpu
from rosa.rwkv7 import MODELS, VOCAB, load_rwkv7
from rosa.adapters import RosaFeatures, RosaInputAdapter
from rosa.semantic import SemanticMemoryAdapter
from rosa.sft import boundary_metrics, load_scene_examples
from rosa.tasks import InitialStateTuner, LocalResidualAdapter, SceneBoundaryHead


class SceneRosaInputAdapter(RosaInputAdapter):
    def forward(self, hidden, aux):
        if aux.get("disable_adapter", False):
            return torch.zeros_like(hidden)
        return super().forward(hidden, aux)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_disjoint(training, validation):
    prompts = {"".join(example["user"].split()) for example in training}
    if any("".join(example["user"].split()) in prompts for example in validation):
        raise ValueError("Train and validation contain an overlapping user prompt")


def batch(example, device, disable_memory=False, disable_adapter=False, disable_state=False,
          rosa_features=None):
    inputs = torch.tensor([example["prompt_ids"]], device=device)
    memory_mask = torch.ones_like(inputs, dtype=torch.bool)
    if disable_memory:
        memory_mask.zero_()
    positions = torch.tensor([example["paragraph_positions"]], device=device, dtype=torch.long)
    labels = torch.tensor([example["boundary_labels"]], device=device, dtype=torch.float32)
    aux = {"memory_mask": memory_mask, "disable_adapter": disable_adapter,
           "disable_state": disable_state}
    if rosa_features is not None:
        aux.update(rosa_features(inputs))
    return inputs, positions, labels, aux


def forward(model, head, example, device, disable_memory=False, disable_adapter=False,
            disable_state=False, rosa_features=None):
    inputs, positions, labels, aux = batch(example, device, disable_memory, disable_adapter,
                                           disable_state, rosa_features)
    _, hidden = model(inputs, aux=aux, return_hidden=True)
    return head(hidden, positions), labels


@torch.no_grad()
def evaluate(model, adapter, head, examples, device, threshold, thresholds, adapter_kind,
             rosa_features=None):
    model.eval()
    head.eval()
    report = {}
    semantic = adapter_kind in ("semantic", "state_semantic")
    original_hops = adapter.hops if semantic else None
    try:
        variants = (("memory_disabled", True, True), ("one_hop", False, False),
                    ("multi_hop", False, False)) if semantic else (
                    ("memory_disabled", True, True), (adapter_kind + "_adapter", False, False))
        if adapter_kind == "none":
            variants = (("head_only", False, False),)
        if adapter_kind == "state":
            variants = (("state_disabled", False, False), ("state_tuned", False, False))
        if adapter_kind == "state_semantic":
            variants += (("state_disabled", False, False),)
        for variant, disabled_memory, disabled_adapter in variants:
            if semantic:
                adapter.hops = 1 if variant == "one_hop" else original_hops
            score_rows = []
            total_loss = 0.0
            total_labels = 0
            for example in examples:
                logits, labels = forward(model, head, example, device, disabled_memory,
                                         disabled_adapter,
                                         variant == "state_disabled",
                                         rosa_features)
                total_loss += float(F.binary_cross_entropy_with_logits(logits, labels)) * labels.numel()
                total_labels += labels.numel()
                score_rows.append(logits[0].sigmoid().cpu().tolist())
            predicted_rows = [[int(score >= threshold) for score in scores] for scores in score_rows]
            metrics = boundary_metrics(examples, predicted_rows)
            metrics.update(binary_nll=total_loss / total_labels, labels=total_labels)
            metrics["threshold"] = threshold
            metrics["threshold_sweep"] = [
                dict(threshold=level, **boundary_metrics(
                    examples, [[int(score >= level) for score in scores] for scores in score_rows]))
                for level in thresholds
            ]
            report[variant] = metrics
    finally:
        if semantic:
            adapter.hops = original_hops
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", default="0.4b")
    parser.add_argument("--adapter-kind", choices=("semantic", "local", "rosa", "state",
                                                    "state_semantic", "none"), default="semantic")
    parser.add_argument("--rosa-k", type=int, default=4)
    parser.add_argument("--vocab", default=VOCAB)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--query-block", type=int, default=128)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--local-hidden", type=int, default=132)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="",
                        help="Optional development-only probability sweep; never select on test")
    args = parser.parse_args()
    if args.mode == "train" and (args.train is None or args.checkpoint is not None):
        parser.error("train requires --train and does not accept --checkpoint")
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    if min(args.steps, args.accum, args.max_length, args.head_hidden,
           args.local_hidden, args.rosa_k) < 1 or args.lr <= 0:
        parser.error("Steps, accumulation, lengths, and learning rate must be positive")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between zero and one")
    try:
        args.thresholds = tuple(float(level) for level in args.thresholds.split(",")) if args.thresholds else ()
    except ValueError as error:
        parser.error(f"invalid --thresholds: {error}")
    if any(not 0 < level < 1 for level in args.thresholds):
        parser.error("--thresholds must contain probabilities between zero and one")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    return args


def main():
    args = parse_args()
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(str(args.vocab))
    validation = load_scene_examples(args.validation, tokenizer, args.max_length)
    training = load_scene_examples(args.train, tokenizer, args.max_length) if args.mode == "train" else []
    if training:
        check_disjoint(training, validation)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_gpu(12000) if args.device == "auto" else args.device
    saved = None
    model_path = str(Path(MODELS.get(args.model, args.model)).resolve())
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["format"] != "scene_boundary_v1":
            raise ValueError("Expected a scene_boundary_v1 checkpoint")
        if saved.get("adapter_kind", "semantic") != args.adapter_kind:
            args.adapter_kind = saved.get("adapter_kind", "semantic")
        if fingerprint(args.vocab) != saved["vocab_sha256"]:
            raise ValueError("Checkpoint tokenizer does not match --vocab")
        model_path = saved["model_path"]
        args.layer = saved["layer"]
    model_hash = fingerprint(model_path)
    if saved is not None and model_hash != saved["model_sha256"]:
        raise ValueError("Backbone contents changed since training")
    model = load_rwkv7(model_path, device=device)
    if not 1 <= args.layer < model.args.n_layer:
        raise ValueError("Choose an intermediate layer in [1, n_layer)")
    if saved is not None:
        adapter_kind = saved.get("adapter_kind", "semantic")
        adapter_config = saved["adapter_config"]
        state_config = saved.get("state_config", {})
    else:
        adapter_kind = args.adapter_kind
        state_config = dict(layers=model.args.n_layer, heads=model.args.n_embd // 64,
                            head_size=64)
        adapter_config = (dict(width=model.args.n_embd, memory_dim=args.memory_dim, hops=args.hops,
                               chunk_size=args.chunk_size, query_block=args.query_block)
                          if adapter_kind in ("semantic", "state_semantic") else
                          dict(width=model.args.n_embd, hidden=args.local_hidden)
                          if adapter_kind == "local" else
                          dict(K=args.rosa_k, hidden=args.head_hidden)
                          if adapter_kind == "rosa" else {})
    head_config = saved["head_config"] if saved is not None else dict(
        width=model.args.n_embd, hidden=args.head_hidden)
    if adapter_kind in ("semantic", "state_semantic"):
        adapter = SemanticMemoryAdapter(**adapter_config).to(device)
    elif adapter_kind == "local":
        adapter = LocalResidualAdapter(**adapter_config).to(device)
    elif adapter_kind == "rosa":
        adapter = SceneRosaInputAdapter(model.emb, **adapter_config).to(device)
    else:
        adapter = None
    state_adapter = (InitialStateTuner(**state_config).to(device)
                     if adapter_kind in ("state", "state_semantic") else None)
    rosa_features = RosaFeatures(adapter_config["K"]) if adapter_kind == "rosa" else None
    torch.manual_seed(args.seed)
    head = SceneBoundaryHead(**head_config).to(device)
    if saved is not None:
        if adapter is not None:
            adapter.load_state_dict(saved["adapter"])
        if state_adapter is not None:
            state_adapter.load_state_dict(saved["state_adapter"])
        head.load_state_dict(saved["head"])
    if adapter is not None:
        model.add_adapter(args.layer, adapter)
    if state_adapter is not None:
        model.add_state_adapter(state_adapter)
    model.grad_ckpt = adapter is not None or state_adapter is not None
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if not name.startswith(("adapters.", "state_adapter."))):
        raise RuntimeError("The RWKV backbone must remain frozen")
    adapter_parameters = list(adapter.parameters()) if adapter is not None else []
    state_parameters = list(state_adapter.parameters()) if state_adapter is not None else []
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(format="scene_boundary_v1", arguments={key: str(value) if isinstance(value, Path) else value
                                                             for key, value in vars(args).items()},
                    model_path=model_path, model_sha256=model_hash,
                    vocab_sha256=fingerprint(args.vocab),
                    validation_sha256=fingerprint(args.validation),
                    train_sha256=fingerprint(args.train) if training else None,
                    adapter_kind=adapter_kind, adapter_config=adapter_config, head_config=head.config,
                    state_config=state_config,
                    checkpoint_sha256=fingerprint(args.checkpoint) if saved is not None else None,
                    adapter_parameters=sum(parameter.numel() for parameter in adapter_parameters),
                    state_parameters=sum(parameter.numel() for parameter in state_parameters),
                    head_parameters=sum(parameter.numel() for parameter in head.parameters()),
                    training_rows=len(training), validation_rows=len(validation))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if training:
        positive = sum(sum(example["boundary_labels"]) for example in training)
        negative = sum(len(example["boundary_labels"]) for example in training) - positive
        pos_weight = torch.tensor([negative / max(1, positive)], device=device)
        trainable = adapter_parameters + state_parameters + list(head.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        with (args.output / "train.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, args.steps + 1):
                model.eval()
                if adapter is not None:
                    adapter.train()
                head.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for _ in range(args.accum):
                    example = rng.choice(training)
                    logits, labels = forward(model, head, example, device,
                                             rosa_features=rosa_features)
                    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight) / args.accum
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at step {step}")
                    loss.backward()
                    loss_sum += float(loss.detach())
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, loss=loss_sum, grad_norm=float(grad_norm))
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step == 1 or step % 10 == 0 or step == args.steps:
                    print(json.dumps(record), flush=True)
        torch.save(dict(format="scene_boundary_v1", adapter_kind=adapter_kind,
                        adapter_config=adapter_config,
                        state_config=state_config,
                        head_config=head.config,
                        adapter={key: value.detach().cpu() for key, value in adapter.state_dict().items()}
                        if adapter is not None else {},
                        state_adapter={key: value.detach().cpu() for key, value in state_adapter.state_dict().items()}
                        if state_adapter is not None else {},
                        head={key: value.detach().cpu() for key, value in head.state_dict().items()},
                        model_path=model_path, model_sha256=model_hash,
                        vocab_sha256=manifest["vocab_sha256"], layer=args.layer, steps=args.steps),
                   args.output / "adapter.pt")
    report = evaluate(model, adapter, head, validation, device, args.threshold, args.thresholds,
                      adapter_kind, rosa_features)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
