"""Train/evaluate semantic multi-hop memory on explicit novel-agent JSONL splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rosa.gpu import pick_gpu
from rosa.rwkv7 import MODELS, VOCAB, load_rwkv7
from rosa.semantic import SemanticMemoryAdapter
from rosa.sft import answer_loss, load_examples, make_batch, score_predictions


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


@torch.no_grad()
def generate(model, tokenizer, example, device, max_new_tokens, disable_memory=False):
    tokens = list(example["prompt_ids"])
    prefix_length = len(tokens)
    generated = []
    for step in range(max_new_tokens):
        inputs = torch.tensor([tokens], device=device)
        memory_mask = torch.arange(len(tokens), device=device)[None] < prefix_length
        if disable_memory:
            memory_mask.zero_()
        logits = model(inputs, aux={"memory_mask": memory_mask}, last_only=True)
        token = int(logits[0, -1].argmax())
        if token == 0:
            break
        tokens.append(token)
        generated.append(token)
        text = tokenizer.decode(generated)
        if "\n\n" in text:
            return text.split("\n\n", 1)[0].strip()
    return tokenizer.decode(generated).strip()


@torch.no_grad()
def evaluate(model, adapter, tokenizer, examples, device, with_generation, max_new_tokens):
    model.eval()
    original_hops = adapter.hops
    report = {}
    predictions = []
    try:
        for variant in ("memory_disabled", "one_hop", "multi_hop"):
            adapter.hops = 1 if variant == "one_hop" else original_hops
            disabled = variant == "memory_disabled"
            total_nll = 0.0
            total_tokens = 0
            texts = []
            for example in examples:
                inputs, labels, aux = make_batch(example, device, disabled)
                logits = model(inputs, aux=aux)
                count = int((labels != -100).sum())
                total_nll += float(answer_loss(logits, labels)) * count
                total_tokens += count
                del logits
                if with_generation:
                    text = generate(model, tokenizer, example, device, max_new_tokens, disabled)
                    texts.append(text)
                    predictions.append(dict(variant=variant, row=example["row"],
                                            prediction=text, target=example["target"]))
            report[variant] = dict(assistant_nll=total_nll / total_tokens,
                                   assistant_tokens=total_tokens, rows=len(examples))
            if with_generation:
                report[variant]["tasks"] = score_predictions(examples, texts)
    finally:
        adapter.hops = original_hops
    return report, predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", default="0.4b", help="Known model name or .pth path")
    parser.add_argument("--vocab", default=VOCAB)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--query-block", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--eval-rows", type=int, default=0,
                        help="Evaluate only the first N validation rows (0 means all)")
    args = parser.parse_args()
    if args.mode == "train" and (args.train is None or args.checkpoint is not None):
        parser.error("train requires --train and does not accept --checkpoint")
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    if min(args.steps, args.accum, args.max_length, args.max_new_tokens) < 1 or args.lr <= 0:
        parser.error("Steps, accumulation, lengths, and learning rate must be positive")
    if args.eval_rows < 0:
        parser.error("--eval-rows must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    return args


def main():
    args = parse_args()
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(str(args.vocab))
    validation = load_examples(args.validation, tokenizer, args.max_length)
    if args.eval_rows:
        validation = validation[:args.eval_rows]
    training = load_examples(args.train, tokenizer, args.max_length) if args.mode == "train" else []
    if training:
        check_disjoint(training, validation)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_gpu(12000) if args.device == "auto" else args.device
    saved = None
    model_path = str(Path(MODELS.get(args.model, args.model)).resolve())
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["format"] != "semantic_memory_v1":
            raise ValueError("Expected a semantic_memory_v1 checkpoint")
        if fingerprint(args.vocab) != saved["vocab_sha256"]:
            raise ValueError("Checkpoint tokenizer does not match --vocab")
        model_path = saved["model_path"]
        args.layer = saved["layer"]
    model_hash = fingerprint(model_path)
    if saved is not None and model_hash != saved["model_sha256"]:
        raise ValueError("Backbone contents changed since training")
    model = load_rwkv7(model_path, device=device)
    if not 1 <= args.layer < model.args.n_layer:
        raise ValueError("Choose an intermediate layer in [1, n_layer); layer 0 lacks contextual features")
    config = saved["config"] if saved is not None else dict(
        width=model.args.n_embd, memory_dim=args.memory_dim, hops=args.hops,
        chunk_size=args.chunk_size, query_block=args.query_block)
    adapter = SemanticMemoryAdapter(**config).to(device)
    if config["width"] != model.args.n_embd:
        raise ValueError("Adapter width does not match backbone")
    if saved is not None:
        adapter.load_state_dict(saved["adapter"])
    model.add_adapter(args.layer, adapter)
    model.grad_ckpt = True
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith("adapters."))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(arguments={key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                    model_path=model_path, model_sha256=model_hash,
                    vocab_sha256=fingerprint(args.vocab),
                    validation_sha256=fingerprint(args.validation),
                    train_sha256=fingerprint(args.train) if training else None,
                    checkpoint_sha256=fingerprint(args.checkpoint) if saved is not None else None,
                    config=adapter.config, trainable_parameters=sum(parameter.numel() for parameter in adapter.parameters()),
                    training_rows=len(training), validation_rows=len(validation))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if training:
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
        with (args.output / "train.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, args.steps + 1):
                model.eval()
                adapter.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for microstep in range(args.accum):
                    example = rng.choice(training)
                    inputs, labels, aux = make_batch(example, device)
                    logits = model(inputs, aux=aux)
                    loss = answer_loss(logits, labels) / args.accum
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at step {step}")
                    loss.backward()
                    loss_sum += float(loss.detach())
                    del logits, loss
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, assistant_nll=loss_sum, grad_norm=float(grad_norm))
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step == 1 or step % 10 == 0 or step == args.steps:
                    print(json.dumps(record), flush=True)
        torch.save(dict(format="semantic_memory_v1", config=adapter.config,
                        adapter={key: value.detach().cpu() for key, value in adapter.state_dict().items()},
                        model_path=model_path, model_sha256=model_hash,
                        vocab_sha256=manifest["vocab_sha256"], layer=args.layer, steps=args.steps),
                   args.output / "adapter.pt")
    report, predictions = evaluate(model, adapter, tokenizer, validation, device,
                                   args.generate, args.max_new_tokens)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if predictions:
        with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as destination:
            for prediction in predictions:
                destination.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
