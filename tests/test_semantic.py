import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rosa.semantic import SemanticMemoryAdapter
from rosa.sft import (answer_loss, boundary_metrics, load_examples, load_scene_examples,
                      make_batch, score_predictions)
from rosa.tasks import InitialStateTuner, LocalResidualAdapter, SceneBoundaryHead


def active_adapter(**kwargs):
    torch.manual_seed(7)
    adapter = SemanticMemoryAdapter(8, memory_dim=4, chunk_size=2, **kwargs)
    nn.init.normal_(adapter.output.weight, std=0.2)
    return adapter


def test_zero_initialization_and_no_memory():
    hidden = torch.randn(2, 7, 8)
    adapter = SemanticMemoryAdapter(8, chunk_size=2)
    assert torch.equal(adapter(hidden), torch.zeros_like(hidden))
    nn.init.normal_(adapter.output.weight)
    empty = {"memory_mask": torch.zeros(2, 7, dtype=torch.bool)}
    assert torch.equal(adapter(hidden, empty), torch.zeros_like(hidden))
    assert torch.equal(adapter(hidden)[:, :2], torch.zeros_like(hidden[:, :2]))


def test_future_tokens_cannot_change_past_reads_or_gradients():
    adapter = active_adapter(query_block=3)
    hidden = torch.randn(1, 9, 8, requires_grad=True)
    full = adapter(hidden)
    for cut in (3, 4, 7):
        torch.testing.assert_close(adapter(hidden[:, :cut]), full[:, :cut])
        changed = hidden.detach().clone()
        changed[:, cut:] = torch.randn_like(changed[:, cut:]) * 30
        torch.testing.assert_close(adapter(changed)[:, :cut], full[:, :cut])
    gradient = torch.autograd.grad(full[:, 4].sum(), hidden)[0]
    assert torch.count_nonzero(gradient[:, 5:]) == 0
    assert torch.count_nonzero(gradient[:, :4]) > 0


def test_prompt_only_memory_does_not_retrieve_assistant_hidden_states():
    adapter = active_adapter()
    hidden = torch.randn(1, 10, 8)
    mask = torch.arange(10)[None] < 4
    expected = adapter(hidden, {"memory_mask": mask})[:, -1]
    changed = hidden.clone()
    changed[:, 4:9] += torch.randn_like(changed[:, 4:9]) * 20
    torch.testing.assert_close(adapter(changed, {"memory_mask": mask})[:, -1], expected)


def test_second_hop_uses_first_read_to_change_address():
    adapter = SemanticMemoryAdapter(2, memory_dim=2, hops=2)
    with torch.no_grad():
        adapter.update.weight.zero_()
        adapter.update.bias.zero_()
        adapter.update.weight[:, 2:] = 5 * torch.eye(2)
    query = torch.tensor([[[1.0, 0.0]]])
    keys = torch.eye(2)[None]
    values = torch.tensor([[[-1.0, 1.0], [1.0, 0.0]]])
    allowed = torch.ones(1, 1, 2, dtype=torch.bool)
    read, trace = adapter.retrieve(query, keys, values, allowed, return_trace=True)
    assert trace[0, 0, 0, 0] > 0.99
    assert trace[0, 1, 0, 1] > 0.99
    torch.testing.assert_close(read, trace[:, -1] @ values)
    altered = values.clone()
    altered[:, 0] = torch.tensor([1.0, -1.0])
    changed_read, changed_trace = adapter.retrieve(query, keys, altered, allowed, return_trace=True)
    assert changed_trace[0, 1, 0, 0] > 0.99
    assert not torch.allclose(read, changed_read)


def test_training_through_frozen_layers_and_checkpoint_roundtrip():
    torch.manual_seed(3)
    backbone = nn.Linear(8, 8)
    backbone.requires_grad_(False)
    before = {name: parameter.clone() for name, parameter in backbone.named_parameters()}
    adapter = SemanticMemoryAdapter(8, memory_dim=4, chunk_size=2)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.02)
    hidden = torch.randn(2, 8, 8)
    target = torch.randn_like(hidden)
    for step in range(3):
        optimizer.zero_grad()
        loss = (backbone(hidden + adapter(hidden)) - target).square().mean()
        loss.backward()
        if step > 0:
            for name in ("query.weight", "key.weight", "value.weight", "update.weight"):
                gradient = dict(adapter.named_parameters())[name].grad
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        optimizer.step()
    for name, parameter in backbone.named_parameters():
        assert parameter.grad is None and torch.equal(parameter, before[name])
    restored = SemanticMemoryAdapter(**adapter.config)
    restored.load_state_dict(adapter.state_dict())
    torch.testing.assert_close(adapter(hidden), restored(hidden))


def test_query_block_partition_and_batch_isolation():
    adapter = active_adapter(query_block=2)
    hidden = torch.randn(2, 9, 8)
    expected = adapter(hidden)
    adapter.query_block = 9
    torch.testing.assert_close(adapter(hidden), expected)
    for index in range(2):
        torch.testing.assert_close(adapter(hidden[index:index + 1]), expected[index:index + 1])


class CharacterTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]


def test_sft_mask_and_no_silent_truncation(tmp_path):
    source = tmp_path / "sample.jsonl"
    row = {"messages": [{"role": "system", "content": "JSON only"},
                        {"role": "user", "content": "[P1] 文本"},
                        {"role": "assistant", "content": '{"boundaries": []}'}]}
    source.write_text(json.dumps(row), encoding="utf-8")
    example = load_examples(source, CharacterTokenizer(), 1000)[0]
    inputs, labels, aux = make_batch(example, "cpu")
    prefix_length = len(example["prompt_ids"])
    assert (labels[:, :prefix_length - 1] == -100).all()
    assert labels[0, prefix_length - 1:].tolist() == example["answer_ids"]
    assert aux["memory_mask"].sum() == prefix_length
    assert not aux["memory_mask"][0, prefix_length:].any()
    logits = torch.randn(1, inputs.shape[1], 256, requires_grad=True)
    answer_loss(logits, labels).backward()
    assert torch.count_nonzero(logits.grad[:, :prefix_length - 1]) == 0
    with pytest.raises(ValueError, match="no silent truncation"):
        load_examples(source, CharacterTokenizer(), 10)


def test_schema_metrics_handle_invalid_outputs_and_noncontiguous_ids():
    examples = [dict(user="[P1] A\n[P2] B", target={"boundaries": [2]}),
                dict(user="[P1] A", target={"boundaries": []}),
                dict(user="[1] A\n[3] B", target={"labels": [
                    {"unit_id": "1", "type": "action"}, {"unit_id": "3", "type": "thought"}]}),
                dict(user="候选:\n- 甲\n- 乙", target={"best_candidate": "甲", "uncertain": False})]
    texts = [json.dumps(example["target"]) for example in examples]
    report = score_predictions(examples, texts)
    assert all(group["exact_match"] == 1 for group in report.values())
    assert report["scene"]["boundary_f1"] == 1
    bad = score_predictions(examples, ['{"boundaries": [true]}', "oops",
                                       '{"labels": [{"unit_id": "1", "type": "other"}]}',
                                       '{"best_candidate": "丙", "uncertain": false}'])
    assert all(group["schema_valid_rate"] == 0 for group in bad.values())
    assert all(group["exact_match"] == 0 for group in bad.values())


def test_scene_examples_use_paragraph_end_positions_and_boundary_metrics(tmp_path):
    source = tmp_path / "scene.jsonl"
    row = {"messages": [{"role": "system", "content": "JSON only"},
                        {"role": "user", "content": "[P1] first\n[P3] second\n[P8] third"},
                        {"role": "assistant", "content": '{"boundaries": [1, 8]}'}]}
    source.write_text(json.dumps(row), encoding="utf-8")
    examples = load_scene_examples(source, CharacterTokenizer(), 1000)
    example = examples[0]
    assert example["paragraph_ids"] == [1, 3, 8]
    assert example["boundary_labels"] == [1, 0, 1]
    assert example["paragraph_positions"] == [len(CharacterTokenizer().encode("JSON only\n\nUser: [P1] first")) - 1,
                                                len(CharacterTokenizer().encode("JSON only\n\nUser: [P1] first\n[P3] second")) - 1,
                                                len(CharacterTokenizer().encode("JSON only\n\nUser: [P1] first\n[P3] second\n[P8] third")) - 1]
    assert boundary_metrics(examples, [[1, 0, 1]])["f1"] == 1
    assert boundary_metrics(examples, [[0, 1, 0]])["f1"] == 0


def test_scene_head_gathers_each_paragraph_and_has_expected_shape():
    torch.manual_seed(5)
    head = SceneBoundaryHead(8, hidden=4)
    hidden = torch.randn(2, 6, 8)
    positions = torch.tensor([[0, 2, 5], [1, 3, 4]])
    logits = head(hidden, positions)
    assert logits.shape == (2, 3)
    with pytest.raises(ValueError):
        head(hidden, torch.tensor([0, 1]))


def test_local_residual_adapter_is_zero_initialized_and_disableable():
    adapter = LocalResidualAdapter(8, hidden=4)
    hidden = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    assert torch.equal(adapter(hidden), torch.zeros_like(hidden))
    assert torch.equal(adapter(hidden, {"disable_adapter": True}), torch.zeros_like(hidden))
    with torch.no_grad():
        adapter.net[-1].weight.fill_(0.1)
    assert torch.count_nonzero(adapter(hidden)) > 0
    changed = hidden.clone()
    changed[:, :3] = torch.randn_like(changed[:, :3])
    torch.testing.assert_close(adapter(changed)[:, 3:], adapter(hidden)[:, 3:])
    assert torch.count_nonzero(adapter(hidden, {"disable_adapter": True})) == 0


@pytest.mark.parametrize("kind", ["none", "local", "rosa", "semantic", "state", "state_semantic"])
def test_scene_evaluation_controls_and_probability_threshold(monkeypatch, kind):
    from scripts import train_scene_head as runner

    examples = [dict(paragraph_ids=[1, 2, 3], boundary_labels=[0, 1, 1])]
    logits = torch.tensor([[-0.2, 0.0, 0.2]])
    labels = torch.tensor([[0.0, 1.0, 1.0]])
    calls = []

    def fake_forward(model, head, example, device, disable_memory, disable_adapter,
                     disable_state, rosa_features):
        calls.append((disable_memory, disable_adapter, disable_state))
        return logits, labels

    monkeypatch.setattr(runner, "forward", fake_forward)
    adapter = SemanticMemoryAdapter(8, hops=2) if kind in ("semantic", "state_semantic") else None
    head = SceneBoundaryHead(8)
    report = runner.evaluate(nn.Identity(), adapter, head, examples, "cpu", 0.5, (0.6,), kind)
    expected_keys = {"none": {"head_only"},
                     "local": {"memory_disabled", "local_adapter"},
                     "rosa": {"memory_disabled", "rosa_adapter"},
                     "semantic": {"memory_disabled", "one_hop", "multi_hop"},
                     "state": {"state_disabled", "state_tuned"},
                     "state_semantic": {"memory_disabled", "one_hop", "multi_hop", "state_disabled"}}
    assert report.keys() == expected_keys[kind]
    assert all(metrics["f1"] == 1 for metrics in report.values())
    assert all(metrics["threshold_sweep"][0]["predicted"] == 0 for metrics in report.values())
    assert calls[0] == ({"none": (False, False, False), "state": (False, False, True)}.get(
        kind, (True, True, False)))
    if kind == "state_semantic":
        assert calls[-1] == (False, False, True)
    assert not head.training
    if adapter is not None:
        assert adapter.hops == 2


def test_scene_rosa_adapter_disables_even_a_learned_bias():
    from scripts.train_scene_head import SceneRosaInputAdapter

    embedding = nn.Embedding(16, 8).requires_grad_(False)
    adapter = SceneRosaInputAdapter(embedding, K=1, hidden=4)
    hidden = torch.randn(1, 3, 8)
    with torch.no_grad():
        adapter.proj.weight.fill_(0.1)
        adapter.ln.bias.fill_(1)
    assert torch.equal(adapter(hidden, {"disable_adapter": True}), torch.zeros_like(hidden))


def test_development_cutoff_selection_has_stable_ties():
    from scripts.evaluate_scene_controls import select_threshold

    metrics = dict(threshold_sweep=[dict(threshold=0.5, f1=0.3),
                                   dict(threshold=0.8, f1=0.4),
                                   dict(threshold=0.6, f1=0.4)])
    assert select_threshold(metrics)["threshold"] == 0.6
    with pytest.raises(ValueError, match="development report"):
        select_threshold({})


def test_initial_state_tuner_expands_shared_zero_state_and_trains():
    tuner = InitialStateTuner(layers=3, heads=2, head_size=4)
    states = tuner.initial_states(5)
    assert len(states) == 3
    assert states[0].shape == (5, 2, 4, 4)
    assert torch.equal(states[0], torch.zeros_like(states[0]))
    loss = sum(state.sum() for state in states)
    loss.backward()
    assert tuner.state.grad is not None
    assert tuner.state.grad.shape == (3, 2, 4, 4)
    torch.testing.assert_close(tuner.state.grad, torch.full_like(tuner.state, 5))
    with torch.no_grad():
        tuner.state[0].fill_(1)
    restored = InitialStateTuner(**tuner.config)
    restored.load_state_dict(tuner.state_dict())
    torch.testing.assert_close(restored.initial_states(2)[0], torch.ones(2, 2, 4, 4))
