import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rosa.rwkv7 import RWKV7, RWKV7_OP, load_kernel
from rosa.tasks import InitialStateTuner


def reference_recurrence(query, decay, key, value, erase, write, initial):
    batch, length, width = query.shape
    heads, head_size = initial.shape[-3:-1]
    query, decay, key, value, erase, write = [
        tensor.float().reshape(batch, length, heads, head_size)
        for tensor in (query, decay, key, value, erase, write)]
    state = initial
    outputs = []
    for position in range(length):
        projection = (state * erase[:, position, :, None, :]).sum(-1)
        state = (state * torch.exp(-torch.exp(decay[:, position, :, None, :]))
                 + projection.unsqueeze(-1) * write[:, position, :, None, :]
                 + value[:, position, :, :, None] * key[:, position, :, None, :])
        outputs.append((state * query[:, position, :, None, :]).sum(-1))
    return torch.stack(outputs, 1).reshape(batch, length, width).to(torch.bfloat16)


def make_inputs(length, batch=2, heads=2):
    torch.manual_seed(19)
    shape = (batch, length, heads * 64)
    query, key, value = [torch.randn(shape, device="cuda") * 0.2 for _ in range(3)]
    decay = -2.0 + torch.randn(shape, device="cuda") * 0.2
    normalized = F.normalize(key.reshape(batch, length, heads, 64), dim=-1).reshape(shape)
    erase = -normalized
    write = normalized * torch.sigmoid(torch.randn(shape, device="cuda"))
    return [tensor.to(torch.bfloat16).requires_grad_()
            for tensor in (query, decay, key, value, erase, write)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the RWKV CUDA kernel")
@pytest.mark.parametrize("length", [16, 48, 256])
def test_state_kernel_matches_reference_outputs_and_all_gradients(length):
    load_kernel()
    inputs = make_inputs(length)
    initial = (torch.randn(2, 2, 64, 64, device="cuda") * 0.1).requires_grad_()
    initial_before = initial.detach().clone()
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    reference_initial = initial.detach().clone().requires_grad_()
    actual = RWKV7_OP(*inputs, initial)
    expected = reference_recurrence(*reference_inputs, reference_initial)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)
    upstream = torch.randn_like(actual) / 10
    actual.backward(upstream)
    expected.backward(upstream)
    for tensor, reference in zip(inputs + [initial], reference_inputs + [reference_initial]):
        assert torch.isfinite(tensor.grad).all()
        torch.testing.assert_close(tensor.grad, reference.grad, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(initial, initial_before, atol=0, rtol=0)
    assert initial.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the RWKV CUDA kernel")
def test_zero_state_matches_legacy_and_state_only_gradient_is_shared_across_batch():
    load_kernel()
    inputs = [tensor.detach() for tensor in make_inputs(32)]
    tuner = InitialStateTuner(1, 2).cuda()
    actual = RWKV7_OP(*inputs, tuner.initial_states(2)[0])
    legacy = RWKV7_OP(*inputs)
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
    actual.float().square().sum().backward()
    full_gradient = tuner.state.grad.clone()
    tuner.zero_grad()
    for sample in range(2):
        single = RWKV7_OP(*[tensor[sample:sample + 1] for tensor in inputs],
                         tuner.initial_states(1)[0])
        single.float().square().sum().backward()
    torch.testing.assert_close(tuner.state.grad, full_gradient, rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(full_gradient) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the RWKV CUDA kernel")
@pytest.mark.parametrize("checkpointing", [False, True])
def test_model_state_reset_padding_freeze_and_reload(checkpointing):
    load_kernel()
    torch.manual_seed(13)
    model = RWKV7(2, 64, vocab_size=128)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    model = model.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False)
    model.grad_ckpt = checkpointing
    tokens = torch.randint(0, 128, (2, 19), device="cuda")
    baseline = model(tokens)
    tuner = InitialStateTuner(2, 1).cuda()
    model.add_state_adapter(tuner)
    torch.testing.assert_close(model(tokens), baseline, rtol=0, atol=0)
    with torch.no_grad():
        tuner.state.normal_(std=0.1)
    before = tuner.state.detach().clone()
    active = model(tokens)
    assert not torch.equal(active, baseline)
    model(tokens.flip(1))
    torch.testing.assert_close(model(tokens), active, rtol=0, atol=0)
    torch.testing.assert_close(model(tokens, aux={"disable_state": True}), baseline, rtol=0, atol=0)
    active.float().square().mean().backward()
    assert torch.isfinite(tuner.state.grad).all() and tuner.state.grad.abs().sum() > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith("state_adapter."))
    torch.testing.assert_close(tuner.state, before, rtol=0, atol=0)
    restored = InitialStateTuner(**tuner.config).cuda()
    restored.load_state_dict(tuner.state_dict())
    model.add_state_adapter(restored)
    torch.testing.assert_close(model(tokens), active, rtol=0, atol=0)
