"""RTP paged KDA replay correctness against the Triton implementation."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from rtp_llm.models_py.triton_kernels import linear_replay
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import FusedRMSNormGated
from rtp_llm.models_py.triton_kernels.test.linear_replay_test import (
    capture_replay_graph,
    make_replay_cache,
)

H, D, T = 48, 128, 4
HD, C = H * D, 3 * H * D
PITCH = (4 * HD + D + H + 15) // 16 * 16
EPS, LOWER = 1e-6, -5.0
DEVICE = torch.device("cuda:0")
BF16 = torch.bfloat16


def packed_projection(batch, generator):
    packed = torch.randn((batch * T, PITCH), device=DEVICE, dtype=BF16, generator=generator)
    fa = packed[:, 4 * HD : 4 * HD + D]
    fa.copy_((fa * 2).round().clamp(-3, 3))
    return packed


def inputs(batch, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def sample(*shape, dtype=BF16):
        return torch.randn(*shape, device=DEVICE, generator=gen, dtype=dtype)

    packed = packed_projection(batch, gen)
    return SimpleNamespace(
        packed=packed, qkv=packed[:, :C], gate=packed[:, C:4 * HD], fa=packed[:, 4 * HD:4 * HD + D],
        beta=packed[:, 4 * HD + D:4 * HD + D + H],
        weight=sample(C, 4) * 0.2,
        fb=(sample(HD, D) * 3).round().clamp(-8, 8) / 64,
        alog=sample(H, dtype=torch.float32) * 0.1,
        bias=sample(HD, dtype=torch.float32) * 0.1,
        norm=sample(D).sigmoid() + 0.5,
        state=sample(batch + 1, H, D, D, dtype=torch.float32) * 0.02,
        conv=sample(batch + 1, 3, C),
    )


def replay_buffers(data, slots):
    shape = (slots, T, H, D)
    cache = make_replay_cache(0, shape, shape, shape, C, device=DEVICE)
    gate_bytes = cache.g.nbytes
    storage = torch.zeros(gate_bytes + cache.conv_inputs.nbytes, dtype=torch.uint8, device=DEVICE)
    cache.g = storage[:gate_bytes].view(torch.float32).view(shape)
    cache.conv_inputs = storage[gate_bytes:].view(BF16).view(slots, T, C)
    return SimpleNamespace(state=data.state.clone(), conv=data.conv.clone(), cache=cache)


def assert_close(actual, ref, error, rtol, atol):
    diff = (actual.float() - ref.float()).abs()
    error.append(float(diff.max().item()))
    torch.testing.assert_close(actual, ref, rtol=rtol, atol=atol)


def verify_rtp_pages(batch, rounds, capture):
    from rtp_llm.models_py.triton_kernels.k3_native_verify import K3NativeVerify

    data = inputs(batch, 8801 + batch)
    # Real K3.1 keeps FP32 convolution weights with BF16 activation history.
    data.weight = data.weight.float()
    pages = torch.arange(1, batch + 1, device=DEVICE, dtype=torch.int32)
    slots = batch + 3
    records = pages.flip((0,)) - 1
    records[1] = slots - 1
    context = SimpleNamespace(
        slot_ids=records.clone(),
        slot_generations=torch.ones_like(records, dtype=torch.int64),
        prev_accept_lengths=torch.zeros_like(records),
        history_valid_lengths=torch.zeros_like(records),
        history_epochs=torch.ones_like(records, dtype=torch.int64),
        verify_epochs=torch.ones_like(records, dtype=torch.int64),
        init_kinds=torch.ones_like(records),
        state_read_block_ids=pages[None].clone(),
        active_block_ids=pages[None].clone(),
    )
    context.slot_ids[0] = -1
    context.init_kinds[-1] = 2
    context.state_read_block_ids[0, -1] = -1

    # Reserve disjoint cache pages to force the destination to move, as in K3.
    data.state = torch.cat((data.state, data.state[1:].clone()))
    data.conv = torch.cat((data.conv, data.conv[1:].clone()))
    reference = replay_buffers(data, slots)
    other = replay_buffers(data, slots)
    triton_replay = linear_replay.LinearReplay(
        data.weight, data.alog, data.bias, reference.state, reference.conv,
        reference.cache, group_id=0, vector_gate=True, state_v_first=False,
        lower_bound=LOWER,
    )

    decoder = SimpleNamespace(
        conv_weights=data.weight, alog=data.alog, dt_bias=data.bias,
        gate_lower_bound=LOWER, _get_conv_states=lambda _: other.conv,
        _get_ssm_states=lambda _: other.state,
    )
    norm = FusedRMSNormGated(data.norm, EPS)
    native = K3NativeVerify(decoder, data.fb, norm)
    kv_cache = SimpleNamespace(
        kv_cache_base=other.state, linear_replay=other.cache, group_id=0,
    )
    assert native.bind(kv_cache)
    assert native.conv_window.data_ptr() == other.cache.g.data_ptr(), (
        "Native convolution window must reuse the legacy G and conv storage"
    )
    assert native.conv_window.shape == (slots, C, T + 2)
    args = SimpleNamespace(linear_replay=context)
    # The logical projection can have a non-aligned row before packing.
    width = 4 * HD + D + H
    projected = data.packed[:, :width].contiguous()

    def run():
        out = native(projected, kv_cache, args)
        linear_replay.finalize_linear_replay(other.cache, context, T)
        return out

    if capture:
        graph, captured, _ = capture_replay_graph(run)
        other.state.copy_(data.state)
        other.conv.copy_(data.conv)
        for name in ("slot_generations", "log_epochs", "valid_counts", "error_flags"):
            getattr(other.cache, name).zero_()

    output_error, state_error = [], []
    for step in range(rounds):
        generator = torch.Generator(device=DEVICE).manual_seed(8801 + batch + step)
        packed = packed_projection(batch, generator)
        data.packed.copy_(packed)
        projected.copy_(packed[:, :width])
        if step:
            context.prev_accept_lengths.copy_((pages + step - 1) % T + 1)
            context.init_kinds.zero_()
            context.history_valid_lengths.fill_(T)
            context.history_epochs.fill_(step)
            context.verify_epochs.fill_(step + 1)
            context.state_read_block_ids.copy_(context.active_block_ids)
            context.active_block_ids.copy_((pages + (step % 2) * batch)[None])
        value = triton_replay(data.qkv, F.linear(data.fa, data.fb), data.beta, context)
        expected = norm(value, data.gate.view(batch * T, H, D)).view(batch * T, HD)
        linear_replay.finalize_linear_replay(reference.cache, context, T)
        if capture:
            graph.replay()
            actual = captured
        else:
            actual = run()
        assert_close(actual, expected, output_error, 0.02, 0.005)
        live_pages = context.active_block_ids[0, 1:].long()
        assert_close(other.state[live_pages], reference.state[live_pages], state_error, 2e-4, 2e-5)
        torch.testing.assert_close(other.conv[live_pages], reference.conv[live_pages], rtol=0, atol=0)
        ids = records[1:].long()
        cached_rows = native.conv_window[ids, :, 2:].transpose(-2, -1)
        torch.testing.assert_close(cached_rows, reference.cache.conv_inputs[ids], rtol=0, atol=0)
        torch.testing.assert_close(other.cache.valid_counts, reference.cache.valid_counts)
        assert (other.cache.error_flags == 0).all().item()
    return dict(batch=batch, rounds=rounds, graph=capture,
                output_max_abs=max(output_error), state_max_abs=max(state_error),
                exact_conv_history=True)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--batch", type=int, nargs="+", default=(8, 16, 32, 64))
    cli.add_argument("--rounds", type=int, default=8)
    cli.add_argument("--graph", action="store_true")
    args = cli.parse_args()
    results = [verify_rtp_pages(b, args.rounds, args.graph) for b in args.batch]
    print("KDA_NATIVE_PARITY", json.dumps(results), flush=True)
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if directory:
        Path(directory, "native_kda_parity.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
