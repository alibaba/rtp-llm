"""SM120 KV projection math, planar stores and dynamic graph publication."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.models_py.modules.dsv4.v41_compressor import V41PairMetadata, V41PairState
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear, quantize_group32
from rtp_llm.models_py.modules.dsv4.v41_projection import V41KVProjection


def fixture(ratio, hidden=5120):
    torch.manual_seed(421)
    device = torch.device("cuda")
    wkv = V41MXFP8Linear(
        (torch.randn(512, hidden, device=device) * 8).to(torch.float8_e4m3fn),
        torch.full((16, hidden // 32), 121, device=device, dtype=torch.uint8),
    )
    sources = (
        V41LayerSources(2, 2, 2, 2, None)
        if ratio == 2
        else V41LayerSources(20, 1, 20, 20, 20)
    )
    freq = precompute_freqs_cis(64, 1024, 65536, 160000, 16, 32, 1, device=device)
    bf = lambda *shape: torch.randn(shape, device=device).bfloat16()
    op = V41KVProjection(
        sources,
        wkv,
        bf(512),
        freq,
        compressor_weight=bf(512, hidden) / hidden**0.5,
        compressor_norm=bf(512),
        gate_weight=bf(512, hidden) / hidden**0.5 if ratio == 2 else None,
        index_weight=bf(128, 512) / 512**0.5,
        index_norm=bf(128),
    )
    state_backing = torch.full((2, 4096 + 64), 173.0, device=device)
    state = V41PairState(state_backing[:, :4096])
    meta = V41PairMetadata(
        torch.tensor([0, 0, 1, 1, -999], device=device),
        torch.tensor([254, 255, 256, 257, -999], device=device),
        torch.tensor([True, True, True, True, False], device=device),
        torch.tensor([-1, 0, -1, 2, 99999], device=device),
        torch.full((5,), -1, device=device, dtype=torch.int64),
        torch.tensor([2, 3, 4, 5, 99999], device=device),
    )
    x = bf(5, hidden)
    x[-1] = float("nan")
    return op, x, state, meta, state_backing


def caches(ratio):
    result = []
    for tag in ("swa_kv", "csa_kv" if ratio == 2 else "hca_kv", "indexer_kv"):
        width = dsv41_page_layouts(1)[tag].stride_bytes
        backing = torch.full((3, width + 64), 173, device="cuda", dtype=torch.uint8)
        result.append((V41PagedCache(backing[:, :width], tag), backing))
    return result


def prepare(op, x, state, meta):
    args = (
        {"pair_state": state, "pair_metadata": meta}
        if op.sources.compress_ratio == 2
        else {}
    )
    return op.prepare(x, meta.positions, meta.valid_rows, **args)


def norm(x, weight):
    y = x.float()
    return (
        y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-20) * weight.cpu()
    ).to(x.dtype)


def rotate(x, frequencies):
    tail = torch.view_as_complex(x[..., -64:].float().reshape(*x.shape[:-1], 32, 2))
    rotated = torch.view_as_real(tail * frequencies.cpu()).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-64], rotated), -1)


class V41ProjectionHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def assert_stores(self, frame, targets, slots, before):
        active = (frame.valid_rows, frame.complete, frame.complete)
        values = (frame.swa, frame.global_kv, frame.index_k)
        for (cache, backing), selected, live, rows, saved in zip(
            targets, slots, active, values, before
        ):
            reference = V41PagedCache(saved.cpu().clone(), cache.tag)
            reference.write(
                rows.cpu(), torch.where(live, selected, -1).cpu(), backend="eager"
            )
            torch.testing.assert_close(cache.raw.cpu(), reference.raw, atol=0, rtol=0)
            self.assertTrue(bool((backing[:, -64:] == 173).all()))

    def test_full_hidden_geometry_reference_projections_and_cache_bytes(self):
        for ratio in (1, 2):
            op, x, state, meta, _ = fixture(ratio)
            before_state = state.raw.clone()
            frame = prepare(op, x, state, meta)
            torch.testing.assert_close(state.raw, before_state, atol=0, rtol=0)
            cpu = x[:4].cpu()
            encoded, scales = quantize_group32(cpu, backend="eager")
            dequant = encoded.float() * torch.exp2(
                scales.float() - 127
            ).repeat_interleave(32, -1)
            projected = F.linear(
                dequant, op.wkv.weight.cpu().float() * 2**-6
            ).bfloat16()
            swa = rotate(
                norm(projected, op.kv_norm), op.frequencies[meta.positions[:4]]
            )
            torch.testing.assert_close(frame.swa[:4].cpu(), swa, atol=0.032, rtol=0.02)
            if ratio == 2:
                values = F.linear(cpu.float(), op.compressor_weight.cpu())
                scores = F.linear(cpu.float(), op.gate_weight.cpu())
                pooled = (
                    (values.reshape(2, 2, 512) * scores.reshape(2, 2, 512).softmax(1))
                    .sum(1)
                    .bfloat16()
                )
                latent = norm(pooled, op.compressor_norm)
                rows = torch.tensor([1, 3])
            else:
                latent = norm(
                    F.linear(cpu, op.compressor_weight.cpu()), op.compressor_norm
                )
                rows = torch.arange(4)
            freqs = op.frequencies[meta.positions[rows] + 1 - ratio]
            keys = rotate(
                norm(F.linear(latent, op.index_weight.cpu()), op.index_norm), freqs
            )
            global_kv = rotate(latent, freqs)
            torch.testing.assert_close(
                frame.global_kv[rows].cpu(), global_kv, atol=0.032, rtol=0.02
            )
            torch.testing.assert_close(
                frame.index_k[rows].cpu(), keys, atol=0.05, rtol=0.03
            )
            targets = caches(ratio)
            slots = [
                torch.tensor(v, device="cuda")
                for v in (
                    [0, 1, 2, 3, -999],
                    [10, 127, 11, 128, -999],
                    [10, 127, 11, 256, -999],
                )
            ]
            before = [cache.raw.clone() for cache, _ in targets]
            frame.publish(
                targets[0][0],
                slots[0],
                global_cache=targets[1][0],
                global_slots=slots[1],
                index_cache=targets[2][0],
                index_slots=slots[2],
            )
            self.assert_stores(frame, targets, slots, before)

    def test_graph_updates_positions_masks_slots_and_state(self):
        for ratio in (1, 2):
            op, x, state, meta, state_backing = fixture(ratio, hidden=128)
            targets = caches(ratio)
            slots = [torch.tensor([0, 1, 2, 3, -999], device="cuda") for _ in range(3)]

            def run():
                frame = prepare(op, x, state, meta)
                frame.publish(
                    targets[0][0],
                    slots[0],
                    global_cache=targets[1][0],
                    global_slots=slots[1],
                    index_cache=targets[2][0],
                    index_slots=slots[2],
                )
                return frame

            for _ in range(2):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for live in (True, False, True):
                x[:4].normal_()
                meta.valid_rows[:4] = live
                meta.positions[:4] += 2
                meta.write_slots[:4] = torch.tensor([0, 1, 6, 7], device="cuda")
                for selected in slots:
                    selected[:4] += 17
                before = [cache.raw.clone() for cache, _ in targets]
                expected_state = state.raw.clone()
                expected = prepare(op, x, state, meta)
                if expected.pair is not None:
                    state_ref = V41PairState(expected_state.cpu())
                    state_ref.write(
                        expected.pair.write_slots.cpu(),
                        expected.pair.pending_rows.cpu(),
                        backend="eager",
                    )
                    expected_state = state_ref.raw.cuda()
                graph.replay()
                torch.cuda.synchronize()
                for actual, reference in (
                    (captured.swa, expected.swa),
                    (captured.global_kv, expected.global_kv),
                    (captured.index_k, expected.index_k),
                ):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                torch.testing.assert_close(state.raw, expected_state, atol=0, rtol=0)
                self.assert_stores(captured, targets, slots, before)
                self.assertTrue(bool((state_backing[:, -64:] == 173).all()))

    def test_stream_and_tf32_contracts(self):
        op, x, state, meta, _ = fixture(2, hidden=128)
        frame = prepare(op, x, state, meta)
        targets = caches(2)
        slots = torch.arange(5, device="cuda")
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "preparation stream"):
                frame.publish(
                    targets[0][0],
                    slots,
                    global_cache=targets[1][0],
                    global_slots=slots,
                    index_cache=targets[2][0],
                    index_slots=slots,
                )
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with self.assertRaisesRegex(RuntimeError, "TF32 disabled"):
                prepare(op, x, state, meta)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old


if __name__ == "__main__":
    unittest.main()
