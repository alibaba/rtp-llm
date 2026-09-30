"""SM120 deferred pair-state writes and reference-ordered pooling."""

import dataclasses
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.v41_compressor import (
    V41PairMetadata,
    V41PairState,
    prepare_ratio2,
)


def oracle(values, scores, norm, state, metadata):
    v, s, n = values.cpu(), scores.cpu(), norm.cpu()
    meta = dataclasses.replace(
        metadata,
        **{
            field.name: getattr(metadata, field.name).cpu()
            for field in dataclasses.fields(metadata)
        }
    )
    raw = V41PairState(state.raw.cpu().clone(), state.gen_num_per_cycle)
    result = torch.zeros_like(v, dtype=torch.bfloat16)
    for row in range(v.shape[0]):
        if not meta.valid_rows[row] or meta.positions[row] % 2 == 0:
            continue
        previous = int(meta.previous_rows[row])
        if previous >= 0:
            pv, ps = v[previous], s[previous]
        else:
            saved = raw.read(meta.previous_slots[row : row + 1], backend="eager")[0]
            pv, ps = saved[:512], saved[512:]
        weights = torch.stack((ps, s[row])).softmax(0)
        pooled = (torch.stack((pv, v[row])) * weights).sum(0).bfloat16().float()
        result[row] = (
            pooled * torch.rsqrt(pooled.square().mean() + 1e-20) * n
        ).bfloat16()
    slots = torch.where(meta.valid_rows, meta.write_slots, -1)
    raw.write(slots, torch.cat((v, s), -1), backend="eager")
    return result, raw.raw


class V41CompressorHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "V4.1 hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(419)

    def fixture(self):
        backing = torch.full((2, 4 * 1024 + 64), 173.0, device="cuda")
        state = V41PairState(backing[:, : 4 * 1024])
        state.write(
            torch.tensor([0], device="cuda"), torch.randn(1, 1024, device="cuda")
        )
        values, scores = torch.randn(4, 512, device="cuda"), torch.randn(
            4, 512, device="cuda"
        )
        values[-1] = scores[-1] = float("nan")
        norm = torch.randn(512, device="cuda").bfloat16()
        meta = V41PairMetadata(
            torch.tensor([0, 0, 0, -999], device="cuda"),
            torch.tensor([1, 2, 3, -999], device="cuda"),
            torch.tensor([True, True, True, False], device="cuda"),
            torch.tensor([-1, -1, 1, 99999], device="cuda"),
            torch.tensor([0, -1, -1, 99999], device="cuda"),
            torch.tensor([1, 2, 3, 99999], device="cuda"),
        )
        return values, scores, norm, state, meta, backing

    def test_pending_state_and_padding_preservation(self):
        values, scores, norm, state, meta, backing = self.fixture()
        expected, expected_state = oracle(values, scores, norm, state, meta)
        saved = backing.clone()
        frame = prepare_ratio2(values, scores, norm, state, meta)
        torch.testing.assert_close(backing, saved, atol=0, rtol=0)
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "preparation stream"):
                frame.commit()
        torch.testing.assert_close(frame.latent.cpu(), expected, atol=0.008, rtol=0.008)
        self.assertEqual(frame.complete.tolist(), [True, False, True, False])
        values.fill_(999)
        scores.fill_(999)
        meta.write_slots.fill_(-1)
        frame.commit()
        torch.testing.assert_close(state.raw.cpu(), expected_state, atol=0, rtol=0)
        torch.testing.assert_close(backing[:, -64:], saved[:, -64:], atol=0, rtol=0)

    def test_graph_reads_old_ring_before_wrap_and_updates_all_metadata(self):
        values, scores, norm, state, meta, backing = self.fixture()
        for _ in range(2):
            prepare_ratio2(values, scores, norm, state, meta).commit()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            frame = prepare_ratio2(values, scores, norm, state, meta)
            frame.commit()
        for live in (True, False, True):
            values.normal_()
            scores.normal_()
            norm.normal_()
            meta.valid_rows[:3] = live
            meta.positions[:3] = torch.tensor([3, 4, 5], device="cuda")
            meta.previous_slots[0] = 2
            meta.write_slots[:3] = torch.tensor([3, 0, 1], device="cuda")
            expected, expected_state = oracle(values, scores, norm, state, meta)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                frame.latent.cpu(), expected, atol=0.008, rtol=0.008
            )
            torch.testing.assert_close(state.raw.cpu(), expected_state, atol=0, rtol=0)
            self.assertTrue(bool((backing[:, -64:] == 173).all()))


if __name__ == "__main__":
    unittest.main()
