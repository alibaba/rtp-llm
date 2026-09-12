"""Actual GPU tail ownership and temporary storage, without model execution."""

import dataclasses
import json
import os
import unittest
from pathlib import Path

import torch
from rtp_llm.models_py.modules.dsv41.attention import V41AttentionCache
from rtp_llm.models_py.modules.dsv41.cache_layout import SWA_WINDOW, CacheLayout
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.indexer import IndexSelection
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.models_py.modules.dsv41.prefill import V41L20Tail
from rtp_llm.models_py.modules.dsv41.transformer import V41L20Output


def _chunk(start, end):
    positions = torch.arange(start, end, dtype=torch.int32, device="cuda")
    history = positions[:, None] - torch.arange(3, 0, -1, device="cuda")
    rows = V41ModelRows(
        positions.clone(),
        (positions % 3 - 1).contiguous(),
        torch.ones(end - start, device="cuda", dtype=torch.bool),
        history.int().contiguous(),
        (history >= 0).contiguous(),
    )
    hc = (positions % 31).bfloat16()[:, None, None].expand(-1, 4, 5120).clone()
    pre_mix = (positions % 7).float()[:, None, None].expand(-1, 4, 1).clone()
    topk = positions[:, None] - torch.arange(512, device="cuda", dtype=torch.int32)
    candidates = positions[:, None] - torch.arange(
        2048, device="cuda", dtype=torch.int32
    )
    selected = IndexSelection(
        topk.contiguous(),
        candidates.contiguous(),
        torch.zeros(end - start, dtype=torch.int32, device="cuda"),
        20,
        20,
        0,
        0,
        0,
    )
    return V41L20Output(rows, hc, pre_mix), selected


def _tensors(l20, selection):
    values = {
        field.name: getattr(l20.rows, field.name)
        for field in dataclasses.fields(V41ModelRows)
    }
    values.update(
        hidden_states=l20.hidden_states,
        pre_mix=l20.pre_mix,
        topk=selection.topk,
        candidates=selection.candidate_blocks,
        status=selection.status,
    )
    return values


class L20TailStorageGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("L20 tail storage requires a real Blackwell GPU")
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        folder = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if folder:
            (Path(folder) / "l20_tail_storage.json").write_text(
                json.dumps(
                    {
                        "scope": "initialized GPU tail data and allocation only; no model, CP, PD or release capacity acceptance",
                        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                        "torch": str(torch.__version__),
                        "observations": cls.records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    @torch.inference_mode()
    def check_append(self, previous_rows, new_rows):
        layout = CacheLayout()
        identity = ReplayConfig(ReplayMode.BOUNDED).cache_identity(
            "2bc89ac599031fa673cab993f1df02fc4a98c673", layout
        )
        end = previous_rows + new_rows
        cache = V41AttentionCache.allocate_local(
            "tail-storage", identity, layout, end, device="cuda"
        )

        def prepare(start, stop, epoch):
            l20, selected = _chunk(start, stop)
            context = cache.begin_forward(epoch=epoch, start=start, end=stop)
            context.published_sources = {2, 8, 14, 20}
            context.completed_layers = set(range(21))
            context.publish_selection(selected)
            return l20, context

        prior = None
        if previous_rows:
            previous, context = prepare(0, previous_rows, 0)
            prior = V41L20Tail.append(None, previous, context)
            del previous, context
        current, context = prepare(previous_rows, end, 1)
        expected, selection = _chunk(max(0, end - SWA_WINDOW), end)
        truth = _tensors(expected, selection)
        warm = V41L20Tail.append(prior, current, context)
        torch.cuda.synchronize()
        del warm
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = V41L20Tail.append(prior, current, context)
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated() - baseline
        peak = torch.cuda.max_memory_allocated() - baseline
        self.records.append(
            {
                "test": self.id(),
                "previous_rows": previous_rows,
                "new_rows": new_rows,
                "retained_rows": len(actual.positions),
                "owned_bytes": actual.storage_bytes,
                "allocated_delta_bytes": live,
                "peak_delta_bytes": peak,
            }
        )
        self.assertGreater(live, 0)
        self.assertLessEqual(
            peak, 2 * live, "append temporary storage must stay bounded by its tail"
        )
        self.assertEqual(
            (actual.positions.start, actual.positions.end),
            (max(0, end - SWA_WINDOW), end),
        )
        actual_tensors = _tensors(actual.l20, actual.selection)
        for name, tensor in actual_tensors.items():
            torch.testing.assert_close(tensor, truth[name], rtol=0, atol=0)
            self.assertEqual(
                tensor.untyped_storage().nbytes(),
                tensor.numel() * tensor.element_size(),
            )
        # Both the encoder chunk and a previous tail may be reused immediately.
        inputs = list(_tensors(current, context.selection_for(20)).values())
        if prior is not None:
            inputs += list(_tensors(prior.l20, prior.selection).values())
        for tensor in inputs:
            tensor.zero_()
        for name, tensor in actual_tensors.items():
            torch.testing.assert_close(tensor, truth[name], rtol=0, atol=0)


def _case(previous_rows, new_rows):
    def run(self):
        self.check_append(previous_rows, new_rows)

    return run


for _previous, _new in [(0, 3), (3, 1)] + [
    (128, count) for count in (1, 3, 127, 128, 129, 12288)
]:
    setattr(
        L20TailStorageGpuTest,
        f"test_owned_bounded_append_{_previous}_{_new}",
        _case(_previous, _new),
    )


if __name__ == "__main__":
    unittest.main()
