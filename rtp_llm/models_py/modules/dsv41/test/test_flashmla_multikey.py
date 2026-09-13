"""Independent exact multi-key output probes, not general error qualification."""

import json
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import test_flashmla as flash_fixture
import torch
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    CompactPages,
    GlobalBinding,
    SwaBinding,
)
from rtp_llm.models_py.modules.dsv41.flashmla import (
    PlanarGlobalBinding,
    PlanarSwaBinding,
    flashmla_compact_attention,
)
from test_compact_reader import _ints


def constant_pages(region, entries):
    swa = region == CacheRegion.SWA
    row_bytes, payload, group = (528, 512, 32) if swa else (288, 256, 16)
    stride = (entries * row_bytes + 511) // 512 * 512
    storage = torch.full((4, stride), 255, dtype=torch.uint8, device="cuda")
    rows = storage[:, : entries * row_bytes].view(4, entries, row_bytes)
    # E4M3 1.0 and two E2M1 2.0 values; each group has a dyadic scale.
    rows[..., :payload] = 0x38 if swa else 0x44
    powers = (torch.arange(512 // group, device="cuda") * group // 32) % 4 - 1
    scales = (
        (powers + 127).to(torch.uint8)
        if swa
        else (2.0**powers).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    rows[..., payload:] = scales
    storage[0].fill_(255)
    return CompactPages(storage[:, : entries * row_bytes], region, entries)


class FlashMLAMultikeyOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flash_fixture.FlashMLAGpuTest.setUpClass.__func__(cls)

    def reference_output(
        self, q, requests, positions, floors, swa, global_kv, selected, sinks
    ):
        planar_swa = PlanarSwaBinding.from_compact(swa)
        planar_global = PlanarGlobalBinding.from_compact(global_kv)
        main, main_lengths, extra, extra_lengths = flash_fixture.independent_indices(
            requests, positions, floors, planar_swa, planar_global, selected
        )

        def scope(pages, layout, indices, lengths):
            return SimpleNamespace(
                indices_in_kvcache=indices,
                topk_length=lengths,
                blocked_k=self.quant.dequantize_k_cache(
                    pages.kernel_view().view(torch.float8_e4m3fn), layout
                ),
            )

        case = SimpleNamespace(
            kv_scope=scope(
                planar_swa.pages,
                self.quant.KVCacheLayout.V41_FP8Sparse,
                main,
                main_lengths,
            ),
            extra_kv_scope=scope(
                planar_global.pages,
                self.quant.KVCacheLayout.V41_FP4,
                extra,
                extra_lengths,
            ),
            q=q.unsqueeze(1),
            sm_scale=1.0 / math.sqrt(512),
            attn_sink=sinks,
        )
        params = SimpleNamespace(
            h_kv=1,
            decode=SimpleNamespace(b=4),
            s_q=1,
            d_qk=512,
            h_q=q.shape[1],
            d_v=512,
        )
        with torch.device(q.device):
            expected, lse_without_sink = type(self).reference(params, case)
        return expected[:, 0], torch.logaddexp(lse_without_sink[:, :, 0], sinks)

    def run_case(self, ratio):
        counts = [2, 4, 64, 128]
        swa = SwaBinding(
            constant_pages(CacheRegion.SWA, 136),
            _ints([1, 2, 3, 1]),
            _ints([120] * 4),
            _ints([256] * 4),
        )
        global_kv = GlobalBinding(
            constant_pages(CacheRegion.GLOBAL, 53), _ints([[1, 2, 3]] * 4), ratio
        )
        query = torch.zeros((4, 64, 512), dtype=torch.bfloat16, device="cuda")
        requests, positions = _ints([0, 1, 2, 3]), _ints([255] * 4)
        floors = _ints([256 - n for n in counts])
        selected = _ints([list(range(n - 1)) + [-1] * (513 - n) for n in counts])
        sinks = torch.zeros(64, device="cuda", dtype=torch.float32)

        def operation():
            return flashmla_compact_attention(
                query,
                requests,
                positions,
                floors,
                swa,
                sinks,
                global_kv=global_kv,
                global_indices=selected,
            )

        result = operation()
        result.check()
        expected, expected_lse = self.reference_output(
            query, requests, positions, floors, swa, global_kv, selected, sinks
        )
        scales = 2.0 ** ((torch.arange(512, device="cuda") // 32) % 4 - 1)
        analytic = (
            torch.stack([(1.5 - 1 / n) * scales for n in counts])[:, None, :]
            .expand_as(query)
            .bfloat16()
        )
        torch.testing.assert_close(expected, analytic, rtol=0, atol=0)
        torch.testing.assert_close(result.output, analytic, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result.lse).all().item())
        max_lse_error = (result.lse - expected_lse).abs().max().item()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            operation()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = operation()
        torch.cuda.synchronize()
        pointer = captured.output.data_ptr()
        records = []
        for step in range(3):
            active_counts = list(reversed(counts)) if step % 2 == 0 else counts
            positions.fill_(254 if step % 2 else 255)
            floors.copy_(positions + 1 - _ints(active_counts))
            selected.copy_(
                _ints([list(range(n - 1)) + [-1] * (513 - n) for n in active_counts])
            )
            swa.page_ids.copy_(_ints([3, 1, 2, 3] if step % 2 else [2, 3, 1, 2]))
            global_kv.page_table.copy_(
                _ints([[3, 1, 2] if step % 2 else [2, 3, 1]] * 4)
            )
            graph.replay()
            torch.cuda.synchronize()
            captured.check()
            self.assertEqual(captured.output.data_ptr(), pointer)
            expected, expected_lse = self.reference_output(
                query, requests, positions, floors, swa, global_kv, selected, sinks
            )
            torch.testing.assert_close(captured.output, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(captured.lse).all().item())
            records.append(
                {
                    "counts": active_counts,
                    "output_mismatches": 0,
                    "lse_max_abs_error": (captured.lse - expected_lse)
                    .abs()
                    .max()
                    .item(),
                }
            )
        folder = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        torch.save(
            {
                "output": captured.output.cpu(),
                "reference": expected.cpu(),
                "lse": captured.lse.cpu(),
                "reference_lse": expected_lse.cpu(),
            },
            folder / f"ratio{ratio}.pt",
        )
        (folder / f"ratio{ratio}.json").write_text(
            json.dumps(
                {
                    "ratio": ratio,
                    "swa_counts": counts,
                    "global_counts": [n - 1 for n in counts],
                    "native_identity": flash_fixture.native_identity("flash-mla"),
                    "official_sources": flash_fixture.SOURCE_HASHES,
                    "output_rtol": 0,
                    "output_atol": 0,
                    "initial_lse_max_abs_error": max_lse_error,
                    "graph_replays": records,
                    "scope": "Exact uniform-score multi-key output component; LSE error recorded without numerical qualification.",
                    "general_numerical_qualification": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_ratio_one_multikey_exact_oracle_and_graph(self):
        self.run_case(1)

    def test_ratio_two_multikey_exact_oracle_and_graph(self):
        self.run_case(2)


if __name__ == "__main__":
    unittest.main()
