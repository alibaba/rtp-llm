"""Real FlashMLA byte-layout, independent-index and CUDA Graph probes."""

import json
import math
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from official_flashmla import SOURCE_HASHES, load_reference
from test_compact_reader import _fixture, _ints, _probe_attention_output_buffers

from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    CompactPages,
    GlobalBinding,
    SwaBinding,
)
from rtp_llm.models_py.modules.dsv41.flashmla import (
    FLASHMLA_REVISION,
    PlanarGlobalBinding,
    PlanarPages,
    PlanarSwaBinding,
    build_indices,
    copy_to_compact,
    flashmla_attention,
    is_supported,
    to_planar,
)
from rtp_llm.models_py.modules.dsv41.native_aot import native_identity


def independent_indices(requests, positions, floors, swa, global_kv, selected):
    requests, positions, floors = [
        tensor.cpu().tolist() for tensor in (requests, positions, floors)
    ]
    swa_ids = swa.page_ids.cpu().tolist()
    table = global_kv.page_table.cpu().tolist() if global_kv is not None else None
    selected = selected.cpu().tolist() if selected is not None else None
    width = max(64, (len(selected[0]) + 63) // 64 * 64) if selected else 64
    main = torch.full((len(requests), 1, 128), -1, dtype=torch.int32)
    extra = torch.full((len(requests), 1, width), -1, dtype=torch.int32)
    main_lengths, extra_lengths = [], []
    for row, (request, position, floor) in enumerate(zip(requests, positions, floors)):
        if position == -1:
            main_lengths.append(0)
            extra_lengths.append(0)
            continue
        tokens = list(range(max(0, floor, position - 127), position + 1))
        main[row, 0, : len(tokens)] = torch.tensor(
            [
                swa_ids[request] * swa.pages.entries_per_page
                + token % swa.pages.entries_per_page
                for token in tokens
            ]
        )
        values = []
        if global_kv is not None:
            for index in selected[row]:
                if 0 <= index < (position + 1) // global_kv.compress_ratio:
                    block, offset = divmod(index, global_kv.pages.entries_per_page)
                    values.append(
                        table[request][block] * global_kv.pages.entries_per_page
                        + offset
                    )
        extra[row, 0, : len(values)] = torch.tensor(values, dtype=torch.int32)
        main_lengths.append(len(tokens))
        extra_lengths.append(len(values))
    return main.cuda(), _ints(main_lengths), extra.cuda(), _ints(extra_lengths)


class FlashMLAGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not is_supported(
            torch.empty(0, device="cuda")
        ):
            raise RuntimeError("FlashMLA required probes need a real SM100/SM103 GPU")
        if os.environ.get("DSV41_FLASHMLA") != "1":
            raise RuntimeError("set DSV41_FLASHMLA=1 for this explicit candidate")
        cls.quant, cls.reference = load_reference(os.environ["DSV41_FLASHMLA_SOURCE"])
        torch.backends.cuda.matmul.allow_tf32 = False
        print(
            json.dumps(
                {
                    "flashmla_revision": FLASHMLA_REVISION,
                    "native_identity": native_identity("flash-mla"),
                    "official_sources": SOURCE_HASHES,
                }
            ),
            flush=True,
        )

    def test_output_aliases_rejected_and_disjoint_graph_buffers_reused(self):
        _probe_attention_output_buffers(self, flashmla_attention, planar=True)

    def test_byte_conversion_and_official_dequantization_are_exact(self):
        for region, entries in ((CacheRegion.SWA, 128), (CacheRegion.GLOBAL, 53)):
            with self.subTest(region=region, entries=entries):
                source, expected = _fixture(region, entries=entries)
                planar = to_planar(source)
                storage = torch.zeros(
                    (source.data.shape[0], source.data.stride(0)),
                    device=source.data.device,
                    dtype=torch.uint8,
                )
                output = CompactPages(
                    storage[:, : source.data.shape[1]], region, entries
                )
                copy_to_compact(planar, output)
                row_bytes = 528 if region == CacheRegion.SWA else 288
                torch.testing.assert_close(
                    output.data[:, : entries * row_bytes],
                    source.data[:, : entries * row_bytes],
                    rtol=0,
                    atol=0,
                )
                layout = (
                    self.quant.KVCacheLayout.V41_FP8Sparse
                    if region == CacheRegion.SWA
                    else self.quant.KVCacheLayout.V41_FP4
                )
                decoded = self.quant.dequantize_k_cache(
                    planar.kernel_view().view(torch.float8_e4m3fn), layout
                )
                torch.testing.assert_close(
                    decoded[1:, :, 0].float().cpu(), expected[1:], rtol=0, atol=0
                )
                with self.assertRaisesRegex(ValueError, "separate storage"):
                    to_planar(source, out=PlanarPages(source.data, region, entries))

    def bindings(self):
        swa_pages, _ = _fixture(CacheRegion.SWA, entries=256)
        global_pages, _ = _fixture(CacheRegion.GLOBAL, entries=53)
        swa = SwaBinding(
            swa_pages, _ints([3, 1, 2]), _ints([0, 0, 0]), _ints([256, 256, 256])
        )
        global_kv = GlobalBinding(
            global_pages, _ints([[1, 2, 3, 1], [3, 2, 1, 3], [2, 1, 3, 2]]), 2
        )
        return swa, global_kv

    def data(self, heads=64):
        requests = _ints([0, 0, 1, 1, 2, -1])
        positions = _ints([0, 3, 127, 128, 255, -1])
        floors = _ints([0, 0, 125, 126, 250, 0])
        selected = _ints([[0, 1, 63, 64, 127] + [-1] * 507] * 6)
        generator = torch.Generator(device="cuda").manual_seed(20260911)
        query = (
            torch.randn(6, heads, 512, generator=generator, device="cuda").bfloat16()
            * 0.125
        )
        sinks = torch.linspace(-2, 2, heads, device="cuda", dtype=torch.float32)
        return query, requests, positions, floors, selected, sinks

    def native_reference(
        self, query, requests, positions, floors, swa, global_kv, selected, sinks
    ):
        from flash_mla.flash_mla_interface import (
            FlashMLASchedMeta,
            flash_mla_with_kvcache,
        )

        main, main_lengths, extra, extra_lengths = independent_indices(
            requests, positions, floors, swa, global_kv, selected
        )
        output, _ = flash_mla_with_kvcache(
            query.unsqueeze(1),
            swa.pages.kernel_view(),
            None,
            None,
            512,
            FlashMLASchedMeta(),
            is_fp8_kvcache=True,
            indices=main,
            attn_sink=sinks,
            topk_length=main_lengths,
            extra_k_cache=(
                global_kv.pages.kernel_view() if global_kv is not None else None
            ),
            extra_indices_in_kvcache=extra if global_kv is not None else None,
            extra_topk_length=extra_lengths if global_kv is not None else None,
        )
        return output[:, 0]

    def test_causal_indices_replay_floor_and_independent_extra_page_sizes(self):
        compact_swa, compact_global = self.bindings()
        swa = PlanarSwaBinding.from_compact(compact_swa)
        global_kv = PlanarGlobalBinding.from_compact(compact_global)
        for heads in (64, 128):
            query, requests, positions, floors, selected, sinks = self.data(heads)
            actual = build_indices(
                requests,
                positions,
                floors,
                swa,
                global_kv=global_kv,
                global_indices=selected,
            )
            expected = independent_indices(
                requests, positions, floors, swa, global_kv, selected
            )
            for left, right in zip(
                (actual.main, actual.main_lengths, actual.extra, actual.extra_lengths),
                expected,
            ):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(actual.main.shape[-1] % 64, 0)
            self.assertEqual(actual.extra.shape[-1] % 64, 0)
            result = flashmla_attention(
                query,
                requests,
                positions,
                floors,
                swa,
                sinks,
                global_kv=global_kv,
                global_indices=selected,
            )
            result.check()
            native = self.native_reference(
                query, requests, positions, floors, swa, global_kv, selected, sinks
            )
            torch.testing.assert_close(result.output, native, rtol=0, atol=0)
            self.assertTrue(torch.isneginf(result.lse[-1]).all().item())

    def test_official_torch_oracle_with_one_visible_key_is_exact(self):
        compact_swa, _ = self.bindings()
        swa = PlanarSwaBinding.from_compact(compact_swa)
        query = torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
        requests, positions, floors = _ints([0, 1]), _ints([7, 29]), _ints([7, 29])
        sinks = torch.full((64,), -torch.inf, device="cuda")
        result = flashmla_attention(query, requests, positions, floors, swa, sinks)
        result.check()
        main, main_lengths, _, _ = independent_indices(
            requests, positions, floors, swa, None, None
        )
        blocked = self.quant.dequantize_k_cache(
            swa.pages.kernel_view().view(torch.float8_e4m3fn),
            self.quant.KVCacheLayout.V41_FP8Sparse,
        )
        parameters = SimpleNamespace(
            h_kv=1, decode=SimpleNamespace(b=2), s_q=1, d_qk=512, h_q=64, d_v=512
        )
        testcase = SimpleNamespace(
            kv_scope=SimpleNamespace(
                indices_in_kvcache=main, topk_length=main_lengths, blocked_k=blocked
            ),
            extra_kv_scope=None,
            q=query.unsqueeze(1),
            sm_scale=1.0 / math.sqrt(512),
            attn_sink=sinks,
        )
        # Upstream test runners set CUDA as the default device for the oracle.
        with torch.device(query.device):
            expected, expected_lse = type(self).reference(parameters, testcase)
        torch.testing.assert_close(result.output, expected[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(result.lse, expected_lse[:, :, 0], rtol=0, atol=0)

    def test_graph_refreshes_queries_lengths_positions_pages_and_cache_bytes(self):
        compact_swa, compact_global = self.bindings()
        swa = PlanarSwaBinding.from_compact(compact_swa)
        global_kv = PlanarGlobalBinding.from_compact(compact_global)
        query, requests, positions, floors, selected, sinks = self.data()

        def operation():
            to_planar(compact_swa.pages, out=swa.pages)
            to_planar(compact_global.pages, out=global_kv.pages)
            return flashmla_attention(
                query,
                requests,
                positions,
                floors,
                swa,
                sinks,
                global_kv=global_kv,
                global_indices=selected,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            operation()
            operation()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = operation()
        graph.replay()
        torch.cuda.synchronize()
        previous = captured.output.clone()
        output_pointer = captured.output.data_ptr()
        for step in range(3):
            query.neg_()
            positions[:2].copy_(_ints([step + 1, step + 5]))
            floors[:2].copy_(_ints([step, step + 4]))
            swa.page_ids.copy_(_ints([1 + step % 3, 3 - step % 3, 2]))
            global_kv.page_table[0].copy_(
                _ints([3, 1, 2, 1] if step % 2 else [2, 3, 1, 2])
            )
            compact_swa.pages.data[3].copy_(compact_swa.pages.data[1])
            selected[2, :3].copy_(_ints([0, 2, 62] if step % 2 else [1, 3, 63]))
            graph.replay()
            torch.cuda.synchronize()
            captured.check()
            self.assertEqual(captured.output.data_ptr(), output_pointer)
            self.assertFalse(torch.equal(captured.output, previous))
            native = self.native_reference(
                query, requests, positions, floors, swa, global_kv, selected, sinks
            )
            torch.testing.assert_close(captured.output, native, rtol=0, atol=0)
            previous.copy_(captured.output)
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            torch.save(
                {
                    "output": captured.output.cpu(),
                    "reference": native.cpu(),
                    "positions": positions.cpu(),
                    "page_ids": swa.page_ids.cpu(),
                },
                Path(directory) / "flashmla_graph.pt",
            )

    def test_invalid_visible_mapping_and_malformed_indices_report_errors(self):
        compact_swa, compact_global = self.bindings()
        swa = PlanarSwaBinding.from_compact(compact_swa)
        global_kv = PlanarGlobalBinding.from_compact(compact_global)
        query, requests, positions, floors, selected, sinks = self.data()
        global_kv.page_table[2, 1] = 0
        selected[2, :3].copy_(_ints([0, 0, -1]))
        result = flashmla_attention(
            query,
            requests,
            positions,
            floors,
            swa,
            sinks,
            global_kv=global_kv,
            global_indices=selected,
        )
        self.assertEqual(result.status[:, 0].cpu().tolist(), [0, 0, 2, 0, 1, 0])
        with self.assertRaisesRegex(RuntimeError, "compact reader rejected"):
            result.check()
        with self.assertRaisesRegex(TypeError, "explicit planar"):
            replace(swa, pages=compact_swa.pages).validate(query.device)


if __name__ == "__main__":
    unittest.main()
