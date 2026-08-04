import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _load_nan_diag():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "_nan_diag_triton.py"))
    spec = importlib.util.spec_from_file_location("_dsv4_nan_diag", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nan_diag = _load_nan_diag()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NanDiagTritonTest(unittest.TestCase):
    @staticmethod
    def _packed_slot_offsets(cache: torch.Tensor, slot: int) -> tuple[int, int, int]:
        block_size = int(cache.shape[1])
        block, pos = divmod(slot, block_size)
        block_base = block * int(cache.stride(0))
        token_data = block_base + pos * 576
        scale_data = block_base + block_size * 576 + pos * 8
        return token_data, token_data + 448, scale_data

    @staticmethod
    def _indexer_slot_offsets(cache: torch.Tensor, slot: int) -> tuple[int, int]:
        block_size = int(cache.shape[1])
        block, pos = divmod(slot, block_size)
        block_base = block * int(cache.stride(0))
        return block_base + pos * 128, block_base + block_size * 128 + pos * 4

    def test_detector_is_read_only(self) -> None:
        x = torch.randn((2, 513), dtype=torch.float32, device="cuda")
        x[0, 7] = float("nan")
        x[0, 300] = float("inf")
        before = x.clone()

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.report_nonfinite(
                x,
                source_id=nan_diag.SOURCE_MOE_INPUT,
                layer_id=17,
            )
            torch.cuda.synchronize()

        torch.testing.assert_close(x, before, equal_nan=True)

    def test_injector_skips_batch_zero_and_injects_mapped_batch(self) -> None:
        x = torch.zeros((1, 8), dtype=torch.bfloat16, device="cuda")
        batch_id = torch.zeros((1,), dtype=torch.int64, device="cuda")

        with patch.object(nan_diag, "ENABLED", True), patch.object(
            nan_diag, "TEST_INJECT", (2, 0, 3)
        ):
            nan_diag.set_batch_context(batch_id)
            nan_diag.maybe_inject_test_nan(x, layer_id=2)
            torch.cuda.synchronize()
            self.assertEqual(float(x[0, 3].item()), 0.0)

            batch_id.fill_(123456)
            nan_diag.maybe_inject_test_nan(x, layer_id=2)
            torch.cuda.synchronize()
            self.assertTrue(torch.isnan(x[0, 3]).item())

    def test_detector_runs_on_every_cuda_graph_replay(self) -> None:
        probe = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
        x = torch.zeros((2, 513), dtype=torch.bfloat16, device="cuda")
        batch_id = torch.tensor([23001], dtype=torch.int64, device="cuda")

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            # JIT must happen outside capture. The live graph shape/strides
            # intentionally differ to verify they do not specialize the kernel.
            nan_diag.report_nonfinite(
                probe,
                source_id=nan_diag.SOURCE_ROUTER_SCORES,
                layer_id=23,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                nan_diag.set_batch_context(batch_id)
                nan_diag.report_nonfinite(
                    x,
                    source_id=nan_diag.SOURCE_ROUTER_SCORES,
                    layer_id=23,
                )

            x[1, 300] = float("nan")
            state_index = nan_diag._report_state_index(
                nan_diag.SOURCE_ROUTER_SCORES, 23
            )
            report_count = nan_diag._REPORT_COUNT_BY_DEVICE[str(x.device)]
            before_count = int(report_count[state_index].item())
            for current_batch in (23002, 23003):
                batch_id.fill_(current_batch)
                graph.replay()
                torch.cuda.synchronize()
            self.assertEqual(
                int(report_count[state_index].item()),
                before_count + 2,
            )
            event_counters, event_records = nan_diag._ensure_event_state(x.device)
            self.assertEqual(int(event_counters[0].item()), 1)
            self.assertEqual(event_records[0, 0].item(), 23003)
            self.assertEqual(
                event_records[0, 1].item(),
                nan_diag.SOURCE_ROUTER_SCORES,
            )
            self.assertEqual(event_records[0, 2].item(), 23)
            torch.testing.assert_close(
                x[1, 300],
                torch.tensor(float("nan"), dtype=x.dtype, device=x.device),
                equal_nan=True,
            )

    def test_rate_limits_a_nan_storm_per_batch_source_and_layer(self) -> None:
        x = torch.full((8, 1024), float("nan"), dtype=torch.float32, device="cuda")
        batch_id = torch.tensor([991001], dtype=torch.int64, device="cuda")
        source_id = 9  # Test-only source slot.
        layer_id = 997

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            state_index = nan_diag._report_state_index(source_id, layer_id)
            _, report_count = nan_diag._ensure_report_state(x.device)
            before_count = int(report_count[state_index].item())

            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 1)

            # Rechecking the same bad tensor in the same model batch is quiet.
            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 1)

            # A new model batch must produce a new event.
            batch_id.fill_(991002)
            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 2)

    def test_activation_event_maps_exact_request_query_and_subrow(self) -> None:
        batch_size, q_len, heads, cols = 3, 2, 4, 11
        x = torch.zeros(
            (batch_size * q_len * heads, cols),
            dtype=torch.float32,
            device="cuda",
        )
        bad_row = 2 * q_len * heads + 1 * heads + 3
        x[bad_row, 5] = float("nan")
        batch_id = torch.tensor([991501], dtype=torch.int64, device="cuda")

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            nan_diag.report_nonfinite(
                x,
                source_id=nan_diag.SOURCE_ATTENTION_OUTPUT,
                layer_id=17,
                batch_size=batch_size,
                q_len=q_len,
            )
            torch.cuda.synchronize()

            counters, records = nan_diag._ensure_event_state(x.device)
            self.assertEqual(int(counters[0].item()), 1)
            record = records[0].cpu().tolist()
            self.assertEqual(record[8], 2)  # request_index
            self.assertEqual(record[9], 1)  # query_index
            self.assertEqual(record[10], bad_row)
            self.assertEqual(record[11], 5)
            self.assertEqual(record[12], 3)  # attention head / subrow
            self.assertEqual(record[15], q_len)

    def test_unmapped_batch_zero_is_recorded_reliably(self) -> None:
        x = torch.full((4, 1024), float("nan"), dtype=torch.float32, device="cuda")
        batch_id = torch.zeros((1,), dtype=torch.int64, device="cuda")
        source_id = 9  # Test-only source slot.
        layer_id = 996

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            state_index = nan_diag._report_state_index(source_id, layer_id)
            _, report_count = nan_diag._ensure_report_state(x.device)
            before_count = int(report_count[state_index].item())

            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 1)
            event_counters, event_records = nan_diag._ensure_event_state(x.device)
            self.assertEqual(int(event_counters[0].item()), 1)
            record = event_records[0].cpu().tolist()
            self.assertEqual(record[:3], [0, source_id, layer_id])
            self.assertGreater(record[4], 0)

            # The first traceable batch must still report after batch-zero work.
            batch_id.fill_(996001)
            nan_diag.report_nonfinite(
                x,
                source_id=source_id,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 2)

    def test_attention_lse_ignores_negative_inf_but_reports_nan(self) -> None:
        lse = torch.zeros((2, 3, 7), dtype=torch.float32, device="cuda")
        lse[0, 0, 0] = -float("inf")
        batch_id = torch.tensor([992001], dtype=torch.int64, device="cuda")

        with patch.object(nan_diag, "ENABLED", True):
            nan_diag.set_batch_context(batch_id)
            state_index = nan_diag._report_state_index(
                nan_diag.SOURCE_CP_ATTENTION_LSE, 12
            )
            _, report_count = nan_diag._ensure_report_state(lse.device)
            before_count = int(report_count[state_index].item())

            nan_diag.report_nonfinite(
                lse,
                source_id=nan_diag.SOURCE_CP_ATTENTION_LSE,
                layer_id=12,
                include_neg_inf=False,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count)

            lse[1, 2, 6] = float("nan")
            nan_diag.report_nonfinite(
                lse,
                source_id=nan_diag.SOURCE_CP_ATTENTION_LSE,
                layer_id=12,
                include_neg_inf=False,
            )
            torch.cuda.synchronize()
            self.assertEqual(int(report_count[state_index].item()), before_count + 1)

    def test_reliable_event_buffer_reports_overflow_instead_of_silent_loss(
        self,
    ) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        device_key = str(device)
        old_counters = nan_diag._EVENT_COUNTERS_BY_DEVICE.get(device_key)
        old_records = nan_diag._EVENT_RECORDS_BY_DEVICE.get(device_key)
        counters = torch.zeros((3,), dtype=torch.int64, device=device)
        records = torch.empty(
            (1, nan_diag._EVENT_FIELDS), dtype=torch.int64, device=device
        )
        nan_diag._EVENT_COUNTERS_BY_DEVICE[device_key] = counters
        nan_diag._EVENT_RECORDS_BY_DEVICE[device_key] = records
        try:
            batch_id = torch.tensor([992501], dtype=torch.int64, device=device)
            x = torch.full((1, 8), float("nan"), dtype=torch.float32, device=device)
            with patch.object(nan_diag, "ENABLED", True):
                nan_diag.set_batch_context(batch_id)
                for layer_id in (998, 999):
                    nan_diag.report_nonfinite(
                        x,
                        source_id=nan_diag.SOURCE_FINAL_HIDDEN,
                        layer_id=layer_id,
                    )
                torch.cuda.synchronize()
            self.assertEqual(counters[:2].cpu().tolist(), [2, 1])
            self.assertEqual(records[0, :3].cpu().tolist(), [992501, 5, 998])

            outputs = SimpleNamespace(hidden_states=x)
            with patch.object(nan_diag, "ENABLED", True):
                self.assertIs(nan_diag.attach_event_buffers(outputs), outputs)
            self.assertIs(outputs.nan_diag_event_counters, counters)
            self.assertIs(outputs.nan_diag_events, records)
        finally:
            if old_counters is None:
                nan_diag._EVENT_COUNTERS_BY_DEVICE.pop(device_key, None)
            else:
                nan_diag._EVENT_COUNTERS_BY_DEVICE[device_key] = old_counters
            if old_records is None:
                nan_diag._EVENT_RECORDS_BY_DEVICE.pop(device_key, None)
            else:
                nan_diag._EVENT_RECORDS_BY_DEVICE[device_key] = old_records

    def test_packed_fp8_kv_cache_reports_all_nan_encodings(self) -> None:
        cache = torch.zeros((2, 4, 584), dtype=torch.uint8, device="cuda")
        indices = torch.tensor([[[0, 5, 7, -1]]], dtype=torch.int32, device="cuda")
        topk_length = torch.tensor([2], dtype=torch.int32, device="cuda")
        batch_id = torch.tensor([993001], dtype=torch.int64, device="cuda")
        source_id = nan_diag.SOURCE_SWA_KV_CACHE_READ
        layer_id = 44
        token_data, rope_data, scale_data = self._packed_slot_offsets(cache, 5)

        with patch.object(nan_diag, "ENABLED", True), patch.object(
            nan_diag, "TEST_KV_CORRUPT", None
        ):
            nan_diag.set_batch_context(batch_id)
            state_index = nan_diag._report_state_index(source_id, layer_id)
            _, report_count = nan_diag._ensure_report_state(cache.device)
            before_count = int(report_count[state_index].item())
            raw = cache.view(-1)

            corruptions = (
                ((token_data, 0x7F),),
                ((rope_data, 0xC1), (rope_data + 1, 0x7F)),
                ((scale_data, 0xFF),),
            )
            for event_index, writes in enumerate(corruptions, start=1):
                cache.zero_()
                for offset, value in writes:
                    raw[offset] = value
                before = cache.clone()
                nan_diag.report_packed_fp8_kv_cache(
                    cache,
                    indices,
                    source_id=source_id,
                    layer_id=layer_id,
                    topk_length=topk_length,
                )
                torch.cuda.synchronize()
                self.assertEqual(
                    int(report_count[state_index].item()),
                    before_count + event_index,
                )
                torch.testing.assert_close(cache, before)
                batch_id.add_(1)

            # Slot 7 is outside topk_length=2 and must not be inspected.
            cache.zero_()
            ignored_token_data, _, _ = self._packed_slot_offsets(cache, 7)
            raw[ignored_token_data] = 0x7F
            nan_diag.report_packed_fp8_kv_cache(
                cache,
                indices,
                source_id=source_id,
                layer_id=layer_id,
                kv_kind=nan_diag.KV_KIND_HCA,
                topk_length=topk_length,
            )
            torch.cuda.synchronize()
            self.assertEqual(
                int(report_count[state_index].item()),
                before_count + len(corruptions),
            )

    def test_packed_fp8_kv_cache_runs_on_cuda_graph_replay(self) -> None:
        cache = torch.zeros((1, 4, 584), dtype=torch.uint8, device="cuda")
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32, device="cuda")
        topk_length = torch.tensor([2], dtype=torch.int32, device="cuda")
        batch_id = torch.tensor([994001], dtype=torch.int64, device="cuda")
        source_id = nan_diag.SOURCE_COMPRESSED_KV_CACHE_READ
        layer_id = 45

        with patch.object(nan_diag, "ENABLED", True), patch.object(
            nan_diag, "TEST_KV_CORRUPT", None
        ):
            nan_diag.set_batch_context(batch_id)
            # Compile the exact HAS_LENGTHS variant before capture.
            nan_diag.report_packed_fp8_kv_cache(
                cache,
                indices,
                source_id=source_id,
                layer_id=layer_id,
                topk_length=topk_length,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                nan_diag.set_batch_context(batch_id)
                nan_diag.report_packed_fp8_kv_cache(
                    cache,
                    indices,
                    source_id=source_id,
                    layer_id=layer_id,
                    kv_kind=nan_diag.KV_KIND_HCA,
                    topk_length=topk_length,
                )

            state_index = nan_diag._report_state_index(source_id, layer_id)
            report_count = nan_diag._REPORT_COUNT_BY_DEVICE[str(cache.device)]
            before_count = int(report_count[state_index].item())
            token_data, _, _ = self._packed_slot_offsets(cache, 1)
            cache.view(-1)[token_data] = 0xFF
            for current_batch in (994002, 994003):
                batch_id.fill_(current_batch)
                graph.replay()
                torch.cuda.synchronize()
            self.assertEqual(
                int(report_count[state_index].item()),
                before_count + 2,
            )
            event_counters, event_records = nan_diag._ensure_event_state(cache.device)
            self.assertEqual(int(event_counters[0].item()), 1)
            self.assertEqual(event_records[0, 0].item(), 994003)
            self.assertEqual(event_records[0, 1].item(), source_id)
            self.assertEqual(event_records[0, 7].item(), 1)
            self.assertEqual(event_records[0, 12].item(), nan_diag.KV_KIND_HCA)

    def test_packed_cache_event_maps_exact_request_query_topk_and_block(self) -> None:
        cache = torch.zeros((3, 4, 584), dtype=torch.uint8, device="cuda")
        indices = torch.tensor(
            [
                [[0, 1, 2], [3, 4, 6]],
                [[7, 8, 5], [9, 10, 11]],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        batch_id = torch.tensor([994501], dtype=torch.int64, device="cuda")
        token_data, _, _ = self._packed_slot_offsets(cache, 5)

        with patch.object(nan_diag, "ENABLED", True), patch.object(
            nan_diag, "TEST_KV_CORRUPT", None
        ):
            nan_diag.set_batch_context(batch_id)
            # Compile the int64-index variant before graph capture.
            nan_diag.report_packed_fp8_kv_cache(
                cache,
                indices,
                source_id=nan_diag.SOURCE_COMPRESSED_KV_CACHE_READ,
                layer_id=46,
                kv_kind=nan_diag.KV_KIND_CSA,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                nan_diag.set_batch_context(batch_id)
                nan_diag.report_packed_fp8_kv_cache(
                    cache,
                    indices,
                    source_id=nan_diag.SOURCE_COMPRESSED_KV_CACHE_READ,
                    layer_id=46,
                    kv_kind=nan_diag.KV_KIND_CSA,
                )

            batch_id.fill_(994502)
            cache.view(-1)[token_data] = 0x7F
            graph.replay()
            torch.cuda.synchronize()

            counters, records = nan_diag._ensure_event_state(cache.device)
            self.assertEqual(int(counters[0].item()), 1)
            record = records[0].cpu().tolist()
            self.assertEqual(record[8], 1)
            self.assertEqual(record[9], 0)
            self.assertEqual(record[10], 2)
            self.assertEqual(record[11], 2)
            self.assertEqual(record[12], nan_diag.KV_KIND_CSA)
            self.assertEqual(record[13], 1)
            self.assertEqual(record[14], 1)
            self.assertEqual(record[15], 2)

    def test_indexer_post_write_maps_exact_request_query_and_slot(self) -> None:
        cache = torch.zeros((3, 4, 132), dtype=torch.uint8, device="cuda")
        slots = torch.tensor(
            [[[-1], [2]], [[6], [9]]], dtype=torch.int64, device="cuda"
        )
        batch_id = torch.tensor([995001], dtype=torch.int64, device="cuda")
        fp8_offset, _ = self._indexer_slot_offsets(cache, 6)

        with patch.object(nan_diag, "ENABLED", True), patch.object(
            nan_diag, "TEST_KV_CORRUPT", None
        ):
            nan_diag.set_batch_context(batch_id)
            cache.view(-1)[fp8_offset] = 0x7F
            before = cache.clone()
            nan_diag.report_packed_fp8_indexer_slots(
                cache,
                slots,
                source_id=nan_diag.SOURCE_INDEXER_KV_CACHE_WRITE,
                layer_id=47,
            )
            torch.cuda.synchronize()

            counters, records = nan_diag._ensure_event_state(cache.device)
            self.assertEqual(int(counters[0].item()), 1)
            record = records[0].cpu().tolist()
            self.assertEqual(record[8], 1)
            self.assertEqual(record[9], 0)
            self.assertEqual(record[10], 2)
            self.assertEqual(record[11], 0)
            self.assertEqual(record[12], nan_diag.KV_KIND_INDEXER)
            self.assertEqual(record[13], 1)
            self.assertEqual(record[14], 2)
            self.assertEqual(record[15], 2)
            torch.testing.assert_close(cache, before)

    def test_indexer_paged_read_maps_request_query_cache_position_under_graph(
        self,
    ) -> None:
        cache = torch.zeros((4, 4, 132), dtype=torch.uint8, device="cuda")
        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
        lengths = torch.tensor([[0, 0], [6, 0]], dtype=torch.int32, device="cuda")
        batch_id = torch.tensor([995501], dtype=torch.int64, device="cuda")
        # request=1, logical cache position=5 -> physical block=3, offset=1.
        fp8_offset, _ = self._indexer_slot_offsets(cache, 13)
        source_id = nan_diag.SOURCE_INDEXER_KV_CACHE_READ
        layer_id = 48

        with patch.object(nan_diag, "ENABLED", True), patch.object(
            nan_diag, "TEST_KV_CORRUPT", None
        ):
            nan_diag.set_batch_context(batch_id)
            nan_diag.report_paged_fp8_indexer_cache(
                cache,
                block_table,
                lengths,
                source_id=source_id,
                layer_id=layer_id,
                max_ctx_len=8,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                nan_diag.set_batch_context(batch_id)
                nan_diag.report_paged_fp8_indexer_cache(
                    cache,
                    block_table,
                    lengths,
                    source_id=source_id,
                    layer_id=layer_id,
                    max_ctx_len=8,
                )

            cache.view(-1)[fp8_offset] = 0xFF
            batch_id.fill_(995502)
            graph.replay()
            torch.cuda.synchronize()

            counters, records = nan_diag._ensure_event_state(cache.device)
            self.assertEqual(int(counters[0].item()), 1)
            record = records[0].cpu().tolist()
            self.assertEqual(record[8], 1)
            self.assertEqual(record[9], 0)
            self.assertEqual(record[10], 2)
            self.assertEqual(record[11], 5)
            self.assertEqual(record[12], nan_diag.KV_KIND_INDEXER)
            self.assertEqual(record[13], 3)
            self.assertEqual(record[14], 1)
            self.assertEqual(record[15], 2)


if __name__ == "__main__":
    unittest.main()
