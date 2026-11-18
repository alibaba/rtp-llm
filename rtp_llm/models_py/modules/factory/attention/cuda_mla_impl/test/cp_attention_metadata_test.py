import os
import json
import statistics
import sys
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory


class CpAttentionMetadataTest(TestCase):
    @skipUnless(os.environ.get("GLM5_METADATA_BENCHMARK") == "1", "opt-in benchmark")
    def test_packed_metadata_benchmark(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.cp_metadata import (
            try_pack_lengths_and_starts,
        )
        from rtp_llm.models_py.modules.base.cuda.indexer_metadata_fusion import (
            try_expand_request_pages,
        )

        for batch in (1, 31, 127, 1031):
            actual = torch.arange(batch, device="cuda", dtype=torch.int64) % 129
            padded = ((actual + 63) // 64 + 1) * 64
            pages = torch.arange(batch * 261, device="cuda", dtype=torch.int32).view(
                batch, 261
            )[:, :257]
            requests = torch.arange(batch * 4, device="cuda", dtype=torch.int32) % batch

            def reference_lengths():
                return (
                    padded.int(),
                    (padded.cumsum(0) - padded).int(),
                    actual.int(),
                    (actual.cumsum(0) - actual).int(),
                )

            pairs = (
                (
                    "lengths",
                    reference_lengths,
                    lambda: try_pack_lengths_and_starts(actual, padded),
                ),
                (
                    "pages",
                    lambda: (pages.index_select(0, requests.long()),),
                    lambda: (try_expand_request_pages(pages, requests),),
                ),
            )
            for name, reference, candidate in pairs:
                expected, result = reference(), candidate()
                for got, want in zip(result, expected):
                    self.assertTrue(torch.equal(got, want))
                operations = (reference, candidate)
                for operation in operations:
                    for _ in range(20):
                        operation()
                torch.cuda.synchronize()
                samples = [[], []]
                for trial in range(20):
                    for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                        begin, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(50):
                            operations[arm]()
                        end.record()
                        end.synchronize()
                        samples[arm].append(begin.elapsed_time(end) * 20)
                graph_samples = [[], []]
                graphs = []
                for operation in operations:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(16):
                            operation()
                    graphs.append(graph)
                for trial in range(20):
                    for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                        begin, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(10):
                            graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        graph_samples[arm].append(begin.elapsed_time(end) * 1000 / 160)
                profiles = []
                for operation in operations:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profile:
                        for _ in range(10):
                            operation()
                        torch.cuda.synchronize()
                    kernels = [
                        event
                        for event in profile.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                        and not event.name.lower().startswith(("memcpy", "memset"))
                    ]
                    profiles.append(
                        {
                            "kernel_us": sum(
                                event.time_range.elapsed_us() for event in kernels
                            )
                            / 10,
                            "launches": len(kernels) / 10,
                            "names": sorted({event.name for event in kernels}),
                        }
                    )
                print(
                    "METADATA_AB "
                    + json.dumps(
                        {
                            "case": name,
                            "batch": batch,
                            "stream_us": [
                                statistics.median(values) for values in samples
                            ],
                            "samples": samples,
                            "graph_us": [
                                statistics.median(values) for values in graph_samples
                            ],
                            "graph_samples": graph_samples,
                            "profiles": profiles,
                        }
                    ),
                    flush=True,
                )

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_packed_lengths_refresh_all_starts_on_graph_replay(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.cp_metadata import (
            try_pack_lengths_and_starts,
        )

        actual = torch.tensor([3, 0, 65], dtype=torch.int64, device="cuda")
        padded = torch.tensor([64, 64, 128], dtype=torch.int64, device="cuda")
        try_pack_lengths_and_starts(actual, padded)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = try_pack_lengths_and_starts(actual, padded)
        pointers = [tensor.data_ptr() for tensor in outputs]
        for current, capacities in (
            ([3, 0, 65], [64, 64, 128]),
            ([0, 2, 1], [128, 64, 64]),
        ):
            actual.copy_(torch.tensor(current, dtype=torch.int64, device="cuda"))
            padded.copy_(torch.tensor(capacities, dtype=torch.int64, device="cuda"))
            graph.replay()
            expected = (
                capacities,
                [sum(capacities[:i]) for i in range(3)],
                current,
                [sum(current[:i]) for i in range(3)],
            )
            for result, values, pointer in zip(outputs, expected, pointers):
                self.assertEqual(result.data_ptr(), pointer)
                self.assertEqual(result.dtype, torch.int32)
                self.assertEqual(result.cpu().tolist(), values)

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_packed_page_rows_follow_live_requests_and_table_on_replay(self):
        from rtp_llm.models_py.modules.base.cuda.indexer_metadata_fusion import (
            try_expand_request_pages,
        )

        storage = torch.arange(3 * 261, dtype=torch.int32, device="cuda").view(3, 261)
        pages = storage[:, :257]
        requests = torch.tensor([2, 0, 0, 1, 2], dtype=torch.int32, device="cuda")
        try_expand_request_pages(pages, requests)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = try_expand_request_pages(pages, requests)
        pointer = result.data_ptr()
        for ids in ([2, 0, 0, 1, 2], [0, 0, 2, 2, 1]):
            requests.copy_(torch.tensor(ids, dtype=torch.int32, device="cuda"))
            pages.add_(3)
            pages[0, 0] = 0
            graph.replay()
            self.assertEqual(result.data_ptr(), pointer)
            self.assertTrue(torch.equal(result, pages.index_select(0, requests.long())))

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_attention_prepare_refreshes_shared_pages_for_new_cache_group(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            flashmla_sparse_impl as sparse,
        )

        SparseMlaImpl = sparse.SparseMlaImpl

        requests = torch.tensor([1, 0, 1], dtype=torch.int32, device="cuda")
        owner = SimpleNamespace(fmha_params=SimpleNamespace(batch_indice_d=requests))
        inputs = SimpleNamespace(
            is_target_verify=True,
            is_prefill=True,
            glm5_mtp_metadata_fusion_enabled=True,
            kv_cache_kernel_block_id_device=torch.arange(
                8, dtype=torch.int32, device="cuda"
            ).view(2, 4),
        )
        # Only the independent DeepGEMM schedule API is unavailable here; drive
        # the real attention-owned table preparation and real CUDA copy.
        with patch.object(sparse, "PREFILL_OPTIMIZATIONS_ENABLED", True), patch.dict(
            os.environ, {"RTP_LLM_ADAPTIVE_VERIFICATION": "1"}
        ), patch.dict(sys.modules, {"deep_gemm": None}):
            SparseMlaImpl._refresh_paged_mqa_schedule_metadata(owner, inputs, False)
            first_output = owner.prepared_paged_indexer_table[1]
            inputs.kv_cache_kernel_block_id_device = (
                inputs.kv_cache_kernel_block_id_device + 100
            )
            requests.copy_(torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda"))
            SparseMlaImpl._refresh_paged_mqa_schedule_metadata(owner, inputs, True)
        source, output = owner.prepared_paged_indexer_table
        self.assertEqual(
            source.data_ptr(), inputs.kv_cache_kernel_block_id_device.data_ptr()
        )
        self.assertEqual(output.data_ptr(), first_output.data_ptr())
        self.assertTrue(torch.equal(output, source.index_select(0, requests.long())))

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_unsupported_or_disabled_packed_metadata_keeps_graph_fallback(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            flashmla_sparse_impl as sparse,
        )

        for enabled, capability, dtype in (
            (True, True, torch.int64),
            (False, True, torch.int32),
            (True, False, torch.int32),
        ):
            owner = SimpleNamespace(
                fmha_params=SimpleNamespace(
                    batch_indice_d=torch.tensor([0], device="cuda", dtype=torch.int32)
                )
            )
            inputs = SimpleNamespace(
                is_target_verify=True,
                is_prefill=True,
                glm5_mtp_metadata_fusion_enabled=capability,
                kv_cache_kernel_block_id_device=torch.ones(
                    (1, 4), device="cuda", dtype=dtype
                ),
            )
            with patch.object(
                sparse, "PREFILL_OPTIMIZATIONS_ENABLED", enabled
            ), patch.dict(
                os.environ, {"RTP_LLM_ADAPTIVE_VERIFICATION": "1"}
            ), patch.dict(
                sys.modules, {"deep_gemm": None}
            ):
                sparse.SparseMlaImpl._refresh_paged_mqa_schedule_metadata(
                    owner, inputs, False
                )
                sparse.SparseMlaImpl._refresh_paged_mqa_schedule_metadata(
                    owner, inputs, True
                )
            self.assertIsNone(owner.prepared_paged_indexer_table)

    def test_cp_factory_does_not_read_device_scalar(self) -> None:
        candidate = MagicMock()
        candidate.__name__ = "CpCandidate"
        candidate.support.return_value = True
        candidate.support_parallelism_config.return_value = True
        candidate.is_sparse.return_value = True
        inputs = SimpleNamespace(is_prefill=True, cu_kv_seqlens=MagicMock())
        inputs.cu_kv_seqlens.max.side_effect = AssertionError("CP needs no scalar read")
        parallelism = SimpleNamespace(prefill_cp_config=MagicMock())
        parallelism.prefill_cp_config.is_enabled.return_value = True
        weights = MagicMock()
        config = SimpleNamespace(is_sparse=True, indexer_topk=128)

        with patch.object(attn_factory, "PREFILL_MLA_IMPS", [candidate]):
            actual = attn_factory.get_mla_impl(
                config, weights, inputs, parallelism_config=parallelism
            )

        self.assertIs(actual, candidate.return_value)
        inputs.cu_kv_seqlens.max.assert_not_called()


if __name__ == "__main__":
    main()
