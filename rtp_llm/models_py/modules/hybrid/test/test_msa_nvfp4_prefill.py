"""Ordinary TP1 FP4 prefill source and GPU contracts.

CPU: python test_msa_nvfp4_prefill.py SourceContractTest
GPU: python test_msa_nvfp4_prefill.py NativeWrapperTest
SourceContractTest imports no RTP, torch or CUDA module.
"""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "msa_attention.py"


class SourceContractTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text()
        self.tree = ast.parse(self.text)
        cls = next(
            n
            for n in self.tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MSAAttention"
        )
        self.methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    def test_dispatch_preserves_cp_and_legacy(self):
        text = ast.get_source_segment(self.text, self.methods["forward"])
        self.assertLess(
            text.index("self._forward_cp_prefill("),
            text.index("self._forward_nvfp4_prefill("),
        )
        self.assertLess(
            text.index("self._forward_nvfp4_prefill("),
            text.index("self._forward_prefill("),
        )
        self.assertIn("self.nvfp4_kv_cache", text)

    def test_cp_only_prepares_fmha_chunks_for_legacy_cache(self):
        method = self.methods["_forward_cp_prefill"]
        gates = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "prepare_fmha_index_score_chunks"
                for statement in node.body
                for call in ast.walk(statement)
            )
        ]
        self.assertEqual(len(gates), 1)
        condition = compile(ast.Expression(gates[0].test), str(SOURCE), "eval")
        for nvfp4 in (False, True):
            for chunk_enabled in (False, True):
                with self.subTest(nvfp4=nvfp4, chunk_enabled=chunk_enabled):
                    actual = eval(
                        condition,
                        {},
                        {
                            "self": SimpleNamespace(nvfp4_kv_cache=nvfp4),
                            "index_score_chunk_enabled": chunk_enabled,
                        },
                    )
                    self.assertEqual(actual, chunk_enabled and not nvfp4)

        # The native reader still needs its own host metadata and chunk cache.
        text = ast.get_source_segment(self.text, method)
        self.assertIn("self.nvfp4_kv_cache or index_score_chunk_enabled", text)
        self.assertIn('index_score_plan["_fp4_host_metadata"]', text)
        self.assertIn("flash_prefill_topk_to_block_tables_fp4(", text)

    def test_cp_host_planning_waits_for_metadata_not_projection(self):
        method = self.methods["_forward_cp_prefill"]
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        record = next(
            call for call in calls if ast.unparse(call.func) == "metadata_ready.record"
        )
        projection = next(
            call for call in calls if ast.unparse(call.func) == "self._project_qkv_idx"
        )
        wait = next(
            call
            for call in calls
            if ast.unparse(call.func) == "metadata_ready.synchronize"
        )
        copies = [call for call in calls if ast.unparse(call.func).endswith(".copy_")]
        self.assertTrue(copies)
        self.assertLess(max(call.lineno for call in copies), record.lineno)
        self.assertLess(record.lineno, projection.lineno)
        self.assertLess(projection.lineno, wait.lineno)
        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "synchronize"
                and isinstance(call.func.value, ast.Call)
                and ast.unparse(call.func.value.func) == "torch.cuda.current_stream"
                for call in calls
            )
        )

    def test_native_only_and_explicit_physical_addressing(self):
        text = ast.get_source_segment(self.text, self.methods["_forward_nvfp4_prefill"])
        self.assertNotIn("all_gather(", text)
        self.assertNotIn("all_reduce(", text)
        self.assertNotIn("_BF16_WORKING_PAGES", text)
        self.assertNotIn("self._kernel_slots_to_paged(", text)
        for call in (
            "table[b, pos // self.page_size]",
            "nvfp4_working_pages(",
            "flash_prefill_topk_to_block_tables_fp4(",
            "sparse_prefill_from_topk_fp4(",
            "common.apply_write_cache_store(",
        ):
            self.assertIn(call, text)

    def test_tp_and_cp_reject_before_any_runtime_import(self):
        text = ast.get_source_segment(self.text, self.methods["_forward_nvfp4_prefill"])
        namespace = {}
        exec(
            compile("from __future__ import annotations\n" + text, str(SOURCE), "exec"),
            namespace,
        )
        forward = namespace["_forward_nvfp4_prefill"]
        for local, raw in ((2, 2), (1, 2), (2, 1)):
            model = SimpleNamespace(
                tp_size=local, parallelism_config=SimpleNamespace(tp_size=raw)
            )
            with self.assertRaisesRegex(RuntimeError, "requires TP1"):
                forward(model, None, None, None)
        model = SimpleNamespace(
            tp_size=1, parallelism_config=SimpleNamespace(tp_size=1), cp_enabled=True
        )
        with self.assertRaisesRegex(RuntimeError, "CP metadata"):
            forward(model, None, SimpleNamespace(context_parallel_info=None), None)
        model.cp_enabled = False
        model._kv_sharded = True
        with self.assertRaisesRegex(RuntimeError, "sharded KV"):
            forward(model, None, SimpleNamespace(context_parallel_info=None), None)


class NativeWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

        from rtp_llm.models_py.modules.hybrid import msa_attention

        cls.torch = torch
        cls.module = msa_attention

    def model(self):
        torch = self.torch
        cls = self.module.MSAAttention
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.tp_size = 1
        model.parallelism_config = SimpleNamespace(tp_size=1)
        model.cp_enabled = False
        model._kv_sharded = False
        model.nvfp4_kv_cache = True
        model.page_size = model.block_size = model.physical_page_size = 128
        model.disable_index_value = True
        model.head_num, model.kv_head_num, model.num_idx_heads = 64, 4, 4
        model.head_dim = model.idx_head_dim = 128
        model.q_size, model.kv_size = 64 * 128, 4 * 128
        model.layer_idx = 0
        model.topk_blocks, model.init_blocks, model.local_blocks = 4, 1, 1
        model.layernorm_eps = 1e-6
        model.idx_q_norm_w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        model.idx_k_norm_w = torch.ones_like(model.idx_q_norm_w)
        model.qk_fuse_norm = None
        model.cos_sin_cache = None
        model._rope_theta = 10000.0
        model._scratch_seq_len = model._scratch_slots = model._scratch_batch_size = 0
        # Deterministic synthetic projections, actual norm/RoPE/writer/restore/
        # native index score/native sparse attention/output path underneath.
        model._project_qkv_idx = lambda hidden, *args: tuple(
            part.clone()
            for part in torch.split(
                hidden, [model.q_size + 2 * model.kv_size, 4 * 128, 128], dim=-1
            )
        )
        model.o_proj = lambda value: value
        return model

    def cache(self, blocks=64):
        torch = self.torch
        return SimpleNamespace(
            kv_cache_base=torch.zeros(
                (blocks, 2 * 4 * 128 * 64), device="cuda", dtype=torch.uint8
            ),
            kv_scale_base=torch.zeros(
                (blocks, 2 * 4 * 128 * 8 + 128 * 64 + 128 * 8),
                device="cuda",
                dtype=torch.uint8,
            ),
        )

    def inputs(self, lengths, prefixes, table, store=False):
        torch = self.torch
        return SimpleNamespace(
            is_target_verify=False,
            is_prefill=True,
            context_parallel_info=None,
            prefix_lengths=torch.tensor(prefixes, device="cuda", dtype=torch.int32),
            input_lengths=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            kv_cache_kernel_block_id_device=table,
            cache_store_inputs=store,
        )

    def run_case(self, totals, prefixes):
        torch = self.torch
        torch.manual_seed(73)
        model = self.model()
        pages_per_request = (max(totals) + 127) // 128
        self.assertLessEqual(len(totals) * pages_per_request, 63)
        table = (
            (torch.randperm(63, device="cuda")[: len(totals) * pages_per_request] + 1)
            .reshape(len(totals), pages_per_request)
            .to(torch.int32)
        )
        hidden = [
            torch.randn(
                (n, model.q_size + 2 * model.kv_size + 5 * 128),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.25
            for n in totals
        ]
        cold_cache, hot_cache = self.cache(), self.cache()
        # Ordinary prefill must never enter a TP collective or BF16 history pool.
        with patch.object(
            self.module,
            "all_gather",
            side_effect=AssertionError("unexpected TP gather"),
        ), patch.object(
            self.module,
            "all_reduce",
            side_effect=AssertionError("unexpected TP reduce"),
        ), patch.object(
            self.module._BF16_WORKING_PAGES,
            "acquire",
            side_effect=AssertionError("BF16 history"),
        ):
            cold = model.forward(
                torch.cat(hidden),
                self.inputs(totals, [0] * len(totals), table),
                cold_cache,
            ).clone()
            seed_rows = [i for i, p in enumerate(prefixes) if p]
            if seed_rows:
                model.forward(
                    torch.cat([hidden[i][: prefixes[i]] for i in seed_rows]),
                    self.inputs(
                        [prefixes[i] for i in seed_rows],
                        [0] * len(seed_rows),
                        table[seed_rows],
                    ),
                    hot_cache,
                )
            hot = model.forward(
                torch.cat([h[p:] for h, p in zip(hidden, prefixes)]),
                self.inputs([n - p for n, p in zip(totals, prefixes)], prefixes, table),
                hot_cache,
            ).clone()
        expected = torch.cat(
            [part[p:] for part, p in zip(cold.split(totals), prefixes)]
        )
        torch.testing.assert_close(hot, expected, rtol=0.02, atol=0.02)
        # Includes all packed main values and the full mixed scale/index side
        # plane, plus untouched/sentinel pages, not only the output tensor.
        self.assertTrue(torch.equal(cold_cache.kv_cache_base, hot_cache.kv_cache_base))
        self.assertTrue(torch.equal(cold_cache.kv_scale_base, hot_cache.kv_scale_base))
        self.assertTrue(torch.isfinite(hot).all().item())
        # Independent physical-slot oracle catches a cold/hot pair that is
        # identically wrong (e.g. writes compact scratch slots into persistence).
        oracle = self.cache()
        for b, rows in enumerate(hidden):
            n = rows.shape[0]
            qkv, iq, ik = model._project_qkv_idx(rows)
            q, k, v = torch.split(
                qkv, [model.q_size, model.kv_size, model.kv_size], dim=-1
            )
            q = q.reshape(n, 64, 128).contiguous()
            k, v = k.reshape(n, 4, 128).contiguous(), v.reshape(n, 4, 128).contiguous()
            iq = self.module._gemma_rmsnorm_per_head(
                iq.reshape(n, 4, 128), model.idx_q_norm_w, 1e-6
            ).contiguous()
            ik = self.module._gemma_rmsnorm_per_head(
                ik.reshape(n, 1, 128), model.idx_k_norm_w, 1e-6
            ).contiguous()
            pos = torch.arange(n, device="cuda", dtype=torch.int64)
            model._apply_rope(q, k, pos)
            model._apply_rope(iq, ik, pos)
            slots = torch.tensor(
                [int(table[b, p // 128].item()) * 128 + p % 128 for p in range(n)],
                device="cuda",
                dtype=torch.int64,
            )
            layout = self.module.nvfp4_cache_layout(
                oracle.kv_cache_base, oracle.kv_scale_base, 4, 128, 128
            )
            self.module.nvfp4_quantize_main_index_rows(
                k, v, ik, slots, layout, mma_scale_layout=True
            )
        self.assertTrue(torch.equal(cold_cache.kv_cache_base, oracle.kv_cache_base))
        self.assertTrue(torch.equal(cold_cache.kv_scale_base, oracle.kv_scale_base))

    def test_token1_odd_suffixes_and_page_boundaries(self):
        for total, prefix in (
            (1, 0),
            (127, 0),
            (128, 0),
            (129, 128),
            (131, 128),
            (257, 128),
            (259, 256),
        ):
            with self.subTest(total=total, prefix=prefix):
                self.run_case([total], [prefix])

    def test_ragged_requests_permuted_physical_pages(self):
        self.run_case([129, 259, 131], [128, 256, 0])

    def test_unwritten_tail_scale_nan_does_not_poison_native_prefill(self):
        acquire = self.module._NVFP4_WORKING_PAGES.acquire

        def poison(*args, **kwargs):
            result = acquire(*args, **kwargs)
            result[1].view(self.torch.uint8).fill_(127)
            result[3].view(self.torch.uint8).fill_(127)
            return result

        with patch.object(
            self.module._NVFP4_WORKING_PAGES, "acquire", side_effect=poison
        ):
            self.run_case([129, 259, 131], [128, 256, 0])
            self.run_case([129], [128])

    def test_sparse_selection_beyond_topk_capacity(self):
        # More than topk=4 pages ensures index selection is not an all-pages
        # special case; reuse crosses both full and partial final pages.
        self.run_case([1025, 1155, 769], [896, 1024, 512])

    def test_chunked_index_score_and_reused_working_pool(self):
        with patch.dict(os.environ, {"M3_MSA_INDEX_SCORE_CHUNK_ROWS": "64"}):
            self.run_case([259, 257, 129], [256, 128, 128])
            self.run_case([129], [128])

    def test_invalid_prefix_and_cache_store_callback(self):
        torch = self.torch
        from rtp_llm.models_py.modules.factory.attention import common

        model, cache = self.model(), self.cache()
        table = torch.tensor([[7, 2, 10]], device="cuda", dtype=torch.int32)
        hidden = torch.randn(
            (1, model.q_size + 2 * model.kv_size + 5 * 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(RuntimeError, "page-aligned"):
            model.forward(hidden, self.inputs([1], [3], table), cache)
        with patch.object(
            common, "create_write_cache_store_impl", return_value="store"
        ) as create, patch.object(common, "apply_write_cache_store") as apply:
            model.forward(hidden, self.inputs([1], [0], table, store=True), cache)
            create.assert_called_once()
            apply.assert_called_once()
            self.assertIs(apply.call_args.args[-1], cache)


if __name__ == "__main__":
    unittest.main()
