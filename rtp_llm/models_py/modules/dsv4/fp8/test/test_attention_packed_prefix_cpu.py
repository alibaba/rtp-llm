"""Exercise real prefix dispatch with CPU tensors and kernel/transport doubles.

GPU byte equality is covered separately by test_swa_fp8_kv_roundtrip.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types
import unittest
from contextlib import nullcontext
from runpy import run_path
from unittest.mock import Mock, patch

import torch

FP8 = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "rtp_llm.models_py.modules.dsv4"


def _load_attention(is_sm120):
    path = FP8 / "attention.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AttentionFP8"
    )
    names = {
        "_attn_via_workspace",
        "_attn_fp8_swa_via_concat",
        "_flash_mla_sparse_fwd_chunked_projected",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in cls.body} == names
    cls.bases = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    env = {
        "torch": torch,
        "is_sm120": is_sm120,
        "record_function_range": lambda *_: nullcontext(),
        "dispose_tensor": lambda *_: None,
        "_DSV4_FP8_KV_ENTRY_BYTES": 584,
    }
    exec(compile(module, str(path), "exec"), env)
    return env["AttentionFP8"]


class PackedPrefixDispatchTest(unittest.TestCase):
    def setUp(self):
        self.sm120 = Mock(return_value=True)
        self.attention = _load_attention(self.sm120)
        self.dequant = Mock()
        self.quantize = Mock()
        self.split = Mock(return_value=(None, None, None, None, 64))
        modules = {}

        def module(name, **attributes):
            if name not in modules:
                value = types.ModuleType(name)
                value.__path__ = []
                modules[name] = value
                parent, _, child = name.rpartition(".")
                if parent:
                    setattr(module(parent), child, value)
            vars(modules[name]).update(attributes)
            return modules[name]

        module(
            PACKAGE + ".fp8._swa_dequant_triton",
            **{
                name: getattr(self.dequant, name)
                for name in (
                    "dequantize_and_gather_k_cache_slots",
                    "dequantize_and_gather_k_cache_slots_cp_byte_sliced",
                    "gather_packed_k_cache_slots_cp_byte_sliced",
                )
            },
        )
        combine = Mock(
            return_value=(
                torch.zeros(2, 1, dtype=torch.int32),
                torch.ones(2, dtype=torch.int32),
            )
        )
        module(
            PACKAGE + ".fp8._swa_ops_triton",
            combine_topk_swa_indices=combine,
            combine_topk_swa_indices_cp=combine,
        )
        module(
            PACKAGE + ".fp8._sm120_prefill_indices",
            split_sm120_combined_tables=self.split,
        )
        module(PACKAGE + ".kv_cache_utils", SWA_KV="swa", CSA_KV="csa", HCA_KV="hca")
        module(
            PACKAGE + ".fp8._swa_kv_insert_triton",
            quantize_and_insert_k_cache=self.quantize,
        )
        module(
            PACKAGE + ".fp8.sm120_sparse_mla",
            SM120_EXTRA_TOPK_WIDTHS=(),
            canonical_topk=Mock(),
            run=Mock(),
            run_chunked_reference=Mock(),
            uses_generic_fallback=Mock(),
        )
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def _exercise(self, *, sm120, byte_sliced, continuation):
        self.sm120.return_value = sm120
        for path in ("swa", "csa", "hca"):
            with self.subTest(path=path):
                self.dequant.reset_mock()
                pool = torch.zeros(1, 64, 584, dtype=torch.uint8)
                prefix = int(continuation)
                slots = torch.tensor([[7]], dtype=torch.int64)
                gathered_pool = pool.clone()
                compact_slots = torch.tensor([[0]], dtype=torch.int64)
                packed_gather = self.dequant.gather_packed_k_cache_slots_cp_byte_sliced
                packed_gather.return_value = (gathered_pool, compact_slots)
                lengths = torch.tensor([prefix], dtype=torch.int32)
                qkv = types.SimpleNamespace(
                    q=torch.zeros(2, 1, 512, dtype=torch.bfloat16),
                    kv_full=torch.ones(2, 512, dtype=torch.bfloat16),
                )
                layer = self.attention()
                layer.head_dim = layer.dim = 512
                layer.compress_ratio = 128 if path == "hca" else 4
                layer.window_size = 128
                layer._swa_cp_byte_sliced = Mock(return_value=byte_sliced)
                layer._pool_view_3d_fp8 = Mock(return_value=pool)
                layer._pool_raw_u8 = Mock(return_value=pool.view(1, -1))
                layer._swa_entries_per_block = Mock(return_value=64)
                layer._should_async_workspace_reads_for_prefill = Mock(
                    return_value=False
                )
                layer._flash_mla_sparse_fwd_chunked_projected = Mock(
                    return_value="output"
                )
                common = types.SimpleNamespace(
                    use_varlen=True,
                    batch_size=1,
                    any_cont=continuation,
                    cp_ctx=types.SimpleNamespace(cp_rank=0, cp_size=8),
                    cp_on=False,
                    freqs_cis=None,
                    workspace=object(),
                )
                if path == "swa":
                    common.swa_meta = types.SimpleNamespace(
                        combined_indices=torch.zeros(2, 1, dtype=torch.int32),
                        combined_lens=torch.ones(2, dtype=torch.int32),
                        cache_seq_lens=lengths,
                        cache_gather_lens=lengths,
                        M=2 + prefix,
                        prefix_len_max=prefix,
                        cache_slot_mapping=slots,
                        cache_compaction=object(),
                        slot_in_flat=torch.arange(2) + prefix,
                    )
                    output = layer._attn_fp8_swa_via_concat(qkv, common)
                else:
                    wm = types.SimpleNamespace(
                        swa_seq_lens=torch.tensor([2 + prefix]),
                        use_cp_raw_q_merge=False,
                        M=3 + prefix,
                        N=1,
                        cmp_reader=Mock(),
                        cmp_seq_lens=torch.tensor([1]),
                        cmp_bt_int32=torch.tensor([[0]]),
                        cmp_eb=64,
                        swa_cache_slot_mapping=slots,
                        swa_cache_gather_lens=lengths,
                        swa_cache_compaction=object(),
                        swa_eb=64,
                        new_k_slot_in_flat=torch.arange(2) + 1 + prefix,
                        dense_cmp_topk=torch.zeros(2, 1, dtype=torch.int32),
                        qsl=torch.tensor([0, 2]),
                        swa_gather_lens=torch.tensor([2 + prefix]),
                    )
                    output = layer._attn_via_workspace(
                        qkv, common, workspace_meta=wm, cmp_topk_runtime=None
                    )
                self.assertEqual(output, "output")
                kwargs = layer._flash_mla_sparse_fwd_chunked_projected.call_args.kwargs
                preserve = sm120 and continuation
                expected_pool = gathered_pool if byte_sliced else pool
                expected_slots = compact_slots if byte_sliced else slots
                self.assertIs(
                    kwargs["sm120_prefix_cache"], expected_pool if preserve else None
                )
                self.assertIs(
                    kwargs["sm120_prefix_slots"], expected_slots if preserve else None
                )
                self.assertIs(
                    kwargs["sm120_prefix_lens"], lengths if preserve else None
                )
                regular = self.dequant.dequantize_and_gather_k_cache_slots
                sliced = self.dequant.dequantize_and_gather_k_cache_slots_cp_byte_sliced
                self.assertEqual(
                    regular.call_count,
                    int(continuation and not preserve and not byte_sliced),
                )
                self.assertEqual(
                    sliced.call_count,
                    int(continuation and byte_sliced and not preserve),
                )
                self.assertEqual(
                    packed_gather.call_count, int(preserve and byte_sliced)
                )

    def test_sm120_continuation_preserves_original_bytes_by_default(self):
        self._exercise(sm120=True, byte_sliced=False, continuation=True)

    def test_cold_prefill_does_not_read_prefix_cache(self):
        self._exercise(sm120=True, byte_sliced=False, continuation=False)
        self._exercise(sm120=True, byte_sliced=True, continuation=False)

    def test_other_architectures_keep_dequantization(self):
        self._exercise(sm120=False, byte_sliced=False, continuation=True)
        self._exercise(sm120=False, byte_sliced=True, continuation=True)

    def test_byte_sliced_cp_preserves_packed_prefix_after_gather(self):
        self._exercise(sm120=True, byte_sliced=True, continuation=True)

    def test_packing_receives_prefix_tables(self):
        layer = self.attention()
        layer.n_heads = 1
        layer.head_dim = 512
        q = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
        prefix = torch.zeros(1, 64, 584, dtype=torch.uint8)
        destination = torch.empty_like(prefix)
        slots = torch.tensor([[7]], dtype=torch.int64)
        lengths = torch.tensor([1], dtype=torch.int32)
        workspace = Mock()
        workspace.prefill_q.return_value = q
        workspace.sm120_packed_cache.return_value = destination
        workspace.sm120_linear_slots.return_value = torch.arange(3)
        self.quantize.side_effect = RuntimeError("packing boundary reached")
        with self.assertRaisesRegex(RuntimeError, "packing boundary reached"):
            layer._flash_mla_sparse_fwd_chunked_projected(
                q=q,
                kv=torch.zeros(3, 1, 512, dtype=torch.bfloat16),
                indices=torch.zeros(2, 1, 1, dtype=torch.int32),
                topk_length=torch.ones(2, dtype=torch.int32),
                freqs_cis=torch.zeros(2, 1),
                prefill_workspace=workspace,
                profile_name="test",
                sm120_prefix_cache=prefix,
                sm120_prefix_slots=slots,
                sm120_prefix_lens=lengths,
            )
        kwargs = self.quantize.call_args.kwargs
        self.assertIs(kwargs["prefix_cache"], prefix)
        self.assertIs(kwargs["prefix_slot_mapping"], slots)
        self.assertIs(kwargs["prefix_gather_lens"], lengths)
        self.assertIs(self.quantize.call_args.args[1], destination)


def setUpModule():
    run_path(str(FP8.parent / "test" / "cpu_test_utils.py"))[
        "isolate_cpu_test_module"
    ]()


if __name__ == "__main__":
    unittest.main()
