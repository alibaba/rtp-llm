"""A fresh process verifies rollback despite conflicting optimized subflags."""

import os

os.environ["RTP_LLM_CP_PREFILL_PIPELINE_OPT"] = "0"
for name in (
    "RTP_LLM_CP_TORCH_SYMM_GATHER",
    "RTP_LLM_CP_KV_OVERLAP",
    "RTP_LLM_CP_KV_CROSS_LAYER",
    "RTP_LLM_CP_INDEXER_OVERLAP",
    "RTP_LLM_CP_HOST_PLAN",
):
    os.environ[name] = "1"
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.prefill_optimization import (
    PREFILL_OPTIMIZATIONS_ENABLED,
    prefill_optimization_setting,
)
from rtp_llm.models_py.distributed import pynccl_cp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)


class PrefillRollbackTest(unittest.TestCase):
    def test_master_overrides_subflags(self):
        self.assertFalse(PREFILL_OPTIMIZATIONS_ENABLED)
        for name in (
            "RTP_LLM_CP_TORCH_SYMM_GATHER",
            "RTP_LLM_CP_KV_OVERLAP",
            "RTP_LLM_CP_KV_CROSS_LAYER",
            "RTP_LLM_CP_INDEXER_OVERLAP",
            "RTP_LLM_CP_HOST_PLAN",
        ):
            self.assertEqual(prefill_optimization_setting(name, "1"), "0")
        self.assertFalse(pynccl_cp._TORCH_SYMM_GATHER)

    def test_preserves_historical_flag_defaults_and_overrides(self):
        self.assertEqual(cp._REMAP_HISTORY_TOPK_DEFAULT, "0")
        flags = {
            "RTP_LLM_CP_KV_INDEX_OPT": "1",
            "RTP_LLM_CP_KV_REGISTERED": "1",
            "RTP_CP_PREPARED_SCATTER": "0",
            "RTP_CP_CONTIGUOUS_ROPE_KPE": "0",
            "RTP_CP_REMAP_HISTORY_TOPK": "0",
            "RTP_LLM_CP_INDEXER_REGISTERED": "0",
        }
        for name, default in flags.items():
            with self.subTest(name=name), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(prefill_optimization_setting(name, default), default)
                for explicit in ("0", "1"):
                    os.environ[name] = explicit
                    self.assertEqual(
                        prefill_optimization_setting(name, default), explicit
                    )

    def test_separate_gathers_restore_exact_rows(self):
        for rows, ids in [(0, []), (1, [1, 0]), (4, [0, 4, 1, 5, 3])]:
            for sharded in (False, True):
                with self.subTest(rows=rows, sharded=sharded):
                    k = torch.arange(rows * 12, dtype=torch.bfloat16).view(rows, 12)[
                        :, ::2
                    ]
                    rope = torch.arange(rows * 8, dtype=torch.bfloat16).view(rows, 8)[
                        :, ::2
                    ]
                    ids_t = torch.tensor(ids, dtype=torch.int64)
                    op = cp.SparseMlaFp8CPOp.__new__(cp.SparseMlaFp8CPOp)
                    op.kv_restore_unpad_indices = ids_t
                    op.kv_cache_sharded = sharded
                    op.sharded_slot_mapping = torch.arange(len(ids))
                    op.mla_params = object()
                    op.kv_cache_write_op = SimpleNamespace(forward=Mock())
                    events = []

                    def gather(tensor, group, role):
                        self.assertTrue(tensor.is_contiguous())
                        events.append(role)
                        return torch.cat((tensor, tensor + 100))

                    with patch.object(cp, "all_gather", side_effect=gather):
                        op.write_kv(k, rope, object())
                    self.assertEqual(events, ["mla_ckv", "mla_kpe"])
                    call = op.kv_cache_write_op.forward.call_args
                    torch.testing.assert_close(
                        call.args[0], torch.cat((k, k + 100))[ids_t], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        call.args[1],
                        torch.cat((rope, rope + 100))[ids_t],
                        rtol=0,
                        atol=0,
                    )
                    self.assertIs(
                        call.kwargs["slot_mapping_override"],
                        op.sharded_slot_mapping if sharded else None,
                    )

    def test_torch_symmetric_dispatch_is_not_entered(self):
        data = torch.ones((3, 8), dtype=torch.bfloat16)
        with patch.object(
            pynccl_cp, "_registered_indexer_all_gather", return_value=None
        ), patch.object(pynccl_cp, "enabled", return_value=True), patch.object(
            pynccl_cp, "_resolve_process_group", return_value="tp"
        ), patch.object(
            pynccl_cp, "_world_size", return_value=4
        ), patch.object(
            pynccl_cp, "_symm_output", return_value=None
        ), patch.object(
            pynccl_cp,
            "_try_torch_symm_all_gather",
            side_effect=AssertionError("optimized dispatch"),
        ), patch.object(
            pynccl_cp, "all_gather_into_tensor"
        ) as gather:
            out = pynccl_cp.all_gather(data, group="tp", role="mla_ckv")
        self.assertEqual(out.shape, (12, 8))
        gather.assert_called_once_with(out, data, group="tp")

    def test_compatibility_prefetch_new_rows_uses_ordinary_collective(self):
        # Exercise the retained compatibility method directly, including a
        # rank with no owned new rows. Normal rollback disables its activation.
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            cp_kv_prefetch as kv,
        )

        for count in (0, 2):
            with self.subTest(count=count):
                gathered = torch.arange(4 * 576).reshape(4, 576).to(torch.bfloat16)
                restore = torch.tensor([3, 0])
                pipeline = SimpleNamespace(
                    owners=[None, 1],
                    new_slots={"hbm": torch.zeros((1, count), dtype=torch.int32)},
                    plan={"new_max": 2, "new_restore": restore},
                    buffers=[torch.empty(2, 576), torch.empty(2, 576)],
                    device=torch.device("cpu"),
                    cache=SimpleNamespace(host_layers=[]),
                    new_lengths=torch.tensor([count], dtype=torch.int32),
                    new_starts=torch.tensor([0], dtype=torch.int32),
                    ready=[Mock(), Mock()],
                    prefetch=Mock(),
                )
                cache = SimpleNamespace(
                    kv_cache_base=torch.zeros(1, 576, dtype=torch.uint8)
                )
                with patch.object(
                    kv, "all_gather", return_value=gathered
                ) as gather, patch.object(
                    kv.rtp_llm_ops, "cp_gather_and_upconvert_fp8_kv_cache_v2"
                ) as upconvert:
                    result = kv.CrossLayerKvPrefetch.prepare(pipeline, 1, cache)
                gather.assert_called_once()
                self.assertEqual(
                    gather.call_args.kwargs,
                    {"group": kv.Group.TP, "role": "mla_new_rows"},
                )
                self.assertEqual(tuple(gather.call_args.args[0].shape), (2, 576))
                self.assertEqual(upconvert.call_count, int(count > 0))
                torch.testing.assert_close(
                    pipeline.pending_new, gathered[restore], rtol=0, atol=0
                )
                pipeline.prefetch.assert_called_once_with(2)
                self.assertIs(result[0], pipeline.buffers[1])
                self.assertIs(result[1], pipeline.ready[1])

    def test_cross_layer_disabled(self):
        fake = SimpleNamespace(overlap_kv=False)
        self.assertIsNone(cp.SparseMlaCpImpl.begin_cross_layer_prefetch(fake, object()))


if __name__ == "__main__":
    unittest.main()
