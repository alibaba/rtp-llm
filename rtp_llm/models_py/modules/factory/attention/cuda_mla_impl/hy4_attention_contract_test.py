import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_impl as sparse,
)


class Hy4AttentionContractTest(unittest.TestCase):
    def test_decode_cp_capability_exception_requires_opt_in(self):
        from rtp_llm.models_py.modules.factory.attention import attn_factory

        config = SimpleNamespace(
            is_sparse=True, use_mla=True, kv_cache_dtype=None, indexer_topk=2048
        )
        for role in (None, "is_target_verify", "is_draft_extend"):
            inputs = SimpleNamespace(is_prefill=role is not None)
            if role is not None:
                setattr(inputs, role, True)
            impl = mock.Mock()
            impl.__name__ = "SparseProbe"
            impl.support.return_value = True
            impl.is_sparse.return_value = True
            impl.support_parallelism_config.return_value = False
            with mock.patch.object(attn_factory, "DECODE_MLA_IMPS", [impl]):
                with self.assertRaisesRegex(Exception, "can not find mla type"):
                    attn_factory.get_mla_impl(config, mock.Mock(), inputs)
                result = attn_factory.get_mla_impl(
                    config, mock.Mock(), inputs, decode_ignores_prefill_cp=True
                )
            self.assertIs(result, impl.return_value)
            impl.support_parallelism_config.assert_called_once()

    def test_fp8_constructor_preserves_glm_heads_and_accepts_explicit_padding(self):
        def initialize(op, *args, **kwargs):
            op.num_heads = 64

        with mock.patch.object(sparse.SparseMlaOp, "__init__", initialize):
            self.assertEqual(sparse.SparseMlaFp8Op().bf16_num_heads, 64)
            self.assertEqual(
                sparse.SparseMlaFp8Op(bf16_prefill_num_heads=128).bf16_num_heads, 128
            )

    def setUp(self):
        self.q = torch.randn(2, 2, 4)
        self.kv = torch.zeros(2, 1, 4, dtype=torch.uint8)
        self.topk = torch.tensor([[[0, -1]], [[1, -1]]], dtype=torch.int32)
        self.physical = torch.tensor([[[5, -1]], [[9, -1]]], dtype=torch.int32)
        self.sink = torch.tensor([0.25, -0.5])
        self.output = torch.randn(2, 2, 3)

    def _op(self, cls):
        op = cls.__new__(cls)
        op.num_heads = 2
        op.bf16_num_heads = 2
        op.kv_lora_rank = 3
        op.qk_rope_head_dim = 1
        op.scale = 0.5
        op._convert_topk_indices_to_global = mock.Mock(
            side_effect=AssertionError("physical indices must not be converted")
        )
        return op

    def test_bf16_physical_indices_and_sink_reach_kernel(self):
        op = self._op(sparse.SparseMlaOp)
        with mock.patch.object(
            sparse, "flash_mla_sparse_fwd", return_value=(self.output, None, None)
        ) as kernel:
            result = op.forward(
                self.q,
                self.kv,
                self.topk,
                physical_indices=self.physical,
                attn_sink=self.sink,
            )
        self.assertIs(kernel.call_args.args[2], self.physical)
        self.assertIs(kernel.call_args.kwargs["attn_sink"], self.sink)
        op._convert_topk_indices_to_global.assert_not_called()
        torch.testing.assert_close(result, self.output)

    def test_fp8_physical_indices_bypass_gather_and_reach_kernel_with_sink(self):
        op = self._op(sparse.SparseMlaFp8Op)
        op._gather = object()
        op._forward_gather = mock.Mock(side_effect=AssertionError("unexpected gather"))
        op._sched_meta = SimpleNamespace(tile_scheduler_metadata=None, num_splits=None)
        op.block_table = torch.zeros(1, 1, dtype=torch.int32)
        with mock.patch.object(
            sparse,
            "flash_mla_with_kvcache",
            return_value=(self.output.unsqueeze(0), None),
        ) as kernel:
            result = op.forward(
                self.q,
                self.kv,
                self.topk,
                physical_indices=self.physical,
                attn_sink=self.sink,
            )
        kwargs = kernel.call_args.kwargs
        torch.testing.assert_close(
            kwargs["indices"], self.physical.squeeze(1).unsqueeze(0)
        )
        self.assertIs(kwargs["attn_sink"], self.sink)
        op._convert_topk_indices_to_global.assert_not_called()
        op._forward_gather.assert_not_called()
        torch.testing.assert_close(result, self.output)

    def test_fp8_gather_keeps_sink_and_masked_indices(self):
        op = self._op(sparse.SparseMlaFp8Op)
        op._gather = SimpleNamespace(
            total_kv_len=2,
            batch_size=1,
            seq_lens=torch.tensor([2], dtype=torch.int32),
            workspace_starts=torch.tensor([0], dtype=torch.int32),
        )
        op.mla_params = SimpleNamespace(batch_indice_d=torch.zeros(2, dtype=torch.long))
        op.block_table = torch.zeros(1, 1, dtype=torch.int32)
        fused = torch.zeros(2, 4, dtype=torch.bfloat16)
        with (
            mock.patch.object(sparse, "_allocate_prefill_fused_kv", return_value=fused),
            mock.patch.object(
                sparse.rtp_llm_ops, "cp_gather_and_upconvert_fp8_kv_cache_v2"
            ) as gather,
            mock.patch.object(
                sparse, "flash_mla_sparse_fwd", return_value=(self.output, None, None)
            ) as kernel,
        ):
            result = op.forward(self.q, self.kv, self.topk, attn_sink=self.sink)
        gather.assert_called_once()
        self.assertIs(kernel.call_args.kwargs["attn_sink"], self.sink)
        torch.testing.assert_close(kernel.call_args.args[2], self.topk)
        torch.testing.assert_close(result, self.output)

    def test_cp_wrapper_forwards_prepared_kv_and_sink_and_releases_workspace(self):
        for fail in (False, True):
            with self.subTest(kernel_raises=fail):
                impl = cp.SparseMlaCpImpl.__new__(cp.SparseMlaCpImpl)
                workspace = mock.Mock()
                workspace.q_transformed.return_value = self.q
                workspace.scatter_output.return_value = self.output
                impl._glm53_prefill_workspace = workspace
                impl.num_heads = 2
                impl.kv_lora_rank = 3
                impl.rope_head_dim = 1
                impl.fmha_params = SimpleNamespace(
                    batch_indice_d=torch.zeros(2, dtype=torch.long)
                )
                backend = mock.Mock()
                backend.total_local_ids = torch.arange(2)
                backend.total_local_ids_is_identity = True
                backend.use_cuda_graph = False
                backend.forward.return_value = self.output
                if fail:
                    backend.forward.side_effect = RuntimeError("attention failure")
                impl.fmha_impl = backend
                impl._apply_input_bmm = mock.Mock(return_value=self.q)
                impl._apply_output_bmm = mock.Mock(return_value=self.output)
                impl._apply_rope = mock.Mock(
                    side_effect=AssertionError("prepared KV already rotated")
                )
                ready = object()
                fused = torch.zeros(2, 4)
                stream = mock.Mock()
                with mock.patch.object(
                    torch.cuda, "current_stream", return_value=stream
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "attention failure"):
                            impl.forward(
                                self.q,
                                self.kv,
                                self.kv,
                                object(),
                                0,
                                self.topk,
                                prepared_kv=(fused, ready),
                                attn_sink=self.sink,
                            )
                    else:
                        result = impl.forward(
                            self.q,
                            self.kv,
                            self.kv,
                            object(),
                            0,
                            self.topk,
                            prepared_kv=(fused, ready),
                            attn_sink=self.sink,
                        )
                        self.assertIs(result, self.output)
                self.assertIs(backend.forward.call_args.kwargs["prepared_kv"], fused)
                self.assertIs(backend.forward.call_args.kwargs["attn_sink"], self.sink)
                stream.wait_event.assert_called_once_with(ready)
                impl._apply_rope.assert_not_called()
                workspace.release.assert_called_once_with()

    def test_cp_forward_binds_prepared_kv_separately_from_sink(self):
        op = self._op(cp.SparseMlaFp8CPOp)
        op.total_local_ids = torch.arange(2)
        op.total_local_ids_is_identity = True
        op._gather = object()
        op.write_cache_store_impl = None
        op.attn_inputs = object()
        op.write_kv = mock.Mock(
            side_effect=AssertionError("prepared KV must not be rewritten")
        )
        fused = torch.zeros(2, 4)
        signature = inspect.signature(cp.SparseMlaFp8CPOp._attend_gather)
        bound_calls = []

        def attend(*args, **kwargs):
            bound = signature.bind(op, *args, **kwargs)
            bound.apply_defaults()
            bound_calls.append(bound.arguments)
            return self.output

        op._attend_gather = mock.Mock(side_effect=attend)
        with mock.patch.object(cp.common, "apply_write_cache_store"):
            result = op.forward(
                self.q,
                self.kv,
                self.kv,
                self.topk,
                torch.zeros(2, dtype=torch.long),
                object(),
                prepared_kv=fused,
                attn_sink=self.sink,
            )
        self.assertEqual(len(bound_calls), 1)
        self.assertIs(bound_calls[0]["prepared_kv"], fused)
        self.assertIs(bound_calls[0]["attn_sink"], self.sink)
        self.assertIs(result, self.output)
        op.write_kv.assert_not_called()

    def test_cp_sharded_fallback_never_binds_sink_as_prepared_kv(self):
        op = self._op(cp.SparseMlaFp8CPOp)
        op.kv_cache_sharded = True
        signature = inspect.signature(cp.SparseMlaFp8CPOp._attend_gather)
        bound_calls = []

        def attend(*args, **kwargs):
            bound = signature.bind(op, *args, **kwargs)
            bound.apply_defaults()
            bound_calls.append(bound.arguments)
            return self.output

        op._attend_gather = mock.Mock(side_effect=attend)
        result = op._attend_with_kvcache(
            self.q, object(), self.topk, 0, attn_sink=self.sink
        )
        self.assertEqual(len(bound_calls), 1)
        self.assertIsNone(bound_calls[0]["prepared_kv"])
        self.assertIs(bound_calls[0]["attn_sink"], self.sink)
        self.assertIs(result, self.output)


if __name__ == "__main__":
    unittest.main()
