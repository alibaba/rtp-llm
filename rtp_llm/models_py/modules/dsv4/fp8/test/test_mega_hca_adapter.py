from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rtp_kernel
import torch

from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import MegaCSARuntime
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    DIM,
    HC,
    HEAD_DIM,
    MAIN_HEADS,
    MAX_BATCH,
    O_GROUPS,
    O_LORA_RANK,
    Q_LORA_RANK,
    ROPE_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter import MegaHCAAdapter
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_weights import (
    HCA_APE_ROWS,
    HCA_COMPRESS_RATIO,
    HCA_STATE_WIDTH,
    MegaHCAWeights,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.utils.model_weight import W


class _IdentityNorm:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _block_stub(adapter: object | None) -> Block:
    block = Block.__new__(Block)
    torch.nn.Module.__init__(block)
    block.layer_id = 3
    block._mega_csa_adapter = None
    block._mega_hca_adapter = adapter
    block._mega_front_adapter = None
    block.attn_norm = _IdentityNorm()
    block.ffn_norm = _IdentityNorm()
    block.attn = MagicMock()
    block.attn.forward_decode.side_effect = lambda value, *_args, **_kwargs: value
    block.attn_hc = MagicMock()
    block.attn_hc.pre.side_effect = lambda value, **_kwargs: (
        value[..., 0, :],
        torch.ones(*value.shape[:-2], 1, 1),
        torch.ones(*value.shape[:-2], 1, 1),
    )
    block.attn_hc.post.side_effect = lambda value, *_args: value.unsqueeze(-2)
    block.ffn_hc = MagicMock()
    block.ffn_hc.pre.side_effect = block.attn_hc.pre.side_effect
    block.ffn_hc.post.side_effect = lambda value, *_args: value.unsqueeze(-2)
    block.ffn = MagicMock(side_effect=lambda value, _input_ids: value)
    return block


class MegaHCARoutingTest(unittest.TestCase):
    def test_transformer_switch_enables_csa_and_hca_together(self) -> None:
        class _Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enable_mega_csa = MagicMock()
                self.enable_mega_hca = MagicMock()
                self.enable_mega_front = MagicMock()

        layer = _Layer()
        global_weights = MagicMock()
        global_weights.__getitem__.return_value = torch.ones(1, dtype=torch.bfloat16)
        model_weights = SimpleNamespace(global_weights=global_weights, weights=[{}])
        args = V4Args(
            n_layers=1,
            n_mtp_layers=0,
            compress_ratios=[128],
            is_decode_role=False,
            fp8_kv_cache=True,
            tp_size=1,
            ep_size=8,
        )

        with patch.dict(os.environ, {"DSV4_MEGA": "1"}, clear=True), patch(
            "rtp_llm.models_py.modules.dsv4.transformer._build_block",
            return_value=layer,
        ), patch(
            "rtp_llm.models_py.modules.dsv4.transformer.EmbeddingTorch",
            return_value=torch.nn.Identity(),
        ), patch(
            "rtp_llm.models_py.modules.dsv4.transformer.RMSNorm",
            return_value=torch.nn.Identity(),
        ), patch(
            "rtp_llm.models_py.modules.dsv4.transformer.build_hc_head",
            return_value=torch.nn.Identity(),
        ), patch(
            "rtp_llm.models_py.modules.dsv4.fp8.decode.mega_support."
            "mega_decode_unavailable_reason",
            return_value=None,
        ):
            transformer = V4Transformer(args, model_weights)

        self.assertIsNotNone(transformer._mega_csa_runtime)
        layer.enable_mega_hca.assert_called_once_with(
            transformer._mega_csa_runtime, model_weights.weights[0]
        )
        layer.enable_mega_csa.assert_called_once_with(
            transformer._mega_csa_runtime, model_weights.weights[0]
        )
        layer.enable_mega_front.assert_called_once_with(required=True)

    def test_decode_q_len_one_uses_complete_hca_sublayer(self) -> None:
        adapter = MagicMock()
        adapter.supports_decode_shape.return_value = True
        adapter.forward_attention_sublayer.side_effect = (
            lambda _block, value, *_a, **_k: value
        )
        block = _block_stub(adapter)
        hidden = torch.zeros(2, 1, 1, 4)
        metadata = SimpleNamespace(q_len_per_req=1)

        result = block.forward_decode(
            hidden, metadata, torch.zeros(2, 1, dtype=torch.long)
        )

        self.assertEqual(tuple(result.shape), tuple(hidden.shape))
        adapter.forward_attention_sublayer.assert_called_once()
        block.attn_hc.pre.assert_not_called()
        block.attn.forward_decode.assert_not_called()

    def test_target_verify_within_flat_token_limit_uses_mega(self) -> None:
        adapter = MagicMock()
        adapter.supports_decode_shape.side_effect = MegaHCAAdapter.supports_decode_shape
        adapter.forward_attention_sublayer.side_effect = (
            lambda _block, value, *_a, **_k: value
        )
        block = _block_stub(adapter)
        hidden = torch.zeros(2, 3, 1, 4)
        metadata = SimpleNamespace(batch_size=2, q_len_per_req=3)

        result = block.forward_decode(
            hidden, metadata, torch.zeros(2, 3, dtype=torch.long)
        )

        self.assertEqual(tuple(result.shape), tuple(hidden.shape))
        adapter.forward_attention_sublayer.assert_called_once()
        block.attn_hc.pre.assert_not_called()
        block.attn.forward_decode.assert_not_called()

    def test_flat_token_count_above_limit_keeps_existing_attention_path(self) -> None:
        adapter = MagicMock(wraps=MegaHCAAdapter.__new__(MegaHCAAdapter))
        block = _block_stub(adapter)
        hidden = torch.zeros(43, 3, 1, 4)
        metadata = SimpleNamespace(batch_size=43, q_len_per_req=3)

        block.forward_decode(hidden, metadata, torch.zeros(43, 3))

        adapter.forward_attention_sublayer.assert_not_called()
        block.attn.forward_decode.assert_called_once()

    def test_hca_failure_is_not_retried_on_existing_path(self) -> None:
        adapter = MagicMock()
        adapter.supports_decode_shape.return_value = True
        adapter.forward_attention_sublayer.side_effect = RuntimeError("kernel failed")
        block = _block_stub(adapter)

        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            block.forward_decode(
                torch.zeros(1, 1, 1, 4),
                SimpleNamespace(q_len_per_req=1),
                torch.zeros(1, 1, dtype=torch.long),
            )
        block.attn_hc.pre.assert_not_called()
        block.attn.forward_decode.assert_not_called()

    def test_adapter_is_attached_only_to_hca_layers(self) -> None:
        runtime = object()
        weights = {}
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter.MegaHCAAdapter"
        ) as adapter_cls:
            hca = Block.__new__(Block)
            torch.nn.Module.__init__(hca)
            hca.attn = SimpleNamespace(compress_ratio=128)
            hca._mega_hca_adapter = None
            hca.enable_mega_hca(runtime, weights)
            adapter_cls.assert_called_once_with(hca, weights, runtime)

            for ratio in (0, 4):
                other = Block.__new__(Block)
                torch.nn.Module.__init__(other)
                other.attn = SimpleNamespace(compress_ratio=ratio)
                other._mega_hca_adapter = None
                other.enable_mega_hca(runtime, weights)
                self.assertIsNone(other._mega_hca_adapter)
            adapter_cls.assert_called_once()


class MegaHCAWeightsTest(unittest.TestCase):
    @staticmethod
    def _meta(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="meta")

    def _layer_weights(self) -> dict[str, torch.Tensor]:
        meta = self._meta
        return {
            W.v4_attn_wq_a_w: meta((1536, DIM), torch.float8_e4m3fn),
            W.v4_attn_wq_a_s: meta((12, 56), torch.float8_e8m0fnu),
            W.v4_attn_wkv_w: meta((512, DIM), torch.float8_e4m3fn),
            W.v4_attn_wkv_s: meta((4, 56), torch.float8_e8m0fnu),
            W.v4_attn_wq_b_w: meta((65536, Q_LORA_RANK), torch.float8_e4m3fn),
            W.v4_attn_wq_b_s: meta((512, 12), torch.float8_e8m0fnu),
            W.v4_compressor_wkv: meta((HCA_STATE_WIDTH, DIM), torch.bfloat16),
            W.v4_compressor_wgate: meta((HCA_STATE_WIDTH, DIM), torch.bfloat16),
            W.v4_attn_q_norm: meta((Q_LORA_RANK,), torch.bfloat16),
            W.v4_attn_kv_norm: meta((HEAD_DIM,), torch.bfloat16),
            W.v4_compressor_norm: meta((HEAD_DIM,), torch.bfloat16),
            W.v4_compressor_ape: meta((HCA_APE_ROWS, HCA_STATE_WIDTH), torch.float32),
            W.v4_hc_attn_fn: meta((24, HC * DIM), torch.float32),
            W.v4_hc_attn_base: meta((24,), torch.float32),
            W.v4_hc_attn_scale: meta((3,), torch.float32),
            W.v4_attn_norm: meta((DIM,), torch.bfloat16),
        }

    def test_production_checkpoint_layout_is_accepted_without_repacking(self) -> None:
        packed = MegaHCAWeights.from_layer_weights(self._layer_weights())
        self.assertEqual(tuple(packed.front_fp8.shape), (2048, DIM))
        self.assertEqual(tuple(packed.front_sf.shape), (16, 56))
        self.assertEqual(tuple(packed.front_bf16.shape), (2 * HCA_STATE_WIDTH, DIM))
        self.assertEqual(tuple(packed.wq_b_fp8.shape), (65536, Q_LORA_RANK))
        self.assertEqual(tuple(packed.wq_b_sf.shape), (512, 12))
        self.assertEqual(
            tuple(packed.compressor_ape.shape), (HCA_APE_ROWS, HCA_STATE_WIDTH)
        )
        self.assertEqual(packed.front_sf.dtype, torch.float8_e8m0fnu)

    def test_csa_shaped_compressor_is_rejected(self) -> None:
        weights = self._layer_weights()
        weights[W.v4_compressor_wkv] = self._meta((1024, DIM), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "hca_compressor_wkv"):
            MegaHCAWeights.from_layer_weights(weights)

    def test_geometry_validation_rejects_non_tp1(self) -> None:
        attn = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            compress_ratio=HCA_COMPRESS_RATIO,
            dim=DIM,
            q_lora_rank=Q_LORA_RANK,
            n_heads=MAIN_HEADS,
            head_dim=HEAD_DIM,
            rope_head_dim=ROPE_DIM,
            n_groups=O_GROUPS,
            o_lora_rank=O_LORA_RANK,
            indexer=None,
        )
        with self.assertRaisesRegex(ValueError, "tp_size=2"):
            MegaHCAAdapter._validate_geometry(SimpleNamespace(attn=attn))

    def test_geometry_validation_rejects_an_indexer(self) -> None:
        attn = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            compress_ratio=HCA_COMPRESS_RATIO,
            dim=DIM,
            q_lora_rank=Q_LORA_RANK,
            n_heads=MAIN_HEADS,
            head_dim=HEAD_DIM,
            rope_head_dim=ROPE_DIM,
            n_groups=O_GROUPS,
            o_lora_rank=O_LORA_RANK,
            indexer=object(),
        )
        with self.assertRaisesRegex(ValueError, "must not have an indexer"):
            MegaHCAAdapter._validate_geometry(SimpleNamespace(attn=attn))

    def test_runtime_check_rejects_an_old_named_but_incompatible_abi(self) -> None:
        old_abi = SimpleNamespace(
            geometry_hca=lambda: {},
            hc_reduce_fuse_out=lambda: None,
            front_mixed_gemm_hca=lambda: None,
            wq_b_proj_gemm_merged_hca=lambda: None,
            q_rmsnorm_rope_cuda_=lambda: None,
            mla_o_inv_rope_quant=lambda: None,
        )
        adapter = MegaHCAAdapter.__new__(MegaHCAAdapter)
        adapter._runtime_checked = False
        with patch.object(rtp_kernel, "dsv4_mega", old_abi, create=True), patch(
            "torch.cuda.get_device_capability", return_value=(10, 3)
        ):
            with self.assertRaisesRegex(RuntimeError, "ABI is incompatible"):
                adapter._require_runtime(torch.device("cuda:0"))


class _FakeSource:
    is_cuda = True
    device = torch.device("cuda:0")
    dtype = torch.int64

    def __init__(self, pointer: int) -> None:
        self.pointer = pointer

    @staticmethod
    def numel() -> int:
        return 8

    @staticmethod
    def is_contiguous() -> bool:
        return True

    def data_ptr(self) -> int:
        return self.pointer

    def __getitem__(self, _key):
        return self


class MegaHCARuntimeTest(unittest.TestCase):
    @staticmethod
    def _metadata(pointer_base: int, is_cuda_graph: bool = False):
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, HCA_STATE, SWA_KV

        sources = [_FakeSource(pointer_base + index) for index in range(3)]
        return SimpleNamespace(
            is_cuda_graph=is_cuda_graph,
            compressor_state_slot_mappings={HCA_STATE: sources[0]},
            pool_write_slot_mappings={HCA_KV: sources[1], SWA_KV: sources[2]},
        )

    def test_slot_access_requires_model_step_boundary(self) -> None:
        runtime = MegaCSARuntime()
        metadata = self._metadata(100)
        with self.assertRaisesRegex(RuntimeError, "begin_decode"):
            runtime.hca_slot_mappings(metadata, 2)

    def test_slots_reuse_framework_tensors_without_allocation(self) -> None:
        runtime = MegaCSARuntime()
        metadata = self._metadata(100)
        with patch("torch.empty", side_effect=AssertionError("unexpected allocation")):
            runtime.begin_decode(metadata)
            slots = runtime.hca_slot_mappings(metadata, 2)

        self.assertIs(
            slots.state_rows,
            next(iter(metadata.compressor_state_slot_mappings.values())),
        )
        actual = (slots.compressed_destinations, slots.window_destinations)
        for source, result in zip(metadata.pool_write_slot_mappings.values(), actual):
            self.assertIs(source, result)

    def test_slots_reject_non_framework_dtype(self) -> None:
        runtime = MegaCSARuntime()
        metadata = self._metadata(100)
        metadata.pool_write_slot_mappings[
            next(iter(metadata.pool_write_slot_mappings))
        ].dtype = torch.int32
        runtime.begin_decode(metadata)
        with self.assertRaisesRegex(TypeError, "must be int64"):
            runtime.hca_slot_mappings(metadata, 2)

    def test_rope_tables_reuse_the_shared_runtime_cache(self) -> None:
        runtime = MegaCSARuntime()
        freqs_cis = torch.polar(
            torch.ones(4, 32), torch.arange(128, dtype=torch.float32).view(4, 32)
        )

        first_cos, first_sin = runtime.rope_tables(freqs_cis)
        second_cos, second_sin = runtime.rope_tables(freqs_cis)

        self.assertEqual(first_cos.data_ptr(), second_cos.data_ptr())
        self.assertEqual(first_sin.data_ptr(), second_sin.data_ptr())
        self.assertTrue(first_cos.is_contiguous())
        self.assertTrue(first_sin.is_contiguous())
        torch.testing.assert_close(first_cos, freqs_cis.real)
        torch.testing.assert_close(first_sin, freqs_cis.imag)


if __name__ == "__main__":
    unittest.main()
