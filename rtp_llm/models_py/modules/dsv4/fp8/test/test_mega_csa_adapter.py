from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rtp_kernel
import torch

from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.decode.forward import forward_layers
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import MegaCSAAdapter
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import MegaCSARuntime
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    DIM,
    HC,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    MAIN_HEADS,
    MAX_BATCH,
    O_GROUPS,
    O_LORA_RANK,
    Q_LORA_RANK,
    ROPE_DIM,
    MegaCSAWeights,
    _cat_rows,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.utils.model_weight import W


class _IdentityNorm:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _block_stub(adapter: object | None) -> Block:
    block = Block.__new__(Block)
    torch.nn.Module.__init__(block)
    block.layer_id = 2
    block._mega_csa_adapter = adapter
    block._mega_hca_adapter = None
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


class MegaCSARoutingTest(unittest.TestCase):
    def test_pdfusion_role_can_attach_mega_adapter(self) -> None:
        class _Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enable_mega_csa = MagicMock()
                self.enable_mega_front = MagicMock()

        layer = _Layer()
        global_weights = MagicMock()
        global_weights.__getitem__.return_value = torch.ones(1, dtype=torch.bfloat16)
        model_weights = SimpleNamespace(global_weights=global_weights, weights=[{}])
        args = V4Args(
            n_layers=1,
            n_mtp_layers=0,
            compress_ratios=[4],
            is_decode_role=False,
            fp8_kv_cache=True,
            tp_size=1,
        )

        with patch.dict(os.environ, {"DSV4_MEGA_CSA": "1"}), patch(
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
        ):
            transformer = V4Transformer(args, model_weights)

        self.assertIsNotNone(transformer._mega_csa_runtime)
        layer.enable_mega_csa.assert_called_once_with(
            transformer._mega_csa_runtime, model_weights.weights[0]
        )
        layer.enable_mega_front.assert_called_once_with()

    def test_decode_q_len_one_uses_complete_mega_sublayer(self) -> None:
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
        adapter.supports_decode_shape.side_effect = MegaCSAAdapter.supports_decode_shape
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

    def test_dspark_attention_override_takes_priority_over_mega(self) -> None:
        adapter = MagicMock()
        adapter.supports_decode_shape.return_value = True
        block = _block_stub(adapter)
        attn_fn = MagicMock(side_effect=lambda value: value)

        block.forward_decode(
            torch.zeros(2, 1, 1, 4),
            SimpleNamespace(batch_size=2, q_len_per_req=1),
            torch.zeros(2, 1, dtype=torch.long),
            attn_fn=attn_fn,
        )

        attn_fn.assert_called_once()
        adapter.supports_decode_shape.assert_not_called()
        adapter.forward_attention_sublayer.assert_not_called()
        block.attn.forward_decode.assert_not_called()

    def test_batch_above_kernel_limit_keeps_existing_attention_path(self) -> None:
        adapter = MagicMock(wraps=MegaCSAAdapter.__new__(MegaCSAAdapter))
        block = _block_stub(adapter)
        hidden = torch.zeros(MAX_BATCH + 1, 1, 1, 4)
        metadata = SimpleNamespace(q_len_per_req=1)

        block.forward_decode(hidden, metadata, torch.zeros(MAX_BATCH + 1, 1))

        adapter.forward_attention_sublayer.assert_not_called()
        block.attn.forward_decode.assert_called_once()

    def test_flat_token_count_above_limit_keeps_existing_attention_path(self) -> None:
        adapter = MagicMock(wraps=MegaCSAAdapter.__new__(MegaCSAAdapter))
        block = _block_stub(adapter)
        hidden = torch.zeros(43, 3, 1, 4)
        metadata = SimpleNamespace(batch_size=43, q_len_per_req=3)

        block.forward_decode(hidden, metadata, torch.zeros(43, 3))

        adapter.forward_attention_sublayer.assert_not_called()
        block.attn.forward_decode.assert_called_once()

    def test_mega_failure_is_not_retried_on_existing_path(self) -> None:
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

    def test_adapter_is_attached_only_to_csa_layers(self) -> None:
        runtime = object()
        weights = {}
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter.MegaCSAAdapter"
        ) as adapter_cls:
            csa = Block.__new__(Block)
            torch.nn.Module.__init__(csa)
            csa.attn = SimpleNamespace(compress_ratio=4)
            csa._mega_csa_adapter = None
            csa.enable_mega_csa(runtime, weights)
            adapter_cls.assert_called_once_with(csa, weights, runtime)

            swa = Block.__new__(Block)
            torch.nn.Module.__init__(swa)
            swa.attn = SimpleNamespace(compress_ratio=0)
            swa._mega_csa_adapter = None
            swa.enable_mega_csa(runtime, weights)
            self.assertIsNone(swa._mega_csa_adapter)
            adapter_cls.assert_called_once()

    def test_production_layer_loop_advances_runtime_first(self) -> None:
        events: list[str] = []

        class _Layer:
            layer_id = 0

            def forward_decode(self, hidden, *_args, **_kwargs):
                events.append("layer")
                return hidden

        class _V4:
            hc_mult = 1
            layers = [_Layer()]
            norm = _IdentityNorm()
            _mtp_hidden_buffer = None
            capture_aux_hidden_layer_ids = ()

            def __init__(self) -> None:
                self.embed = torch.nn.Embedding(8, 4)

            def begin_decode(self, _metadata) -> None:
                events.append("begin")

            @staticmethod
            def _hc_head_reduce(hidden):
                return hidden.squeeze(2)

            @staticmethod
            def finish_aux_hidden_capture(_capture) -> None:
                return None

        metadata = SimpleNamespace(batch_size=2, q_len_per_req=1)
        result = forward_layers(_V4(), None, torch.tensor([1, 2]), metadata)

        self.assertEqual(events, ["begin", "layer"])
        self.assertEqual(tuple(result.shape), (2, 1, 4))


class MegaCSAWeightsTest(unittest.TestCase):
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
            W.v4_indexer_wq_b_w: meta((8192, Q_LORA_RANK), torch.float8_e4m3fn),
            W.v4_indexer_wq_b_s: meta((64, 12), torch.float8_e8m0fnu),
            W.v4_compressor_wkv: meta((1024, DIM), torch.bfloat16),
            W.v4_compressor_wgate: meta((1024, DIM), torch.bfloat16),
            W.v4_indexer_compressor_wkv: meta((256, DIM), torch.bfloat16),
            W.v4_indexer_compressor_wgate: meta((256, DIM), torch.bfloat16),
            W.v4_indexer_weights_proj_w: meta((INDEX_HEADS, DIM), torch.bfloat16),
            W.v4_attn_q_norm: meta((Q_LORA_RANK,), torch.bfloat16),
            W.v4_attn_kv_norm: meta((HEAD_DIM,), torch.bfloat16),
            W.v4_indexer_compressor_norm: meta((INDEX_HEAD_DIM,), torch.bfloat16),
            W.v4_compressor_norm: meta((HEAD_DIM,), torch.bfloat16),
            W.v4_compressor_ape: meta((4, 1024), torch.float32),
            W.v4_indexer_compressor_ape: meta((4, 256), torch.float32),
            W.v4_hc_attn_fn: meta((24, HC * DIM), torch.float32),
            W.v4_hc_attn_base: meta((24,), torch.float32),
            W.v4_hc_attn_scale: meta((3,), torch.float32),
            W.v4_attn_norm: meta((DIM,), torch.bfloat16),
        }

    def test_production_checkpoint_layout_is_accepted_without_repacking(self) -> None:
        packed = MegaCSAWeights.from_layer_weights(self._layer_weights())
        self.assertEqual(tuple(packed.front_fp8.shape), (2048, DIM))
        self.assertEqual(tuple(packed.front_sf.shape), (16, 56))
        self.assertEqual(tuple(packed.front_bf16.shape), (2624, DIM))
        self.assertEqual(tuple(packed.wq_b_fp8.shape), (73728, Q_LORA_RANK))
        self.assertEqual(tuple(packed.wq_b_sf.shape), (576, 12))
        self.assertEqual(packed.front_sf.dtype, torch.float8_e8m0fnu)

    def test_row_concatenation_preserves_declared_order(self) -> None:
        first = torch.tensor([[1.0], [2.0]])
        second = torch.tensor([[3.0]])
        result = _cat_rows("rows", (first, second), (3, 1))
        self.assertEqual(result.flatten().tolist(), [1.0, 2.0, 3.0])

    def test_geometry_validation_rejects_non_tp1(self) -> None:
        indexer = SimpleNamespace(n_heads=INDEX_HEADS, head_dim=INDEX_HEAD_DIM)
        attn = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            compress_ratio=4,
            dim=DIM,
            q_lora_rank=Q_LORA_RANK,
            n_heads=MAIN_HEADS,
            head_dim=HEAD_DIM,
            rope_head_dim=ROPE_DIM,
            n_groups=O_GROUPS,
            o_lora_rank=O_LORA_RANK,
            indexer=indexer,
        )
        with self.assertRaisesRegex(ValueError, "tp_size=2"):
            MegaCSAAdapter._validate_geometry(SimpleNamespace(attn=attn))

    def test_runtime_check_rejects_an_old_named_but_incompatible_abi(self) -> None:
        old_abi = SimpleNamespace(
            geometry_csa=lambda: {},
            hc_reduce_fuse_out=lambda: None,
            front_mixed_gemm_csa=lambda: None,
            wq_b_proj_gemm_merged_csa=lambda: None,
            mqa_logits_fp8_decode_out=lambda: None,
            mla_o_inv_rope_quant=lambda: None,
        )
        adapter = MegaCSAAdapter.__new__(MegaCSAAdapter)
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


class MegaCSARuntimeTest(unittest.TestCase):
    @staticmethod
    def _metadata(pointer_base: int, is_cuda_graph: bool = False):
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            CSA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        sources = [_FakeSource(pointer_base + index) for index in range(5)]
        return SimpleNamespace(
            is_cuda_graph=is_cuda_graph,
            compressor_state_slot_mappings={
                CSA_STATE: sources[0],
                INDEXER_STATE: sources[1],
            },
            pool_write_slot_mappings={
                CSA_KV: sources[2],
                INDEXER_KV: sources[3],
                SWA_KV: sources[4],
            },
        )

    def test_slot_access_requires_model_step_boundary(self) -> None:
        runtime = MegaCSARuntime()
        metadata = self._metadata(100)
        with self.assertRaisesRegex(RuntimeError, "begin_decode"):
            runtime.slot_mappings(metadata, 2)

    def test_slots_reuse_framework_tensors_without_allocation(self) -> None:
        runtime = MegaCSARuntime()
        metadata = self._metadata(100)
        with patch("torch.empty", side_effect=AssertionError("unexpected allocation")):
            runtime.begin_decode(metadata)
            slots = runtime.slot_mappings(metadata, 2)

        sources = (
            *metadata.compressor_state_slot_mappings.values(),
            *metadata.pool_write_slot_mappings.values(),
        )
        actual = (
            slots.main_state_rows,
            slots.indexer_state_rows,
            slots.main_destinations,
            slots.indexer_destinations,
            slots.swa_destinations,
        )
        for source, result in zip(sources, actual):
            self.assertIs(source, result)

    def test_slots_follow_active_metadata(self) -> None:
        runtime = MegaCSARuntime()
        first_meta = self._metadata(100, is_cuda_graph=True)
        second_meta = self._metadata(200, is_cuda_graph=True)
        runtime.begin_decode(first_meta)
        first = runtime.slot_mappings(first_meta, 2)
        runtime.begin_decode(second_meta)
        second = runtime.slot_mappings(second_meta, 2)

        self.assertEqual(first.main_state_rows.data_ptr(), 100)
        self.assertEqual(second.main_state_rows.data_ptr(), 200)

    def test_slots_reject_non_framework_dtype(self) -> None:
        runtime = MegaCSARuntime()
        metadata = self._metadata(100)
        metadata.pool_write_slot_mappings[
            next(iter(metadata.pool_write_slot_mappings))
        ].dtype = torch.int32
        runtime.begin_decode(metadata)
        with self.assertRaisesRegex(TypeError, "must be int64"):
            runtime.slot_mappings(metadata, 2)


if __name__ == "__main__":
    unittest.main()
