import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.linear_attn_weight import LinearAttnAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.load_quant_w8a8_int8_weight import (
    LoadQuantW8A8Int8PerChannelWeight,
    W8A8QuantizingDatabase,
    validate_w8a8_source,
)
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.database import BaseDatabase, CkptDatabase
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    transpose_stack_moe_w1,
)
from rtp_llm.utils.w8a8_int8_quant import (
    quantize_weight_per_output_channel,
    scale_name_for,
)
from safetensors.torch import save_file

_STACKED_GATE_UP = "model.language_model.layers.{i}.mlp.experts.gate_up_proj"
_DENSE_Q = "model.layers.0.self_attn.q_proj.weight"


@contextmanager
def _database(tensors):
    with tempfile.TemporaryDirectory() as checkpoint_path:
        save_file(tensors, os.path.join(checkpoint_path, "model.safetensors"))
        database = CkptDatabase(checkpoint_path)
        try:
            yield database
        finally:
            for checkpoint in database.pretrain_file_list:
                checkpoint.close_safetensor_handle()


class _CpuDevice:
    def maybe_rewrite_weight_by_key(self, _key, tensor, **_kwargs):
        return tensor

    def shuffle_moe_weight(self, tensor, *_args):
        return tensor


def _load_config():
    return LoadConfig.model_construct(
        database=BaseDatabase(),
        num_layers=1,
        hidden_size=4,
        head_num=1,
        head_num_kv=1,
        size_per_head=4,
        moe_pure_tp_mode=False,
        align_size=1,
        moe_align_size=1,
        moe_layer_index=[],
        moe_n_group=1,
        expert_num=2,
        enable_eplb=False,
        phy_exp_num=0,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        lm_head_tp_size=1,
        lm_head_tp_rank=0,
        ffn_tp_size=1,
        ffn_tp_rank=0,
        num_nodes=1,
        compute_dtype=torch.bfloat16,
        exported_device=_CpuDevice(),
    )


class W8A8QuantizingDatabaseTest(unittest.TestCase):
    def _tensors(self):
        return {
            _DENSE_Q: torch.tensor(
                [[-2.0, 0.0, 1.0, 4.0], [3.0, -6.0, 0.5, 0.0]],
                dtype=torch.bfloat16,
            ),
            _STACKED_GATE_UP.format(i="0"): torch.arange(2 * 4 * 5, dtype=torch.float32)
            .reshape(2, 4, 5)
            .sub_(17)
            .to(torch.bfloat16),
        }

    def test_proxy_matches_offline_recipe_for_2d_and_3d_and_scale_order(self):
        tensors = self._tensors()
        with _database(tensors) as database:
            for weight_name, source in tensors.items():
                expected_weight, expected_scale = quantize_weight_per_output_channel(
                    source
                )
                scale_name = scale_name_for(weight_name)

                scale_first = W8A8QuantizingDatabase(database, chunk_rows=2)
                actual_scale = scale_first.load_tensor_slice(
                    scale_name, (), torch.float32
                )
                actual_weight = scale_first.load_tensor_slice(
                    weight_name, (), torch.int8
                )

                weight_first = W8A8QuantizingDatabase(database, chunk_rows=3)
                later_weight = weight_first.load_tensor_slice(
                    weight_name, (), torch.int8
                )
                later_scale = weight_first.load_tensor_slice(
                    scale_name, (), torch.float32
                )

                with self.subTest(weight_name=weight_name):
                    torch.testing.assert_close(
                        actual_weight, expected_weight, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual_scale, expected_scale, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        later_weight, expected_weight, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        later_scale, expected_scale, rtol=0, atol=0
                    )

    def test_k_slice_keeps_scales_from_complete_k_dimension(self):
        tensors = self._tensors()
        with _database(tensors) as database:
            for weight_name, source in tensors.items():
                expected_weight, expected_scale = quantize_weight_per_output_channel(
                    source
                )
                view = W8A8QuantizingDatabase(database, chunk_rows=1)
                prefix = (slice(0, 1),) if source.dim() == 3 else ()
                requested_weight = view.load_tensor_slice(
                    weight_name, (*prefix, slice(1, 3), slice(1, 4)), torch.int8
                )
                requested_scale = view.load_tensor_slice(
                    scale_name_for(weight_name),
                    (*prefix, slice(1, 3), slice(None)),
                    torch.float32,
                )
                expected_weight_slice = expected_weight[
                    (*prefix, slice(1, 3), slice(1, 4))
                ]
                expected_scale_slice = expected_scale[
                    (*prefix, slice(1, 3), slice(None))
                ]

                with self.subTest(weight_name=weight_name):
                    torch.testing.assert_close(
                        requested_weight, expected_weight_slice, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        requested_scale, expected_scale_slice, rtol=0, atol=0
                    )

    def test_quantized_reads_never_use_full_bf16_loader(self):
        with _database(self._tensors()) as database, patch.object(
            database, "load_tensor", wraps=database.load_tensor
        ) as full_read, patch.object(
            database, "load_tensor_slice", wraps=database.load_tensor_slice
        ) as slice_read:
            view = W8A8QuantizingDatabase(database, chunk_rows=2)
            view.load_tensor_slice(_DENSE_Q, (), torch.int8)

            self.assertFalse(full_read.called)
            self.assertTrue(slice_read.called)
            self.assertTrue(
                all(
                    call.args[1][-1] == slice(None)
                    for call in slice_read.call_args_list
                )
            )

    def test_rejects_nonfinite_sources_and_source_quantization_conflicts(self):
        with _database({_DENSE_Q: torch.tensor([[float("nan"), 1.0]])}) as database:
            view = W8A8QuantizingDatabase(database, chunk_rows=1)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                view.load_tensor_slice(_DENSE_Q, (), torch.int8)
            with self.assertRaisesRegex(ValueError, "requires torch.int8"):
                view.load_tensor_slice(_DENSE_Q, (), torch.bfloat16)

        with _database({_DENSE_Q: torch.ones(2, 2, dtype=torch.int8)}) as database:
            with self.assertRaisesRegex(ValueError, "must be floating point"):
                validate_w8a8_source(database)

        with _database(
            {
                _DENSE_Q: torch.ones(2, 2),
                scale_name_for(_DENSE_Q): torch.ones(2, 1),
            }
        ) as database:
            with self.assertRaisesRegex(ValueError, "already contains"):
                validate_w8a8_source(database)


class LoadQuantW8A8WeightParityTest(unittest.TestCase):
    @staticmethod
    def _offline_tensors(float_tensors):
        tensors = {}
        for name, weight in float_tensors.items():
            quantized, scale = quantize_weight_per_output_channel(weight)
            tensors[name] = quantized
            tensors[scale_name_for(name)] = scale
        return tensors

    def _assert_online_matches_prequantized(self, source, float_tensors):
        online = WeightModule.create(
            source,
            CompressedW8A8Int8PerChannelQuantConfig(
                is_quanted=False, load_chunk_rows=1
            ),
        )
        offline = WeightModule.create(
            source, CompressedW8A8Int8PerChannelQuantConfig(is_quanted=True)
        )
        self.assertIsInstance(online, LoadQuantW8A8Int8PerChannelWeight)

        with _database(float_tensors) as float_database, _database(
            self._offline_tensors(float_tensors)
        ) as quantized_database:
            for preshard in (False, True):
                config = _load_config()
                config.moe_pure_tp_mode = preshard
                config.moe_pure_tp_preshard = preshard
                actual = online.load(
                    DatabaseTensorSource(float_database), 0, "cpu", config
                )
                expected = offline.load(
                    DatabaseTensorSource(quantized_database), 0, "cpu", config
                )
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=0, atol=0
                    )

    def test_split_expert_layout_fails_before_loading_weights(self):
        source = MoeAtomicWeight(
            W.moe_w2,
            [
                CkptWeightInfo(
                    "model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight"
                )
            ],
            process_fun=stack_,
            config=MoeConfig(expert_num=2),
        )
        with self.assertRaisesRegex(ValueError, "split-expert checkpoint layouts"):
            WeightModule.create(
                source, CompressedW8A8Int8PerChannelQuantConfig(is_quanted=False)
            )

    def _source(self):
        return MoeAtomicWeight(
            W.moe_w1,
            [CkptWeightInfo(_STACKED_GATE_UP, identity)],
            process_fun=transpose_stack_moe_w1,
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
            enable_pure_tp_preshard=True,
        )

    def test_online_weight_load_matches_prequantized_stacked_moe_weight(self):
        source = self._source()
        source_name = _STACKED_GATE_UP.format(i="0")
        float_weight = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
        float_weight = float_weight.sub_(13).to(torch.bfloat16)
        self._assert_online_matches_prequantized(source, {source_name: float_weight})

    def test_online_weight_load_matches_prequantized_stacked_moe_down_proj(self):
        source_name = "model.language_model.layers.{i}.mlp.experts.down_proj"
        source = MoeAtomicWeight(
            W.moe_w2,
            [CkptWeightInfo(source_name, identity)],
            process_fun=stack_,
            config=MoeConfig(expert_num=2),
            stacked_ckpt_keys=True,
            enable_pure_tp_preshard=True,
        )
        float_weight = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
        self._assert_online_matches_prequantized(
            source, {source_name.format(i="0"): float_weight.to(torch.bfloat16)}
        )

    def test_online_weight_load_matches_prequantized_gdn_qkv_plus_z(self):
        qkv_name = "model.layers.{i}.linear_attn.in_proj_qkv.weight"
        z_name = "model.layers.{i}.linear_attn.in_proj_z.weight"
        source = LinearAttnAtomicWeight(
            W.linear_attn_qkvz_w,
            [CkptWeightInfo(qkv_name, identity), CkptWeightInfo(z_name, identity)],
            identity,
            config=object(),
        )
        self._assert_online_matches_prequantized(
            source,
            {
                qkv_name.format(i="0"): torch.arange(6 * 4, dtype=torch.float32)
                .reshape(6, 4)
                .to(torch.bfloat16),
                z_name.format(i="0"): torch.arange(2 * 4, dtype=torch.float32)
                .reshape(2, 4)
                .sub_(4)
                .to(torch.bfloat16),
            },
        )

    def test_online_weight_load_matches_prequantized_attention_qkv(self):
        q_name = "model.layers.{i}.self_attn.q_proj.weight"
        k_name = "model.layers.{i}.self_attn.k_proj.weight"
        v_name = "model.layers.{i}.self_attn.v_proj.weight"
        source = AttnAtomicWeight(
            W.attn_qkv_w,
            [
                CkptWeightInfo(q_name, identity),
                CkptWeightInfo(k_name, identity),
                CkptWeightInfo(v_name, identity),
            ],
            config=AttnConfig(),
        )
        self._assert_online_matches_prequantized(
            source,
            {
                q_name.format(i="0"): torch.arange(2 * 4, dtype=torch.float32)
                .reshape(2, 4)
                .to(torch.bfloat16),
                k_name.format(i="0"): torch.arange(2 * 4, dtype=torch.float32)
                .reshape(2, 4)
                .sub_(3)
                .to(torch.bfloat16),
                v_name.format(i="0"): torch.arange(2 * 4, dtype=torch.float32)
                .reshape(2, 4)
                .sub_(6)
                .to(torch.bfloat16),
            },
        )


if __name__ == "__main__":
    unittest.main()
