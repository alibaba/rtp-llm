import os
import unittest
from unittest.mock import Mock, patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.cpp.cache.test.libcache_config_creator_py_test import validate_basic_config
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next
from rtp_llm.ops import DataType, HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType
from rtp_llm.tools.convert import weights_convert
from rtp_llm.tools.convert.weights_convert import WeightConverter
from rtp_llm.tools.convert.weights_convert_utils import (
    ModelConfigPostBuilder,
    apply_layer_override_and_post_build,
)


class WeightsConvertParallelismTest(unittest.TestCase):
    def test_explicit_world_size_covers_all_conversion_ranks(self):
        converter = WeightConverter.__new__(WeightConverter)
        for rank in range(4):
            with self.subTest(rank=rank), patch.dict(
                os.environ,
                {
                    "TP_SIZE": "2",
                    "DP_SIZE": "2",
                    "WORLD_SIZE": "4",
                    "WORLD_RANK": str(rank),
                    "LOCAL_WORLD_SIZE": "4",
                },
                clear=True,
            ):
                config = converter._build_parallelism_config()
                self.assertEqual(config.world_size, 4)
                self.assertEqual(
                    (
                        config.tp_rank,
                        config.dp_rank,
                        config.ep_rank,
                        config.local_rank,
                    ),
                    (rank % 2, rank // 2, rank, rank),
                )

    def test_missing_world_size_defaults_to_tp_times_dp(self):
        converter = WeightConverter.__new__(WeightConverter)
        for tp_size, dp_size in ((1, 1), (4, 1), (2, 2)):
            world_size = tp_size * dp_size
            for rank in range(world_size):
                with self.subTest(tp=tp_size, dp=dp_size, rank=rank), patch.dict(
                    os.environ,
                    {
                        "TP_SIZE": str(tp_size),
                        "DP_SIZE": str(dp_size),
                        "WORLD_RANK": str(rank),
                        "LOCAL_WORLD_SIZE": str(world_size),
                    },
                    clear=True,
                ):
                    config = converter._build_parallelism_config()
                    self.assertEqual(config.world_size, world_size)
                    self.assertEqual(
                        (
                            config.tp_rank,
                            config.dp_rank,
                            config.ep_rank,
                            config.local_rank,
                        ),
                        (rank % tp_size, rank // tp_size, rank, rank),
                    )

    def test_explicit_world_size_rejects_out_of_range_rank(self):
        converter = WeightConverter.__new__(WeightConverter)
        with patch.dict(
            os.environ,
            {
                "TP_SIZE": "2",
                "DP_SIZE": "2",
                "WORLD_SIZE": "4",
                "WORLD_RANK": "4",
                "LOCAL_WORLD_SIZE": "4",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError, r"world_rank 4 out of range \[0, 4\)"
            ):
                converter._build_parallelism_config()

    def test_convert_without_world_size_covers_all_ranks(self):
        for tp_size, dp_size in ((4, 1), (2, 2)):
            converter = WeightConverter.__new__(WeightConverter)
            converter.env_params = {"TP_SIZE": str(tp_size), "DP_SIZE": str(dp_size)}
            converter.model_path = "/model"
            converter.model_type = "test_model"
            converter.model_cls = Mock()
            with (
                self.subTest(tp=tp_size, dp=dp_size),
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    converter, "_estimate_convert_parallel_num", return_value=1
                ),
                patch.object(converter, "_save_converted") as save_converted,
                patch.object(weights_convert.torch.cuda, "device_count", return_value=0),
                patch.object(
                    weights_convert.torch.cuda, "is_available", return_value=False
                ),
                patch.object(
                    weights_convert,
                    "fetch_remote_file_to_local",
                    side_effect=lambda path, *args: path,
                ),
                patch.object(weights_convert, "build_model_config"),
                patch.object(weights_convert, "apply_layer_override_and_post_build"),
            ):
                self.assertEqual(converter.convert("/converted"), 0)
                self.assertNotIn("WORLD_SIZE", os.environ)

                world_size = tp_size * dp_size
                configs = [
                    call.kwargs["parallelism_config"]
                    for call in converter.model_cls.from_config.call_args_list
                ]
                self.assertEqual(
                    [(c.world_size, c.tp_rank, c.dp_rank, c.ep_rank) for c in configs],
                    [
                        (world_size, rank % tp_size, rank // tp_size, rank)
                        for rank in range(world_size)
                    ],
                )
                model = converter.model_cls.from_config.return_value
                loader = model.create_model_loader.return_value
                self.assertEqual(loader.dump_weight_as_ft_style.call_count, world_size)
                save_converted.assert_called_once_with("/model", "/converted")


class WeightsConvertLayerOverrideTest(unittest.TestCase):
    def _model_config(self) -> ModelConfig:
        model_config = ModelConfig()
        model_config.num_layers = 4
        model_config.data_type = "fp16"
        model_config.attn_config.head_num = 2
        model_config.attn_config.kv_head_num = 2
        model_config.attn_config.size_per_head = 16
        model_config.attn_config.tokens_per_block = 4
        model_config.hybrid_attention_config.enable_hybrid_attention = True
        model_config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ]
        model_config.linear_attention_config.linear_conv_kernel_dim = 2
        model_config.linear_attention_config.linear_key_head_dim = 8
        model_config.linear_attention_config.linear_value_head_dim = 8
        model_config.linear_attention_config.linear_num_key_heads = 2
        model_config.linear_attention_config.linear_num_value_heads = 2
        model_config.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP16
        model_config.linear_attention_config.conv_state_dtype = DataType.TYPE_FP16
        model_config.moe_layer_index = [1, 3]
        stale_desc = KVCacheSpecDesc()
        stale_desc.tag = "stale"
        stale_desc.cache_type = KVCacheSpecType.MHA
        model_config.kv_cache_spec_descs = [[stale_desc] for _ in range(4)]
        return model_config

    def _assert_real_post_build(self, model_cls: type[ModelConfigPostBuilder]) -> None:
        model_config = self._model_config()

        result = apply_layer_override_and_post_build(
            model_config, model_cls, {"HACK_LAYER_NUM": "2"}
        )

        self.assertIs(result, model_config)
        self.assertEqual(result.num_layers, 2)
        self.assertEqual(result.moe_layer_index, [1])
        self.assertEqual(
            result.hybrid_attention_config.hybrid_attention_types,
            [HybridAttentionType.LINEAR, HybridAttentionType.NONE],
        )
        self.assertEqual(len(result.kv_cache_spec_descs), 2)
        self.assertEqual(
            [layer_descs[0].tag for layer_descs in result.kv_cache_spec_descs],
            ["linear", "full"],
        )
        self.assertEqual(
            [layer_descs[0].cache_type for layer_descs in result.kv_cache_spec_descs],
            [KVCacheSpecType.LINEAR, KVCacheSpecType.MHA],
        )
        validate_basic_config(result)

    def test_qwen3_next_real_post_build_uses_overridden_layer_metadata(self):
        self._assert_real_post_build(Qwen3Next)

    def test_kimi_real_post_build_replaces_stale_layer_metadata(self):
        self._assert_real_post_build(KimiLinear)

    def test_qwen3_next_linear_only_override_uses_one_independent_group(self):
        model_config = self._model_config()
        model_config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ]
        result = apply_layer_override_and_post_build(
            model_config, Qwen3Next, {"HACK_LAYER_NUM": "2"}
        )

        self.assertEqual(result.num_layers, 2)
        self.assertEqual(
            [descs[0].tag for descs in result.kv_cache_spec_descs],
            ["linear", "linear"],
        )
        validate_basic_config(result)

    def test_invalid_layer_override_preserves_metadata_and_can_retry(self):
        model_config = self._model_config()
        with self.assertRaisesRegex(ValueError, "layer override must be"):
            apply_layer_override_and_post_build(
                model_config, Qwen3Next, {"HACK_LAYER_NUM": "5"}
            )

        self.assertEqual(model_config.num_layers, 4)
        self.assertEqual(
            model_config.hybrid_attention_config.hybrid_attention_types,
            [
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
            ],
        )
        self.assertEqual(
            [layer_descs[0].tag for layer_descs in model_config.kv_cache_spec_descs],
            ["stale"] * 4,
        )

        apply_layer_override_and_post_build(
            model_config, Qwen3Next, {"HACK_LAYER_NUM": "2"}
        )
        self.assertEqual(model_config.num_layers, 2)
        self.assertEqual(len(model_config.kv_cache_spec_descs), 2)


if __name__ == "__main__":
    unittest.main()
