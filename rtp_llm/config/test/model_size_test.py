import json
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.ops import ProfilingDebugLoggingConfig, VitSeparation


class ModelSizeTest(unittest.TestCase):
    def test_visual_estimation_matches_deployment(self):
        modes = (
            (None, True),
            (VitSeparation.VIT_SEPARATION_LOCAL, True),
            (VitSeparation.VIT_SEPARATION_ROLE, True),
            (VitSeparation.VIT_SEPARATION_REMOTE, False),
        )
        with TemporaryDirectory() as ckpt_path:
            args = ModelArgs()
            args.ckpt_path = args.tokenizer_path = ckpt_path
            args.model_type = "unused_visual_model"
            args.max_seq_len = 16
            for mode, owns_vit in modes:
                for multimodal, mtp in ((True, False), (True, True), (False, False)):
                    with self.subTest(mode=mode, multimodal=multimodal, mtp=mtp):
                        config = ModelConfig()
                        config.hidden_size = 8
                        config.vocab_size = 16
                        config.num_layers = 1
                        config.inter_size = 16
                        config.attn_config.head_num = 2
                        config.attn_config.kv_head_num = 1
                        config.attn_config.size_per_head = 4
                        vit = None
                        if mode is not None:
                            vit = VitConfig()
                            vit.vit_separation = mode
                        build_model_config(
                            config,
                            args,
                            KVCacheConfig(),
                            ProfilingDebugLoggingConfig(),
                            vit_config=vit,
                        )
                        baseline = (
                            config.eval_model_weight_size(),
                            config.eval_model_size(),
                            config.model_param_count(),
                        )
                        config.mm_model_config.is_multimodal = multimodal
                        config.mm_model_config.mm_sep_tokens = [[200025], [200026]]
                        config.is_mtp = mtp
                        include_vit = owns_vit and multimodal and not mtp
                        with patch(
                            "rtp_llm.config.model_config.get_multimodal_mixin_cls"
                        ) as factory:
                            factory.return_value.eval_mm_model_size.return_value = 1024
                            factory.return_value.eval_mm_model_param_count.return_value = (
                                512
                            )
                            if not include_vit:
                                factory.side_effect = AssertionError(
                                    "must not load ViT"
                                )
                            actual = (
                                config.eval_model_weight_size(),
                                config.eval_model_size(),
                                config.model_param_count(),
                            )
                            self.assertEqual(
                                actual,
                                tuple(
                                    base + (extra if include_vit else 0)
                                    for base, extra in zip(baseline, (1024, 1024, 512))
                                ),
                            )
                        self.assertEqual(config.is_multimodal(), multimodal)
                        self.assertEqual(
                            config.mm_model_config.mm_sep_tokens, [[200025], [200026]]
                        )


# --- Routed-expert storage-width pricing ------------------------------------
#
# DeepSeek-V4 checkpoints declare ``expert_dtype: fp4`` while their
# ``quantization_config`` describes only the dense FP8 block-wise scheme:
# routed experts occupy 0.5 byte/param (two FP4 values packed per I8 byte)
# plus a UE8M0 group-32 scale (1 byte per 32 values).  Pricing experts at
# the model-wide 8-bit width made the loader see V4-Pro as 1562GB against
# 893GB on disk and pushed it off the fastsafetensors path.  The tests
# below lock the corrected expert pricing and the byte-identical legacy
# pricing for models without narrow experts.

# DeepSeek-V4-Pro-DSpark geometry (config.json): 61 layers x 384 experts,
# 3072 moe_inter, 7168 hidden, MQA head_dim 512, one shared expert.
V4_PRO_SHAPE = dict(
    layers=61,
    hidden=7168,
    vocab=129280,
    head_num=128,
    kv_head_num=1,
    size_per_head=512,
    expert_num=384,
    moe_inter=3072,
    inter=3072,
)
# DeepSeek-V4-Flash geometry (config.json).
V4_FLASH_SHAPE = dict(
    layers=43,
    hidden=4096,
    vocab=129280,
    head_num=64,
    kv_head_num=1,
    size_per_head=512,
    expert_num=256,
    moe_inter=2048,
    inter=2048,
)

FP8_BLOCKWISE_QUANT_CONFIG = {
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
}


def _write_quant_config(ckpt_path: str) -> None:
    with open(f"{ckpt_path}/config.json", "w") as writer:
        json.dump(FP8_BLOCKWISE_QUANT_CONFIG, writer)


def _moe_model_config(ckpt_path: str, shape: dict, expert_weight_dtype=None):
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.hidden_size = shape["hidden"]
    config.vocab_size = shape["vocab"]
    config.num_layers = shape["layers"]
    config.inter_size = shape["inter"]
    config.moe_inter_size = shape["moe_inter"]
    config.expert_num = shape["expert_num"]
    config.moe_style = 2
    config.moe_layer_index = list(range(shape["layers"]))
    config.attn_config.head_num = shape["head_num"]
    config.attn_config.kv_head_num = shape["kv_head_num"]
    config.attn_config.size_per_head = shape["size_per_head"]
    if expert_weight_dtype is not None:
        config.expert_weight_dtype = expert_weight_dtype
    # Loads the FP8 block-wise quantization_config from ckpt_path and sets
    # quant_algo (weight bits 8).
    config.init_precision_config(kv_cache_config=None, act_type="BF16")
    return config


def _legacy_layer_param_count(shape: dict) -> int:
    """Independent mirror of layer_weight_param_count for moe_style == 2.

    Uses the default gated activation (3 FFN matrices) like the V4 configs.
    """
    layers, hidden = shape["layers"], shape["hidden"]
    kv_width = shape["kv_head_num"] * shape["size_per_head"]
    return (
        layers * hidden * hidden  # qkv projection
        + layers * hidden * kv_width * 2  # kv projection
        + layers * hidden * hidden  # attention output projection
        + layers * shape["inter"] * hidden * 3  # shared expert FFN
        + layers * shape["moe_inter"] * hidden * 3 * shape["expert_num"]  # routed
        + layers * hidden * 11  # norms and other small tensors
    )


def _legacy_weight_size(shape: dict) -> int:
    # word embedding + lm_head, each bf16 (2 bytes/param).
    embedding = 2 * (shape["vocab"] * shape["hidden"])
    return embedding + _legacy_layer_param_count(shape) + embedding


class ExpertWeightDtypeSizeTest(unittest.TestCase):
    def test_v4_pro_fp4_experts_priced_at_checkpoint_width(self):
        with TemporaryDirectory() as ckpt_path:
            _write_quant_config(ckpt_path)
            config = _moe_model_config(ckpt_path, V4_PRO_SHAPE, "fp4")
            self.assertEqual(config.quant_algo.getWeightBits(), 8)

            moe_params = config.moe_weight_param_count()
            size = config.eval_model_weight_size()
            # Only the routed-expert slice is repriced (1.0 -> 0.54 B/param);
            # the dense slice and the embeddings keep the legacy pricing.
            embedding = 2 * (V4_PRO_SHAPE["vocab"] * V4_PRO_SHAPE["hidden"])
            dense_params = _legacy_layer_param_count(V4_PRO_SHAPE) - moe_params
            self.assertAlmostEqual(
                size,
                embedding + dense_params + moe_params * 0.54 + embedding,
                delta=2.0,
            )
            # DeepSeek-V4-Pro-DSpark occupies 892.7GB on disk (66 safetensors
            # files); the corrected estimate must land within 5% of that.
            self.assertLessEqual(size, 893e9 * 1.05)
            self.assertGreaterEqual(size, 893e9 * 0.95)

            # Pre-fix behavior for the record: without the expert-width
            # declaration the same shape prices experts at the model-wide
            # 8-bit width and lands at 1561.85GB -- 1.81x the checkpoint
            # (the expert slice alone accounts for the 2x on 99.3% of the
            # parameters).  Reverting the fix flips this back.
            config.expert_weight_dtype = None
            legacy_size = config.eval_model_weight_size()
            self.assertEqual(legacy_size, _legacy_weight_size(V4_PRO_SHAPE))
            self.assertAlmostEqual(legacy_size / 1e9, 1561.85, delta=0.5)

    def test_flash_fp8_experts_estimate_unchanged(self):
        # A Flash-shaped model whose routed experts share the model-wide
        # FP8 width (no narrow-expert declaration) must keep the exact
        # legacy estimate -- the fix only reprices 4-bit experts.
        with TemporaryDirectory() as ckpt_path:
            _write_quant_config(ckpt_path)
            config = _moe_model_config(ckpt_path, V4_FLASH_SHAPE)
            self.assertEqual(
                config.eval_model_weight_size(), _legacy_weight_size(V4_FLASH_SHAPE)
            )

            # The shipped V4-Flash checkpoints also declare
            # ``expert_dtype: fp4`` (their experts are FP4-packed too), so
            # their estimate drops the same way; only the routed-expert
            # slice changes and every decision margin only widens.
            config.expert_weight_dtype = "fp4"
            moe_params = config.moe_weight_param_count()
            embedding = 2 * (V4_FLASH_SHAPE["vocab"] * V4_FLASH_SHAPE["hidden"])
            dense_params = _legacy_layer_param_count(V4_FLASH_SHAPE) - moe_params
            self.assertAlmostEqual(
                config.eval_model_weight_size(),
                embedding + dense_params + moe_params * 0.54 + embedding,
                delta=2.0,
            )

    def test_whole_model_4bit_experts_stay_at_model_width(self):
        # A whole-model 4-bit config already prices everything at 0.54
        # B/param; declaring 4-bit experts must not change the estimate.
        with TemporaryDirectory() as ckpt_path:
            _write_quant_config(ckpt_path)
            config = _moe_model_config(ckpt_path, V4_FLASH_SHAPE)
            config.quant_algo.setQuantAlgo("gptq", 4, 128)
            self.assertEqual(config.quant_algo.getWeightBits(), 4)
            legacy_size = config.eval_model_weight_size()
            config.expert_weight_dtype = "fp4"
            self.assertAlmostEqual(
                config.eval_model_weight_size(), legacy_size, delta=2.0
            )

    def test_unknown_expert_dtype_fails_fast(self):
        with TemporaryDirectory() as ckpt_path:
            _write_quant_config(ckpt_path)
            config = _moe_model_config(ckpt_path, V4_PRO_SHAPE, "e5m2")
            with self.assertRaises(ValueError):
                config.eval_model_weight_size()


class WeightConvertDeviceDecisionReplayTest(unittest.TestCase):
    """Replay the v4_pro smoke failure with the measured CI numbers.

    The v4_pro_cp4_ep4_basic_sm100 case runs cp4/ep4 on SM103 (~288GB
    cards).  Measured free device memory at load time: 265.97GiB; largest
    of the 66 checkpoint files: 14118826460 bytes (~13.15GiB).  Before the
    expert-width fix the estimate was 1561.85GB, i.e. 363.65GiB per rank:
    ``_choose_weight_convert_device`` returned cpu, which short-circuited
    the AUTO chain's fastsafetensors condition (loader.py
    ``_load_weight``) and left the case on the scratch path (1370s loads
    vs the fastsafetensors ~30s scale).  With experts priced at their real
    FP4 width the estimate drops to ~850GB (197.9GiB/rank) and both gates
    pass.
    """

    FREE_MEM_GIB = 265.97
    MAX_FILE_BYTES = 14118826460

    @staticmethod
    def _loader(config, free_gib, max_file_bytes):
        # Bypass ModelLoader.__init__: only the fields touched by the two
        # decision helpers are populated, no GPU or database needed.
        loader = object.__new__(ModelLoader)
        loader._load_config = SimpleNamespace(
            force_cpu_load_weights=False,
            ep_size=4,
            tp_size=4,
            exported_device=SimpleNamespace(
                get_mem_info=lambda: SimpleNamespace(free=free_gib * 1024**3)
            ),
            database=SimpleNamespace(get_max_file_size=lambda: max_file_bytes),
        )
        loader._weights_info = SimpleNamespace(model_config=config)
        return loader

    def test_v4_pro_smoke_decision_replay(self):
        with TemporaryDirectory() as ckpt_path:
            _write_quant_config(ckpt_path)
            config = _moe_model_config(ckpt_path, V4_PRO_SHAPE, "fp4")
            loader = self._loader(config, self.FREE_MEM_GIB, self.MAX_FILE_BYTES)

            # Corrected estimate: ~850.05GB -> 197.9GiB/rank < 0.9 * 265.97.
            self.assertEqual(loader._choose_weight_convert_device("cuda:0"), "cuda:0")
            # fastsafetensors margin: 265.97GiB - 197.9GiB = 68.1GiB free
            # after weights vs 3 * 13.15GiB for the largest file.
            self.assertTrue(loader._is_memory_enough_for_fastsafetensor())

            # Pre-fix estimate (experts at the model-wide 8-bit width):
            # 363.65GiB/rank exceeds 0.9 * 265.97GiB and forces the cpu
            # convert device -- the exact fault this fix removes.
            config.expert_weight_dtype = None
            self.assertEqual(loader._choose_weight_convert_device("cuda:0"), "cpu")


if __name__ == "__main__":
    unittest.main()
