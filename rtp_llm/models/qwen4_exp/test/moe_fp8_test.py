import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.model_loader.ffn_weight import (
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models.qwen4_exp.moe_fp8 import (
    Qwen4MoeFp8WeightInfo,
    validate_moe_fp8_checkpoint,
    validate_moe_fp8_manifest,
)
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class Qwen4MoeFp8Test(unittest.TestCase):
    def setUp(self):
        self.config = {
            "text_config": {
                "num_hidden_layers": 2,
                "num_experts": 2,
                "hidden_size": 256,
                "moe_intermediate_size": 128,
            },
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head"],
            },
        }
        self.keys = {}
        for prefix, layers in (("model.language_model", 2), ("mtp", 1)):
            for layer in range(layers):
                for expert in range(2):
                    for projection in ("gate", "up", "down"):
                        name = (
                            f"{prefix}.layers.{layer}.mlp.experts.{expert}."
                            f"{projection}_proj.weight"
                        )
                        self.keys[name] = "model.safetensors"
                        self.keys[name + "_scale_inv"] = "model.safetensors"

    def test_accepts_complete_target_and_draft(self):
        validate_moe_fp8_manifest(self.config, self.keys)

    def test_rejects_missing_projection_and_scale(self):
        keys = dict(self.keys)
        name = next(iter(keys))
        del keys[name]
        del keys[name + "_scale_inv"]
        with self.assertRaisesRegex(ValueError, "missing target"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_non_expert_quantization(self):
        keys = dict(self.keys)
        keys["model.language_model.layers.0.self_attn.q_proj.weight_scale_inv"] = "x"
        with self.assertRaisesRegex(ValueError, "exactly the split MoE"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_scale_coverage_mismatch(self):
        keys = dict(self.keys)
        del keys[next(name for name in keys if name.endswith("_scale_inv"))]
        with self.assertRaisesRegex(ValueError, "exactly the split MoE"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_wrong_layer_with_same_key_count(self):
        keys = {
            name.replace("layers.1.", "layers.9."): file
            for name, file in self.keys.items()
        }
        with self.assertRaisesRegex(ValueError, "missing target"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_partial_draft(self):
        keys = dict(self.keys)
        name = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
        del keys[name]
        del keys[name + "_scale_inv"]
        with self.assertRaisesRegex(ValueError, "missing draft"):
            validate_moe_fp8_manifest(self.config, keys)

    def test_rejects_unsupported_quantization(self):
        for change in (
            {"activation_scheme": "static"},
            {"weight_block_size": [64, 128]},
            {"modules_to_not_convert": []},
        ):
            config = copy.deepcopy(self.config)
            config["quantization_config"].update(change)
            with self.assertRaises(ValueError):
                validate_moe_fp8_manifest(config, self.keys)

    def _checkpoint_tensors(self, scale_dtype=torch.bfloat16):
        tensors = {}
        for name in self.keys:
            shape = (256, 128) if "down_proj" in name else (128, 256)
            if name.endswith("_scale_inv"):
                shape = tuple(size // 128 for size in shape)
                tensors[name] = torch.ones(shape, dtype=scale_dtype)
            else:
                tensors[name] = torch.zeros(shape).to(torch.float8_e4m3fn)
        return tensors

    def _validate_checkpoint(self, tensors, keys=None):
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            validate_moe_fp8_checkpoint(
                self.config, self.keys if keys is None else keys, directory
            )

    def test_checkpoint_accepts_official_scale_dtypes(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(scale_dtype=dtype):
                self._validate_checkpoint(self._checkpoint_tensors(dtype))

    def test_checkpoint_rejects_expert_dtype_and_shape_mismatch(self):
        weight = next(name for name in self.keys if not name.endswith("_scale_inv"))
        for name, value in (
            (weight, torch.zeros(128, 256, dtype=torch.bfloat16)),
            (weight, torch.zeros(129, 256).to(torch.float8_e4m3fn)),
            (weight + "_scale_inv", torch.ones(1, 2, dtype=torch.int64)),
            (weight + "_scale_inv", torch.ones(2, 2, dtype=torch.bfloat16)),
        ):
            with self.subTest(tensor=name, dtype=value.dtype, shape=value.shape):
                tensors = self._checkpoint_tensors()
                tensors[name] = value
                with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
                    self._validate_checkpoint(tensors)

    def test_checkpoint_rejects_non_expert_fp8_and_unindexed_tensor(self):
        tensors = self._checkpoint_tensors()
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        tensors[name] = torch.zeros(256, 256).to(torch.float8_e4m3fn)
        keys = {**self.keys, name: "model.safetensors"}
        with self.assertRaisesRegex(ValueError, "non-expert tensor"):
            self._validate_checkpoint(tensors, keys)
        with self.assertRaisesRegex(ValueError, "shard/index keys disagree"):
            self._validate_checkpoint(tensors)

    def test_checkpoint_rejects_nonrelative_shard(self):
        for shard in ("/tmp/model.safetensors", "../model.safetensors"):
            with self.subTest(shard=shard):
                keys = {name: shard for name in self.keys}
                with self.assertRaisesRegex(ValueError, "relative safetensors"):
                    self._validate_checkpoint(self._checkpoint_tensors(), keys)

    def test_checkpoint_ple_requires_finite_global_scale(self):
        name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
        scale_name = name.rsplit(".shard_", 1)[0] + ".weight_scale"
        tensors = self._checkpoint_tensors()
        tensors[name] = torch.ones(2, 160).to(torch.float8_e4m3fn)
        keys = {**self.keys, name: "model.safetensors"}
        with self.assertRaisesRegex(ValueError, "missing its global scale"):
            self._validate_checkpoint(tensors, keys)
        keys[scale_name] = "model.safetensors"
        for value in (0.00019931793212890625, 0.0, -1.0, float("nan")):
            with self.subTest(value=value):
                tensors[scale_name] = torch.tensor([value], dtype=torch.bfloat16)
                if value > 0:
                    self._validate_checkpoint(tensors, keys)
                else:
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        self._validate_checkpoint(tensors, keys)

    def test_converts_only_expert_projections(self):
        config = MoeConfig(expert_num=2, align_size=128)
        gate = MoeAtomicWeight(
            W.moe_gate, [CkptWeightInfo("mlp.gate.weight")], config=config
        )
        experts = [
            MoeAtomicWeight(
                name,
                [
                    CkptWeightInfo("mlp.experts.{expert_id}." + projection + ".weight")
                    for projection in projections
                ],
                config=config,
            )
            for name, projections in (
                (W.moe_w1, ("up_proj", "gate_proj")),
                (W.moe_w2, ("down_proj",)),
            )
        ]
        plain = [
            AtomicWeight(name, [CkptWeightInfo(name + ".weight")])
            for name in (
                W.attn_qkv_w,
                W.linear_attn_qkvz_w,
                W.ffn_w1,
                W.qwen4_indexer_qk_proj_w,
            )
        ]
        head = AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight")])
        info = Qwen4MoeFp8WeightInfo(
            [head], [plain + [MoeWeight([gate] + experts, config)]]
        )
        result = info.to_quant_weight_info(Fp8BlockWiseQuantConfig(is_quanted=True))
        self.assertIs(result.weights[0], head)
        for got, original in zip(result.layer_weights[0][:-1], plain):
            self.assertIs(got, original)
        moe = result.layer_weights[0][-1]
        self.assertIs(moe.sub_weights[W.moe_gate], gate)
        self.assertIsInstance(moe.sub_weights[W.moe_w1], PerBlockFp8Weight)
        self.assertIsInstance(moe.sub_weights[W.moe_w2], PerBlockFp8Weight)

    def test_load_guard_uses_final_ffn_sp_topology(self):
        # EP8 alone leaves FFN TP8. Exercise the real setup used by serving;
        # a mocked get_ffn_tp_size() would hide this configuration boundary.
        for sp_size in (1, 8):
            with self.subTest(ffn_sp_size=sp_size):
                parallelism = ParallelismConfig()
                parallelism.world_size = parallelism.local_world_size = 8
                parallelism.tp_size = parallelism.ep_size = 8
                parallelism.dp_size = 1
                parallelism.ffn_sp_size = sp_size
                set_parallelism_config(parallelism, world_rank=0)
                self.assertEqual(parallelism.get_ffn_tp_size(), 8 // sp_size)
                model = Qwen4Exp.__new__(Qwen4Exp)
                model.model_config = SimpleNamespace(
                    _qwen4_moe_only_fp8=True,
                    quant_config=Fp8BlockWiseQuantConfig(is_quanted=True),
                    enable_qwen4_qsa=False,
                    enable_qwen4_ple=False,
                )
                model.parallelism_config = parallelism
                with mock.patch.dict(
                    os.environ, {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true"}
                ), mock.patch.object(
                    Qwen35Moe, "load", return_value="parent"
                ) as parent:
                    if sp_size == 1:
                        with self.assertRaisesRegex(RuntimeError, "FFN TP=1"):
                            model.load()
                        parent.assert_not_called()
                    else:
                        self.assertEqual(model.load(), "parent")
                        parent.assert_called_once_with(skip_python_model=False)

    def test_real_constructor_keeps_shared_and_expert_alignment_separate(self):
        config = ModelConfig()
        config._qwen4_moe_only_fp8 = True
        config.quant_config = Fp8BlockWiseQuantConfig(is_quanted=True)
        quant = config.quant_config
        config.quant_algo.setQuantAlgo(
            quant.get_algo().lower(), quant.bits, quant.group_size()
        )
        config.hidden_size = 256
        config.expert_num = 16
        config.num_layers = 1
        config.n_shared_experts = 1
        config.inter_size = config.moe_inter_size = 128
        for sp_size in (1, 8):
            parallel = ParallelismConfig()
            parallel.world_size = parallel.local_world_size = 8
            parallel.tp_size = parallel.ep_size = 8
            parallel.dp_size = 1
            parallel.ffn_sp_size = sp_size
            set_parallelism_config(parallel, world_rank=0)
            kwargs = dict(
                model_config=config,
                parallelism_config=parallel,
                hw_kernel_config=HWKernelConfig(),
                kv_cache_config=KVCacheConfig(),
            )
            if sp_size == 1:
                with self.assertRaisesRegex(ValueError, "FFN TP=1"):
                    Qwen4ExpWeight(**kwargs)
            else:
                weight = Qwen4ExpWeight(**kwargs)
                self.assertEqual(weight._align_size, 0)
                modules = weight._create_ffn_weight()
                shared = next(
                    module for module in modules if isinstance(module, FfnWeight)
                )
                moe = next(
                    module for module in modules if isinstance(module, MoeWeight)
                )
                self.assertEqual(shared.config.align_size, 0)
                self.assertEqual(moe.config.align_size, 128)


if __name__ == "__main__":
    unittest.main()
