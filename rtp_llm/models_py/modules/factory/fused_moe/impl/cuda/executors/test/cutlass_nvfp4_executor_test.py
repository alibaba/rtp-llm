import json
import os
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.device.device_impl import CudaImpl
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    quantize_expert_to_nvfp4,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
    CutlassNvfp4Executor,
)
from rtp_llm.utils.model_weight import W


class CutlassNvfp4ExecutorTest(unittest.TestCase):

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 12,
        "requires SM120 NVIDIA GPU",
    )
    def test_offline_checkpoint_loader_tp2_matches_independent_nvfp4_reference(self):
        from rtp_llm.model_loader.tensor_source import TensorSource

        class Source(TensorSource):
            def __init__(self, values):
                self.values = values

            def load_tensor(self, name, data_type=torch.bfloat16):
                return [self.values[name].to(data_type)]

            def has_tensor(self, name):
                return name in self.values

            def get_database(self):
                return None

        experts, hidden, inter = 2, 2560, 640
        values = {}
        # Author an offline checkpoint directly from E2M1 codes. No online
        # weight quantizer or loader output participates in the reference.
        for expert in range(experts):
            for projection in ("up", "gate", "down"):
                rows, columns = (
                    (hidden, inter) if projection == "down" else (inter, hidden)
                )
                codes = torch.zeros(rows, columns, dtype=torch.uint8)
                row = torch.arange(rows)
                if projection == "down":
                    # Each output uses both TP partitions, with distinct signs.
                    codes[row, row % (inter // 2)] = 2
                    codes[row, row % (inter // 2) + inter // 2] = 9 if expert else 4
                else:
                    column = (row % 32) * 2 + int(projection == "gate")
                    codes[row, column] = 4 if projection == "gate" else 2 + expert
                prefix = f"layers.0.experts.{expert}.{projection}_proj."
                values[prefix + "weight"] = codes[:, 0::2] | (codes[:, 1::2] << 4)
                values[prefix + "weight_scale"] = (
                    (row[:, None] % 4 + torch.arange(columns // 16)[None, :] % 2 + 1)
                    / 4
                ).to(torch.float8_e4m3fn)
                values[prefix + "weight_scale_2"] = torch.tensor(
                    (expert + 1) / (8 if projection == "down" else 4)
                ).reshape((1,) if projection == "gate" else ())
                values[prefix + "input_scale"] = torch.tensor(1 / 256).reshape(
                    (1, 1) if projection == "down" else (1,)
                )

        self._check_offline_loader_tp2(
            Source(values), "layers.{i}.experts.{expert_id}.", experts, hidden, inter
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 12,
        "requires SM120 NVIDIA GPU",
    )
    def test_real_offline_checkpoint_layer0_tp2(self):
        from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
        from rtp_llm.utils.database import CkptDatabase

        directory = os.getenv("QWEN4_NVFP4_CKPT_DIR")
        if not directory:
            self.skipTest(
                "set QWEN4_NVFP4_CKPT_DIR to a complete offline NVFP4 checkpoint"
            )
        with open(os.path.join(directory, "config.json")) as stream:
            config = json.load(stream)
        text = config["text_config"]
        self.assertEqual(text["num_experts"], 512)
        self.assertEqual(text["hidden_size"], 2560)
        self.assertEqual(text["moe_intermediate_size"], 640)
        # Load every expert in layer 0 through the real safetensors database;
        # independently decode only the two experts selected by the payload.
        self._check_offline_loader_tp2(
            DatabaseTensorSource(CkptDatabase(directory)),
            "model.language_model.layers.{i}.mlp.experts.{expert_id}.",
            text["num_experts"],
            text["hidden_size"],
            text["moe_intermediate_size"],
        )

    def _check_offline_loader_tp2(self, source, prefix, experts, hidden, inter):
        from unittest import mock

        from flashinfer.autotuner import AutoTuner

        from rtp_llm.config.quant_config import QWEN4_NVFP4_RECIPE, ModelOptFp4Config
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
        from rtp_llm.models.qwen4_exp.offline_nvfp4 import Qwen4Nvfp4MoeWeight
        from rtp_llm.ops import ParallelismConfig
        from rtp_llm.utils.model_weight import CkptWeightInfo, stack_, stack_moe_w1

        quant = ModelOptFp4Config(
            bits=4, group_size=16, is_quanted=True, model_recipe=QWEN4_NVFP4_RECIPE
        )
        descriptors = []
        for name, projections, process in (
            (W.moe_w1, ("up", "gate"), stack_moe_w1),
            (W.moe_w2, ("down",), stack_),
        ):
            descriptors.append(
                Qwen4Nvfp4MoeWeight(
                    MoeAtomicWeight(
                        name,
                        [
                            CkptWeightInfo(prefix + p + "_proj.weight")
                            for p in projections
                        ],
                        process_fun=process,
                        config=MoeConfig(expert_num=experts),
                    ),
                    quant,
                )
            )

        # Use the actual CUDA preparation methods without parsing process argv.
        device = CudaImpl.__new__(CudaImpl)
        device.py_env_configs = SimpleNamespace(
            moe_config=SimpleNamespace(fp4_moe_op="cutlass")
        )
        executors = []
        for rank in range(2):
            lc = SimpleNamespace(
                compute_dtype=torch.bfloat16,
                tp_size=2,
                tp_rank=rank,
                ffn_tp_size=2,
                ffn_tp_rank=rank,
                ep_size=1,
                ep_rank=0,
                dp_size=1,
                dp_rank=0,
                hidden_size=hidden,
                head_num=24,
                head_num_kv=2,
                size_per_head=256,
                moe_pure_tp_mode=True,
                moe_pure_tp_preshard=False,
                merge_lora=False,
                use_swizzleA=False,
                bit=4,
                exported_device=device,
                get_selected_experts=lambda layer, count: range(count),
            )
            loaded = {}
            for descriptor in descriptors:
                loaded.update(descriptor.load(source, 0, "cuda", lc))
            self.assertEqual(loaded[W.moe_w1].shape, (experts, inter, hidden // 2))
            self.assertEqual(loaded[W.moe_w2].shape, (experts, hidden, inter // 4))
            self.assertEqual(loaded[W.moe_w1_s2].shape, (experts,))
            self.assertEqual(loaded[W.moe_w2_i_s].shape, (experts,))
            pc = ParallelismConfig()
            pc.tp_size = 2
            pc.tp_rank = rank
            pc.ep_size = 1
            pc.ffn_sp_size = 1
            pc.world_size = 2
            pc.local_world_size = 2
            pc.world_rank = rank
            set_parallelism_config(pc)
            executors.append(
                CutlassNvfp4Executor(
                    SimpleNamespace(
                        ep_size=1,
                        ep_rank=0,
                        parallelism_config=pc,
                        model_config=SimpleNamespace(enable_qwen4_qsa=False),
                    ),
                    FusedMoEQuantConfig(quant_dtype=torch.uint8, block_shape=[16, 16]),
                    loaded,
                )
            )

        levels = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def checkpoint_tensor(projection, expert, suffix, dtype):
            name = prefix.format(i=0, expert_id=expert) + projection + "_proj." + suffix
            tensors = source.load_tensor(name, dtype)
            self.assertEqual(len(tensors), 1)
            return tensors[0].to("cuda")

        def decode(projection, expert):
            packed = checkpoint_tensor(projection, expert, "weight", torch.uint8)
            nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            block = checkpoint_tensor(
                projection, expert, "weight_scale", torch.float8_e4m3fn
            ).float()
            outer = checkpoint_tensor(
                projection, expert, "weight_scale_2", torch.float32
            ).item()
            return levels[nibbles.long()] * block.repeat_interleave(16, -1) * outer

        def quantize_activation(tensor, input_scale):
            # Independent nearest-value E2M1 reference. Prefer even ordinal on
            # exact ties, and apply both the FP8 block and FP32 global scales.
            block = tensor.float().reshape(*tensor.shape[:-1], -1, 16)
            sf = (block.abs().amax(-1) / (6 * input_scale)).clamp(max=448)
            sf = sf.to(torch.float8_e4m3fn).float() * input_scale
            divisor = torch.where(sf == 0, torch.ones_like(sf), sf)
            normalized = block / divisor.unsqueeze(-1)
            candidates = levels[torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device="cuda")]
            closest = (normalized.abs().unsqueeze(-1) - candidates).abs().argmin(-1)
            rounded = candidates[closest] * normalized.sign()
            return (rounded * sf.unsqueeze(-1)).reshape_as(tensor)

        selected = (0, experts - 1)
        weights = {
            (expert, p): decode(p, expert)
            for expert in selected
            for p in ("up", "gate", "down")
        }
        first_input_scale = checkpoint_tensor(
            "up", 0, "input_scale", torch.float32
        ).item()
        second_input_scale = checkpoint_tensor(
            "down", 0, "input_scale", torch.float32
        ).item()
        ids = torch.tensor([selected, selected[::-1]], dtype=torch.int32, device="cuda")
        routing = torch.tensor([[0.25, 0.75], [0.625, 0.375]], device="cuda")
        x = torch.zeros(2, hidden, device="cuda", dtype=torch.bfloat16)
        x[:, :64:2] = 0.5
        x[:, 1:64:2] = 1
        x[1] *= 0.5
        for amplitude in (0.5, 1.0, 2.0):
            current = x * amplitude
            payload = ExpertForwardPayload(
                expert_x=current,
                expert_x_origin_dtype=torch.bfloat16,
                expert_topk_ids=ids,
                expert_topk_weights=routing,
            )
            tuner = AutoTuner.get()
            choose_one = tuner.choose_one
            tactics = []

            def record_tactic(*args, **kwargs):
                result = choose_one(*args, **kwargs)
                tactics.append((args[0], result[1]))
                return result

            # Observe selection without replacing it: this reference models
            # the default, unfused-finalize CUTLASS tactic (sentinel -1).
            with mock.patch.object(tuner, "choose_one", side_effect=record_tactic):
                got_parts = [
                    executor.execute(
                        payload, "SiGLU", None, None, False, None
                    ).fused_expert_output.float()
                    for executor in executors
                ]
            self.assertEqual(
                tactics,
                [
                    ("trtllm::fused_moe::gemm1", -1),
                    ("trtllm::fused_moe::gemm2", -1),
                ]
                * 2,
            )
            quantized_input = quantize_activation(current, first_input_scale)
            expected_parts = []
            for rank, got in enumerate(got_parts):
                shard = slice(rank * (inter // 2), (rank + 1) * (inter // 2))
                expert_outputs = {}
                for expert in selected:
                    # CUTLASS stores FC1 in the BF16 backbone dtype. Its
                    # activation quantizer also rounds the SwiGLU product to
                    # BF16 before computing the NVFP4 block scale and codes.
                    up = (
                        (quantized_input @ weights[expert, "up"][shard].T)
                        .to(torch.bfloat16)
                        .float()
                    )
                    gate = (
                        (quantized_input @ weights[expert, "gate"][shard].T)
                        .to(torch.bfloat16)
                        .float()
                    )
                    product = (up * torch.nn.functional.silu(gate)).to(torch.bfloat16)
                    intermediate = quantize_activation(product, second_input_scale)
                    # The unfused finalize path reads BF16 FC2 expert outputs,
                    # accumulates routing weights in FP32, then writes BF16.
                    expert_outputs[expert] = (
                        (intermediate @ weights[expert, "down"][:, shard].T)
                        .to(torch.bfloat16)
                        .float()
                    )
                expected = torch.zeros_like(got)
                for slot in range(ids.shape[1]):
                    slot_output = torch.zeros_like(got)
                    for expert in selected:
                        slot_output += (
                            ids[:, slot : slot + 1] == expert
                        ) * expert_outputs[expert]
                    expected += slot_output * routing[:, slot : slot + 1]
                expected = expected.to(torch.bfloat16).float()
                expected_parts.append(expected)
                with self.subTest(experts=experts, amplitude=amplitude, rank=rank):
                    torch.testing.assert_close(got, expected, rtol=0.02, atol=1e-4)
            # Compare the FP32 sum of already-rounded rank outputs as well as
            # each rank, so cancellation cannot hide a shard-loading error.
            with self.subTest(experts=experts, amplitude=amplitude, rank="sum"):
                torch.testing.assert_close(
                    sum(got_parts), sum(expected_parts), rtol=0.02, atol=1e-4
                )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 12,
        "requires SM120 NVIDIA GPU",
    )
    def test_tensor_parallel_loader_split_and_dynamic_graph(self):
        from rtp_llm.config.quant_config import ModelOptFp4Config
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            OnlineModelOptFp4MoeWeight,
        )
        from rtp_llm.ops import ParallelismConfig
        from rtp_llm.utils.model_weight import CkptWeightInfo

        torch.manual_seed(85)
        experts, hidden, inter = 4, 2560, 640
        raw = {}
        descriptors = []
        for wn, sn, an, ins, shape in [
            (W.moe_w1, W.moe_s1, W.moe_w1_s2, W.moe_w1_i_s, (2 * inter, hidden)),
            (W.moe_w2, W.moe_s2, W.moe_w2_s2, W.moe_w2_i_s, (hidden, inter)),
        ]:
            quantized = [
                quantize_expert_to_nvfp4(
                    torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
                )
                for _ in range(experts)
            ]
            q, s, a = zip(*quantized)
            raw[wn], raw[sn], raw[an] = map(torch.stack, (q, s, a))
            raw[ins] = torch.full((experts,), 32 / 2688.0, device="cuda")
            descriptors.append(
                OnlineModelOptFp4MoeWeight(
                    MoeAtomicWeight(
                        wn,
                        [CkptWeightInfo("unused")],
                        config=MoeConfig(expert_num=experts),
                    ),
                    ModelOptFp4Config(bits=4, group_size=16, is_quanted=False),
                )
            )

        def prepared(values):
            return {
                name: (
                    CudaImpl.swizzle_blockscale(value)
                    if name in (W.moe_s1, W.moe_s2)
                    else value
                )
                for name, value in values.items()
            }

        qc = FusedMoEQuantConfig(quant_dtype=torch.uint8, block_shape=[16, 16])
        full = CutlassNvfp4Executor(
            SimpleNamespace(
                ep_size=1,
                ep_rank=0,
                model_config=SimpleNamespace(enable_qwen4_qsa=False),
            ),
            qc,
            prepared(raw),
        )
        parts = []
        for rank in range(2):
            lc = SimpleNamespace(
                tp_size=2,
                tp_rank=rank,
                ep_size=1,
                ep_rank=0,
                dp_size=1,
                dp_rank=0,
                ffn_tp_size=2,
                ffn_tp_rank=rank,
                hidden_size=hidden,
                head_num=24,
                head_num_kv=2,
                size_per_head=256,
                moe_pure_tp_mode=True,
                bit=4,
            )
            local = {}
            for descriptor in descriptors:
                local.update(descriptor._split(raw, lc))
            self.assertEqual(local[W.moe_w1].shape, (experts, inter, hidden // 2))
            self.assertEqual(local[W.moe_w2].shape, (experts, hidden, inter // 4))
            self.assertTrue(torch.equal(local[W.moe_w1_s2], raw[W.moe_w1_s2]))
            self.assertTrue(torch.equal(local[W.moe_w2_s2], raw[W.moe_w2_s2]))
            pc = ParallelismConfig()
            pc.tp_size = 2
            pc.tp_rank = rank
            pc.ep_size = 1
            pc.ffn_sp_size = 1
            pc.world_size = 2
            pc.local_world_size = 2
            pc.world_rank = rank
            set_parallelism_config(pc)
            parts.append(
                CutlassNvfp4Executor(
                    SimpleNamespace(
                        ep_size=1,
                        ep_rank=0,
                        parallelism_config=pc,
                        model_config=SimpleNamespace(enable_qwen4_qsa=False),
                    ),
                    qc,
                    prepared(local),
                )
            )
            self.assertEqual(parts[-1].tp_size, 2)
            self.assertEqual(parts[-1].tp_rank, rank)

        x = torch.randn(8, hidden, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = (
            torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
            .expand(8, -1)
            .contiguous()
        )
        weights = (
            torch.tensor([0.2, 0.3, 0.5], device="cuda").expand(8, -1).contiguous()
        )
        payload = ExpertForwardPayload(
            expert_x=x,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=ids,
            expert_topk_weights=weights,
        )

        def run(executor):
            return executor.execute(
                payload, "SiGLU", None, None, False, None
            ).fused_expert_output

        def combined():
            return run(parts[0]).float() + run(parts[1]).float()

        for _ in range(3):
            combined()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = combined()
        for _ in range(3):
            x.mul_(0.8)
            ids.add_(1).remainder_(experts)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(got, combined(), atol=0, rtol=0)
            torch.testing.assert_close(got, run(full).float(), atol=0.0005, rtol=0.02)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 12,
        "requires SM120 NVIDIA GPU",
    )
    def test_offline_recipe_is_selected_but_generic_prequantized_weights_are_not(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.config.quant_config import QWEN4_NVFP4_RECIPE, ModelOptFp4Config
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
            CudaNvfp4CutlassStrategy,
        )
        from rtp_llm.ops import MoeConfig, ParallelismConfig

        model = ModelConfig()
        model.data_type = "bf16"
        model.expert_num = 512
        model.moe_k = 10
        model.moe_inter_size = 640
        model.hidden_size = 2560
        model.n_shared_experts = 1
        model._qwen4_full_fp8_nvfp4 = False
        pc = ParallelismConfig()
        pc.tp_size = 2
        pc.ep_size = 1
        pc.ffn_sp_size = 1
        pc.world_size = 2
        pc.local_world_size = 2
        set_parallelism_config(pc)
        moe = MoeConfig()
        moe.moe_strategy = "nvfp4_cutlass"
        moe.fp4_moe_op = "cutlass"
        moe.use_all_gather = True
        moe.use_deepep_moe = False
        strategy = CudaNvfp4CutlassStrategy()
        for offline_source, recipe, expected in (
            (True, QWEN4_NVFP4_RECIPE, True),
            (True, None, False),
            (False, QWEN4_NVFP4_RECIPE, False),
            (False, None, False),
        ):
            with self.subTest(offline_source=offline_source, recipe=recipe):
                model._qwen4_nvfp4_source = offline_source
                model.quant_config = ModelOptFp4Config(
                    bits=4, group_size=16, is_quanted=True, model_recipe=recipe
                )
                # can_handle includes both the router and executor predicates.
                self.assertEqual(
                    strategy.can_handle(MoEConfigAdapter(model, pc, moe)), expected
                )

    def test_tensor_parallel_selection_is_gated_and_rejects_sp_and_world_mismatch(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.config.quant_config import ModelOptFp4Config
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
            CudaNvfp4CutlassStrategy,
        )
        from rtp_llm.ops import MoeConfig, ParallelismConfig

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("requires SM120 NVIDIA GPU")
        model = ModelConfig()
        model.data_type = "bf16"
        model.expert_num = 512
        model.moe_k = 10
        model.moe_inter_size = 640
        model.hidden_size = 2560
        model.n_shared_experts = 1
        model._qwen4_full_fp8_nvfp4 = True
        model.quant_config = ModelOptFp4Config(
            bits=4, group_size=16, is_quanted=False, activation_amax=32.0
        )
        pc = ParallelismConfig()
        pc.tp_size = 2
        pc.ep_size = 1
        pc.ffn_sp_size = 1
        pc.world_size = 2
        pc.local_world_size = 2
        set_parallelism_config(pc)
        moe = MoeConfig()
        moe.moe_strategy = "nvfp4_cutlass"
        moe.fp4_moe_op = "cutlass"
        moe.use_all_gather = True
        moe.use_deepep_moe = False
        strategy = CudaNvfp4CutlassStrategy()
        self.assertTrue(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))
        model._qwen4_full_fp8_nvfp4 = False
        self.assertFalse(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))
        model._qwen4_full_fp8_nvfp4 = True
        pc.world_size = 4
        self.assertFalse(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))
        pc.world_size = 2
        pc.ffn_sp_size = 2
        set_parallelism_config(pc)
        self.assertFalse(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))

    def test_selection_requires_matching_attention_tp_ep_and_online_weights(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.config.quant_config import ModelOptFp4Config
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
            CudaNvfp4CutlassStrategy,
        )
        from rtp_llm.ops import MoeConfig, ParallelismConfig

        if (
            not torch.cuda.is_available()
            or torch.version.hip is not None
            or torch.cuda.get_device_capability()[0] != 12
        ):
            self.skipTest("requires SM120 NVIDIA GPU")
        model = ModelConfig()
        model.data_type = "bf16"
        model.expert_num = 512
        model.moe_k = 10
        model.moe_inter_size = 640
        model.hidden_size = 2560
        model.n_shared_experts = 1
        model.quant_config = ModelOptFp4Config(
            bits=4, group_size=16, is_quanted=False, activation_amax=32.0
        )
        pc = ParallelismConfig()
        pc.tp_size = 2
        pc.ep_size = 2
        pc.ffn_sp_size = 2
        pc.world_size = 2
        pc.local_world_size = 2
        set_parallelism_config(pc)
        moe = MoeConfig()
        moe.moe_strategy = "nvfp4_cutlass"
        moe.fp4_moe_op = "cutlass"
        moe.use_all_gather = True
        moe.use_deepep_moe = False
        strategy = CudaNvfp4CutlassStrategy()
        self.assertTrue(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))
        pc.ep_size = 1
        set_parallelism_config(pc)
        self.assertFalse(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))
        pc.ep_size = 2
        set_parallelism_config(pc)
        model.quant_config = ModelOptFp4Config(bits=4, group_size=16, is_quanted=True)
        self.assertFalse(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 12,
        "requires SM120 NVIDIA GPU",
    )
    def test_graph_replay_updates_tokens_and_global_expert_routes(self):
        torch.manual_seed(83)
        E, K, N, M, T = 512, 2560, 640, 8, 10
        values = {}
        for wn, sn, an, ins, shape in [
            (W.moe_w1, W.moe_s1, W.moe_w1_s2, W.moe_w1_i_s, (2 * N, K)),
            (W.moe_w2, W.moe_s2, W.moe_w2_s2, W.moe_w2_i_s, (K, N)),
        ]:
            q = []
            sf = []
            a = []
            for _ in range(E):
                p, sc, alpha = quantize_expert_to_nvfp4(
                    torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
                )
                q.append(p)
                sf.append(sc)
                a.append(alpha)
            values[wn] = torch.stack(q)
            values[sn] = CudaImpl.swizzle_blockscale(torch.stack(sf))
            values[an] = torch.stack(a)
            values[ins] = torch.full((E,), 32 / 2688.0, device="cuda")
        ex = CutlassNvfp4Executor(
            SimpleNamespace(
                ep_size=1,
                ep_rank=0,
                model_config=SimpleNamespace(enable_qwen4_qsa=False),
            ),
            FusedMoEQuantConfig(quant_dtype=torch.uint8, block_shape=[16, 16]),
            values,
        )
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = torch.arange(M * T, device="cuda", dtype=torch.int32).reshape(M, T) % E
        wt = torch.full((M, T), 1 / T, device="cuda")
        payload = ExpertForwardPayload(
            expert_x=x,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=ids,
            expert_topk_weights=wt,
        )

        def run():
            return ex.execute(
                payload, "SiGLU", None, None, False, None
            ).fused_expert_output

        # EP partitions retain global IDs, including cross-partition routes.
        ids.copy_(
            torch.tensor(
                [0, 1, 255, 256, 257, 511, 31, 95, 320, 450],
                device="cuda",
                dtype=torch.int32,
            ).expand(M, -1)
        )
        wt.copy_(
            torch.arange(1, T + 1, device="cuda", dtype=torch.float32).expand(M, -1)
            / 55
        )
        expected = run()
        parts = []
        for rank in range(2):
            local = {
                name: value[rank * (E // 2) : (rank + 1) * (E // 2)]
                for name, value in values.items()
            }
            part = CutlassNvfp4Executor(
                SimpleNamespace(
                    ep_size=2,
                    ep_rank=rank,
                    model_config=SimpleNamespace(enable_qwen4_qsa=False),
                ),
                FusedMoEQuantConfig(quant_dtype=torch.uint8, block_shape=[16, 16]),
                local,
            )
            parts.append(
                part.execute(
                    payload, "SiGLU", None, None, False, None
                ).fused_expert_output
            )
        torch.testing.assert_close(
            parts[0].float() + parts[1].float(), expected.float(), rtol=0.02, atol=1e-4
        )
        for amplitude in (1.0, 4.0):
            x.normal_(std=amplitude)
            self.assertTrue(torch.isfinite(run()).all())
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = run()
        for _ in range(3):
            x.mul_(0.9)
            ids.add_(17).remainder_(E)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, run(), rtol=0, atol=0)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 12,
        "requires SM120 NVIDIA GPU",
    )
    def test_exact_quantized_swiglu_reference(self):
        E, K, N = 2, 256, 128
        w1 = torch.zeros(E, 2 * N, K, device="cuda", dtype=torch.bfloat16)
        w2 = torch.zeros(E, K, N, device="cuda", dtype=torch.bfloat16)
        # Explicit [up,gate] ordering, different branches and expert values.
        for e in range(E):
            w1[e, :N, 0] = float(e + 1)
            w1[e, N:, 1] = 0.5
            w2[e, :, 0] = 1.0
        values = {}
        for wn, sn, an, ins, w in (
            (W.moe_w1, W.moe_s1, W.moe_w1_s2, W.moe_w1_i_s, w1),
            (W.moe_w2, W.moe_s2, W.moe_w2_s2, W.moe_w2_i_s, w2),
        ):
            qs, ss, aa = zip(*(quantize_expert_to_nvfp4(e) for e in w))
            values[wn] = torch.stack(qs)
            values[sn] = CudaImpl.swizzle_blockscale(torch.stack(ss))
            values[an] = torch.stack(aa)
            values[ins] = torch.full((E,), 32 / 2688.0, device="cuda")
        ex = CutlassNvfp4Executor(
            SimpleNamespace(
                ep_size=1,
                ep_rank=0,
                model_config=SimpleNamespace(enable_qwen4_qsa=False),
            ),
            FusedMoEQuantConfig(quant_dtype=torch.uint8, block_shape=[16, 16]),
            values,
        )
        x = torch.zeros(1, K, device="cuda", dtype=torch.bfloat16)
        x[0, 0] = 0.5
        x[0, 1] = 1.0
        ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
        wt = torch.tensor([[0.25, 0.75]], device="cuda")
        payload = ExpertForwardPayload(
            expert_x=x,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=ids,
            expert_topk_weights=wt,
        )
        got = ex.execute(payload, "SiGLU", None, None, False, None).fused_expert_output
        # Quantized intermediate is independently checked against the exact
        # SwiGLU scalar; tolerates the explicit final NVFP4 activation rounding.
        expected = torch.nn.functional.silu(torch.tensor(0.5, device="cuda")) * (
            0.25 * 0.5 + 0.75 * 1.0
        )
        torch.testing.assert_close(
            got.float(), expected.expand_as(got).float(), rtol=0.12, atol=0.002
        )


if __name__ == "__main__":
    unittest.main()
