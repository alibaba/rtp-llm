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
    def test_selection_rejects_unquantized_bf16_models(self):
        from rtp_llm.config.model_config import ModelConfig
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
        model.quant_config = None
        moe = MoeConfig()
        moe.fp4_moe_op = "cutlass"
        moe.use_all_gather = True
        moe.use_deepep_moe = False
        for tp_size in (1, 2):
            pc = ParallelismConfig()
            pc.tp_size = tp_size
            pc.ep_size = tp_size
            pc.ffn_sp_size = tp_size
            pc.world_size = tp_size
            pc.local_world_size = tp_size
            set_parallelism_config(pc)
            for requested in ("auto", "nvfp4_cutlass"):
                with self.subTest(tp_size=tp_size, requested=requested):
                    moe.moe_strategy = requested
                    config = MoEConfigAdapter(model, pc, moe)
                    self.assertFalse(CudaNvfp4CutlassStrategy().can_handle(config))

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

    def test_tensor_parallel_selection_uses_format_and_rejects_sp_and_world_mismatch(
        self,
    ):
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
            bits=4,
            group_size=16,
            is_quanted=False,
            activation_amax=32.0,
            source_fp8_block_size=128,
            require_contiguous_experts=True,
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
        self.assertTrue(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))
        model.quant_config.require_contiguous_experts = False
        self.assertFalse(strategy.can_handle(MoEConfigAdapter(model, pc, moe)))
        model.quant_config.require_contiguous_experts = True
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
            bits=4,
            group_size=16,
            is_quanted=False,
            activation_amax=32.0,
            source_fp8_block_size=128,
            require_contiguous_experts=True,
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
