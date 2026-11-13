import gc
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import Nvfp4MarlinMoE
from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.test.marlin_fixture import (
    MarlinNvfp4TestFixture,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.marlin_nvfp4_executor import (
    _MARLIN_NAMES,
    Qwen4DecodeMarlinExecutor,
    Qwen4MarlinExecutor,
    _linear_scales,
    _prepare_marlin_weights,
)
from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
    PreparedWeightDict,
    prepare_compute_weights,
)
from rtp_llm.resident_weights import ResidentWeights, resident_init_tensors, store
from rtp_llm.utils.model_weight import W

_EXECUTOR_MODULE = Qwen4MarlinExecutor.__module__
_RESIDENT_FINGERPRINT = {
    "model_type": "qwen4_exp_mtp",
    "checkpoint_path": "/test/modelopt-nvfp4",
}


def _source_mtp_config():
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.config.quant_config import ModelOptFp4Config
    from rtp_llm.config.server_config_setup import set_parallelism_config
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
        MoEConfigAdapter,
    )
    from rtp_llm.ops import MoeConfig, ParallelismConfig

    model = ModelConfig()
    model.model_type, model.data_type = "qwen4_exp_mtp", "bf16"
    model.is_mtp = True
    model.expert_num, model.hidden_size, model.moe_inter_size, model.moe_k = (
        8,
        2560,
        640,
        3,
    )
    model._qwen4_nvfp4_source = True
    model.quant_config = ModelOptFp4Config(bits=4, group_size=16, is_quanted=True)
    pc = ParallelismConfig()
    pc.tp_size = pc.ep_size = pc.world_size = pc.local_world_size = 2
    set_parallelism_config(pc)
    moe = MoeConfig()
    moe.moe_strategy, moe.fp4_moe_op = "nvfp4_marlin", "marlin"
    moe.use_all_gather, moe.use_deepep_moe = True, False
    return MoEConfigAdapter(model, pc, moe)


def _packed_snapshot(executor):
    return [
        (
            str(tensor.dtype),
            list(tensor.shape),
            hashlib.sha256(
                tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest(),
        )
        for tensor in (
            executor.moe.w1,
            executor.moe.w2,
            executor.moe.s1,
            executor.moe.s2,
            executor.moe.g1,
            executor.moe.g2,
        )
    ]


def _resident_restart(path):
    from rtp_llm.model_loader.model_weight_info import ModelWeights

    expected = torch.load(path, weights_only=True)
    manager = ResidentWeights("draft", 0, _RESIDENT_FINGERPRINT, False)
    restored = manager.attach()
    assert restored is not None
    weights = ModelWeights(1, "cuda:0", torch.bfloat16)
    for item in restored:
        weights.weights[item.scope][item.name] = item.tensor
    manager.bind(weights)
    # A replay must not depend on raw source entries, even if canonical views exist.
    for name in _MARLIN_NAMES:
        weights.weights[0].pop(name, None)
    compute = prepare_compute_weights(weights)
    config = _source_mtp_config()
    with patch(
        _EXECUTOR_MODULE + "._prepare_marlin_weights",
        side_effect=AssertionError("warm start repacked weights"),
    ), patch(
        _EXECUTOR_MODULE + ".resident_init_tensors",
        wraps=resident_init_tensors,
    ) as replay:
        for _ in range(2):
            executor = Qwen4MarlinExecutor(config, None, compute.weights[0])
            assert _packed_snapshot(executor) == expected["packed"]
            for name in _MARLIN_NAMES:
                assert weights.weights[0][name] is compute.weights[0][name]
            args = [tensor.cuda() for tensor in expected["inputs"]]
            actual = executor.moe(*args)
            torch.testing.assert_close(actual.cpu(), expected["output"], rtol=0, atol=0)
            del executor
        assert replay.call_count == 1
    manager.finalize()


@unittest.skipUnless(
    Nvfp4MarlinMoE.is_supported(),
    "requires a supported NVFP4 Marlin runtime and GPU",
)
class MarlinNvfp4Test(MarlinNvfp4TestFixture, unittest.TestCase):

    @unittest.skipUnless(
        os.environ.get("QWEN4_NVFP4_FIXTURE"), "real NVFP4 fixture not supplied"
    )
    def test_checkpoint_experts_against_dequantized_reference(self):
        from safetensors import safe_open

        with safe_open(os.environ["QWEN4_NVFP4_FIXTURE"], framework="pt") as tensors:
            for prefix in ("model.language_model", "mtp"):

                def load(projection, suffix):
                    return torch.stack(
                        [
                            tensors.get_tensor(
                                f"{prefix}.layers.0.mlp.experts.{expert}.{projection}_proj.{suffix}"
                            )
                            for expert in (0, 117, 0, 117)
                        ]
                    ).cuda()

                self.q1 = torch.cat((load("up", "weight"), load("gate", "weight")), 1)
                self.q2 = load("down", "weight")
                self.s1 = torch.cat(
                    (load("up", "weight_scale"), load("gate", "weight_scale")), 1
                )
                self.s2 = load("down", "weight_scale")
                self.g1, self.g2 = load("up", "weight_scale_2"), load(
                    "down", "weight_scale_2"
                )
                torch.testing.assert_close(
                    self.g1, load("gate", "weight_scale_2"), rtol=0, atol=0
                )
                lut = torch.tensor(
                    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
                    device="cuda",
                )

                def dequant(q, scale, outer):
                    codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
                    return (
                        lut[codes]
                        * scale.float().repeat_interleave(16, -1)
                        * outer[:, None, None]
                    )

                self.w1 = dequant(self.q1, self.s1, self.g1)
                self.w2 = dequant(self.q2, self.s2, self.g2)
                for rank in (0, 1):
                    model = Nvfp4MarlinMoE(
                        self.q1, self.s1, self.g1, self.q2, self.s2, self.g2, 2, rank
                    )
                    for rows in (1, 7, 33):
                        with self.subTest(prefix=prefix, rank=rank, rows=rows):
                            x, ids, weights = self.inputs(rows, rank)
                            actual = model(x, ids, weights)
                            expected = self.reference(x, ids, weights, rank)
                            self.assert_reference(actual, expected)
                            rmse = (
                                (
                                    (actual.float() - expected).square().mean()
                                    / expected.square().mean().clamp_min(1e-30)
                                )
                                .sqrt()
                                .item()
                            )
                            print(
                                "REAL_NVFP4_RELATIVE_RMSE",
                                prefix,
                                rank,
                                rows,
                                rmse,
                                flush=True,
                            )

    def test_repack_rebinds_final_storage_and_executor_contract(self):
        def swizzle(sf):
            e, n, k = sf.shape
            return (
                sf.view(torch.uint8)
                .reshape(e, n // 128, 4, 32, k // 4, 4)
                .permute(0, 1, 4, 3, 2, 5)
                .contiguous()
                .reshape(e, n, k)
                .view(torch.int32)
            )

        q1, q2 = self.q1.clone(), self.q2.clone()
        sf1, sf2 = swizzle(self.s1), swizzle(self.s2)
        torch.testing.assert_close(
            _linear_scales(sf1, 4, 1280, 160).float(), self.s1.float(), rtol=0, atol=0
        )
        native = SimpleNamespace(
            config=None,
            quant_config=None,
            weights=PreparedWeightDict(),
            ep_size=2,
            ep_rank=1,
            tp_size=1,
            tp_rank=0,
            w1=q1.view(torch.int64),
            w2=q2.view(torch.int64),
            block_scales=(sf1, sf2),
        )
        native.weights.update(
            {
                W.moe_w1: q1,
                W.moe_w2: q2,
                W.moe_s1: sf1,
                W.moe_s2: sf2,
                W.moe_w1_s2: self.g1,
                W.moe_w2_s2: self.g2,
            }
        )
        ptrs = [v.data_ptr() for v in (q1, q2, sf1, sf2)]
        executor = Qwen4DecodeMarlinExecutor(native)
        final_ptrs = [
            v.data_ptr()
            for v in (
                executor.moe.w1,
                executor.moe.w2,
                executor.moe.s1,
                executor.moe.s2,
            )
        ]
        self.assertTrue(all(old != new for old, new in zip(ptrs, final_ptrs)))
        self.assertEqual(
            final_ptrs,
            [v.data_ptr() for v in (native.w1, native.w2, *native.block_scales)],
        )
        torch.testing.assert_close(q1, self.q1, rtol=0, atol=0)
        torch.testing.assert_close(q2, self.q2, rtol=0, atol=0)
        torch.testing.assert_close(
            _linear_scales(sf1, 4, 1280, 160).float(), self.s1.float(), rtol=0, atol=0
        )
        args = self.inputs(4, 1)
        payload = ExpertForwardPayload(
            expert_x=args[0], expert_topk_ids=args[1], expert_topk_weights=args[2]
        )
        actual = executor.execute(
            payload, "silu", None, None, False, None
        ).fused_expert_output
        self.assert_reference(actual, self.reference(*args, 1))
        self.assertIs(executor.weights, native.weights)
        self.assertIsNotNone(native.weights[W.moe_w1]._qwen4_marlin_layout)
        del executor
        # A later model can reuse the packed tensors after the first model dies.
        # It must rebuild views and workspaces without repacking their bytes.
        for _ in range(2):
            executor = Qwen4DecodeMarlinExecutor(native)
            repeated = executor.execute(
                payload, "silu", None, None, False, None
            ).fused_expert_output
            torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
            one_row = self.inputs(1, 1)
            one_payload = ExpertForwardPayload(
                expert_x=one_row[0],
                expert_topk_ids=one_row[1],
                expert_topk_weights=one_row[2],
            )
            one_output = executor.execute(
                one_payload, "silu", None, None, False, None
            ).fused_expert_output
            self.assert_reference(one_output, self.reference(*one_row, 1))
        for activation, expert_map, apply_input in [
            ("relu", None, False),
            ("silu", torch.zeros(8, device="cuda"), False),
            ("silu", None, True),
        ]:
            with self.assertRaises(ValueError):
                executor.execute(
                    payload, activation, expert_map, None, apply_input, None
                )

    def test_resident_source_mtp_restart(self):
        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory

        config = _source_mtp_config()
        name = f"rtp_llm_marlin_test.{uuid.uuid4().hex}"
        with patch.dict(os.environ, {"RTP_LLM_RESIDENT_STORE": name}):
            store.ensure_running(name, owner_pid=os.getpid())
            try:
                weights = ModelWeights(1, "cuda:0", torch.bfloat16)
                weights.weights[0].update(
                    {
                        W.moe_w1: self.q1.clone(),
                        W.moe_w2: self.q2.clone(),
                        W.moe_s1: CudaImpl.swizzle_blockscale(self.s1.cpu()),
                        W.moe_s2: CudaImpl.swizzle_blockscale(self.s2.cpu()),
                        W.moe_w1_s2: self.g1.clone(),
                        W.moe_w2_s2: self.g2.clone(),
                        W.moe_w1_i_s: torch.ones(self.experts, device="cuda"),
                        W.moe_w2_i_s: torch.ones(self.experts, device="cuda"),
                    }
                )
                manager = ResidentWeights("draft", 0, _RESIDENT_FINGERPRINT, False)
                manager.move_to_vmm(weights)
                raw_allocations = [
                    weakref.ref(manager.allocation_of(weights.weights[0][key]))
                    for key in (W.moe_w1, W.moe_w2)
                ]
                compute = prepare_compute_weights(weights)
                with patch(
                    _EXECUTOR_MODULE + "._prepare_marlin_weights",
                    wraps=_prepare_marlin_weights,
                ) as pack:
                    executor = (
                        FusedMoeFactory()
                        .create_fused_moe(config, compute.weights[0])
                        .fused_experts
                    )
                    self.assertIsInstance(executor, Qwen4MarlinExecutor)
                    packed = _packed_snapshot(executor)
                    args = self.inputs(4)
                    output = executor.moe(*args)
                    self.assert_reference(output, self.reference(*args, 0))
                    del executor
                    executor = Qwen4MarlinExecutor(config, None, compute.weights[0])
                    self.assertEqual(pack.call_count, 1)
                    self.assertEqual(_packed_snapshot(executor), packed)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in raw_allocations))
                for key, tensor in zip(
                    _MARLIN_NAMES,
                    (
                        executor.moe.w1,
                        executor.moe.w2,
                        executor.moe.s1,
                        executor.moe.s2,
                        executor.moe.g1,
                        executor.moe.g2,
                    ),
                ):
                    self.assertIs(weights.weights[0][key], compute.weights[0][key])
                    self.assertEqual(
                        weights.weights[0][key].untyped_storage().data_ptr(),
                        tensor.untyped_storage().data_ptr(),
                    )
                manager.finalize()
                self.assertTrue(manager._published)
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "expected.pt")
                    torch.save(
                        {
                            "packed": packed,
                            "inputs": [x.cpu() for x in args],
                            "output": output.cpu(),
                        },
                        path,
                    )
                    child = subprocess.run(
                        [sys.executable, __file__, "--resident-marlin-restart", path],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
            finally:
                store.stop_store(name)

    def test_explicit_strategy_uses_prepared_weights(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.config.quant_config import (
            QWEN4_FP8_NVFP4_RECIPE,
            ModelOptFp4Config,
        )
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
            CudaNvfp4MarlinStrategy,
        )
        from rtp_llm.ops import MoeConfig, ParallelismConfig

        model = ModelConfig()
        model.model_type = "qwen4_exp"
        model.data_type = "bf16"
        model.expert_num = 8
        model.hidden_size, model.moe_inter_size, model.moe_k = self.k, self.n, 3
        model.quant_config = ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=False,
            model_recipe=QWEN4_FP8_NVFP4_RECIPE,
            source_fp8_block_size=128,
            require_contiguous_experts=True,
        )
        pc = ParallelismConfig()
        pc.tp_size = pc.ep_size = pc.world_size = pc.local_world_size = 2
        set_parallelism_config(pc)
        from rtp_llm.config.qwen4_exp_config import resolve_qwen4_parallelism

        pc = resolve_qwen4_parallelism(model, pc)
        moe = MoeConfig()
        moe.moe_strategy, moe.fp4_moe_op = "nvfp4_marlin", "marlin"
        moe.use_all_gather, moe.use_deepep_moe = True, False
        config = MoEConfigAdapter(model, pc, moe)
        strategy = CudaNvfp4MarlinStrategy()
        config.moe_strategy = "auto"
        with patch.object(
            strategy,
            "get_attributes",
            side_effect=AssertionError("auto imported Marlin"),
        ):
            self.assertFalse(strategy.can_handle(config))
        config.moe_strategy = "nvfp4_marlin"
        self.assertTrue(strategy.can_handle(config))
        factory = FusedMoeFactory()
        self.assertIsInstance(
            factory.registry.get_strategy(config), CudaNvfp4MarlinStrategy
        )
        weights = PreparedWeightDict(
            {
                W.moe_w1: self.q1.clone(),
                W.moe_w2: self.q2.clone(),
                W.moe_s1: CudaImpl.swizzle_blockscale(self.s1.cpu()),
                W.moe_s2: CudaImpl.swizzle_blockscale(self.s2.cpu()),
                W.moe_w1_s2: self.g1,
                W.moe_w2_s2: self.g2,
                W.moe_w1_i_s: torch.ones(self.experts, device="cuda"),
                W.moe_w2_i_s: torch.ones(self.experts, device="cuda"),
            }
        )
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "cutlass_nvfp4_executor.CutlassNvfp4Executor.__init__",
            side_effect=AssertionError("Marlin must not initialize a CUTLASS executor"),
        ):
            executor = factory.create_fused_moe(config, weights).fused_experts
        self.assertIs(executor.weights, weights)
        for rows in (1, 64, 256):
            args = self.inputs(rows)
            payload = ExpertForwardPayload(
                expert_x=args[0], expert_topk_ids=args[1], expert_topk_weights=args[2]
            )
            actual = executor.execute(
                payload, "silu", None, None, False, None
            ).fused_expert_output
            self.assert_reference(actual, self.reference(*args, 0))
        model.model_type = "other"
        self.assertFalse(strategy.can_handle(config))
        model.model_type = "qwen4_exp"
        moe.fp4_moe_op = "cutlass"
        self.assertFalse(strategy.can_handle(config))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resident-marlin-restart":
        _resident_restart(sys.argv[2])
    else:
        unittest.main()
