import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _MEGA_MOE_JIT_WARMED_KEYS,
    MegaMoeExecutor,
    _activate_mega_moe_rank_nvcc_tmpdir,
    _mega_moe_rank_nvcc_tmpdir,
    _restore_tmpdir,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    clamp_token_counts,
    generate_jit_token_counts_from_signature,
    generate_mega_moe_jit_token_counts,
    mega_moe_config_signature,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


class MegaMoeJitWarmupTest(unittest.TestCase):
    def test_backend_signatures_deduplicate_nonmonotonic_buckets(self):
        signatures = ["a", "a", "b", "a", "c", "b"]
        self.assertEqual(
            generate_jit_token_counts_from_signature(signatures.__getitem__, 5),
            [0, 2, 4],
        )
        self.assertEqual(
            generate_jit_token_counts_from_signature(
                signatures.__getitem__, 5, include_cap=True
            ),
            [0, 4, 5],
        )

    def test_fp8_resolver_only_needs_existing_block_m_api(self):
        executor = MegaMoeFp8Executor.__new__(MegaMoeFp8Executor)
        torch.nn.Module.__init__(executor)
        executor.cfg = types.SimpleNamespace(
            max_tokens_per_rank=128,
            warmup_include_capacity=True,
            ep_size=4,
            n_routed_experts=512,
            n_activated_experts=10,
        )
        executor._mega_buf = types.SimpleNamespace(num_max_tokens_per_rank=1920)
        query = mock.Mock(return_value=16)
        backend = types.SimpleNamespace(get_block_m_for_mega_moe_fp8=query)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules, {"deep_gemm": types.SimpleNamespace(mega_fp8=backend)}
        ):
            self.assertEqual(executor._resolve_jit_warmup_token_counts(148), [1, 128])
        self.assertEqual(query.call_count, 130)
        self.assertEqual(query.call_args.args, (4, 512, 1920, 128, 10))

    def test_fp4_resolvers_use_allocated_buffer_capacity(self):
        for cls in (MegaMoeExecutor, MegaMoeSEExecutor):
            executor = cls.__new__(cls)
            torch.nn.Module.__init__(executor)
            executor.cfg = types.SimpleNamespace(
                max_tokens_per_rank=128,
                warmup_include_capacity=True,
                ep_size=4,
                n_routed_experts=512,
                n_activated_experts=10,
            )
            executor._mega_buf = types.SimpleNamespace(num_max_tokens_per_rank=1920)
            query = mock.Mock(return_value=16)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
                sys.modules,
                {"deep_gemm": types.SimpleNamespace(get_block_m_for_mega_moe=query)},
            ):
                self.assertEqual(
                    executor._resolve_jit_warmup_token_counts(148), [1, 128]
                )
            self.assertEqual(query.call_args.args, (4, 512, 1920, 128, 10, "fp8xfp4"))

    def test_pack_only_suppresses_all_deepgemm_launches(self):
        for cls in (MegaMoeExecutor, MegaMoeSEExecutor, MegaMoeFp8Executor):
            executor = cls.__new__(cls)
            torch.nn.Module.__init__(executor)
            executor._jit_pack_only = True
            # No weights/config/backend is needed: a pack-only launch exits
            # before touching DeepGEMM or any pre-kernel collective.
            executor._launch(torch.empty(1), 1, torch.device("cpu"))

    def test_fp8_packer_tokens_follow_launch_selectors(self):
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            _gate_pack_block_m,
            _ordinary_pack_block_m,
            mega_moe_input_pack_warmup_token_counts,
        )

        for override in (None, "1", "2", "4", "8"):
            env = {} if override is None else {"MEGA_MOE_PACK_BLOCK_M": override}
            with mock.patch.dict(os.environ, env, clear=True):
                for cap in (1, 1023, 1024, 2047, 2048, 4096):
                    reps = mega_moe_input_pack_warmup_token_counts(cap)
                    signature = lambda t: (
                        _ordinary_pack_block_m(t),
                        _gate_pack_block_m(t),
                    )
                    self.assertEqual(
                        {signature(t) for t in reps},
                        {signature(t) for t in range(1, cap + 1)},
                    )

    def test_signature_covers_non_block_m_template_branches(self):
        params = dict(num_ranks=4, num_experts=512, num_topk=10)
        signature = lambda t, bm, fp8: mega_moe_config_signature(
            **params, num_tokens=t, block_m=bm, fp8_weights=fp8
        )
        self.assertEqual(signature(1500, 192, True), (192, 32, 256, True))
        self.assertEqual(signature(3277, 192, True), (192, 32, 256, False))
        self.assertEqual(signature(212, 64, True), (64, 32, 128, True))
        self.assertEqual(signature(417, 64, False), (64, 16, 256, False))

    def test_fp8_packers_warm_before_deepgemm_execution(self):
        executor = MegaMoeFp8Executor.__new__(MegaMoeFp8Executor)
        torch.nn.Module.__init__(executor)
        executor.cfg = Fp8Fp4MoeRuntimeConfig(
            layer_id=1,
            hidden_size=128,
            moe_inter_dim=128,
            expert_num=16,
            moe_k=4,
            n_shared_experts=0,
            swiglu_limit=0.0,
            ep_size=2,
            ep_rank=0,
            max_tokens_per_rank=2048,
            moe_strategy="mega_moe_fp8",
        )
        executor._mega_l1_w = torch.empty(1)
        executor._mega_group = object()
        events = []

        def forward(x, weights, indices):
            self.assertEqual(weights.dtype, torch.float32)
            self.assertEqual(indices.dtype, torch.int64)
            self.assertEqual(weights.stride(), (4, 1))
            self.assertEqual(indices.stride(), (4, 1))
            events.append(("ordinary", executor._jit_pack_only, x.size(0)))

        def gate(x, payload):
            events.append(("gate", executor._jit_pack_only, x.size(0)))

        with mock.patch.dict(os.environ, {"MODEL_WARM_UP": "1"}), mock.patch.object(
            executor, "forward", side_effect=forward
        ), mock.patch.object(
            executor, "forward_gate_pack", side_effect=gate
        ), mock.patch(
            "torch.cuda.synchronize"
        ), mock.patch(
            "torch.distributed.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.barrier",
            side_effect=lambda **kw: events.append("barrier"),
        ):
            executor.warmup_jit([0, 1, 2048])
        phase = [
            (path, True, tokens)
            for tokens in (0, 1, 2048)
            for path in ("ordinary", "gate")
        ]
        self.assertEqual(
            events,
            phase
            + ["barrier"]
            + [
                event
                for tokens in (0, 1, 2048)
                for event in (
                    "barrier",
                    ("ordinary", False, tokens),
                    ("gate", False, tokens),
                )
            ]
            + ["barrier"],
        )
        self.assertFalse(executor._jit_pack_only)

    def test_failed_packing_does_not_enter_collective_phase(self):
        executor = MegaMoeFp8Executor.__new__(MegaMoeFp8Executor)
        torch.nn.Module.__init__(executor)
        executor.cfg = types.SimpleNamespace(
            dim=128,
            n_routed_experts=16,
            n_activated_experts=4,
            local_expert_start=0,
            n_local_experts=8,
        )
        executor._mega_l1_w = torch.empty(1)
        with mock.patch.dict(os.environ, {"MODEL_WARM_UP": "1"}), mock.patch.object(
            executor, "forward", side_effect=RuntimeError("compile failed")
        ), mock.patch.object(executor, "forward_gate_pack") as gate, mock.patch(
            "torch.distributed.barrier"
        ) as barrier:
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                executor.warmup_jit([1])
        self.assertFalse(executor._jit_pack_only)
        gate.assert_not_called()
        barrier.assert_not_called()

    def test_model_warmup_switch(self):
        with mock.patch.dict(os.environ, {"MODEL_WARM_UP": "0"}):
            self.assertFalse(mega_moe_jit_warmup_enabled())

    def test_rank_local_nvcc_directory(self):
        with mock.patch.dict(
            os.environ, {"DG_JIT_CACHE_DIR": "/tmp/dg-cache"}, clear=True
        ):
            self.assertEqual(
                _mega_moe_rank_nvcc_tmpdir(7),
                "/tmp/dg-cache/rtp_llm_mega_moe_nvcc/rank_7",
            )

    def test_tmpdir_is_restored_after_warmup_failure(self):
        executor = MegaMoeExecutor.__new__(MegaMoeExecutor)
        torch.nn.Module.__init__(executor)
        executor.cfg = Fp8Fp4MoeRuntimeConfig(
            layer_id=1,
            hidden_size=7168,
            moe_inter_dim=2048,
            expert_num=256,
            moe_k=6,
            n_shared_experts=1,
            swiglu_limit=7.0,
            ep_size=8,
            ep_rank=5,
            max_tokens_per_rank=4096,
            moe_strategy="mega_moe",
        )
        executor._input_packer = FusedMegaMoEInputPacker()
        executor._mega_group = object()
        executor.warmup_jit = mock.Mock(side_effect=RuntimeError("compile failed"))
        executor._resolve_jit_warmup_token_counts = mock.Mock(return_value=[1])
        fake_deep_gemm = types.SimpleNamespace(get_num_sms=lambda: 148)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "MEGA_MOE_NVCC_TMPDIR": tmpdir,
                "MODEL_WARM_UP": "1",
                "TMPDIR": "/old/tmp",
            },
            clear=True,
        ), mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
        ), mock.patch(
            "torch.distributed.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.get_rank", return_value=5
        ):
            _MEGA_MOE_JIT_WARMED_KEYS.clear()
            self.addCleanup(_MEGA_MOE_JIT_WARMED_KEYS.clear)
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                executor._maybe_warmup_jit_once()
            self.assertFalse(_MEGA_MOE_JIT_WARMED_KEYS)
            self.assertEqual(os.environ["TMPDIR"], "/old/tmp")
            self.assertTrue(
                os.path.isdir(os.path.join(tmpdir, "rtp_llm_mega_moe_nvcc", "rank_5"))
            )

    def test_warmup_compiles_ordinary_and_both_gate_pack_variants(self):
        executor = MegaMoeExecutor.__new__(MegaMoeExecutor)
        torch.nn.Module.__init__(executor)
        executor.cfg = Fp8Fp4MoeRuntimeConfig(
            layer_id=1,
            hidden_size=128,
            moe_inter_dim=128,
            expert_num=16,
            moe_k=4,
            n_shared_experts=0,
            swiglu_limit=7.0,
            ep_size=2,
            ep_rank=0,
            max_tokens_per_rank=4,
            moe_strategy="mega_moe",
            route_scale=2.5,
        )
        executor._mega_l1_w = torch.empty(1)
        executor._input_packer = FusedMegaMoEInputPacker()
        executor._mega_group = object()

        with mock.patch.dict(
            os.environ,
            {
                "MODEL_WARM_UP": "1",
                "MEGA_MOE_INPUT_PACKER_IMPL": "optimized",
            },
            clear=True,
        ), mock.patch.object(executor, "forward") as forward, mock.patch.object(
            executor, "forward_gate_pack"
        ) as forward_gate_pack, mock.patch(
            "torch.distributed.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.barrier"
        ) as barrier, mock.patch(
            "torch.cuda.synchronize"
        ):
            executor.warmup_jit([1, 4])

        self.assertEqual(forward.call_count, 4)
        self.assertEqual(forward_gate_pack.call_count, 8)
        payloads = [call.args[1] for call in forward_gate_pack.call_args_list]
        nonhash = [payload for payload in payloads if payload.bias is not None]
        hashed = [payload for payload in payloads if payload.tid2eid is not None]
        self.assertEqual(len(nonhash), 4)
        self.assertEqual(len(hashed), 4)
        self.assertTrue(all(payload.route_scale == 2.5 for payload in payloads))
        self.assertTrue(all(payload.input_ids is not None for payload in hashed))
        self.assertEqual(barrier.call_count, 4)

    def test_generator_adds_packer_boundaries_and_covers_backend_variants(self):
        params = dict(num_ranks=4, num_experts=512, num_topk=10)
        block_m = lambda t: 16 if t < 1500 else 192
        for cap in (1, 1100, 2200, 4096):
            for fp8 in (False, True):
                for include_cap in (False, True):
                    tokens = generate_mega_moe_jit_token_counts(
                        **params,
                        get_block_m=block_m,
                        max_tokens_per_rank=cap,
                        fp8_weights=fp8,
                        include_cap=include_cap,
                    )
                    self.assertLessEqual(max(tokens), cap)
                    self.assertIn(1, tokens)
                    for boundary in (1024, 2048):
                        if cap >= boundary:
                            self.assertIn(boundary, tokens)
                    if include_cap:
                        self.assertIn(cap, tokens)
                    if cap >= 3277 and fp8:
                        self.assertIn(cap if include_cap else 3277, tokens)

    def test_override_tokens_are_generic_sorted_and_clamped(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "4098,2,2,999999,0,-1"},
        ):
            tokens = parse_mega_moe_jit_warmup_tokens_override()
        self.assertEqual(tokens, [2, 4098, 999999])
        self.assertEqual(clamp_token_counts(tokens or [], 65536), [2, 4098, 65536])

    def test_invalid_override_falls_back_to_generated_counts(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "2,not-a-token"},
            clear=True,
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(parse_mega_moe_jit_warmup_tokens_override())
        self.assertIn("invalid MEGA_MOE_JIT_WARMUP_TOKENS", "\n".join(logs.output))

    def test_override_without_positive_tokens_falls_back(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "0,-1,-20"},
            clear=True,
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(parse_mega_moe_jit_warmup_tokens_override())
        self.assertIn("contains no positive token counts", "\n".join(logs.output))

    def test_generic_adapter_supports_default_warmup_for_both_executors(self):
        model_config = ModelConfig()
        model_config.expert_num = 8
        model_config.moe_k = 2
        model_config.moe_inter_size = 128
        model_config.n_shared_experts = 1
        model_config.max_seq_len = 256
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 2
        parallelism_config.ep_rank = 0
        parallelism_config.world_size = 2
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )

        cases = (
            (
                MegaMoeExecutor,
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "mega_moe.generate_mega_moe_jit_token_counts",
            ),
            (
                MegaMoeSEExecutor,
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                "mega_moe_se.generate_mega_moe_se_jit_token_counts",
            ),
        )
        for executor_type, generator_path in cases:
            with self.subTest(executor=executor_type.__name__):
                executor = executor_type.__new__(executor_type)
                torch.nn.Module.__init__(executor)
                executor.cfg = config
                with mock.patch(generator_path, return_value=[1]) as generator:
                    self.assertEqual(
                        executor._resolve_jit_warmup_token_counts(128), [1]
                    )
                self.assertFalse(generator.call_args.kwargs["include_cap"])


if __name__ == "__main__":
    unittest.main()
