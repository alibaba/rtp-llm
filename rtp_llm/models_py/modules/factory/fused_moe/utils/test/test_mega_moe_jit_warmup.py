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
    generate_mega_moe_jit_token_counts,
    mega_moe_config_signature,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


class MegaMoeJitWarmupTest(unittest.TestCase):
    def test_model_warmup_switch(self):
        with mock.patch.dict(os.environ, {"MODEL_WARM_UP": "0"}):
            self.assertFalse(mega_moe_jit_warmup_enabled())

    def test_rank_local_nvcc_directory(self):
        with mock.patch.dict(
            os.environ, {"DG_JIT_CACHE_DIR": "/tmp/dg-cache"}, clear=True
        ):
            scratch = _mega_moe_rank_nvcc_tmpdir(7)
            self.assertNotEqual(scratch, _mega_moe_rank_nvcc_tmpdir(8))
            self.assertEqual(scratch, _mega_moe_rank_nvcc_tmpdir(7))
            self.assertLess(len(os.fsencode(scratch)) + 40, 108)
            self.assertEqual(os.environ["DG_JIT_CACHE_DIR"], "/tmp/dg-cache")

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
        executor._activation = "swiglu"
        executor._activation_kwargs = {}
        executor._input_packer = FusedMegaMoEInputPacker()
        executor.warmup_jit = mock.Mock(side_effect=RuntimeError("compile failed"))
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
            executor.warmup_jit.assert_called_once()
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
            "torch.distributed.barrier"
        ) as barrier, mock.patch(
            "torch.cuda.synchronize"
        ):
            executor.warmup_jit([1, 4])

        self.assertEqual(forward.call_count, 2)
        self.assertEqual(forward_gate_pack.call_count, 4)
        payloads = [call.args[1] for call in forward_gate_pack.call_args_list]
        nonhash = [payload for payload in payloads if payload.bias is not None]
        hashed = [payload for payload in payloads if payload.tid2eid is not None]
        self.assertEqual(len(nonhash), 2)
        self.assertEqual(len(hashed), 2)
        self.assertTrue(all(payload.route_scale == 2.5 for payload in payloads))
        self.assertTrue(all(payload.input_ids is not None for payload in hashed))
        self.assertEqual(barrier.call_count, 3)

    def test_unchunked_generated_counts_use_bucket_representative(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=4,
            num_experts=256,
            num_experts_per_rank=64,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=16384,
        )
        self.assertLess(tokens[-1], 16384)

    def test_chunked_generated_counts_cover_runtime_cap(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=4,
            num_experts=256,
            num_experts_per_rank=64,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=16384,
            include_cap=True,
        )
        self.assertEqual(tokens[-1], 16384)

    def test_generated_counts_cover_every_unique_bucket_for_ep_sizes(self):
        for ep_size in (2, 4, 8):
            with self.subTest(ep_size=ep_size):
                params = dict(
                    num_ranks=ep_size,
                    num_experts=256,
                    num_experts_per_rank=256 // ep_size,
                    num_topk=6,
                    intermediate_hidden=2048,
                    num_sms=148,
                )
                tokens = generate_mega_moe_jit_token_counts(
                    **params,
                    max_tokens_per_rank=4096,
                )
                self.assertTrue(any(1 < token < 4096 for token in tokens))
                signatures = [
                    mega_moe_config_signature(**params, num_tokens=token)
                    for token in tokens
                ]
                expected_signatures = []
                previous = None
                for token in range(1, 4097):
                    signature = mega_moe_config_signature(**params, num_tokens=token)
                    if signature != previous:
                        expected_signatures.append(signature)
                        previous = signature
                self.assertEqual(signatures, expected_signatures)

    def test_generated_counts_never_exceed_runtime_cap(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=8,
            num_experts=256,
            num_experts_per_rank=32,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=257,
        )
        self.assertTrue(tokens)
        self.assertLessEqual(max(tokens), 257)

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


class _K3CpuPacker:
    name = "reference"

    def pack(self, x, routing, ids, buf, tokens):
        buf.x[:tokens].copy_(x)
        buf.topk_weights[:tokens].copy_(routing)
        buf.topk_idx[:tokens].copy_(ids)


class K3FusedSharedJitWarmupTest(unittest.TestCase):
    def setUp(self):
        import deep_gemm

        from rtp_llm.models_py.modules.kimi_k3 import moe

        self.k3_dg, self.k3_moe = deep_gemm, moe
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patches = [
            mock.patch.dict(
                os.environ,
                {
                    "MODEL_WARM_UP": "1",
                    "MEGA_MOE_JIT_WARMUP_TOKENS": "1,7,32",
                    "MEGA_MOE_NVCC_TMPDIR": temporary.name,
                    "TMPDIR": "/before",
                },
            ),
            mock.patch.object(moe, "_FUSED_SHARED_JIT_WARMED_KEYS", set()),
            mock.patch.object(deep_gemm, "get_num_sms", return_value=148),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("torch.distributed.barrier"),
            mock.patch("torch.distributed.is_initialized", return_value=False),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _k3_experts(self, *, capacity=32, shared_hidden=256):
        experts = self.k3_moe._FusedSharedExperts.__new__(
            self.k3_moe._FusedSharedExperts
        )
        torch.nn.Module.__init__(experts)
        experts.cfg = types.SimpleNamespace(
            ep_size=2,
            n_routed_experts=16,
            n_local_experts=8,
            n_activated_experts=2,
            local_expert_start=8,
            dim=128,
            moe_inter_dim=256,
            warmup_include_capacity=False,
            layer_id=3,
            activation_beta=4.0,
            activation_linear_beta=25.0,
        )
        experts._max_tokens_per_rank = capacity
        experts._shared_intermediate = 256
        experts.buf = types.SimpleNamespace(
            num_max_tokens_per_rank=capacity,
            x=torch.empty(capacity, 128, dtype=torch.bfloat16),
            x_sf=torch.empty(capacity, 1),
            topk_weights=torch.empty(capacity, 2),
            topk_idx=torch.empty(capacity, 2, dtype=torch.long),
        )
        experts.routed_y = torch.empty(capacity, 128, dtype=torch.bfloat16)
        experts.shared_x = torch.empty(capacity, shared_hidden, dtype=torch.bfloat16)
        experts.shared_y = torch.empty_like(experts.shared_x)
        experts.routed_l1, experts.routed_l2 = object(), object()
        experts.shared_l1, experts.shared_l2 = object(), object()
        experts.kernel = lambda *args, **kwargs: self.k3_dg.fp8_fp4_mega_moe_k3(
            *args, **kwargs
        )
        experts.activation_kwargs = {
            "activation_beta": experts.cfg.activation_beta,
            "activation_linear_beta": experts.cfg.activation_linear_beta,
        }
        experts.packer = _K3CpuPacker()
        experts.fp4_block = 32
        return experts

    def test_k3_shared_forward_deduplicates_matching_geometry(self):
        calls = []

        def launch(y, l1, l2, buf, **kwargs):
            tokens = y.shape[0]
            self.assertEqual(kwargs["activation"], "situ")
            self.assertEqual(kwargs["activation_beta"], 4.0)
            self.assertEqual(kwargs["activation_linear_beta"], 25.0)
            self.assertEqual(kwargs["recipe"], (1, 1, 32))
            self.assertEqual(tuple(buf.x[:tokens].shape), (tokens, 128))
            self.assertTrue(
                torch.equal(
                    buf.topk_idx[:tokens], torch.tensor([8, 9]).expand(tokens, -1)
                )
            )
            self.assertEqual(torch.count_nonzero(buf.topk_weights[:tokens]).item(), 0)
            y.zero_()
            kwargs["shared_y"][:tokens].copy_(kwargs["shared_x"][:tokens])
            calls.append((tokens, kwargs["shared_y"].shape[1]))

        # Keep the K3 tensor preparation and forward real; replace the external launch.
        with mock.patch.object(self.k3_dg, "fp8_fp4_mega_moe_k3", side_effect=launch):
            self._k3_experts()._maybe_warmup_jit_once()
            count = len(calls)
            self.assertGreater(count, 0)
            self._k3_experts()._maybe_warmup_jit_once()
            self.assertEqual(len(calls), count)
            self._k3_experts(shared_hidden=512)._maybe_warmup_jit_once()
            self.assertGreater(len(calls), count)
        self.assertEqual(
            set(calls),
            {
                (1, 256),
                (7, 256),
                (32, 256),
                (1, 512),
                (7, 512),
                (32, 512),
            },
        )

    def test_k3_fused_pack_prepares_both_valid_mask_layouts(self):
        from rtp_llm.models_py.triton_kernels.moe import mega_moe_input_pack

        experts = self._k3_experts()
        experts.packer = types.SimpleNamespace(name="fused")
        masks = []

        def pack(x, routing, ids, out, scales, out_ids, out_routing, **kwargs):
            masks.append(kwargs["valid_mask"] is not None)
            out.copy_(x)
            out_ids.copy_(ids)
            out_routing.copy_(routing)
            kwargs["shared_out"].copy_(kwargs["shared_input"])

        with (
            mock.patch.object(
                mega_moe_input_pack, "fused_pack_mega_moe_inputs", side_effect=pack
            ),
            mock.patch.object(self.k3_dg, "fp8_fp4_mega_moe_k3"),
        ):
            experts.warmup_jit([1, 7])
        self.assertEqual(masks, [False, True, False, True])

    def test_k3_automatic_plan_includes_shared_pack_block_boundary(self):
        experts = self._k3_experts(capacity=2048)
        experts.packer = types.SimpleNamespace(name="fused")
        with mock.patch.dict(os.environ, clear=True):
            tokens = experts._resolve_jit_warmup_token_counts(148)
        self.assertIn(2048, tokens)

    def test_k3_failed_compile_restores_tmpdir_without_marking_prepared(self):
        with mock.patch.object(
            self.k3_dg,
            "fp8_fp4_mega_moe_k3",
            side_effect=RuntimeError("compiler failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                self._k3_experts()._maybe_warmup_jit_once()
        self.assertEqual(os.environ["TMPDIR"], "/before")
        self.assertFalse(self.k3_moe._FUSED_SHARED_JIT_WARMED_KEYS)


if __name__ == "__main__":
    unittest.main()
