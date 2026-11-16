"""Regression coverage for the exact-SM120 DSv4 worker startup contract.

``_validate_sm120_dsv4_worker_profile`` is the fail-fast gate that runs inside
``setup_and_configure_server`` before model processes spawn. These tests are
pure-Python (mocked configs and a patched architecture probe); no hardware is
required and nothing here asserts runtime behavior.
"""

import inspect
import re
import textwrap
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.config.server_config_setup import _validate_sm120_dsv4_worker_profile
from rtp_llm.ops import RoleType, SpeculativeType


def _configs(
    role_type,
    *,
    pp_size=1,
    tp_size=1,
    dp_size=1,
    ep_size=1,
    world_size=1,
    local_world_size=1,
    cp_enabled=False,
    spec_type=SpeculativeType.NONE,
    moe_strategy="auto",
    cuda_graph=False,
    buckets=(1, 4),
    eplb=False,
    multi_task_prompt_tokens=None,
):
    return SimpleNamespace(
        role_config=SimpleNamespace(role_type=role_type),
        parallelism_config=SimpleNamespace(
            pp_size=pp_size,
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
            world_size=world_size,
            local_world_size=local_world_size,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: cp_enabled),
        ),
        sp_config=SimpleNamespace(type=spec_type),
        moe_config=SimpleNamespace(moe_strategy=moe_strategy),
        py_hw_kernel_config=SimpleNamespace(
            enable_cuda_graph=cuda_graph, decode_capture_batch_sizes=buckets
        ),
        concurrency_config=SimpleNamespace(concurrency_limit=8),
        eplb_config=SimpleNamespace(enable_eplb=lambda: eplb),
        kv_cache_config=SimpleNamespace(
            multi_task_prompt_tokens=multi_task_prompt_tokens or {}
        ),
    )


class Sm120Dsv4WorkerProfileTest(TestCase):
    """Accepted profiles: PREFILL TP4/CP4/DP1/EP4/world4, DECODE
    TP1 DP=EP=world in {4, 8} CP-off, aligned PDFUSION TP1 DP=EP=world."""

    def _validate(self, configs, sm120=True):
        with (
            patch(
                "rtp_llm.config.server_config_setup."
                "_auto_deepep_supported_on_visible_devices",
                return_value=not sm120,
            ),
            patch("torch.cuda.device_count", return_value=8),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            _validate_sm120_dsv4_worker_profile(configs)

    def test_prefill_profile_accepted_and_strategy_resolved(self):
        cfg = _configs(
            RoleType.PREFILL,
            tp_size=4,
            dp_size=1,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            cp_enabled=True,
        )
        self._validate(cfg)
        self.assertEqual(cfg.moe_config.moe_strategy, "sm120_fused_moe")

    def test_decode_profile_accepted_and_strategy_resolved(self):
        cfg = _configs(
            RoleType.DECODE,
            tp_size=1,
            dp_size=4,
            ep_size=4,
            world_size=4,
            local_world_size=4,
        )
        self._validate(cfg)
        self.assertEqual(cfg.moe_config.moe_strategy, "sm120_a2a_decode")

    def test_decode_eager_profiles(self):
        for world in (4, 8):
            for spec_type in (SpeculativeType.NONE, SpeculativeType.MTP):
                for strategy in ("auto", "sm120_a2a_decode"):
                    with self.subTest(
                        world=world, spec_type=spec_type, strategy=strategy
                    ):
                        cfg = _configs(
                            RoleType.DECODE,
                            dp_size=world,
                            ep_size=world,
                            world_size=world,
                            local_world_size=world,
                            spec_type=spec_type,
                            moe_strategy=strategy,
                        )
                        self._validate(cfg)
                        self.assertEqual(
                            cfg.moe_config.moe_strategy, "sm120_a2a_decode"
                        )

    def test_multihost_decode_eager_profiles(self):
        for strategy in ("auto", "sm120_a2a_decode"):
            with self.subTest(strategy=strategy):
                cfg = _configs(
                    RoleType.DECODE,
                    dp_size=4,
                    ep_size=4,
                    world_size=4,
                    local_world_size=2,
                    moe_strategy=strategy,
                )
                self._validate(cfg)
                self.assertEqual(cfg.moe_config.moe_strategy, "sm120_a2a_decode")

    def test_decode_graph_profiles(self):
        for world in (4, 8):
            for strategy in ("auto", "sm120_graph_decode"):
                with self.subTest(world=world, strategy=strategy):
                    cfg = _configs(
                        RoleType.DECODE,
                        dp_size=world,
                        ep_size=world,
                        world_size=world,
                        local_world_size=world,
                        moe_strategy=strategy,
                        cuda_graph=True,
                    )
                    self._validate(cfg)
                    self.assertEqual(cfg.moe_config.moe_strategy, "sm120_graph_decode")

    def test_decode_graph_rejects_speculation(self):
        for spec_type in (SpeculativeType.MTP, SpeculativeType.DSPARK):
            for strategy in ("auto", "sm120_graph_decode"):
                with self.subTest(spec_type=spec_type, strategy=strategy):
                    cfg = _configs(
                        RoleType.DECODE,
                        dp_size=8,
                        ep_size=8,
                        world_size=8,
                        local_world_size=8,
                        spec_type=spec_type,
                        moe_strategy=strategy,
                        cuda_graph=True,
                    )
                    with self.assertRaisesRegex(ValueError, "ordinary decode only"):
                        self._validate(cfg)

    def test_decode_graph_rejects_incompatible_strategy_and_topology(self):
        for changes, message in (
            ({"buckets": (1, 1)}, "unique positive buckets"),
            ({"moe_strategy": "sm120_a2a_decode"}, "requires eager execution"),
            (
                {"cuda_graph": False, "moe_strategy": "sm120_graph_decode"},
                "enable_cuda_graph",
            ),
            ({"local_world_size": 4}, "single-host"),
            ({"cp_enabled": True}, "CP off"),
            ({"buckets": ()}, "DECODE_CAPTURE_CONFIG"),
            ({"buckets": (1, 9)}, "CONCURRENCY_LIMIT"),
            ({"eplb": True}, "fixed expert ownership"),
            ({"multi_task_prompt_tokens": {"system": [1]}}, "local system prompts"),
        ):
            with self.subTest(changes=changes):
                kwargs = dict(
                    dp_size=8,
                    ep_size=8,
                    world_size=8,
                    local_world_size=8,
                    cuda_graph=True,
                )
                kwargs.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    self._validate(_configs(RoleType.DECODE, **kwargs))

    def test_decode_graph_workspace_bounds(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.graph_decode import (
            Sm120DecodeMoeGeometry,
            decode_workspace_geometry,
        )

        one = Sm120DecodeMoeGeometry(16, 8, 4096, 2048, 256, 6)
        self.assertEqual(one.expert_capacity, 128)
        self.assertEqual(one.local_experts, 32)
        self.assertEqual(
            decode_workspace_geometry([16, 16], [1], 8, 4096, 2048, 256, 6),
            one,
        )
        with self.assertRaisesRegex(ValueError, "kernel extents"):
            Sm120DecodeMoeGeometry(1 << 22, 8, 4096, 2048, 256, 6)
        with self.assertRaisesRegex(ValueError, "geometry"):
            Sm120DecodeMoeGeometry(16, 8, 4096, 2048, 254, 6)

    def test_decode_graph_rejects_mixed_devices(self):
        cfg = _configs(
            RoleType.DECODE,
            dp_size=4,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            cuda_graph=True,
        )
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
            patch(
                "torch.cuda.get_device_capability",
                side_effect=lambda rank: (12, 0) if rank < 3 else (12, 1),
            ),
            self.assertRaisesRegex(ValueError, "homogeneous exact SM120"),
        ):
            _validate_sm120_dsv4_worker_profile(cfg)

    def test_pdfusion_non_mtp_profile_rejected_before_backend(self):
        # The exact retired profile: non-speculative sm120_a2a_decode PDFUSION
        # over DP>1 has no phase protocol keeping EP peers inside the same
        # collective order, so it is rejected at startup whether the strategy
        # is auto-resolved or explicit — under both eager and captured-graph
        # configurations. Active MTP or separate PREFILL/DECODE roles remain
        # supported.
        for moe_strategy in ("auto", "sm120_a2a_decode"):
            for cuda_graph in (False, True):
                with self.subTest(moe_strategy=moe_strategy, cuda_graph=cuda_graph):
                    cfg = _configs(
                        RoleType.PDFUSION,
                        tp_size=1,
                        dp_size=4,
                        ep_size=4,
                        world_size=4,
                        local_world_size=4,
                        moe_strategy=moe_strategy,
                        cuda_graph=cuda_graph,
                    )
                    with self.assertRaisesRegex(
                        ValueError, "non-speculative sm120_a2a_decode PDFUSION"
                    ):
                        self._validate(cfg)

    def test_pdfusion_single_gpu_accepted_without_strategy_override(self):
        cfg = _configs(
            RoleType.PDFUSION,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            world_size=1,
            local_world_size=1,
        )
        self._validate(cfg)
        self.assertEqual(cfg.moe_config.moe_strategy, "auto")

    def test_pdfusion_dp_with_speculation_eager_accepted(self):
        # Speculative fused decode is allowed on the eager path: the
        # all-to-all executor exchanges per-rank token counts each round, so
        # uneven or idle DP ranks stay collective-aligned. Only the captured
        # CUDA-graph variant is rejected.
        cfg = _configs(
            RoleType.PDFUSION,
            tp_size=1,
            dp_size=4,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            spec_type=SpeculativeType.MTP,
        )
        self._validate(cfg)
        self.assertEqual(cfg.moe_config.moe_strategy, "sm120_a2a_decode")

    def test_pdfusion_dp_with_speculation_and_cuda_graphs_rejected(self):
        cfg = _configs(
            RoleType.PDFUSION,
            tp_size=1,
            dp_size=4,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            spec_type=SpeculativeType.MTP,
            cuda_graph=True,
        )
        with self.assertRaisesRegex(ValueError, "eager execution"):
            self._validate(cfg)

    def test_pdfusion_single_dp_with_speculation_accepted(self):
        cfg = _configs(
            RoleType.PDFUSION,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            world_size=1,
            local_world_size=1,
            spec_type=SpeculativeType.MTP,
            cuda_graph=True,
        )
        self._validate(cfg)

    def test_prefill_without_cp_rejected(self):
        cfg = _configs(
            RoleType.PREFILL,
            tp_size=4,
            dp_size=1,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            cp_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, "Required: TP4/CP4"):
            self._validate(cfg)

    def test_decode_wrong_topology_reports_actual_vs_required(self):
        cfg = _configs(
            RoleType.DECODE,
            tp_size=2,
            dp_size=4,
            ep_size=4,
            world_size=8,
            local_world_size=8,
        )
        with self.assertRaisesRegex(ValueError, "tp=2") as ctx:
            self._validate(cfg)
        self.assertIn("Required: TP1 DECODE", str(ctx.exception))

    def test_pp_size_must_be_one(self):
        cfg = _configs(
            RoleType.PREFILL,
            pp_size=2,
            tp_size=4,
            dp_size=1,
            ep_size=4,
            world_size=8,
            local_world_size=4,
            cp_enabled=True,
        )
        with self.assertRaisesRegex(ValueError, "pp_size=1"):
            self._validate(cfg)

    def test_conflicting_explicit_strategy_rejected(self):
        cfg = _configs(
            RoleType.DECODE,
            tp_size=1,
            dp_size=4,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            moe_strategy="sm120_fused_moe",
        )
        with self.assertRaisesRegex(ValueError, "moe_strategy"):
            self._validate(cfg)

    def test_non_sm120_participants_skip_validation(self):
        # Even a topology that would fail on the target lane must pass
        # untouched when participating devices are not exact SM120.
        cfg = _configs(
            RoleType.PREFILL,
            tp_size=2,
            dp_size=2,
            ep_size=2,
            world_size=4,
            local_world_size=4,
        )
        self._validate(cfg, sm120=False)
        self.assertEqual(cfg.moe_config.moe_strategy, "auto")

    def test_frontend_role_skips_validation(self):
        cfg = _configs(RoleType.FRONTEND)
        self._validate(cfg)


class MoeCpActivationSemanticsTest(TestCase):
    """``moe_cp_enabled`` must stay enabled for every non-decode role,
    including PDFUSION — the role is supported on the target lane."""

    def test_pdfusion_prefill_cp_keeps_moe_cp_enabled(self):
        import rtp_llm.models_py.model_desc.deepseek_v4_model as model_mod

        source = inspect.getsource(model_mod)
        match = re.search(r"moe_cp_enabled = ([^\n]+)", source)
        self.assertIsNotNone(match)
        expr = textwrap.dedent(match.group(1)).strip()

        stub = SimpleNamespace(_prefill_cp_size=4, _is_decode_role=False)
        enabled = eval(expr, {}, {"self": stub})  # noqa: S307 - test-local expr
        self.assertTrue(enabled)

        stub._is_decode_role = True
        self.assertFalse(eval(expr, {}, {"self": stub}))  # noqa: S307

        # DECODE-role workers never get MoE CP even with a nonzero cp size.
        # PREFILL and PDFUSION both report is_decode_role=False, so the
        # expression above already proves PDFUSION keeps CP semantics.
        self.assertIn("not self._is_decode_role", expr)


if __name__ == "__main__":
    main()
