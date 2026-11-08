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
        py_hw_kernel_config=SimpleNamespace(enable_cuda_graph=cuda_graph),
    )


class Sm120Dsv4WorkerProfileTest(TestCase):
    """Accepted profiles: PREFILL TP4/CP4/DP1/EP4/world4, DECODE
    TP1/DP4/EP4/world4 CP-off, aligned PDFUSION TP1 DP=EP=world."""

    def _validate(self, configs, sm120=True):
        with patch(
            "rtp_llm.config.server_config_setup."
            "_auto_deepep_supported_on_visible_devices",
            return_value=not sm120,
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

    def test_decode_eager_with_and_without_mtp_accepted(self):
        # With graphs disabled the canonical sm120_a2a_decode backend runs
        # eagerly for both auto-resolved and explicit strategies, with or
        # without speculation.
        for moe_strategy in ("auto", "sm120_a2a_decode"):
            for spec_type in (SpeculativeType.NONE, SpeculativeType.MTP):
                with self.subTest(moe_strategy=moe_strategy, spec_type=spec_type):
                    cfg = _configs(
                        RoleType.DECODE,
                        tp_size=1,
                        dp_size=4,
                        ep_size=4,
                        world_size=4,
                        local_world_size=4,
                        spec_type=spec_type,
                        moe_strategy=moe_strategy,
                    )
                    self._validate(cfg)
                    self.assertEqual(cfg.moe_config.moe_strategy, "sm120_a2a_decode")

    def test_decode_cuda_graph_rejected_before_backend(self):
        # sm120_a2a_decode is eager-only: a valid TP1/DP4/EP4/world4 DECODE
        # profile must fail at startup when CUDA graphs are enabled, whether
        # the strategy is auto-resolved or explicit, with or without MTP.
        for moe_strategy in ("auto", "sm120_a2a_decode"):
            for spec_type in (SpeculativeType.NONE, SpeculativeType.MTP):
                with self.subTest(moe_strategy=moe_strategy, spec_type=spec_type):
                    cfg = _configs(
                        RoleType.DECODE,
                        tp_size=1,
                        dp_size=4,
                        ep_size=4,
                        world_size=4,
                        local_world_size=4,
                        spec_type=spec_type,
                        moe_strategy=moe_strategy,
                        cuda_graph=True,
                    )
                    with self.assertRaisesRegex(ValueError, "sm120_a2a_decode") as ctx:
                        self._validate(cfg)
                    self.assertIn("eager execution", str(ctx.exception))
                    self.assertIn("collective protocol", str(ctx.exception))

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
        self.assertIn("Required: TP1/DP4/EP4/world4", str(ctx.exception))

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
