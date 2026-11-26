"""Regression coverage for the SM120 DSv4 worker startup contract.

``validate_sm120_dsv4_worker_profile`` is the fail-fast gate invoked from
``setup_and_configure_server`` before model processes spawn.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.config.server_config_setup import _validate_sm120_dsv4_worker_profile
from rtp_llm.config.sm120_dsv4_policy import validate_sm120_dsv4_worker_profile
from rtp_llm.models_py.model_desc.deepseek_v4_model import resolve_moe_cp_size
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
            "rtp_llm.config.sm120_dsv4_policy.participating_sm120_state",
            return_value="all" if sm120 else "none",
        ):
            validate_sm120_dsv4_worker_profile(configs)

    def test_server_config_setup_delegates_to_policy_module(self):
        cfg = _configs(
            RoleType.PREFILL,
            tp_size=4,
            dp_size=1,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            cp_enabled=True,
        )
        with patch(
            "rtp_llm.config.sm120_dsv4_policy.participating_sm120_state",
            return_value="all",
        ):
            _validate_sm120_dsv4_worker_profile(cfg)
        self.assertEqual(cfg.moe_config.moe_strategy, "sm120_fused_moe")

    def test_mixed_sm120_and_non_sm120_participating_gpus_rejected(self):
        decode_kwargs = dict(
            tp_size=1,
            dp_size=4,
            ep_size=4,
            world_size=4,
            local_world_size=4,
        )
        for caps, expected_strategy, should_fail in (
            ([(10, 0), (12, 0), (10, 0), (12, 0)], "auto", True),
            ([(12, 0)] * 4, "sm120_a2a_decode", False),
            ([(10, 0)] * 4, "auto", False),
        ):
            with (
                self.subTest(caps=caps),
                patch("rtp_llm.device.cuda_capability.is_cuda", return_value=True),
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.device_count", return_value=len(caps)),
                patch("torch.cuda.get_device_capability", side_effect=caps.__getitem__),
            ):
                cfg = _configs(RoleType.DECODE, **decode_kwargs)
                if should_fail:
                    with self.assertRaisesRegex(ValueError, "mixed SM120 and non-SM120"):
                        validate_sm120_dsv4_worker_profile(cfg)
                else:
                    validate_sm120_dsv4_worker_profile(cfg)
                self.assertEqual(cfg.moe_config.moe_strategy, expected_strategy)

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

    def test_prefill_eager_accepted_for_auto_and_explicit_strategy(self):
        for moe_strategy in ("auto", "sm120_fused_moe"):
            with self.subTest(moe_strategy=moe_strategy):
                cfg = _configs(
                    RoleType.PREFILL,
                    tp_size=4,
                    dp_size=1,
                    ep_size=4,
                    world_size=4,
                    local_world_size=4,
                    cp_enabled=True,
                    moe_strategy=moe_strategy,
                )
                self._validate(cfg)
                self.assertEqual(cfg.moe_config.moe_strategy, "sm120_fused_moe")

    def test_prefill_cuda_graph_rejected_before_backend(self):
        for moe_strategy in ("auto", "sm120_fused_moe"):
            for spec_type in (SpeculativeType.NONE, SpeculativeType.MTP):
                with self.subTest(moe_strategy=moe_strategy, spec_type=spec_type):
                    cfg = _configs(
                        RoleType.PREFILL,
                        tp_size=4,
                        dp_size=1,
                        ep_size=4,
                        world_size=4,
                        local_world_size=4,
                        cp_enabled=True,
                        spec_type=spec_type,
                        moe_strategy=moe_strategy,
                        cuda_graph=True,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "sm120_fused_moe CP4 PREFILL requires eager execution",
                    ) as ctx:
                        self._validate(cfg)
                    self.assertIn("enable_cuda_graph=false", str(ctx.exception))

    def test_prefill_cuda_graph_keeps_earlier_errors(self):
        cfg = _configs(
            RoleType.PREFILL,
            tp_size=4,
            dp_size=1,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            cp_enabled=False,
            cuda_graph=True,
        )
        with self.assertRaisesRegex(ValueError, "Required: TP4/CP4"):
            self._validate(cfg)
        cfg = _configs(
            RoleType.PREFILL,
            tp_size=4,
            dp_size=1,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            cp_enabled=True,
            moe_strategy="sm120_a2a_decode",
            cuda_graph=True,
        )
        with self.assertRaisesRegex(ValueError, "Required: sm120_fused_moe"):
            self._validate(cfg)

    def test_pdfusion_non_mtp_profile_rejected_before_backend(self):
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
                        ValueError, "sm120_a2a_decode PDFUSION profile without MTP"
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

    def test_pdfusion_dp_requires_phase_padding_speculation(self):
        for spec_type, accepted in (
            (SpeculativeType.EAGLE, True),
            (SpeculativeType.DSPARK, True),
            (SpeculativeType.DFLASH, True),
            (SpeculativeType.VANILLA, False),
            (SpeculativeType.EAGLE3, False),
            (SpeculativeType.DETERMINISTIC, False),
        ):
            with self.subTest(spec_type=spec_type):
                cfg = _configs(
                    RoleType.PDFUSION,
                    tp_size=1,
                    dp_size=4,
                    ep_size=4,
                    world_size=4,
                    local_world_size=4,
                    spec_type=spec_type,
                )
                if accepted:
                    self._validate(cfg)
                    self.assertEqual(cfg.moe_config.moe_strategy, "sm120_a2a_decode")
                else:
                    with self.assertRaisesRegex(
                        ValueError, "without MTP, EAGLE, DSPARK or DFLASH"
                    ):
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

    def test_pdfusion_single_dp_rejects_ep_strategies_and_keeps_grouped_fp4(self):
        for strategy in ("auto", " auto ", "grouped_fp4"):
            with self.subTest(strategy=strategy):
                cfg = _configs(
                    RoleType.PDFUSION,
                    tp_size=1,
                    dp_size=1,
                    ep_size=1,
                    world_size=1,
                    local_world_size=1,
                    spec_type=SpeculativeType.MTP,
                    moe_strategy=strategy,
                    cuda_graph=True,
                )
                self._validate(cfg)
                self.assertEqual(cfg.moe_config.moe_strategy, strategy)

        for strategy in ("sm120_fused_moe", "sm120_a2a_decode", "mega_moe", "local_loop"):
            with self.subTest(rejected=strategy):
                cfg = _configs(
                    RoleType.PDFUSION,
                    tp_size=1,
                    dp_size=1,
                    ep_size=1,
                    world_size=1,
                    local_world_size=1,
                    spec_type=SpeculativeType.MTP,
                    moe_strategy=strategy,
                    cuda_graph=True,
                )
                with self.assertRaisesRegex(ValueError, "auto or grouped_fp4"):
                    self._validate(cfg)

    def test_strategy_whitespace_is_normalized_and_deepep_rejected_on_sm120(self):
        cfg = _configs(
            RoleType.DECODE,
            tp_size=1,
            dp_size=4,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            moe_strategy=" auto ",
        )
        self._validate(cfg)
        self.assertEqual(cfg.moe_config.moe_strategy, "sm120_a2a_decode")

        for field in ("use_deepep_moe", "use_mori_ep"):
            bad = _configs(
                RoleType.DECODE,
                tp_size=1,
                dp_size=4,
                ep_size=4,
                world_size=4,
                local_world_size=4,
            )
            setattr(bad.moe_config, field, True)
            with self.assertRaisesRegex(ValueError, "unsupported on SM120"):
                self._validate(bad)

    def test_legacy_vit_separation_role_normalizes_before_dsv4_validation(self):
        from rtp_llm.config.server_config_setup import setup_and_configure_server
        from rtp_llm.ops import VitSeparation

        cfg = SimpleNamespace(
            vit_config=SimpleNamespace(vit_separation=VitSeparation.VIT_SEPARATION_ROLE),
            role_config=SimpleNamespace(role_type=RoleType.PDFUSION),
            parallelism_config=SimpleNamespace(
                pp_size=1, tp_size=1, dp_size=1, ep_size=1, world_size=1, local_world_size=1
            ),
            moe_config=SimpleNamespace(),
            deep_ep_config=SimpleNamespace(),
            sp_config=SimpleNamespace(type=SpeculativeType.NONE, gen_num_per_cycle=0),
            server_config=SimpleNamespace(ip="127.0.0.1"),
            model_args=SimpleNamespace(model_type="deepseek_v4"),
        )
        seen_roles = []
        with (
            patch("rtp_llm.config.server_config_setup.setup_default_args"),
            patch("rtp_llm.config.server_config_setup.fetch_model_files_to_local"),
            patch("rtp_llm.config.server_config_setup.resolve_max_generate_batch_size", return_value=1),
            patch(
                "rtp_llm.config.server_config_setup.auto_configure_deepep",
                side_effect=lambda **kw: seen_roles.append(("deepep", kw["role_type"])),
            ),
            patch(
                "rtp_llm.config.server_config_setup._validate_sm120_dsv4_worker_profile",
                side_effect=lambda c: seen_roles.append(("profile", c.role_config.role_type)),
            ),
        ):
            setup_and_configure_server(cfg)
        self.assertEqual(cfg.role_config.role_type, RoleType.VIT)
        self.assertEqual(
            seen_roles, [("deepep", RoleType.VIT), ("profile", RoleType.VIT)]
        )

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

    def test_non_sm120_cp_prefill_cuda_graph_skips_validation(self):
        cfg = _configs(
            RoleType.PREFILL,
            tp_size=4,
            dp_size=1,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            cp_enabled=True,
            cuda_graph=True,
        )
        self._validate(cfg, sm120=False)
        self.assertEqual(cfg.moe_config.moe_strategy, "auto")

    def test_frontend_role_skips_validation(self):
        cfg = _configs(RoleType.FRONTEND)
        self._validate(cfg)


class MoeCpActivationSemanticsTest(TestCase):
    """``moe_cp_enabled`` stays enabled for non-decode roles (PREFILL and PDFUSION)."""

    def test_resolve_moe_cp_size_handles_prefill_pdfusion_and_decode(self):
        self.assertEqual(resolve_moe_cp_size(4, 4, is_decode_role=False), 4)
        self.assertEqual(resolve_moe_cp_size(4, 4, is_decode_role=True), 1)
        self.assertEqual(resolve_moe_cp_size(1, 4, is_decode_role=False), 1)

    def test_deepseek_v4_model_initialize_wires_resolve_moe_cp_size(self):
        model_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "models_py/model_desc/deepseek_v4_model.py"
        )
        tree = ast.parse(model_path.read_text())
        model_cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV4Model"
        )
        init_impl = next(
            n
            for n in model_cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_initialize_impl"
        )
        called = {
            n.func.id
            for n in ast.walk(init_impl)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertIn("resolve_moe_cp_size", called)


if __name__ == "__main__":
    main()
