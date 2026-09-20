"""Config-rejection tests for the canonical release profile and fail-closed startup validation.

These pin the behaviour a serving group depends on: a configuration that would produce a divergent
collective, an uncaptured batch shape, a silently slow prefill path or a token envelope decode cannot
finish is REJECTED before model loading, and a correct configuration is accepted. They also pin the
fail-closed posture -- a knob that cannot be resolved from the configuration object is a violation,
never a default -- and the developer-mode guarantee that nothing runs unless a profile is selected.

Pure Python: no pybind config types, no CUDA, no distributed runtime. The snapshot is the same
representation production validates, built here directly; `snapshot_from_env_configs` is exercised
against a stub so the dotted paths into the real config object are pinned too.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from rtp_llm.config import release_profile as rp


def decode_snapshot(**overrides):
    """A valid N2 decode-rank snapshot; overrides model one misconfiguration at a time."""
    base = dict(
        role_type="RoleType.DECODE",
        rank_id=1,
        world_size=4,
        tp_size=1,
        dp_size=4,
        ep_size=4,
        concurrency_limit=2,
        enable_cuda_graph=True,
        prefill_cp_method="CPRotateMethod.PREFILL_CP",
        cp_computes=False,
        cp_consumes_prefill=True,
        prefill_cp_kv_cache_sharded=True,
        prefill_cp_size=4,
        sp_type="DSpark",
        gen_num_per_cycle=3,
        max_seq_len=32832,
        kernel_seq_size_per_block=256,
        decode_capture_batch_sizes=[1, 2],
        reuse_cache=False,
        max_batch_tokens_size=32832,
        max_context_batch_size=1,
        fake_balance_expert=False,
        use_batch_decode_scheduler=False,
        decode_fixed_bs=2,
        nccl_p2p_level=None,
        moe_fp4_backend=None,
    )
    base.update(overrides)
    return rp.ReleaseSnapshot(**base)


def prefill_snapshot(**overrides):
    """A valid prefill-rank snapshot for the same group."""
    base = dict(
        role_type="RoleType.PREFILL",
        rank_id=0,
        world_size=4,
        tp_size=4,
        dp_size=1,
        ep_size=4,
        concurrency_limit=2,
        enable_cuda_graph=False,
        prefill_cp_method="CPRotateMethod.ALL_GATHER",
        cp_computes=True,
        cp_consumes_prefill=False,
        prefill_cp_kv_cache_sharded=True,
        prefill_cp_size=rp.UNSET,
        sp_type="DSpark",
        gen_num_per_cycle=3,
        max_seq_len=133120,
        kernel_seq_size_per_block=256,
        decode_capture_batch_sizes=[],
        reuse_cache=False,
        max_batch_tokens_size=64000,
        max_context_batch_size=1,
        fake_balance_expert=False,
        use_batch_decode_scheduler=False,
        decode_fixed_bs=0,
        nccl_p2p_level=None,
        moe_fp4_backend=None,
    )
    base.update(overrides)
    return rp.ReleaseSnapshot(**base)


PROFILE = rp.RELEASE_PROFILES["sm120_dp4ep4_n2"]


class AcceptsTheCanonicalProfile(unittest.TestCase):
    def test_valid_decode_rank_has_no_violations(self):
        self.assertEqual([], rp.validate(decode_snapshot(), PROFILE))

    def test_valid_prefill_rank_has_no_violations(self):
        self.assertEqual([], rp.validate(prefill_snapshot(), PROFILE))

    def test_every_decode_rank_of_the_group_is_accepted(self):
        for rank in range(4):
            self.assertEqual([], rp.validate(decode_snapshot(rank_id=rank), PROFILE), f"rank {rank}")


class RejectsDecodeMisconfiguration(unittest.TestCase):
    def assertRejected(self, snapshot, needle):
        violations = rp.validate(snapshot, PROFILE)
        self.assertTrue(violations, "expected a rejection, got none")
        self.assertTrue(
            any(needle in v for v in violations),
            f"expected a violation mentioning {needle!r}, got: {violations}",
        )

    def test_padding_disabled_is_rejected(self):
        # Without fixed padding, ranks replay divergent graph keys under uneven batch and the
        # expert-parallel collectives deadlock. This is the single most important rejection.
        self.assertRejected(decode_snapshot(decode_fixed_bs=0), "fixed decode padding must be enabled")

    def test_padding_malformed_value_is_rejected(self):
        self.assertRejected(decode_snapshot(decode_fixed_bs=-1), "fixed decode padding must be enabled")

    def test_padding_not_matching_the_profile_is_rejected(self):
        self.assertRejected(decode_snapshot(decode_fixed_bs=4), "profile requires 2")

    def test_capture_set_missing_the_padded_batch_is_rejected(self):
        self.assertRejected(decode_snapshot(decode_capture_batch_sizes=[1]), "does not contain")

    def test_admission_cap_above_the_padded_batch_is_rejected(self):
        # A rank would admit more streams than it can replay, and the batch fails.
        self.assertRejected(decode_snapshot(concurrency_limit=4), "must equal the padded batch")

    def test_admission_cap_below_the_padded_batch_is_rejected(self):
        self.assertRejected(decode_snapshot(concurrency_limit=1), "must equal the padded batch")

    def test_topology_mismatch_is_rejected(self):
        self.assertRejected(decode_snapshot(dp_size=2), "dp_size=2")
        self.assertRejected(decode_snapshot(ep_size=2), "ep_size=2")
        self.assertRejected(decode_snapshot(tp_size=4), "tp_size=4")

    def test_geometry_other_than_256_is_rejected(self):
        self.assertRejected(decode_snapshot(kernel_seq_size_per_block=128), "kernel_seq_size_per_block=128")

    def test_speculative_depth_mismatch_is_rejected(self):
        self.assertRejected(decode_snapshot(gen_num_per_cycle=1), "gen_num_per_cycle=1")

    def test_a_different_speculative_mechanism_at_the_right_depth_is_rejected(self):
        # Depth alone is not the MTP shape: the wrong mechanism would silently change behaviour.
        self.assertRejected(decode_snapshot(sp_type="SpeculativeType.EAGLE"), "speculative mechanism")

    def test_the_expected_speculative_mechanism_is_accepted(self):
        self.assertEqual([], rp.validate(decode_snapshot(sp_type="SpeculativeType.DSPARK"), PROFILE))

    def test_token_envelope_too_small_is_rejected(self):
        self.assertRejected(decode_snapshot(max_seq_len=8192), "max_seq_len=8192")

    def test_cache_reuse_enabled_without_qualification_is_rejected(self):
        self.assertRejected(decode_snapshot(reuse_cache=True), "reuse_cache=True")


class RejectsTestOnlyAndNonDefaultOverrides(unittest.TestCase):
    def assertRejected(self, snapshot, needle):
        violations = rp.validate(snapshot, PROFILE)
        self.assertTrue(any(needle in v for v in violations), f"expected {needle!r} in {violations}")

    def test_forced_expert_routing_is_rejected(self):
        self.assertRejected(decode_snapshot(fake_balance_expert=True), "fake_balance_expert")

    def test_exact_batch_test_scheduler_is_rejected(self):
        self.assertRejected(decode_snapshot(use_batch_decode_scheduler=True), "use_batch_decode_scheduler")

    def test_forced_transport_override_is_rejected(self):
        self.assertRejected(decode_snapshot(nccl_p2p_level="PHB"), "non-default NCCL transport")

    def test_unapproved_experimental_moe_backend_is_rejected(self):
        self.assertRejected(decode_snapshot(moe_fp4_backend="deepgemm"), "unapproved experimental MoE backend")

    def test_an_approved_override_is_accepted(self):
        approved = rp.ReleaseProfile(
            name="approved-transport",
            approved_nccl_p2p_levels=("PHB",),
            approved_moe_backends=("deepgemm",),
        )
        snapshot = decode_snapshot(nccl_p2p_level="PHB", moe_fp4_backend="deepgemm")
        self.assertEqual([], rp.validate(snapshot, approved))


class RejectsPrefillMisconfiguration(unittest.TestCase):
    def test_admission_bound_above_the_all_to_all_guard_is_rejected(self):
        # Above the guard, prefill falls back to the fixed-EP path, which allocates an
        # O(world * tokens * hidden) FP32 reduction buffer per layer: a large latency cliff.
        violations = rp.validate(prefill_snapshot(max_batch_tokens_size=133120), PROFILE)
        self.assertTrue(any("all-to-all payload guard" in v for v in violations), violations)

    def test_auto_derived_admission_bound_is_rejected(self):
        violations = rp.validate(prefill_snapshot(max_batch_tokens_size=0), PROFILE)
        self.assertTrue(any("auto-derived" in v for v in violations), violations)

    def test_admission_bound_above_the_profile_value_is_rejected(self):
        violations = rp.validate(prefill_snapshot(max_batch_tokens_size=65000), PROFILE)
        self.assertTrue(any("profile value" in v for v in violations), violations)

    def test_the_profile_bound_is_accepted(self):
        self.assertEqual([], rp.validate(prefill_snapshot(max_batch_tokens_size=64000), PROFILE))

    def test_prefill_ep_mismatch_is_rejected(self):
        violations = rp.validate(prefill_snapshot(ep_size=2), PROFILE)
        self.assertTrue(any("ep_size=2" in v for v in violations), violations)


class RejectsGraphTopologyRankAndMembershipCases(unittest.TestCase):
    """The cases an independent audit reproduced as accepted, plus the group-level equivalents.

    Each of these is individually plausible and collectively broken: a decode leg that never replays
    its captured graphs, a prefill leg running a different parallelism arrangement, an unqualified
    geometry on the prefill side, a rank number outside the group, a partial group, and a manifest
    that identifies no artifact at all.
    """

    def assertRejected(self, snapshot, needle):
        violations = rp.validate(snapshot, PROFILE)
        self.assertTrue(violations, f"expected a rejection for {needle!r}, got none")
        self.assertTrue(
            any(needle in v for v in violations),
            f"expected a violation mentioning {needle!r}, got: {violations}",
        )

    # --- audit case 1 -------------------------------------------------------------------------
    def test_decode_with_cuda_graph_disabled_is_rejected(self):
        # Configured capture sizes are not evidence that the captured shapes are replayed.
        self.assertRejected(decode_snapshot(enable_cuda_graph=False), "enable_cuda_graph=False")

    def test_decode_with_unresolved_graph_setting_is_rejected(self):
        self.assertRejected(decode_snapshot(enable_cuda_graph=rp.UNSET), "not resolvable")

    # --- audit case 2 -------------------------------------------------------------------------
    def test_prefill_running_decode_topology_is_rejected(self):
        # TP1/DP4 on the prefill leg is the decode arrangement, not the CP4 leg.
        self.assertRejected(prefill_snapshot(tp_size=1), "tp_size=1")
        self.assertRejected(prefill_snapshot(dp_size=4), "dp_size=4")

    def test_prefill_without_context_parallelism_is_rejected(self):
        # is_enabled() is the engine predicate for 'this leg computes context parallelism'; a CP4 leg
        # that reports false is not the qualified prefill arrangement.
        self.assertRejected(prefill_snapshot(cp_computes=False), "is_enabled() is false")

    def test_decode_not_declaring_a_cp_prefill_is_rejected(self):
        # is_prefill_enabled() is the decode-side pairing predicate, and with a sharded KV the engine
        # CHECKs the matching size at allocation time.
        self.assertRejected(decode_snapshot(cp_consumes_prefill=False), "is_prefill_enabled() is false")

    def test_sharded_kv_without_an_explicit_cp_size_is_rejected(self):
        self.assertRejected(decode_snapshot(prefill_cp_size=1), "requires an explicit prefill CP size > 1")

    def test_prefill_with_the_decode_cp_method_is_rejected(self):
        self.assertRejected(
            prefill_snapshot(prefill_cp_method="CPRotateMethod.PREFILL_CP"), "prefill leg of this profile expects"
        )

    # --- audit case 3 -------------------------------------------------------------------------
    def test_prefill_geometry_128_is_rejected(self):
        self.assertRejected(prefill_snapshot(kernel_seq_size_per_block=128), "prefill leg must use the qualified")

    # --- audit case 4 -------------------------------------------------------------------------
    def test_rank_outside_the_group_is_rejected(self):
        self.assertRejected(decode_snapshot(rank_id=99), "outside the group")

    def test_wrong_world_size_is_rejected(self):
        self.assertRejected(decode_snapshot(world_size=8), "world_size=8")

    def test_decode_leg_naming_the_wrong_prefill_cp_size_is_rejected(self):
        self.assertRejected(decode_snapshot(prefill_cp_size=8), "prefill_cp_size=8")

    def test_pd_kv_sharing_mismatch_is_rejected_on_either_leg(self):
        self.assertRejected(decode_snapshot(prefill_cp_kv_cache_sharded=False), "prefill_cp_kv_cache_sharded=False")
        self.assertRejected(prefill_snapshot(prefill_cp_kv_cache_sharded=False), "prefill_cp_kv_cache_sharded=False")

    # --- audit case 5 -------------------------------------------------------------------------
    def test_group_with_a_single_rank_is_rejected(self):
        manifests = {
            "decode-0": rp.build_manifest(decode_snapshot(rank_id=0), PROFILE, {"git_sha": "abc"}),
            "prefill-0": rp.build_manifest(prefill_snapshot(rank_id=0), PROFILE, {"git_sha": "abc"}),
        }
        violations = rp.check_group_consistency(manifests)
        self.assertTrue(any("INCOMPLETE MEMBERSHIP" in v for v in violations), violations)

    def test_group_with_wrong_rank_ids_is_rejected(self):
        manifests = {}
        for i in range(4):
            manifests[f"decode-{i}"] = rp.build_manifest(
                decode_snapshot(rank_id=i + 10), PROFILE, {"git_sha": "abc"})
            manifests[f"prefill-{i}"] = rp.build_manifest(
                prefill_snapshot(rank_id=i + 10), PROFILE, {"git_sha": "abc"})
        violations = rp.check_group_consistency(manifests)
        self.assertTrue(any("MEMBERSHIP MISMATCH" in v for v in violations), violations)

    # --- audit case 6 -------------------------------------------------------------------------
    def test_group_without_artifact_identity_is_rejected(self):
        manifests = {}
        for i in range(4):
            manifests[f"decode-{i}"] = rp.build_manifest(decode_snapshot(rank_id=i), PROFILE, None)
            manifests[f"prefill-{i}"] = rp.build_manifest(prefill_snapshot(rank_id=i), PROFILE, None)
        violations = rp.check_group_consistency(manifests)
        self.assertTrue(any("MISSING ARTIFACT IDENTITY" in v for v in violations), violations)

    def test_mixed_release_profiles_are_rejected(self):
        other = rp.ReleaseProfile(name="another-profile")
        manifests = {
            "decode-0": rp.build_manifest(decode_snapshot(), PROFILE, {"a": 1}),
            "decode-1": rp.build_manifest(decode_snapshot(), other, {"a": 1}),
        }
        violations = rp.check_group_consistency(manifests)
        self.assertTrue(any("MIXED RELEASE PROFILES" in v for v in violations), violations)

    def test_startup_rejects_an_empty_artifact_record(self):
        class _Cfg:
            pass

        with self.assertRaises(rp.ReleaseProfileError) as ctx:
            rp.enforce_release_profile(
                _Cfg(), env={rp.RELEASE_PROFILE_ENV: "sm120_dp4ep4_n2"}, artifact_ids={}
            )
        self.assertIn("artifact identity is empty", str(ctx.exception))


class FailsClosedOnUnresolvableConfiguration(unittest.TestCase):
    def test_a_missing_knob_is_a_violation_not_a_default(self):
        violations = rp.validate(decode_snapshot(concurrency_limit=rp.UNSET), PROFILE)
        self.assertTrue(any("concurrency_limit: not resolvable" in v for v in violations), violations)

    def test_a_missing_capture_set_is_a_violation(self):
        violations = rp.validate(decode_snapshot(decode_capture_batch_sizes=rp.UNSET), PROFILE)
        self.assertTrue(any("decode_capture_batch_sizes: not resolvable" in v for v in violations), violations)

    def test_an_unknown_role_is_a_violation(self):
        violations = rp.validate(decode_snapshot(role_type="RoleType.FRONTEND"), PROFILE)
        self.assertTrue(any("neither a decode nor a prefill role" in v for v in violations), violations)

    def test_unset_is_falsy_so_it_cannot_be_mistaken_for_a_value(self):
        self.assertFalse(rp.UNSET)


class SnapshotFromResolvedConfig(unittest.TestCase):
    """Pin the dotted paths into the real config object, plus the environment-only knobs."""

    def stub_config(self):
        return SimpleNamespace(
            role_config=SimpleNamespace(role_type="RoleType.DECODE"),
            distribute_config=SimpleNamespace(rank_id=2),
            parallelism_config=SimpleNamespace(tp_size=1, dp_size=4, ep_size=4, world_size=4),
            concurrency_config=SimpleNamespace(concurrency_limit=2),
            sp_config=SimpleNamespace(type="DSpark", gen_num_per_cycle=3, checkpoint_path=None),
            model_args=SimpleNamespace(max_seq_len=32832, ckpt_path=None, tokenizer_path=None),
            kv_cache_config=SimpleNamespace(kernel_seq_size_per_block=256, reuse_cache=False),
            py_hw_kernel_config=SimpleNamespace(decode_capture_batch_sizes=[1, 2], enable_cuda_graph=True),
            prefill_cp_config=SimpleNamespace(
                method="CPRotateMethod.PREFILL_CP",
                kv_cache_sharded=True,
                prefill_cp_size=4,
                is_enabled=lambda: False,
                is_prefill_enabled=lambda: True,
            ),
            runtime_config=SimpleNamespace(
                fifo_scheduler_config=SimpleNamespace(max_batch_tokens_size=32832, max_context_batch_size=1),
                use_batch_decode_scheduler=False,
            ),
            moe_config=SimpleNamespace(fake_balance_expert=False),
        )

    def test_every_path_resolves(self):
        snap = rp.snapshot_from_env_configs(
            self.stub_config(), env={rp.DECODE_FIXED_BS_ENV: "2"}
        )
        self.assertEqual([], rp.validate(snap, PROFILE))
        self.assertEqual(2, snap.decode_fixed_bs)
        self.assertEqual([1, 2], list(snap.decode_capture_batch_sizes))
        self.assertEqual(256, snap.kernel_seq_size_per_block)

    def test_a_renamed_or_missing_path_fails_closed(self):
        cfg = self.stub_config()
        cfg.py_hw_kernel_config = SimpleNamespace()  # capture sizes moved/renamed
        snap = rp.snapshot_from_env_configs(cfg, env={rp.DECODE_FIXED_BS_ENV: "2"})
        self.assertIs(rp.UNSET, snap.decode_capture_batch_sizes)
        self.assertTrue(rp.validate(snap, PROFILE))

    def test_padding_padding_absent_from_env_means_disabled(self):
        snap = rp.snapshot_from_env_configs(self.stub_config(), env={})
        self.assertEqual(0, snap.decode_fixed_bs)
        self.assertTrue(any("padding must be enabled" in v for v in rp.validate(snap, PROFILE)))

    def test_malformed_padding_value_is_not_read_as_the_default(self):
        snap = rp.snapshot_from_env_configs(self.stub_config(), env={rp.DECODE_FIXED_BS_ENV: "two"})
        self.assertEqual(-1, snap.decode_fixed_bs)
        self.assertTrue(rp.validate(snap, PROFILE))

    def test_transport_and_backend_overrides_are_captured(self):
        snap = rp.snapshot_from_env_configs(
            self.stub_config(),
            env={rp.DECODE_FIXED_BS_ENV: "2", rp.NCCL_P2P_LEVEL_ENV: "PHB", rp.MOE_FP4_BACKEND_ENV: "deepgemm"},
        )
        self.assertEqual("PHB", snap.nccl_p2p_level)
        self.assertEqual("deepgemm", snap.moe_fp4_backend)
        self.assertTrue(rp.validate(snap, PROFILE))


class Manifest(unittest.TestCase):
    def manifest(self, snapshot, artifact=None):
        return rp.build_manifest(snapshot, PROFILE, artifact)

    def test_digest_is_stable_and_order_independent(self):
        a = self.manifest(decode_snapshot(), {"git_sha": "abc", "image": "sha256:1"})
        b = self.manifest(decode_snapshot(), {"image": "sha256:1", "git_sha": "abc"})
        self.assertEqual(a["manifest_digest"], b["manifest_digest"])

    def test_a_configuration_change_changes_the_digest(self):
        a = self.manifest(decode_snapshot())
        b = self.manifest(decode_snapshot(kernel_seq_size_per_block=128))
        self.assertNotEqual(a["config_digest"], b["config_digest"])

    def test_an_artifact_change_changes_the_digest(self):
        a = self.manifest(decode_snapshot(), {"git_sha": "abc"})
        b = self.manifest(decode_snapshot(), {"git_sha": "def"})
        self.assertEqual(a["config_digest"], b["config_digest"])
        self.assertNotEqual(a["artifact_digest"], b["artifact_digest"])
        self.assertNotEqual(a["manifest_digest"], b["manifest_digest"])

    def test_manifest_is_sanitized_to_configuration_and_identifiers(self):
        m = self.manifest(decode_snapshot(), {"git_sha": "abc"})
        self.assertEqual({"profile", "config", "artifact", "config_digest", "group_config_digest",
                          "artifact_digest", "manifest_digest"}, set(m))
        self.assertNotIn("env", m)
        self.assertNotIn("environ", m)

    def test_per_rank_identity_is_in_the_rank_digest_but_not_the_group_digest(self):
        # Ranks of one role differ only by rank_id; if that leaked into the group digest every rank
        # would look like a mixed configuration and the startup comparison would be useless.
        a = self.manifest(decode_snapshot(rank_id=0))
        b = self.manifest(decode_snapshot(rank_id=1))
        self.assertNotEqual(a["config_digest"], b["config_digest"])
        self.assertEqual(a["group_config_digest"], b["group_config_digest"])

    def test_a_real_configuration_difference_does_change_the_group_digest(self):
        a = self.manifest(decode_snapshot(rank_id=0))
        b = self.manifest(decode_snapshot(rank_id=1, kernel_seq_size_per_block=128))
        self.assertNotEqual(a["group_config_digest"], b["group_config_digest"])

    def test_an_unresolved_field_is_visible_in_the_manifest(self):
        m = self.manifest(decode_snapshot(concurrency_limit=rp.UNSET))
        self.assertIsNone(m["config"]["concurrency_limit"])


class GroupConsistency(unittest.TestCase):
    def group(self, decode_artifact=None, prefill_artifact=None, **decode_overrides):
        manifests = {}
        for rank in range(4):
            manifests[f"decode-{rank}"] = rp.build_manifest(
                decode_snapshot(rank_id=rank, **decode_overrides), PROFILE, decode_artifact)
        for rank in range(4):
            manifests[f"prefill-{rank}"] = rp.build_manifest(
                prefill_snapshot(rank_id=rank), PROFILE, prefill_artifact)
        return manifests

    def test_a_consistent_group_has_no_violations(self):
        art = {"git_sha": "abc"}
        self.assertEqual([], rp.check_group_consistency(self.group(art, art)))

    def test_the_candidate_token_envelope_gap_is_surfaced_but_not_silently_accepted(self):
        # The candidate group really does have prefill advertising more than decode can finish. That is
        # not rejected here, because the prefill value sizes its own workspace and the bound that
        # protects a request is the public/ingress envelope, which the token-limit work package adds.
        # What must not happen is the gap disappearing from view, so it is reported explicitly.
        art = {"git_sha": "abc"}
        group = self.group(art, art)
        self.assertEqual([], rp.check_group_consistency(group))
        self.assertEqual((133120, 32832), rp.group_token_envelope(group))

    def test_the_envelope_rule_rejects_once_enforcement_is_enabled(self):
        art = {"git_sha": "abc"}
        violations = rp.check_group_consistency(self.group(art, art), enforce_token_envelope=True)
        self.assertTrue(any("TOKEN ENVELOPE MISMATCH" in v for v in violations), violations)

    def test_mixed_artifacts_are_detected(self):
        violations = rp.check_group_consistency(
            self.group({"git_sha": "abc"}, {"git_sha": "def"}))
        self.assertTrue(any("MIXED ARTIFACTS" in v for v in violations), violations)

    def test_mixed_configuration_within_a_role_is_detected(self):
        manifests = self.group({"git_sha": "abc"}, {"git_sha": "abc"})
        # One decode rank resolved a different geometry: exactly the "mixed configuration" hazard.
        manifests["decode-2"] = rp.build_manifest(
            decode_snapshot(rank_id=2, kernel_seq_size_per_block=128), PROFILE, {"git_sha": "abc"})
        violations = rp.check_group_consistency(manifests)
        self.assertTrue(any("MIXED CONFIGURATION" in v for v in violations), violations)

    def test_token_envelope_mismatch_is_detected(self):
        # Prefill advertising more than decode can finish is the defect that lets a request in that
        # decode cannot complete.
        manifests = {}
        for rank in range(4):
            manifests[f"decode-{rank}"] = rp.build_manifest(
                decode_snapshot(rank_id=rank, max_seq_len=32832), PROFILE, {"a": 1})
            manifests[f"prefill-{rank}"] = rp.build_manifest(
                prefill_snapshot(rank_id=rank, max_seq_len=133120), PROFILE, {"a": 1})
        violations = rp.check_group_consistency(manifests, enforce_token_envelope=True)
        self.assertTrue(any("TOKEN ENVELOPE MISMATCH" in v for v in violations), violations)

    def test_rank_ids_may_repeat_across_roles_but_not_within_one(self):
        # Each leg of a PD group numbers its ranks from 0, so 0..3 in both roles is normal.
        art = {"git_sha": "abc"}
        self.assertEqual([], rp.check_group_consistency(self.group(art, art)))
        # Two decode ranks claiming the same slot is not.
        manifests = self.group(art, art)
        manifests["decode-3"] = rp.build_manifest(decode_snapshot(rank_id=2), PROFILE, art)
        violations = rp.check_group_consistency(manifests)
        self.assertTrue(any("DUPLICATE rank ids among ranks with role" in v for v in violations), violations)

    def test_empty_input_is_a_violation(self):
        self.assertEqual(["no rank manifests supplied"], rp.check_group_consistency({}))


class CanonicalSettingsArtifact(unittest.TestCase):
    """The checked-in runnable settings must be exactly what the profile describes.

    A benchmark or deployment consumes the JSON while validation uses the profile definition; if the two
    could drift, a run could be launched with settings the validator never agreed to, which is the
    failure this artifact exists to prevent.
    """

    def test_the_checked_in_json_matches_the_profile(self):
        path = rp.profile_json_path(PROFILE.name)
        self.assertTrue(os.path.exists(path), f"canonical settings artifact is missing: {path}")
        with open(path) as f:
            on_disk = f.read()
        self.assertEqual(
            rp.dump_profile_json(PROFILE.name),
            on_disk,
            "the canonical settings artifact has drifted from the profile definition; regenerate it",
        )

    def test_the_json_parses_and_carries_both_legs(self):
        with open(rp.profile_json_path(PROFILE.name)) as f:
            payload = json.load(f)
        self.assertEqual(PROFILE.name, payload["profile"])
        for leg in ("decode", "prefill"):
            self.assertIn(leg, payload["per_leg"])
        self.assertEqual(PROFILE.decode_fixed_bs, payload["per_leg"]["decode"]["fixed_bs"])
        self.assertEqual(PROFILE.decode_fixed_bs, payload["per_leg"]["decode"]["concurrency_limit"])
        self.assertIn(PROFILE.decode_fixed_bs, payload["per_leg"]["decode"]["capture_batch_sizes"])
        self.assertTrue(payload["per_leg"]["decode"]["enable_cuda_graph"])
        self.assertEqual(PROFILE.prefill_max_batch_tokens_size, payload["per_leg"]["prefill"]["max_batch_tokens_size"])

    def test_the_canonical_settings_are_self_consistent_with_the_validator(self):
        # Every value the artifact publishes must be one the validator accepts, so a runner that applies
        # the artifact verbatim cannot be rejected for a value another field contradicts.
        self.assertEqual(
            PROFILE.expected_decode_ranks, PROFILE.expected_prefill_ranks
        )
        self.assertLessEqual(PROFILE.prefill_max_batch_tokens_size, rp.A2A_PAYLOAD_TOKEN_BOUND)


class EnforceReleaseProfile(unittest.TestCase):
    def stub_config(self):
        return SnapshotFromResolvedConfig().stub_config()

    def test_developer_mode_is_untouched_when_no_profile_is_selected(self):
        self.assertIsNone(rp.enforce_release_profile(self.stub_config(), env={}))

    def test_an_unknown_profile_is_rejected(self):
        with self.assertRaises(rp.ReleaseProfileError) as ctx:
            rp.enforce_release_profile(self.stub_config(), env={rp.RELEASE_PROFILE_ENV: "nope"})
        self.assertIn("not a known release profile", str(ctx.exception))

    def test_a_valid_decode_rank_returns_its_manifest(self):
        manifest = rp.enforce_release_profile(
            self.stub_config(),
            env={rp.RELEASE_PROFILE_ENV: "sm120_dp4ep4_n2", rp.DECODE_FIXED_BS_ENV: "2"},
            artifact_ids={"git_sha": "abc"},
        )
        self.assertIsNotNone(manifest)
        self.assertEqual("sm120_dp4ep4_n2", manifest["profile"])
        self.assertEqual(64, len(manifest["manifest_digest"]))

    def test_a_violation_raises_and_lists_every_reason(self):
        with self.assertRaises(rp.ReleaseProfileError) as ctx:
            rp.enforce_release_profile(
                self.stub_config(), env={rp.RELEASE_PROFILE_ENV: "sm120_dp4ep4_n2"}
            )
        message = str(ctx.exception)
        self.assertIn("padding must be enabled", message)
        self.assertIn("refusing to start", message)


class GroupRendezvous(unittest.TestCase):
    """Startup membership and provenance: bounded, attempt-scoped, group-wide.

    A rank must not begin loading a model until the profile's declared membership is present with
    agreeing configuration and artifact digests. A peer that never publishes has to stop every rank
    (not leave the survivors entering capture alone), and a record from an earlier start must never be
    mistaken for this one's.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rp_group_")
        self.artifact = {"git_sha": "abc"}

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def publish_all(self, attempt="a1", skip=(), artifact_for=None):
        artifact_for = artifact_for or {}
        manifests = []
        for rank in range(PROFILE.expected_decode_ranks):
            if ("DECODE", rank) in skip:
                continue
            m = rp.build_manifest(
                decode_snapshot(rank_id=rank), PROFILE,
                artifact_for.get(("DECODE", rank), self.artifact))
            rp.publish_rank_manifest(m, self.dir, attempt)
            manifests.append(m)
        for rank in range(PROFILE.expected_prefill_ranks):
            if ("PREFILL", rank) in skip:
                continue
            m = rp.build_manifest(
                prefill_snapshot(rank_id=rank), PROFILE,
                artifact_for.get(("PREFILL", rank), self.artifact))
            rp.publish_rank_manifest(m, self.dir, attempt)
            manifests.append(m)
        return manifests

    def join_env(self):
        return {rp.GROUP_DIR_ENV: self.dir, rp.GROUP_ATTEMPT_ENV: "a1"}

    def test_published_manifest_is_attempt_scoped_and_round_trips(self):
        manifest = rp.build_manifest(decode_snapshot(rank_id=1), PROFILE, self.artifact)
        path = rp.publish_rank_manifest(manifest, self.dir, "attempt-7")
        self.assertTrue(path.endswith("attempt-7__RoleType-DECODE__1.json"), path)
        with open(path) as f:
            self.assertEqual(manifest, json.load(f))
        # No temporary file is left behind for a peer to trip over.
        self.assertEqual([os.path.basename(path)], sorted(os.listdir(self.dir)))

    def test_a_complete_group_is_verified_without_violations(self):
        self.publish_all()
        manifests, stale, timed_out, duplicates = rp.collect_group_manifests(
            self.dir, "a1", expected_total=8, timeout_s=0)
        self.assertEqual(8, len(manifests))
        self.assertFalse(timed_out)
        self.assertEqual(0, stale)
        self.assertEqual({}, duplicates)
        self.assertEqual([], rp.check_group_consistency(manifests))

    def test_a_missing_peer_is_reported_after_a_bounded_wait(self):
        self.publish_all(skip={("DECODE", 3)})
        now = [0.0]
        slept = []

        def fake_sleep(seconds):
            slept.append(seconds)
            now[0] += seconds

        manifests, _stale, timed_out, duplicates = rp.collect_group_manifests(
            self.dir, "a1", expected_total=8, timeout_s=0.5,
            clock=lambda: now[0], sleep=fake_sleep)
        self.assertTrue(timed_out)
        self.assertEqual(7, len(manifests))
        self.assertEqual({}, duplicates)
        # The wait is bounded by the budget, not by the peer eventually appearing.
        self.assertLessEqual(len(slept), 3, slept)
        self.assertGreaterEqual(now[0], 0.5)

    def test_records_from_another_attempt_are_ignored_and_counted(self):
        self.publish_all(attempt="old-run")
        self.publish_all(attempt="current")
        manifests, stale, timed_out, _dup = rp.collect_group_manifests(
            self.dir, "current", expected_total=8, timeout_s=0)
        self.assertFalse(timed_out)
        self.assertEqual(8, len(manifests))
        self.assertEqual(8, stale, "the earlier attempt's manifests must be ignored, not merged")

    def test_a_second_claimant_cannot_overwrite_the_incumbent(self):
        # The audit's probe: eight ranks published, then a ninth claimant for decode rank 0. The slot has
        # exactly one publisher, so the claim must fail and the incumbent's manifest must survive.
        self.publish_all()
        with open(os.path.join(self.dir, "a1__RoleType-DECODE__0.json")) as f:
            incumbent = json.load(f)
        impostor = rp.build_manifest(decode_snapshot(rank_id=0), PROFILE, {"git_sha": "impostor"})
        with self.assertRaises(rp.ReleaseProfileError) as ctx:
            rp.publish_rank_manifest(impostor, self.dir, "a1")
        self.assertIn("already claimed", str(ctx.exception))
        with open(os.path.join(self.dir, "a1__RoleType-DECODE__0.json")) as f:
            self.assertEqual(incumbent, json.load(f))

    def test_a_duplicate_claim_by_a_peer_is_preserved_and_rejected(self):
        # A peer that bypassed the exclusive claim (a copied file, another publisher) must not be merged
        # away: the collision itself is the violation.
        self.publish_all()
        role, rank = "RoleType.DECODE", 0
        copy = os.path.join(self.dir, "a1__RoleType-DECODE__0-copy.json")
        with open(os.path.join(self.dir, "a1__RoleType-DECODE__0.json")) as f:
            manifest = json.load(f)
        with open(copy, "w") as f:
            json.dump(manifest, f)
        manifests, _stale, _timed_out, duplicates = rp.collect_group_manifests(
            self.dir, "a1", expected_total=8, timeout_s=0)
        self.assertEqual([(role, rank)], list(duplicates))
        violations = rp.check_group_consistency(manifests, duplicates=duplicates)
        self.assertTrue(any("DUPLICATE RANK OWNERSHIP" in v for v in violations), violations)
        self.assertTrue(any(f"{role}#{rank}" in v for v in violations), violations)

    def test_a_non_finite_join_timeout_is_rejected(self):
        # nan never compares greater than the clock and inf never expires: both turn the bounded
        # rendezvous into an unbounded wait, so neither is accepted as a budget.
        for bad in ("nan", "inf", "-inf"):
            with self.assertRaises(rp.ReleaseProfileError) as ctx:
                rp.collect_group_manifests(
                    self.dir, "a1", expected_total=8,
                    env={rp.GROUP_TIMEOUT_ENV: bad}, clock=lambda: 1.0, sleep=lambda s: None)
            self.assertIn("finite", str(ctx.exception))
        with self.assertRaises(rp.ReleaseProfileError):
            rp.collect_group_manifests(
                self.dir, "a1", expected_total=8, timeout_s=float("nan"),
                clock=lambda: 1.0, sleep=lambda s: None)

    def test_join_without_a_configured_directory_is_a_violation(self):
        manifest = rp.build_manifest(decode_snapshot(rank_id=0), PROFILE, self.artifact)
        violations = rp.join_group_and_verify(manifest, env={}, attempt="a1", timeout_s=0)
        self.assertTrue(any("GROUP VERIFICATION NOT CONFIGURED" in v for v in violations), violations)

    def test_join_without_an_attempt_id_is_a_violation(self):
        manifest = rp.build_manifest(decode_snapshot(rank_id=0), PROFILE, self.artifact)
        violations = rp.join_group_and_verify(
            manifest, env={rp.GROUP_DIR_ENV: self.dir}, timeout_s=0)
        self.assertTrue(any("ATTEMPT" in v for v in violations), violations)

    def test_an_incomplete_group_stops_the_rank(self):
        # Two ranks never publish; the joining rank (which claims the last free slot) must time out
        # rather than proceed with a partial group.
        self.publish_all(skip={("PREFILL", 3), ("DECODE", 2), ("DECODE", 3)})
        manifest = rp.build_manifest(decode_snapshot(rank_id=3), PROFILE, self.artifact)
        violations = rp.join_group_and_verify(manifest, env=self.join_env(), timeout_s=0)
        self.assertTrue(any("GROUP INCOMPLETE" in v for v in violations), violations)

    def test_a_rank_built_from_another_artifact_stops_the_rank(self):
        self.publish_all(
            skip={("DECODE", 3)},
            artifact_for={("DECODE", 2): {"git_sha": "different"}},
        )
        manifest = rp.build_manifest(decode_snapshot(rank_id=3), PROFILE, self.artifact)
        violations = rp.join_group_and_verify(manifest, env=self.join_env(), timeout_s=0)
        self.assertTrue(any("MIXED ARTIFACTS" in v for v in violations), violations)

    def test_a_rank_outside_the_declared_membership_stops_the_rank(self):
        self.publish_all(skip={("DECODE", 3)})
        extra = rp.build_manifest(decode_snapshot(rank_id=9), PROFILE, self.artifact)
        rp.publish_rank_manifest(extra, self.dir, "a1")
        manifest = rp.build_manifest(decode_snapshot(rank_id=3), PROFILE, self.artifact)
        violations = rp.join_group_and_verify(manifest, env=self.join_env(), timeout_s=0)
        self.assertTrue(any("MEMBERSHIP" in v for v in violations), violations)

    def test_join_group_is_opt_in_on_the_enforce_path(self):
        # Developer mode stays untouched, and the parent leg process (no rank identity) must not join.
        stub = SnapshotFromResolvedConfig().stub_config()
        self.assertIsNone(rp.enforce_release_profile(stub, env={}))
        with self.assertRaises(rp.ReleaseProfileError) as ctx:
            rp.enforce_release_profile(
                stub,
                env={rp.RELEASE_PROFILE_ENV: "sm120_dp4ep4_n2", rp.DECODE_FIXED_BS_ENV: "2"},
                artifact_ids={"git_sha": "abc"},
                join_group=True,
            )
        self.assertIn("GROUP VERIFICATION NOT CONFIGURED", str(ctx.exception))


class ContentIdentity(unittest.TestCase):
    """Artifact identity must be CONTENT, and must name the binary that actually runs.

    The audit's probe: identity used size+mtime of the Python initializer, so a different library or a
    different checkpoint could pass as identical. These tests pin content hashing, the sampled/full
    distinction, coverage enforcement for a packaging-supplied digest list, and that the discovered
    library paths are the compiled shared objects rather than a .py file.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rp_content_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, data):
        if isinstance(data, str):
            data = data.encode()
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_same_size_and_mtime_but_different_content_differs(self):
        a = self.write("a.bin", b"A" * 4096)
        b = self.write("b.bin", b"B" * 4096)
        os.utime(a, (1_700_000_000, 1_700_000_000))
        os.utime(b, (1_700_000_000, 1_700_000_000))
        ia = rp._content_digest(a, 10 * 1024 * 1024, 1024)
        ib = rp._content_digest(b, 10 * 1024 * 1024, 1024)
        self.assertEqual(ia["size"], ib["size"])
        self.assertEqual("full", ia["mode"])
        self.assertNotEqual(ia["sha256"], ib["sha256"])

    def test_a_large_file_is_marked_sampled_and_a_small_one_full(self):
        small = self.write("small.bin", b"x" * 1024)
        large = self.write("large.bin", b"y" * 8192)
        self.assertEqual("full", rp._content_digest(small, 4096, 512)["mode"])
        sampled = rp._content_digest(large, 4096, 512)
        self.assertEqual("sampled", sampled["mode"])
        self.assertEqual(512, sampled["sample_bytes"])

    def test_a_tree_identity_changes_when_a_file_changes(self):
        self.write("config.json", b'{"a": 1}')
        self.write("shard-1.safetensors", b"weights-1" * 1000)
        first = rp._tree_identity(self.dir, 4096, 256)
        self.write("shard-1.safetensors", b"weights-2" * 1000)
        second = rp._tree_identity(self.dir, 4096, 256)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["file_count"], second["file_count"])

    def test_a_weights_manifest_must_cover_every_file(self):
        self.write("config.json", b"{}")
        shard = self.write("shard-1.safetensors", b"w" * 4096)
        tree = rp._tree_identity(self.dir, 1024, 256)
        manifest = self.write(
            "weights.sha256",
            "%s  shard-1.safetensors\n" % ("0" * 64),
        )
        ident, errors = rp._weights_manifest(manifest, tree)
        self.assertTrue(any("does not cover" in e for e in errors), errors)
        self.assertEqual(1, ident["declared_files"])
        self.assertFalse(ident["verified"])
        # A manifest that covers the shard but not the metadata file is still incomplete.
        with open(manifest, "a") as f:
            f.write("%s  config.json\n" % ("1" * 64))
        _ident, errors2 = rp._weights_manifest(manifest, tree)
        self.assertEqual([], errors2)
        self.assertTrue(shard)

    def test_a_malformed_digest_line_is_an_error(self):
        self.write("config.json", b"{}")
        tree = rp._tree_identity(self.dir, 1024, 256)
        manifest = self.write("weights.sha256", "not-a-digest  config.json\n")
        _ident, errors = rp._weights_manifest(manifest, tree)
        self.assertTrue(any("malformed" in e for e in errors), errors)

    def test_discovered_libraries_are_compiled_shared_objects(self):
        selected, shadowed, errors = rp._resolve_libraries()
        self.assertTrue(selected, "no library path was discovered in the deployed layout")
        self.assertTrue(all(p.endswith(".so") for p in selected.values()), selected)
        self.assertTrue(any("libth_transformer" in n for n in selected), selected)
        self.assertTrue(any("librtp_compute_ops" in n for n in selected), selected)
        self.assertEqual([], errors, "the deployed layout must not be ambiguous")
        self.assertEqual({}, {k: v for k, v in shadowed.items() if v})

    def test_shadowed_library_copies_are_an_error_not_an_overwrite(self):
        # Two different files with one basename: the loader picks one, so the manifest cannot claim to
        # describe the running binary, and the FIRST copy must not be silently replaced by the second.
        a = self.write("one/libth_transformer.so", b"first")
        b = self.write("two/libth_transformer.so", b"second")
        selected, shadowed, errors = rp._select_libraries([a, b])
        self.assertEqual(a, selected["libth_transformer.so"])
        self.assertEqual([b], shadowed["libth_transformer.so"])
        self.assertTrue(any("more than one distinct file" in e for e in errors), errors)
        # A symlink alias of the SAME file is not a second copy.
        alias = os.path.join(self.dir, "alias", "libth_transformer.so")
        os.makedirs(os.path.dirname(alias))
        os.symlink(a, alias)
        selected2, shadowed2, errors2 = rp._select_libraries([a, alias])
        self.assertEqual(a, selected2["libth_transformer.so"])
        self.assertEqual([], shadowed2["libth_transformer.so"])
        self.assertEqual([], errors2)

    def test_a_file_that_cannot_be_read_fails_the_tree(self):
        self.write("config.json", b"{}")
        shard = self.write("shard.safetensors", b"w" * 4096)
        tree = rp._tree_identity(self.dir, 1024, 256)
        self.assertTrue(tree["complete"])
        self.assertNotIn("errors", tree)
        # A dangling symlink cannot be stat()ed whatever the uid, unlike a chmod-0 file for root.
        os.symlink(os.path.join(self.dir, "missing-target"), shard + ".dangling")
        broken = rp._tree_identity(self.dir, 1024, 256)
        self.assertFalse(broken["complete"])
        self.assertEqual("incomplete", broken["mode"])
        self.assertTrue(any("could not be read" in e for e in broken["errors"]), broken["errors"])

    def test_the_full_inventory_is_kept_when_the_listing_is_capped(self):
        for i in range(rp.MAX_LISTED_FILES + 2):
            self.write(f"f{i:03d}.bin", b"x" * 32)
        tree = rp._tree_identity(self.dir, 1024, 256)
        self.assertEqual(rp.MAX_LISTED_FILES + 2, tree["file_count"])
        self.assertEqual(rp.MAX_LISTED_FILES + 2, len(tree["files"]), "inventory must be complete")
        report = rp._tree_for_report(tree)
        self.assertEqual(rp.MAX_LISTED_FILES, len(report["files"]))
        self.assertEqual(rp.MAX_LISTED_FILES + 2, report["files_listed"])
        self.assertEqual(tree["sha256"], report["sha256"])

    def test_an_empty_manifest_over_a_capped_tree_is_rejected(self):
        # The audit's probe: 66 sampled files + an empty manifest passed strict mode with covered: 0.
        for i in range(rp.MAX_LISTED_FILES + 2):
            self.write(f"shard-{i:03d}.safetensors", b"w" * 32)
        manifest = self.write("weights.sha256", "# empty\n")
        tree = rp._tree_identity(self.dir, 1024, 256)
        ident, errors = rp._weights_manifest(manifest, tree)
        self.assertTrue(any("does not cover" in e for e in errors), errors)
        self.assertEqual(0, ident["covered"])
        self.assertFalse(ident["verified"])

    def test_declared_digests_are_not_proof_until_verified(self):
        payload = b"weights" * 1000
        shard = self.write("shard.safetensors", payload)
        tree = rp._tree_identity(self.dir, 1024, 256)
        zero = self.write("zeros.sha256", "%s  shard.safetensors\n" % ("0" * 64))
        ident, errors = rp._weights_manifest(zero, tree, verify_full=True)
        self.assertFalse(ident["verified"])
        self.assertTrue(any("do not match" in e for e in errors), errors)
        # A correct declaration over the whole tree verifies.
        import hashlib as _h
        good = self.write("good.sha256", "%s  shard.safetensors\n" % _h.sha256(payload).hexdigest())
        ident2, errors2 = rp._weights_manifest(good, tree, verify_full=True)
        self.assertEqual([], errors2)
        self.assertTrue(ident2["verified"])
        self.assertEqual(1, ident2["verified_files"])
        self.assertTrue(shard)

    def test_strict_mode_rejects_an_unverified_declaration(self):
        stub = SnapshotFromResolvedConfig().stub_config()
        model = os.path.join(self.dir, "model")
        os.makedirs(model)
        with open(os.path.join(model, "shard.safetensors"), "wb") as f:
            f.write(b"w" * 8192)
        stub.model_args.ckpt_path = model
        zeros = os.path.join(self.dir, "zero.sha256")
        with open(zeros, "w") as f:
            f.write("%s  shard.safetensors\n" % ("0" * 64))
        ids = rp.artifact_identity_from(stub, {
            rp.FULL_HASH_MAX_ENV: "0.001",
            rp.REQUIRE_FULL_WEIGHTS_ENV: "1",
            rp.WEIGHTS_MANIFEST_ENV: zeros,
        })
        self.assertFalse(ids["model"]["weights_manifest"]["verified"])
        self.assertTrue(any("verified full content digest" in e for e in ids.get("errors", [])), ids)
        self.assertEqual("sampled", ids["model"]["mode"])

    def test_require_full_weights_rejects_a_sampled_identity(self):
        stub = SnapshotFromResolvedConfig().stub_config()
        model = os.path.join(self.dir, "model")
        os.makedirs(model)
        with open(os.path.join(model, "config.json"), "wb") as f:
            f.write(b"{}")
        with open(os.path.join(model, "shard.safetensors"), "wb") as f:
            f.write(b"w" * 8192)
        stub.model_args.ckpt_path = model
        env = {rp.FULL_HASH_MAX_ENV: "0.001", rp.SAMPLE_BYTES_ENV: "0.0005",
               rp.REQUIRE_FULL_WEIGHTS_ENV: "1"}
        ids = rp.artifact_identity_from(stub, env)
        self.assertTrue(any("requires a verified full content digest" in e for e in ids.get("errors", [])),
                        ids.get("errors"))
        ids2 = rp.artifact_identity_from(stub, {rp.FULL_HASH_MAX_ENV: "1"})
        self.assertFalse(
            [e for e in ids2.get("errors", []) if "requires a verified full content digest" in e], ids2)
        self.assertEqual("full", ids2["model"]["mode"])


if __name__ == "__main__":
    unittest.main()
