"""flexlb_cfg unit tests (SSOT render contracts).

Golden hashes pin the five base profiles byte-for-byte (any intentional
value change must update the hash WITH a deliberate reason — the golden
diff is the migration safety net).  Structural asserts spell out the
layering (base < profile < override), the OMIT sentinel, the
"existing-fields-only" rule and the shell override parser.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from flexlb_cfg import (  # noqa: E402
    DSV4_PREFILL_EXPRESSION,
    OMIT,
    PROFILES,
    STRESS_PROFILE,
    ConfigOverride,
    parse_overrides,
    render_env,
    render_process_config,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class GoldenProfileTest(unittest.TestCase):
    """Base render snapshots (byte-for-byte, sha256 pinned)."""

    GOLDEN = {
        "batch-window": "b5313755f180838cbe6cd3e39b83909fb934ad9ee61c74517fb3f5fa2077691b",
        "single-nonbatch": "0e4154649598be46bbc7ca4d3c332a5921db1b553c905db0ebef12255942ea3b",
        "single-batch": "17008ef16c95cefcbd32ff931a3a49e886a6b32bd9df88d1c9d1d73ced6ce30e",
        "window-nonbatch": "46a24d13d8f495c98f905767d2509ffea514b9bcb55a68c6566db0b4f57c86a4",
        # the former data/config/master_fixed_window.json FLEXLB_CONFIG
        "stress-na130": "d3b763a3ca18e3df2d3a533ad51de1f0177134f7194789be540e7b8796979a81",
    }

    def test_functional_golden(self) -> None:
        for profile in PROFILES:
            with self.subTest(profile=profile):
                self.assertEqual(self.GOLDEN[profile], sha(render_env(profile)))

    def test_stress_golden(self) -> None:
        self.assertEqual(self.GOLDEN[STRESS_PROFILE], sha(render_env(STRESS_PROFILE)))

    def test_stress_golden_fields(self):
        doc = json.loads(render_env(STRESS_PROFILE))
        self.assertEqual(3, doc["schemaVersion"])
        self.assertNotIn("capacity", doc["scheduler"])
        self.assertNotIn("lifecycle", doc["scheduler"])
        self.assertEqual(
            {"request": {"timeoutMs": 300000}, "decision": {"lifetime": 2.0}},
            doc["requestLifecycle"],
        )
        self.assertEqual(
            {"type": "BATCH", "maxInflightPerPrefillWorker": 2}, doc["dispatcher"]
        )
        self.assertNotIn("candidateChoice", doc["router"]["roles"]["prefill"])
        self.assertEqual(
            {"availability": {"maxKvUsagePercent": 95, "maxEngineRequests": 384}},
            doc["router"]["roles"]["decode"],
        )
        self.assertEqual(3000, doc["workerRegistry"]["health"]["cleanupIntervalMs"])
        self.assertEqual(
            DSV4_PREFILL_EXPRESSION,
            doc["router"]["roles"]["prefill"]["executionTimeEstimator"]["expression"],
        )

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(ValueError):
            render_env("no-such-profile")


class LayeringTest(unittest.TestCase):
    """base < profile < override one-way layering."""

    def test_override_beats_profile_value(self) -> None:
        base = json.loads(render_env("batch-window"))
        self.assertEqual(60000, base["scheduler"]["queueTimeoutMs"])
        doc = json.loads(
            render_env(
                "batch-window",
                ConfigOverride(queue_timeout_ms=1500, request_timeout_ms=12345),
            )
        )
        self.assertEqual(1500, doc["scheduler"]["queueTimeoutMs"])
        self.assertEqual(12345, doc["requestLifecycle"]["request"]["timeoutMs"])
        # untouched profile keys survive
        self.assertEqual(base["scheduler"]["decision"], doc["scheduler"]["decision"])

    def test_none_keeps_profile_value(self) -> None:
        doc = json.loads(render_env("batch-window", ConfigOverride()))
        self.assertEqual(60000, doc["scheduler"]["queueTimeoutMs"])
        self.assertEqual(2, doc["dispatcher"]["maxInflightPerPrefillWorker"])

    def test_omit_removes_key(self) -> None:
        doc = json.loads(
            render_env("batch-window", ConfigOverride(queue_timeout_ms=OMIT))
        )
        self.assertNotIn("queueTimeoutMs", doc["scheduler"])

    def test_omit_on_always_present_field_raises(self) -> None:
        with self.assertRaises(ValueError):
            render_env("batch-window", ConfigOverride(request_timeout_ms=OMIT))

    def test_priority_axes(self) -> None:
        doc = json.loads(
            render_env(
                "batch-window",
                ConfigOverride(
                    ordering="priority",
                    default_priority=50,
                    preemption={
                        "allowed_victim_stages": [
                            "PREFILL_QUEUED",
                            "DECODE_RESERVED",
                            "DECODE_ENGINE_OWNED",
                        ],
                        "timeout_ms": 1000,
                    },
                    decode_max_engine_requests=1,
                ),
            )
        )
        ordering = doc["scheduler"]["ordering"]
        self.assertEqual("PRIORITY", ordering["type"])
        self.assertEqual(50, ordering["defaultPriority"])
        self.assertEqual(
            {
                "allowedVictimStages": [
                    "PREFILL_QUEUED",
                    "DECODE_RESERVED",
                    "DECODE_ENGINE_OWNED",
                ],
                "timeoutMs": 1000,
            },
            ordering["preemption"],
        )
        self.assertEqual(
            1,
            doc["router"]["roles"]["decode"]["availability"]["maxEngineRequests"],
        )

    def test_fifo_rejects_priority_keys(self) -> None:
        with self.assertRaises(ValueError):
            render_env(
                "batch-window",
                ConfigOverride(ordering="fifo", default_priority=50),
            )

    def test_functional_kv_limit_is_explicit(self):
        doc = json.loads(
            render_env("batch-window", ConfigOverride(decode_max_kv_usage_percent=85))
        )
        self.assertEqual(
            85, doc["router"]["roles"]["decode"]["availability"]["maxKvUsagePercent"]
        )


class StressOverrideTest(unittest.TestCase):
    """stress-na130 override edit-in-place contracts."""

    def test_decode_max_engine_requests(self) -> None:
        doc = json.loads(
            render_env(STRESS_PROFILE, ConfigOverride(decode_max_engine_requests=5000))
        )
        self.assertEqual(
            5000,
            doc["router"]["roles"]["decode"]["availability"]["maxEngineRequests"],
        )

    def test_strip_preemption(self) -> None:
        doc = json.loads(
            render_env(STRESS_PROFILE, ConfigOverride(strip_preemption=True))
        )
        self.assertNotIn("preemption", doc["scheduler"]["ordering"])

    def test_strip_preemption_functional_noop(self) -> None:
        # functional bases have no preemption block to strip
        doc = json.loads(
            render_env("batch-window", ConfigOverride(strip_preemption=True))
        )
        self.assertNotIn("preemption", doc["scheduler"]["ordering"])

    def test_queue_timeout_omit(self) -> None:
        doc = json.loads(
            render_env(STRESS_PROFILE, ConfigOverride(queue_timeout_ms=OMIT))
        )
        self.assertNotIn("queueTimeoutMs", doc["scheduler"])

    def test_kv_usage_percent_edit(self) -> None:
        doc = json.loads(
            render_env(STRESS_PROFILE, ConfigOverride(decode_max_kv_usage_percent=80))
        )
        self.assertEqual(
            80,
            doc["router"]["roles"]["decode"]["availability"]["maxKvUsagePercent"],
        )

    def test_dispatcher_non_batch_drops_batch_keys(self) -> None:
        doc = json.loads(
            render_env(
                STRESS_PROFILE,
                ConfigOverride(
                    dispatcher="non_batch", max_inflight_per_prefill_worker=1
                ),
            )
        )
        self.assertEqual(
            {
                "type": "NON_BATCH",
                "maxInflightPerPrefillWorker": 1,
            },
            doc["dispatcher"],
        )

    def test_unified_limit_applies_to_both_dispatchers(self):
        for dispatcher in ("batch", "non_batch"):
            doc = json.loads(
                render_env(
                    STRESS_PROFILE,
                    ConfigOverride(
                        dispatcher=dispatcher, max_inflight_per_prefill_worker=7
                    ),
                )
            )
            self.assertEqual(7, doc["dispatcher"]["maxInflightPerPrefillWorker"])

    def test_removed_knobs_fail_closed(self):
        for key in (
            "max_outstanding",
            "max_inflight_batches",
            "enqueue_rpc_timeout_ms",
            "stale_inflight_ms",
        ):
            with self.assertRaises(TypeError):
                ConfigOverride(**{key: 1})
            with self.assertRaises(ValueError):
                parse_overrides(f"{key}=1")

    def test_decision_single_drops_window_keys(self) -> None:
        doc = json.loads(render_env(STRESS_PROFILE, ConfigOverride(decision="single")))
        self.assertEqual({"type": "SINGLE"}, doc["scheduler"]["decision"])


class ProcessConfigTest(unittest.TestCase):
    """envelope projection (single render, two projections)."""

    def test_stress_envelope_golden(self) -> None:
        # byte-for-byte the retired data/config/master_fixed_window.json
        self.assertEqual(
            "7d0cf2e99c11d92cac417baea73355096a291e99b7a834f08b3b14c9435ae4d0",
            sha(render_process_config(STRESS_PROFILE)),
        )

    def test_envelope_structure(self) -> None:
        text = render_process_config("batch-window", jvm_heap="16g")
        doc = json.loads(text)
        self.assertEqual("master", doc["zone_name"])
        envs = doc["zone_process_setting"]["process_info"]["envs"]
        self.assertEqual(
            ["FLEXLB_CONFIG", "FLEXLB_JVM_HEAP_SIZE"], [e[0] for e in envs]
        )
        self.assertEqual("16g", envs[1][1])
        # env payload == render_env of the same profile
        self.assertEqual(render_env("batch-window"), envs[0][1])

    def test_envelope_env_and_file_cannot_diverge(self) -> None:
        overrides = ConfigOverride(request_timeout_ms=9999)
        envs = json.loads(render_process_config("single-batch", overrides))[
            "zone_process_setting"
        ]["process_info"]["envs"]
        self.assertEqual(render_env("single-batch", overrides), envs[0][1])

    def test_raw_config_projection(self) -> None:
        # negative-test bypass: the raw string lands in the envelope verbatim
        raw = '{"schemaVersion":2,"autoTpmEnabled":true}'
        envs = json.loads(render_process_config("batch-window", raw_config=raw))[
            "zone_process_setting"
        ]["process_info"]["envs"]
        self.assertEqual(raw, envs[0][1])
        self.assertEqual("32g", envs[1][1])


class OverrideParsingTest(unittest.TestCase):
    """FLEXLB_CONFIG_OVERRIDE shell knob parsing."""

    def test_empty(self) -> None:
        self.assertIsNone(parse_overrides(""))
        self.assertIsNone(parse_overrides(None))
        self.assertIsNone(parse_overrides("   "))

    def test_int_and_flag(self) -> None:
        ov = parse_overrides("request_timeout_ms=30000,strip_preemption")
        self.assertEqual(30000, ov.request_timeout_ms)
        self.assertTrue(ov.strip_preemption)

    def test_str_and_bool(self) -> None:
        ov = parse_overrides("ordering=priority,strip_preemption=false")
        self.assertEqual("priority", ov.ordering)
        self.assertFalse(ov.strip_preemption)

    def test_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_overrides("bogus=1")

    def test_bad_int_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_overrides("request_timeout_ms=abc")


class VocabTest(unittest.TestCase):
    """closed vocabulary (frozen dataclass)."""

    def test_unknown_field_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            ConfigOverride(bogus=1)

    def test_frozen(self) -> None:
        ov = ConfigOverride(request_timeout_ms=2)
        with self.assertRaises(Exception):
            ov.max_outstanding = 3


class JavaFixtureLockstepTest(unittest.TestCase):
    """The Java schema guard's fixture must stay in lockstep with the
    Python render (flexlb-mock-engine's ConfigSchemaGuardTest parses the
    snapshot; a render change without refreshing the fixture fails HERE
    before it can surprise the Java side)."""

    FIXTURE = (
        SCRIPT_DIR.parents[1]
        / "flexlb-mock-engine"
        / "src"
        / "test"
        / "resources"
        / "master-config-stress-na130.json"
    )

    def test_fixture_byte_equal(self) -> None:
        self.assertTrue(self.FIXTURE.is_file(), f"missing fixture: {self.FIXTURE}")
        self.assertEqual(
            render_process_config(STRESS_PROFILE, None),
            self.FIXTURE.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
