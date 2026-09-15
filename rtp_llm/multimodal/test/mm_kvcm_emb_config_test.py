import copy
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.multimodal.kvcm import RtpKvMetaObjectConfigError
from rtp_llm.multimodal.kvcm import _config as kvcm_config
from rtp_llm.multimodal.kvcm._config import (
    KVE_INSTANCE_PREFIX,
    RtpKvMetaObjectClientConfig,
)


def _primary_config(**overrides):
    config = {
        "enable_vipserver": False,
        "vipserver_domain": "",
        "instance_group": "shared-group",
        "instance_id": "shared-instance",
        "address": ["127.0.0.1:19001", "127.0.0.2:19001"],
        "block_size": 128,
        "location_spec_infos": {"tp0": 4096},
        "location_spec_groups": {"all": ["tp0"]},
        "meta_channel_config": {
            "retry_time": 3,
            "connection_timeout": 6000,
            "call_timeout": 1500,
        },
        "sdk_config": {
            "thread_num": 4,
            "queue_size": 2000,
            "sdk_backend_configs": [
                {
                    "type": "file",
                    "sdk_log_file_path": "",
                    "sdk_log_level": "ERROR",
                }
            ],
            "timeout_config": {
                "put_timeout_ms": 12_000,
                "get_timeout_ms": 12_000,
            },
        },
        "model_deployment": {
            "model_name": "model-a",
            "dtype": "fp16",
            "use_mla": False,
            "tp_size": 2,
            "dp_size": 1,
            "pp_size": 1,
            "extra": "",
            "user_data": "deployment-data",
        },
        "future_field": {"preserved": True},
    }
    config.update(overrides)
    return config


def _serialized(config_map=None):
    return json.dumps({"": _primary_config()} if config_map is None else config_map)


def _split_reco_config(**overrides):
    values = {
        "reco_client_config": "",
        "reco_enable_vipserver": False,
        "reco_vipserver_domain": "",
        "reco_server_address": "127.0.0.1:19001",
        "reco_instance_group": "shared-group",
        "reco_instance_id_salt": "",
        "reco_meta_channel_retry_time": 3,
        "reco_meta_channel_connection_timeout": 6000,
        "reco_meta_channel_call_timeout": 1500,
        "reco_storage_thread_num": 4,
        "reco_storage_queue_size": 2000,
        "reco_put_timeout_ms": 12_000,
        "reco_get_timeout_ms": 12_000,
        "reco_model_sdk_config": json.dumps(
            [
                {
                    "type": "file",
                    "sdk_log_file_path": "",
                    "sdk_log_level": "ERROR",
                }
            ]
        ),
        "reco_model_user_data": "deployment-data",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _online_env(**overrides):
    values = {
        "RECO_ENABLE_VIPSERVER": "1",
        "RECO_VIPSERVER_DOMAIN": "kvcm-na130-m3-bailian-grpc-2.vipserver",
        "RECO_INSTANCE_GROUP": "pace_group_m3",
        "RECO_PUT_TIMEOUT_MS": "100000",
        "RECO_GET_TIMEOUT_MS": "100000",
        "RECO_MODEL_SDK_CONFIG": json.dumps(
            [
                {
                    "type": "pace",
                    "sdk_log_file_path": "logs/pace_client.log",
                    "sdk_log_level": "INFO",
                }
            ]
        ),
        # Process/SDK variables are intentionally ignored by the RTP adapter.
        "TAIR_MEMPOOL_KMONITOR_SINK_ADDRESS": "127.0.0.1:4141",
        "KVCM_LOG_LEVEL": "INFO",
    }
    values.update(overrides)
    return values


class RtpKvMetaExplicitConfigTest(unittest.TestCase):
    def test_derives_prefixed_identity_and_exact_object_schema(self):
        source = _primary_config()
        original = copy.deepcopy(source)
        kv_cache = _split_reco_config(reco_client_config=_serialized({"": source}))

        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(kv_cache)
        transfer = json.loads(derived.transfer_client_config)

        self.assertEqual(derived.instance_group, "kve_shared-group")
        self.assertEqual(derived.instance_id, "kve_shared-instance")
        self.assertEqual(
            derived.addresses,
            ("127.0.0.1:19001", "127.0.0.2:19001"),
        )
        self.assertEqual(derived.user_data, "deployment-data")
        self.assertEqual(derived.call_timeout_ms, 1500)
        self.assertEqual(derived.write_timeout_seconds, 30)
        self.assertEqual(derived.max_object_bytes, 1024 * 1024 * 1024)
        self.assertEqual(transfer["instance_group"], derived.instance_group)
        self.assertEqual(transfer["instance_id"], derived.instance_id)
        self.assertEqual(transfer["block_size"], 1)
        self.assertEqual(transfer["location_spec_infos"], {"value": 1})
        self.assertEqual(transfer["location_spec_groups"], {})
        self.assertEqual(transfer["address"], list(derived.addresses))
        self.assertFalse(transfer["enable_vipserver"])
        self.assertEqual(transfer["vipserver_domain"], "")
        self.assertEqual(transfer["sdk_config"], source["sdk_config"])
        self.assertEqual(transfer["model_deployment"], source["model_deployment"])
        self.assertEqual(transfer["future_field"], {"preserved": True})
        self.assertEqual(source, original)

    def test_config_repr_does_not_expose_transfer_or_user_data(self):
        secret = "private-deployment-secret"
        source = _primary_config(
            model_deployment={
                **_primary_config()["model_deployment"],
                "user_data": secret,
            },
            future_field={"credential": secret},
        )

        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(
            _split_reco_config(reco_client_config=_serialized({"": source}))
        )

        self.assertNotIn(secret, repr(derived))
        self.assertNotIn("transfer_client_config", repr(derived))
        self.assertNotIn("user_data", repr(derived))

    def test_selects_single_named_or_explicit_empty_key_primary(self):
        single = _split_reco_config(
            reco_client_config=_serialized({"model": _primary_config()})
        )
        self.assertEqual(
            RtpKvMetaObjectClientConfig.from_kv_cache_config(single).instance_id,
            "kve_shared-instance",
        )

        explicit = _split_reco_config(
            reco_client_config=_serialized(
                {
                    "z-model": _primary_config(instance_id="secondary"),
                    "": _primary_config(instance_id="primary"),
                }
            )
        )
        self.assertEqual(
            RtpKvMetaObjectClientConfig.from_kv_cache_config(explicit).instance_id,
            "kve_primary",
        )

    def test_rejects_ambiguous_multiple_named_entries(self):
        value = _serialized(
            {
                "model-a": _primary_config(instance_id="a"),
                "model-b": _primary_config(instance_id="b"),
            }
        )
        with self.assertRaisesRegex(RtpKvMetaObjectConfigError, "empty-key primary"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(
                _split_reco_config(reco_client_config=value)
            )

    def test_rejects_non_strict_json_without_echoing_config_content(self):
        cases = (
            "[]",
            '{"": {"instance_id": "secret", "instance_id": "duplicate"}}',
            '{"": {"number": NaN}}',
            '{"super-secret":',
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(RtpKvMetaObjectConfigError) as caught:
                    RtpKvMetaObjectClientConfig.from_kv_cache_config(
                        _split_reco_config(reco_client_config=value)
                    )
                self.assertNotIn("super-secret", str(caught.exception))
                self.assertNotIn("duplicate", str(caught.exception))

    def test_rejects_empty_map_and_oversized_serialized_result(self):
        with self.assertRaisesRegex(RtpKvMetaObjectConfigError, "empty map"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(
                _split_reco_config(reco_client_config="{}")
            )

        primary = _primary_config(
            block_size=1,
            location_spec_infos={"value": 1},
            location_spec_groups={},
            future_field={"padding": ""},
        )
        limit = 1024 * 1024
        serialized = json.dumps({"": primary}, separators=(",", ":"))
        padding = limit - len(serialized.encode("utf-8"))
        self.assertGreater(padding, 0)
        primary["future_field"]["padding"] = "x" * padding
        serialized = json.dumps({"": primary}, separators=(",", ":"))
        self.assertEqual(len(serialized.encode("utf-8")), limit)
        with self.assertRaisesRegex(
            RtpKvMetaObjectConfigError, "transfer client config exceeds"
        ):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(
                _split_reco_config(reco_client_config=serialized)
            )

    def test_rejects_invalid_explicit_fields(self):
        def mutate(path, value):
            primary = _primary_config()
            target = primary
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = value
            return _split_reco_config(reco_client_config=_serialized({"": primary}))

        cases = (
            (("instance_id",), ""),
            (("instance_group",), 7),
            (("enable_vipserver",), 1),
            (("meta_channel_config",), []),
            (("meta_channel_config", "retry_time"), -1),
            (("meta_channel_config", "connection_timeout"), False),
            (("meta_channel_config", "call_timeout"), True),
            (("meta_channel_config", "call_timeout"), 600_001),
            (("sdk_config",), None),
            (("sdk_config", "thread_num"), 0),
            (("sdk_config", "queue_size"), 63),
            (("sdk_config", "sdk_backend_configs"), {}),
            (("sdk_config", "sdk_backend_configs"), ["not-an-object"]),
            (("sdk_config", "sdk_backend_configs"), [{}]),
            (("sdk_config", "timeout_config"), []),
            (("sdk_config", "timeout_config", "put_timeout_ms"), 0),
            (("sdk_config", "timeout_config", "get_timeout_ms"), False),
            (("model_deployment",), []),
            (("model_deployment", "user_data"), 5),
        )
        for path, value in cases:
            with (
                self.subTest(path=path, value=value),
                self.assertRaises(RtpKvMetaObjectConfigError),
            ):
                RtpKvMetaObjectClientConfig.from_kv_cache_config(mutate(path, value))

    def test_rejects_invalid_static_addresses(self):
        cases = (
            None,
            "127.0.0.1:1",
            [],
            [""],
            [" 127.0.0.1:1"],
            ["127.0.0.1: 1"],
            ["127.0.0.1:\t1"],
            ["127.0.0.1:1\x00"],
            ["127.0.0.1:1", "127.0.0.1:1"],
            ["x"] * 65,
            ["x" * 1025],
        )
        for addresses in cases:
            with self.subTest(addresses=addresses):
                config = _split_reco_config(
                    reco_client_config=_serialized(
                        {"": _primary_config(address=addresses)}
                    )
                )
                with self.assertRaises(RtpKvMetaObjectConfigError):
                    RtpKvMetaObjectClientConfig.from_kv_cache_config(config)

    def test_identity_limits_count_utf8_bytes_after_prefix(self):
        accepted = _split_reco_config(
            reco_client_config=_serialized(
                {"": _primary_config(instance_id="界" * 169)}
            )
        )
        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(accepted)
        self.assertEqual(derived.instance_id, KVE_INSTANCE_PREFIX + "界" * 169)

        rejected = _split_reco_config(
            reco_client_config=_serialized(
                {"": _primary_config(instance_group="界" * 170)}
            )
        )
        with self.assertRaisesRegex(RtpKvMetaObjectConfigError, "exceeds 512"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(rejected)

        invalid_utf8 = _split_reco_config(reco_instance_group="bad\ud800")
        with self.assertRaisesRegex(RtpKvMetaObjectConfigError, "valid UTF-8"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(invalid_utf8)

    def test_write_lease_strictly_exceeds_metadata_and_put_budget(self):
        primary = _primary_config()
        primary["meta_channel_config"]["call_timeout"] = 10_000
        primary["sdk_config"]["timeout_config"]["put_timeout_ms"] = 10_000
        config = _split_reco_config(reco_client_config=_serialized({"": primary}))
        self.assertEqual(
            RtpKvMetaObjectClientConfig.from_kv_cache_config(
                config
            ).write_timeout_seconds,
            41,
        )

        primary["meta_channel_config"]["call_timeout"] = 600_000
        config = _split_reco_config(reco_client_config=_serialized({"": primary}))
        with self.assertRaisesRegex(RtpKvMetaObjectConfigError, "write lease limit"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(config)

    def test_serialization_failure_is_redacted(self):
        config = _split_reco_config(reco_client_config=_serialized())
        with (
            patch.object(
                kvcm_config.json,
                "dumps",
                side_effect=TypeError("private-serializer-detail"),
            ),
            self.assertRaisesRegex(
                RtpKvMetaObjectConfigError, "cannot be serialized safely"
            ) as caught,
        ):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(config)
        self.assertNotIn("private-serializer-detail", str(caught.exception))

    def test_resolves_vipserver_once_and_freezes_static_snapshot(self):
        primary = _primary_config(
            enable_vipserver=True,
            vipserver_domain="kvcm.example",
            address=[],
        )
        calls = []

        def resolve(domain):
            calls.append(domain)
            return [
                SimpleNamespace(ip="10.0.0.1", port=18001),
                SimpleNamespace(ip="10.0.0.2", port=18002),
            ]

        config = _split_reco_config(reco_client_config=_serialized({"": primary}))
        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(
            config, vipserver_resolver=resolve
        )
        transfer = json.loads(derived.transfer_client_config)
        self.assertEqual(calls, ["kvcm.example"])
        self.assertEqual(derived.addresses, ("10.0.0.1:18001", "10.0.0.2:18002"))
        self.assertEqual(transfer["address"], list(derived.addresses))

    def test_rejects_malformed_vipserver_results_without_leaking_domain(self):
        primary = _primary_config(
            enable_vipserver=True,
            vipserver_domain="secret-domain",
            address=[],
        )
        config = _split_reco_config(reco_client_config=_serialized({"": primary}))
        results = (
            None,
            [],
            "10.0.0.1:80",
            [SimpleNamespace(ip="", port=80)],
            [SimpleNamespace(ip="::1", port=80)],
            [SimpleNamespace(ip="not-an-ip", port=80)],
            [SimpleNamespace(ip="300.0.0.1", port=80)],
            [SimpleNamespace(ip="10.0.0.1", port=0)],
            [SimpleNamespace(ip="10.0.0.1", port=True)],
            [SimpleNamespace(ip="10.0.0.1", port=80)] * 2,
        )
        for result in results:
            with self.subTest(result=result):
                with self.assertRaises(RtpKvMetaObjectConfigError) as caught:
                    RtpKvMetaObjectClientConfig.from_kv_cache_config(
                        config,
                        vipserver_resolver=lambda _domain, value=result: value,
                    )
                self.assertNotIn("secret-domain", str(caught.exception))

        class BrokenHost:
            @property
            def ip(self):
                raise RuntimeError("provider-secret")

        with self.assertRaises(RtpKvMetaObjectConfigError) as caught:
            RtpKvMetaObjectClientConfig.from_kv_cache_config(
                config, vipserver_resolver=lambda _domain: [BrokenHost()]
            )
        self.assertNotIn("provider-secret", str(caught.exception))

    def test_default_vipserver_resolver_is_lazy_and_usable(self):
        fake_vipserver = SimpleNamespace(
            get_host_list_by_domain_now=lambda domain: (
                [SimpleNamespace(ip="10.0.0.9", port=19001)]
                if domain == "kvcm.example"
                else self.fail("unexpected VIPServer domain")
            )
        )
        config = _split_reco_config(
            reco_enable_vipserver=True,
            reco_vipserver_domain="kvcm.example",
            reco_server_address="",
        )
        with patch.dict(sys.modules, {"rtp_llm.vipserver": fake_vipserver}):
            derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(config)
        self.assertEqual(derived.addresses, ("10.0.0.9:19001",))


class RtpKvMetaSplitConfigTest(unittest.TestCase):
    def test_online_environment_reuses_existing_reco_settings(self):
        calls = []

        def resolve(domain):
            calls.append(domain)
            return [SimpleNamespace(ip="10.23.1.7", port=19001)]

        derived = RtpKvMetaObjectClientConfig.from_env(
            environ=_online_env(), vipserver_resolver=resolve
        )
        transfer = json.loads(derived.transfer_client_config)

        self.assertEqual(calls, ["kvcm-na130-m3-bailian-grpc-2.vipserver"])
        self.assertEqual(derived.addresses, ("10.23.1.7:19001",))
        self.assertEqual(derived.instance_group, "kve_pace_group_m3")
        self.assertEqual(derived.instance_id, "kve_pace_group_m3")
        self.assertEqual(derived.call_timeout_ms, 1500)
        self.assertEqual(derived.write_timeout_seconds, 105)
        self.assertEqual(
            transfer["sdk_config"]["timeout_config"],
            {"put_timeout_ms": 100_000, "get_timeout_ms": 100_000},
        )
        self.assertEqual(
            transfer["sdk_config"]["sdk_backend_configs"],
            [
                {
                    "type": "pace",
                    "sdk_log_file_path": "logs/pace_client.log",
                    "sdk_log_level": "INFO",
                }
            ],
        )
        self.assertEqual(
            transfer["model_deployment"],
            {
                "model_name": "__kv_meta_object__",
                "dtype": "opaque_bytes",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "extra": "kv_meta_v1",
                "user_data": "",
            },
        )

    def test_parsed_kv_cache_config_uses_salt_and_is_not_mutated(self):
        source = _split_reco_config(reco_instance_id_salt="stable-deployment")
        before = copy.deepcopy(source.__dict__)

        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(source)

        self.assertEqual(derived.instance_group, "kve_shared-group")
        self.assertEqual(derived.instance_id, "kve_stable-deployment")
        self.assertEqual(source.__dict__, before)

    def test_explicit_client_config_precedes_invalid_split_fields(self):
        source = _split_reco_config(
            reco_client_config=_serialized(),
            reco_enable_vipserver="not-a-bool",
            reco_model_sdk_config="not-json",
        )

        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(source)

        self.assertEqual(derived.instance_group, "kve_shared-group")
        self.assertEqual(derived.instance_id, "kve_shared-instance")

    def test_explicit_config_does_not_read_split_properties(self):
        class ExplicitOnlyConfig:
            reco_client_config = _serialized()

            def __getattr__(self, name):
                raise AssertionError(f"split property was read: {name}")

        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(ExplicitOnlyConfig())
        self.assertEqual(derived.instance_id, "kve_shared-instance")

    def test_explicit_environment_ignores_stale_split_values(self):
        derived = RtpKvMetaObjectClientConfig.from_env(
            environ={
                "RECO_CLIENT_CONFIG": _serialized(),
                "RECO_ENABLE_VIPSERVER": "not-a-bool",
                "RECO_META_CHANNEL_CALL_TIMEOUT": "not-an-int",
                "RECO_INSTANCE_GROUP": 7,
            }
        )
        self.assertEqual(derived.instance_group, "kve_shared-group")
        self.assertEqual(derived.instance_id, "kve_shared-instance")

    def test_rejects_invalid_split_config_values(self):
        cases = (
            ("reco_client_config", None),
            ("reco_enable_vipserver", 1),
            ("reco_instance_group", ""),
            ("reco_instance_id_salt", 1),
            ("reco_server_address", "127.0.0.1: 19001"),
            ("reco_meta_channel_retry_time", -1),
            ("reco_meta_channel_connection_timeout", True),
            ("reco_meta_channel_call_timeout", 600_001),
            ("reco_storage_thread_num", 0),
            ("reco_storage_queue_size", 63),
            ("reco_put_timeout_ms", 0),
            ("reco_get_timeout_ms", False),
            ("reco_model_sdk_config", None),
            ("reco_model_sdk_config", "x" * (1024 * 1024 + 1)),
            ("reco_model_sdk_config", "{}"),
            ("reco_model_sdk_config", '["not-an-object"]'),
            ("reco_model_sdk_config", '[{"type":"pace","type":"file"}]'),
            ("reco_model_user_data", 7),
        )
        for name, value in cases:
            with (
                self.subTest(name=name, value=value),
                self.assertRaises(RtpKvMetaObjectConfigError),
            ):
                RtpKvMetaObjectClientConfig.from_kv_cache_config(
                    _split_reco_config(**{name: value})
                )

    def test_requires_one_connection_source(self):
        with self.assertRaisesRegex(RtpKvMetaObjectConfigError, "address count"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(
                _split_reco_config(reco_server_address="")
            )

        with self.assertRaisesRegex(RtpKvMetaObjectConfigError, "vipserver_domain"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(
                _split_reco_config(
                    reco_enable_vipserver=True,
                    reco_vipserver_domain="",
                    reco_server_address="",
                ),
                vipserver_resolver=lambda _domain: self.fail("must not resolve"),
            )

    def test_preserves_zero_retry_and_connection_timeout(self):
        derived = RtpKvMetaObjectClientConfig.from_kv_cache_config(
            _split_reco_config(
                reco_meta_channel_retry_time=0,
                reco_meta_channel_connection_timeout=0,
            )
        )
        transfer = json.loads(derived.transfer_client_config)
        self.assertEqual(transfer["meta_channel_config"]["retry_time"], 0)
        self.assertEqual(transfer["meta_channel_config"]["connection_timeout"], 0)

    def test_environment_values_are_typed_and_bounded(self):
        cases = (
            ({"RECO_ENABLE_VIPSERVER": "maybe"}, "boolean"),
            ({"RECO_META_CHANNEL_CALL_TIMEOUT": "not-int"}, "integer"),
            ({"RECO_STORAGE_QUEUE_SIZE": "63"}, "at least 64"),
            ({"RECO_PUT_TIMEOUT_MS": "0"}, "valid range"),
            ({"RECO_INSTANCE_GROUP": 7}, "must be a string"),
        )
        base = {
            "RECO_SERVER_ADDRESS": "127.0.0.1:19001",
            "RECO_INSTANCE_GROUP": "shared-group",
        }
        for override, message in cases:
            with self.subTest(override=override):
                environment = dict(base)
                environment.update(override)
                with self.assertRaisesRegex(RtpKvMetaObjectConfigError, message):
                    RtpKvMetaObjectClientConfig.from_env(environ=environment)

    def test_environment_input_must_be_mapping(self):
        with self.assertRaisesRegex(TypeError, "mapping"):
            RtpKvMetaObjectClientConfig.from_env(environ=[])

    def test_real_process_environment_path_and_snapshot_failures(self):
        environment = {
            "RECO_SERVER_ADDRESS": "127.0.0.1:19001",
            "RECO_INSTANCE_GROUP": "shared-group",
        }
        with patch.dict(os.environ, environment, clear=True):
            derived = RtpKvMetaObjectClientConfig.from_env()
        self.assertEqual(derived.instance_group, "kve_shared-group")

        class BrokenClientConfigMapping(dict):
            def get(self, name, default=None):
                if name == "RECO_CLIENT_CONFIG":
                    raise RuntimeError("private-detail")
                return super().get(name, default)

        class BrokenSplitMapping(dict):
            def get(self, name, default=None):
                if name == "RECO_CLIENT_CONFIG":
                    return ""
                raise RuntimeError("private-detail")

        for source in (BrokenClientConfigMapping(), BrokenSplitMapping()):
            with self.subTest(source=type(source).__name__):
                with self.assertRaises(RtpKvMetaObjectConfigError) as caught:
                    RtpKvMetaObjectClientConfig.from_env(environ=source)
                self.assertNotIn("private-detail", str(caught.exception))

        with self.assertRaisesRegex(
            RtpKvMetaObjectConfigError, "RECO_CLIENT_CONFIG must be a string"
        ):
            RtpKvMetaObjectClientConfig.from_env(environ={"RECO_CLIENT_CONFIG": 1})

    def test_max_object_bytes_is_validated_before_client_construction(self):
        environment = {
            "RECO_SERVER_ADDRESS": "127.0.0.1:19001",
            "RECO_INSTANCE_GROUP": "shared-group",
        }
        for value in (0, True, 1024 * 1024 * 1024 + 1):
            with (
                self.subTest(value=value),
                self.assertRaises(RtpKvMetaObjectConfigError),
            ):
                RtpKvMetaObjectClientConfig.from_env(
                    environ=environment, max_object_bytes=value
                )
        self.assertEqual(
            RtpKvMetaObjectClientConfig.from_env(
                environ=environment, max_object_bytes=4096
            ).max_object_bytes,
            4096,
        )

    def test_missing_attribute_is_reported_without_provider_exception_text(self):
        class BrokenConfig:
            @property
            def reco_client_config(self):
                raise RuntimeError("private-provider-detail")

        with self.assertRaises(RtpKvMetaObjectConfigError) as caught:
            RtpKvMetaObjectClientConfig.from_kv_cache_config(BrokenConfig())
        self.assertNotIn("private-provider-detail", str(caught.exception))

        with self.assertRaisesRegex(TypeError, "must not be None"):
            RtpKvMetaObjectClientConfig.from_kv_cache_config(None)


if __name__ == "__main__":
    unittest.main()
