import copy
import json
import unittest
from types import SimpleNamespace

from rtp_llm.config.mm_kvcm_config import (
    KVE_INSTANCE_PREFIX,
    MMKvcmConfigError,
    configure_mm_kvcm_client,
    derive_mm_kvcm_client_config,
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
    config = {
        "reco_client_config": "",
        "reco_enable_vipserver": False,
        "reco_vipserver_domain": "",
        "reco_server_address": "127.0.0.1:19001",
        "reco_instance_group": "shared-group",
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
        "reco_model_extra_info": "unused-by-kvmeta",
        "reco_instance_id_salt": "",
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _runtime_config(mode="kvcm", shared=None):
    kvcm = SimpleNamespace(
        addresses=[],
        instance_id="",
        instance_group="",
        user_data="",
        transfer_client_config="",
        call_timeout_ms=3000,
        write_timeout_seconds=30,
        object_gc_timeout_ms=180_000,
        max_object_bytes=1024,
        max_receipt_bytes=8192,
        max_pending_objects=64,
        max_pending_bytes=65536,
    )
    return SimpleNamespace(
        vit_config=SimpleNamespace(
            output_transport=SimpleNamespace(mode=mode, kvcm=kvcm)
        ),
        kv_cache_config=_split_reco_config(
            reco_client_config=_serialized() if shared is None else shared
        ),
    )


class DeriveMMKvcmClientConfigTest(unittest.TestCase):
    def test_derives_prefixed_identity_and_exact_object_schema(self):
        source = _primary_config()
        original = copy.deepcopy(source)

        derived = derive_mm_kvcm_client_config(_serialized({"": source}))
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

    def test_accepts_one_named_entry(self):
        derived = derive_mm_kvcm_client_config(
            _serialized({"model-a": _primary_config()})
        )
        self.assertEqual(derived.instance_id, "kve_shared-instance")

    def test_empty_key_is_explicit_primary_entry(self):
        primary = _primary_config(instance_id="primary")
        secondary = _primary_config(instance_id="secondary")
        derived = derive_mm_kvcm_client_config(
            _serialized({"z-model": secondary, "": primary})
        )
        self.assertEqual(derived.instance_id, "kve_primary")

    def test_rejects_ambiguous_multiple_named_entries(self):
        with self.assertRaisesRegex(MMKvcmConfigError, "empty-key primary"):
            derive_mm_kvcm_client_config(
                _serialized(
                    {
                        "model-a": _primary_config(instance_id="a"),
                        "model-b": _primary_config(instance_id="b"),
                    }
                )
            )

    def test_write_lease_strictly_exceeds_full_completion_budget(self):
        primary = _primary_config()
        primary["meta_channel_config"]["call_timeout"] = 10_000
        primary["sdk_config"]["timeout_config"]["put_timeout_ms"] = 10_000
        derived = derive_mm_kvcm_client_config(_serialized({"": primary}))
        self.assertEqual(derived.write_timeout_seconds, 41)

        primary["meta_channel_config"]["call_timeout"] = 600_000
        with self.assertRaisesRegex(MMKvcmConfigError, "write lease limit"):
            derive_mm_kvcm_client_config(_serialized({"": primary}))

    def test_preserves_zero_retry_and_connection_timeout_compatibility(self):
        primary = _primary_config()
        primary["meta_channel_config"]["retry_time"] = 0
        primary["meta_channel_config"]["connection_timeout"] = 0

        derived = derive_mm_kvcm_client_config(_serialized({"": primary}))
        transfer = json.loads(derived.transfer_client_config)

        self.assertEqual(transfer["meta_channel_config"]["retry_time"], 0)
        self.assertEqual(transfer["meta_channel_config"]["connection_timeout"], 0)

    def test_identity_limits_count_utf8_bytes_after_prefix(self):
        accepted = _primary_config(instance_id="界" * 169)
        # 4-byte ASCII prefix + 169*3 UTF-8 bytes = 511 bytes.
        derived = derive_mm_kvcm_client_config(_serialized({"": accepted}))
        self.assertEqual(derived.instance_id, KVE_INSTANCE_PREFIX + "界" * 169)

        rejected = _primary_config(instance_group="界" * 170)
        with self.assertRaisesRegex(MMKvcmConfigError, "exceeds 512"):
            derive_mm_kvcm_client_config(_serialized({"": rejected}))

    def test_rejects_non_strict_or_malformed_json_without_echoing_content(self):
        cases = (
            "",
            "[]",
            '{"": {"instance_id": "secret", "instance_id": "duplicate"}}',
            '{"": {"number": NaN}}',
            '{"super-secret":',
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(MMKvcmConfigError) as caught:
                    derive_mm_kvcm_client_config(value)
                self.assertNotIn("super-secret", str(caught.exception))
                self.assertNotIn("duplicate", str(caught.exception))

    def test_rejects_invalid_shared_fields(self):
        def mutate(path, value):
            primary = _primary_config()
            target = primary
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = value
            return _serialized({"": primary})

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
            with self.subTest(path=path, value=value):
                with self.assertRaises(MMKvcmConfigError):
                    derive_mm_kvcm_client_config(mutate(path, value))

    def test_rejects_invalid_direct_addresses(self):
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
                with self.assertRaises(MMKvcmConfigError):
                    derive_mm_kvcm_client_config(
                        _serialized({"": _primary_config(address=addresses)})
                    )

    def test_resolves_vipserver_once_and_freezes_addresses(self):
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

        derived = derive_mm_kvcm_client_config(
            _serialized({"": primary}), vipserver_resolver=resolve
        )
        transfer = json.loads(derived.transfer_client_config)
        self.assertEqual(calls, ["kvcm.example"])
        self.assertEqual(derived.addresses, ("10.0.0.1:18001", "10.0.0.2:18002"))
        self.assertEqual(transfer["address"], list(derived.addresses))

    def test_rejects_bad_vipserver_results_without_leaking_domain(self):
        primary = _primary_config(
            enable_vipserver=True,
            vipserver_domain="secret-domain",
            address=[],
        )
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
                with self.assertRaises(MMKvcmConfigError) as caught:
                    derive_mm_kvcm_client_config(
                        _serialized({"": primary}),
                        vipserver_resolver=lambda _domain, value=result: value,
                    )
                self.assertNotIn("secret-domain", str(caught.exception))


class ConfigureMMKvcmClientTest(unittest.TestCase):
    def test_non_kvcm_mode_has_no_shared_config_or_env_side_effect(self):
        config = _runtime_config(mode="grpc", shared="not-json")
        before = copy.deepcopy(config.vit_config.output_transport.kvcm.__dict__)
        configure_mm_kvcm_client(
            config,
            environ={"MM_KVCM_INSTANCE_ID": "stale"},
            vipserver_resolver=lambda _domain: self.fail("must not resolve"),
        )
        self.assertEqual(config.vit_config.output_transport.kvcm.__dict__, before)

    def test_applies_shared_client_fields_and_preserves_policy_fields(self):
        config = _runtime_config()
        policy_before = {
            key: value
            for key, value in config.vit_config.output_transport.kvcm.__dict__.items()
            if key.startswith("max_") or key == "object_gc_timeout_ms"
        }
        configure_mm_kvcm_client(config, environ={})
        kvcm = config.vit_config.output_transport.kvcm
        self.assertEqual(kvcm.instance_group, "kve_shared-group")
        self.assertEqual(kvcm.instance_id, "kve_shared-instance")
        self.assertEqual(kvcm.call_timeout_ms, 1500)
        self.assertEqual(kvcm.write_timeout_seconds, 30)
        self.assertEqual(
            {
                key: value
                for key, value in kvcm.__dict__.items()
                if key.startswith("max_") or key == "object_gc_timeout_ms"
            },
            policy_before,
        )

    def test_split_reco_fields_reuse_online_vipserver_configuration(self):
        config = _runtime_config(shared="")
        config.kv_cache_config = _split_reco_config(
            reco_enable_vipserver=True,
            reco_vipserver_domain="kvcm-na130-m3-bailian-grpc-2.vipserver",
            reco_server_address="",
            reco_instance_group="pace_group_m3",
            reco_put_timeout_ms=100_000,
            reco_get_timeout_ms=100_000,
            reco_model_sdk_config=json.dumps(
                [
                    {
                        "type": "pace",
                        "sdk_log_file_path": "logs/pace_client.log",
                        "sdk_log_level": "INFO",
                    }
                ]
            ),
            reco_model_user_data="production-model",
        )
        source_before = copy.deepcopy(config.kv_cache_config.__dict__)
        calls = []

        def resolve(domain):
            calls.append(domain)
            return [SimpleNamespace(ip="10.23.1.7", port=19001)]

        configure_mm_kvcm_client(
            config,
            environ={
                "TAIR_MEMPOOL_KMONITOR_SINK_ADDRESS": "127.0.0.1:4141",
                "KVCM_LOG_LEVEL": "INFO",
            },
            vipserver_resolver=resolve,
        )

        kvcm = config.vit_config.output_transport.kvcm
        transfer = json.loads(kvcm.transfer_client_config)
        self.assertEqual(calls, ["kvcm-na130-m3-bailian-grpc-2.vipserver"])
        self.assertEqual(kvcm.addresses, ["10.23.1.7:19001"])
        self.assertEqual(kvcm.instance_group, "kve_pace_group_m3")
        self.assertEqual(kvcm.instance_id, "kve_pace_group_m3")
        self.assertEqual(kvcm.user_data, "production-model")
        self.assertEqual(kvcm.call_timeout_ms, 1500)
        self.assertEqual(kvcm.write_timeout_seconds, 105)
        self.assertEqual(transfer["instance_group"], "kve_pace_group_m3")
        self.assertEqual(transfer["instance_id"], "kve_pace_group_m3")
        self.assertFalse(transfer["enable_vipserver"])
        self.assertEqual(transfer["vipserver_domain"], "")
        self.assertEqual(transfer["address"], ["10.23.1.7:19001"])
        self.assertEqual(transfer["block_size"], 1)
        self.assertEqual(transfer["location_spec_infos"], {"value": 1})
        self.assertEqual(transfer["location_spec_groups"], {})
        self.assertEqual(transfer["meta_channel_config"]["retry_time"], 3)
        self.assertEqual(transfer["meta_channel_config"]["connection_timeout"], 6000)
        self.assertEqual(transfer["meta_channel_config"]["call_timeout"], 1500)
        self.assertEqual(transfer["sdk_config"]["thread_num"], 4)
        self.assertEqual(transfer["sdk_config"]["queue_size"], 2000)
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
                "user_data": "production-model",
            },
        )
        self.assertEqual(config.kv_cache_config.__dict__, source_before)

    def test_split_reco_fields_prefer_existing_instance_id_salt(self):
        config = _runtime_config(shared="")
        config.kv_cache_config.reco_instance_id_salt = "stable-deployment"

        configure_mm_kvcm_client(config, environ={})

        kvcm = config.vit_config.output_transport.kvcm
        self.assertEqual(kvcm.instance_group, "kve_shared-group")
        self.assertEqual(kvcm.instance_id, "kve_stable-deployment")

    def test_explicit_client_config_takes_precedence_over_split_fields(self):
        config = _runtime_config()
        config.kv_cache_config.reco_enable_vipserver = "not-a-bool"
        config.kv_cache_config.reco_model_sdk_config = "not-json"

        configure_mm_kvcm_client(config, environ={})

        kvcm = config.vit_config.output_transport.kvcm
        self.assertEqual(kvcm.instance_group, "kve_shared-group")
        self.assertEqual(kvcm.instance_id, "kve_shared-instance")

    def test_split_reco_fields_reject_invalid_values_without_partial_apply(self):
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
            ("reco_model_sdk_config", "{}"),
            ("reco_model_sdk_config", '["not-an-object"]'),
            ("reco_model_sdk_config", '[{"type":"pace","type":"file"}]'),
            ("reco_model_user_data", 7),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                config = _runtime_config(shared="")
                setattr(config.kv_cache_config, name, value)
                before = copy.deepcopy(config.vit_config.output_transport.kvcm.__dict__)
                with self.assertRaises(MMKvcmConfigError):
                    configure_mm_kvcm_client(config, environ={})
                self.assertEqual(
                    config.vit_config.output_transport.kvcm.__dict__, before
                )

    def test_split_reco_fields_require_one_connection_source(self):
        config = _runtime_config(shared="")
        config.kv_cache_config.reco_server_address = ""

        with self.assertRaisesRegex(MMKvcmConfigError, "address count"):
            configure_mm_kvcm_client(config, environ={})

        config.kv_cache_config.reco_enable_vipserver = True
        with self.assertRaisesRegex(MMKvcmConfigError, "vipserver_domain"):
            configure_mm_kvcm_client(
                config,
                environ={},
                vipserver_resolver=lambda _domain: self.fail("must not resolve"),
            )

    def test_split_reco_fields_preserve_zero_retry_timeout_compatibility(self):
        config = _runtime_config(shared="")
        config.kv_cache_config.reco_meta_channel_retry_time = 0
        config.kv_cache_config.reco_meta_channel_connection_timeout = 0

        configure_mm_kvcm_client(config, environ={})

        transfer = json.loads(
            config.vit_config.output_transport.kvcm.transfer_client_config
        )
        self.assertEqual(transfer["meta_channel_config"]["retry_time"], 0)
        self.assertEqual(transfer["meta_channel_config"]["connection_timeout"], 0)

    def test_rejects_removed_client_env_vars_before_mutation(self):
        removed = (
            "MM_KVCM_ADDRESSES",
            "MM_KVCM_INSTANCE_ID",
            "MM_KVCM_INSTANCE_GROUP",
            "MM_KVCM_USER_DATA",
            "MM_KVCM_TRANSFER_CLIENT_CONFIG",
            "MM_KVCM_CALL_TIMEOUT_MS",
            "MM_KVCM_WRITE_TIMEOUT_SECONDS",
        )
        for name in removed:
            with self.subTest(name=name):
                config = _runtime_config()
                before = copy.deepcopy(config.vit_config.output_transport.kvcm.__dict__)
                with self.assertRaisesRegex(MMKvcmConfigError, name):
                    configure_mm_kvcm_client(config, environ={name: ""})
                self.assertEqual(
                    config.vit_config.output_transport.kvcm.__dict__, before
                )

    def test_invalid_shared_config_does_not_partially_apply(self):
        config = _runtime_config(shared='{"": {}}')
        before = copy.deepcopy(config.vit_config.output_transport.kvcm.__dict__)
        with self.assertRaises(MMKvcmConfigError):
            configure_mm_kvcm_client(config, environ={})
        self.assertEqual(config.vit_config.output_transport.kvcm.__dict__, before)


if __name__ == "__main__":
    unittest.main()
