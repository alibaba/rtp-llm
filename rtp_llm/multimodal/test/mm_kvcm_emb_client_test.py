import importlib.util
import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rtp_llm.multimodal.kvcm as kvcm_package
from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient
from rtp_llm.multimodal.kvcm import client as kvcm_client
from rtp_llm.multimodal.kvcm._config import RtpKvMetaObjectClientConfig


def _client_config(**overrides):
    values = {
        "addresses": ("10.0.0.1:19001", "10.0.0.2:19001"),
        "instance_id": "kve_rtp-emb-1",
        "instance_group": "kve_epd-emb",
        "user_data": "rtp",
        "transfer_client_config": '{"block_size":1}',
        "call_timeout_ms": 1234,
        "write_timeout_seconds": 45,
        "max_object_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return RtpKvMetaObjectClientConfig(**values)


def _kv_cache_config(**overrides):
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
            [{"type": "file", "sdk_log_file_path": "", "sdk_log_level": "ERROR"}]
        ),
        "reco_model_user_data": "rtp",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RtpKvMetaObjectClientTest(unittest.TestCase):
    def _create(self, config=None, *, generic_client=None):
        underlying = MagicMock() if generic_client is None else generic_client
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        client_type = MagicMock(return_value=underlying)
        with patch.object(
            kvcm_client,
            "_load_kvcm_client_types",
            return_value=(client_type, config_type),
        ):
            result = RtpKvMetaObjectClient._from_config(
                _client_config() if config is None else config
            )
        return result, underlying, config_type, client_type

    def test_maps_validated_rtp_config_and_registers_once(self):
        config = _client_config()

        client, generic_client, config_type, client_type = self._create(config)

        config_type.assert_called_once_with(
            addresses=config.addresses,
            instance_id=config.instance_id,
            instance_group=config.instance_group,
            user_data=config.user_data,
            transfer_client_config=config.transfer_client_config,
            call_timeout_ms=config.call_timeout_ms,
            write_timeout_seconds=config.write_timeout_seconds,
            max_object_bytes=config.max_object_bytes,
        )
        client_type.assert_called_once()
        self.assertIsInstance(client_type.call_args.args[0], SimpleNamespace)
        self.assertIs(client._client, generic_client)
        self.assertIs(client._config, config)
        self.assertEqual(client.instance_id, config.instance_id)
        self.assertEqual(client.instance_group, config.instance_group)

    def test_rejects_wrong_config_before_loading_optional_wheel(self):
        with (
            patch.object(kvcm_client, "_load_kvcm_client_types") as loader,
            self.assertRaisesRegex(TypeError, "RtpKvMetaObjectClientConfig"),
        ):
            RtpKvMetaObjectClient._from_config(SimpleNamespace())
        loader.assert_not_called()

    def test_delegates_small_operation_surface_without_reinterpreting_inputs(self):
        client, generic_client, _, _ = self._create()
        keys = ["one", "two"]
        tensors = [object(), object()]

        client.save(keys, tensors)
        client.load(keys, tensors, trace_id="load-trace")
        client.remove(keys, trace_id="remove-trace")
        client.close()

        generic_client.save.assert_called_once_with(keys, tensors, trace_id=None)
        generic_client.load.assert_called_once_with(
            keys, tensors, trace_id="load-trace"
        )
        generic_client.remove.assert_called_once_with(keys, trace_id="remove-trace")
        generic_client.close.assert_called_once_with()

    def test_single_embedding_helpers_remove_sequence_boilerplate(self):
        client, generic_client, _, _ = self._create()
        tensor = object()

        client.save_one("one", tensor, trace_id="save-one")
        loaded = client.load_one("one", tensor, trace_id="load-one")
        client.remove_one("one", trace_id="remove-one")

        self.assertIs(loaded, tensor)
        generic_client.save.assert_called_once_with(
            ("one",), (tensor,), trace_id="save-one"
        )
        generic_client.load.assert_called_once_with(
            ("one",), (tensor,), trace_id="load-one"
        )
        generic_client.remove.assert_called_once_with(("one",), trace_id="remove-one")

    def test_public_package_exports_only_the_supported_rtp_surface(self):
        self.assertEqual(
            kvcm_package.__all__,
            ["RtpKvMetaObjectClient", "RtpKvMetaObjectConfigError"],
        )
        self.assertFalse(hasattr(kvcm_package, "RtpKvMetaObjectClientConfig"))
        self.assertFalse(hasattr(kvcm_package, "KVE_INSTANCE_PREFIX"))
        self.assertEqual(list(inspect.signature(RtpKvMetaObjectClient).parameters), [])

    def test_context_manager_returns_facade_and_never_suppresses_body_error(self):
        client, generic_client, _, _ = self._create()
        generic_client.__exit__.return_value = True

        self.assertIs(client.__enter__(), client)
        suppressed = client.__exit__(ValueError, ValueError("body"), None)

        generic_client.__enter__.assert_called_once_with()
        generic_client.__exit__.assert_called_once()
        self.assertIsNone(suppressed)

    def test_missing_or_incompatible_wheel_has_stable_rtp_error(self):
        with (
            patch.dict(
                sys.modules,
                {"kv_cache_manager": None, "kv_cache_manager.client": None},
            ),
            self.assertRaisesRegex(RuntimeError, "kvcm_py_client wheel") as caught,
        ):
            RtpKvMetaObjectClient._from_config(_client_config())

        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_native_extension_import_failure_is_translated(self):
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        client_type = MagicMock(side_effect=ImportError("private loader detail"))
        with (
            patch.object(
                kvcm_client,
                "_load_kvcm_client_types",
                return_value=(client_type, config_type),
            ),
            self.assertRaisesRegex(RuntimeError, "kvcm_py_client wheel") as caught,
        ):
            RtpKvMetaObjectClient._from_config(_client_config())

        self.assertIsInstance(caught.exception.__cause__, ImportError)
        self.assertNotIn("private loader detail", str(caught.exception))

    def test_config_and_registration_failures_are_not_masked(self):
        config_error = ValueError("invalid generic config")
        with (
            patch.object(
                kvcm_client,
                "_load_kvcm_client_types",
                return_value=(MagicMock(), MagicMock(side_effect=config_error)),
            ),
            self.assertRaises(ValueError) as invalid,
        ):
            RtpKvMetaObjectClient._from_config(_client_config())
        self.assertIs(invalid.exception, config_error)

        registration_error = RuntimeError("registration rejected")
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        with (
            patch.object(
                kvcm_client,
                "_load_kvcm_client_types",
                return_value=(MagicMock(side_effect=registration_error), config_type),
            ),
            self.assertRaises(RuntimeError) as rejected,
        ):
            RtpKvMetaObjectClient._from_config(_client_config())
        self.assertIs(rejected.exception, registration_error)

    def test_from_env_supports_an_explicit_object_limit(self):
        environment = {
            "RECO_SERVER_ADDRESS": "127.0.0.1:19001",
            "RECO_INSTANCE_GROUP": "pace_group_m3",
        }
        generic_client = MagicMock()
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        client_type = MagicMock(return_value=generic_client)

        with (
            patch.object(
                kvcm_client,
                "_load_kvcm_client_types",
                return_value=(client_type, config_type),
            ),
        ):
            client = RtpKvMetaObjectClient.from_env(
                environ=environment, max_object_bytes=4096
            )

        self.assertEqual(client.instance_group, "kve_pace_group_m3")
        self.assertEqual(client.instance_id, "kve_pace_group_m3")
        self.assertEqual(client._config.max_object_bytes, 4096)
        client_type.assert_called_once()

    def test_no_arg_constructor_reuses_online_vip_settings(self):
        environment = {
            "RECO_ENABLE_VIPSERVER": "1",
            "RECO_VIPSERVER_DOMAIN": "kvcm.example.vipserver",
            "RECO_INSTANCE_GROUP": "pace_group_m3",
            "RECO_PUT_TIMEOUT_MS": "100000",
            "RECO_GET_TIMEOUT_MS": "100000",
            "RECO_MODEL_SDK_CONFIG": json.dumps([{"type": "pace"}]),
        }
        fake_vipserver = SimpleNamespace(
            get_host_list_by_domain_now=lambda domain: [
                SimpleNamespace(ip="10.0.0.9", port=19001)
            ]
        )
        generic_client = MagicMock()
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        client_type = MagicMock(return_value=generic_client)

        with (
            patch.object(
                kvcm_client,
                "_load_kvcm_client_types",
                return_value=(client_type, config_type),
            ),
            patch.dict(os.environ, environment, clear=True),
            patch.dict(sys.modules, {"rtp_llm.vipserver": fake_vipserver}),
        ):
            client = RtpKvMetaObjectClient()

        self.assertEqual(client._config.addresses, ("10.0.0.9:19001",))
        self.assertEqual(client.instance_group, "kve_pace_group_m3")
        self.assertEqual(client.instance_id, "kve_pace_group_m3")
        client_type.assert_called_once()

    def test_from_parsed_kv_cache_config_honors_cli_resolved_values(self):
        generic_client = MagicMock()
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        client_type = MagicMock(return_value=generic_client)
        source = _kv_cache_config(reco_instance_id_salt="deployment-a")

        with patch.object(
            kvcm_client,
            "_load_kvcm_client_types",
            return_value=(client_type, config_type),
        ):
            client = RtpKvMetaObjectClient.from_kv_cache_config(source)

        self.assertEqual(client.instance_group, "kve_shared-group")
        self.assertEqual(client.instance_id, "kve_deployment-a")
        client_type.assert_called_once()

    def test_module_load_does_not_import_optional_kvcm_package(self):
        module_path = Path(kvcm_client.__file__)
        spec = importlib.util.spec_from_file_location(
            "rtp_llm.multimodal.kvcm._lazy_client_probe", module_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        real_import = __import__

        def reject_kvcm(name, *args, **kwargs):
            if name.startswith("kv_cache_manager"):
                raise AssertionError(f"optional KVCM package imported: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_kvcm):
            spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, "RtpKvMetaObjectClient"))


if __name__ == "__main__":
    unittest.main()
