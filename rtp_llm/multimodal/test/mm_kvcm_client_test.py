import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.multimodal.transport.kvcm import client as kvcm_client
from rtp_llm.multimodal.transport.kvcm.client import RtpKvMetaObjectClient


def _rtp_config(**overrides):
    values = {
        "addresses": ["10.0.0.1:19001", "10.0.0.2:19001"],
        "instance_id": "kve_rtp-emb-1",
        "instance_group": "kve_epd-emb",
        "user_data": "rtp",
        "transfer_client_config": '{"block_size":1}',
        "call_timeout_ms": 1234,
        "write_timeout_seconds": 45,
        "max_object_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RtpKvMetaObjectClientTest(TestCase):
    def _create(self, config=None, *, client=None):
        generic_client = MagicMock() if client is None else client
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        client_type = MagicMock(return_value=generic_client)
        loader = patch.object(
            kvcm_client,
            "_load_kvcm_client_types",
            return_value=(client_type, config_type),
        )
        with loader:
            result = RtpKvMetaObjectClient(_rtp_config() if config is None else config)
        return result, generic_client, config_type, client_type

    def test_maps_canonical_rtp_config_and_registers_once(self):
        config = _rtp_config()

        client, generic_client, config_type, client_type = self._create(config)

        config_type.assert_called_once_with(
            addresses=tuple(config.addresses),
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
        self.assertEqual(client.instance_id, config.instance_id)
        self.assertEqual(client.instance_group, config.instance_group)

    def test_snapshots_registration_identity_exactly_once(self):
        class MutableIdentityConfig:
            addresses = ["127.0.0.1:19001"]
            user_data = "rtp"
            transfer_client_config = '{"block_size":1}'
            call_timeout_ms = 3000
            write_timeout_seconds = 30
            max_object_bytes = 4096

            def __init__(self):
                self.id_reads = 0
                self.group_reads = 0

            @property
            def instance_id(self):
                self.id_reads += 1
                return f"kve_instance-{self.id_reads}"

            @property
            def instance_group(self):
                self.group_reads += 1
                return f"kve_group-{self.group_reads}"

        config = MutableIdentityConfig()

        client, _, config_type, _ = self._create(config)

        self.assertEqual(config.id_reads, 1)
        self.assertEqual(config.group_reads, 1)
        self.assertEqual(client.instance_id, "kve_instance-1")
        self.assertEqual(client.instance_group, "kve_group-1")
        self.assertEqual(
            config_type.call_args.kwargs["instance_id"], client.instance_id
        )
        self.assertEqual(
            config_type.call_args.kwargs["instance_group"], client.instance_group
        )

    def test_delegates_the_small_rtp_operation_surface(self):
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

    def test_context_manager_returns_rtp_facade_and_delegates_cleanup(self):
        client, generic_client, _, _ = self._create()
        generic_client.__exit__.return_value = True

        self.assertIs(client.__enter__(), client)
        suppressed = client.__exit__(ValueError, ValueError("body"), None)

        generic_client.__enter__.assert_called_once_with()
        generic_client.__exit__.assert_called_once()
        self.assertTrue(suppressed)

    def test_missing_or_incompatible_wheel_has_stable_rtp_error(self):
        with patch.dict(
            sys.modules,
            {"kv_cache_manager": None, "kv_cache_manager.client": None},
        ):
            with self.assertRaisesRegex(RuntimeError, "kvcm_py_client wheel") as caught:
                RtpKvMetaObjectClient(_rtp_config())

        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_native_extension_import_failure_is_translated(self):
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        client_type = MagicMock(side_effect=ImportError("private loader detail"))
        with patch.object(
            kvcm_client,
            "_load_kvcm_client_types",
            return_value=(client_type, config_type),
        ):
            with self.assertRaisesRegex(RuntimeError, "kvcm_py_client wheel") as caught:
                RtpKvMetaObjectClient(_rtp_config())

        self.assertIsInstance(caught.exception.__cause__, ImportError)
        self.assertNotIn("private loader detail", str(caught.exception))

    def test_config_and_registration_failures_are_not_masked(self):
        config_error = ValueError("invalid shared config")
        with patch.object(
            kvcm_client,
            "_load_kvcm_client_types",
            return_value=(MagicMock(), MagicMock(side_effect=config_error)),
        ):
            with self.assertRaises(ValueError) as invalid:
                RtpKvMetaObjectClient(_rtp_config())
        self.assertIs(invalid.exception, config_error)

        registration_error = RuntimeError("registration rejected")
        config_type = MagicMock(side_effect=lambda **values: SimpleNamespace(**values))
        with patch.object(
            kvcm_client,
            "_load_kvcm_client_types",
            return_value=(MagicMock(side_effect=registration_error), config_type),
        ):
            with self.assertRaises(RuntimeError) as rejected:
                RtpKvMetaObjectClient(_rtp_config())
        self.assertIs(rejected.exception, registration_error)

    def test_module_load_does_not_import_optional_kvcm_package(self):
        module_path = Path(kvcm_client.__file__)
        spec = importlib.util.spec_from_file_location(
            "_rtp_kvcm_client_lazy_import_probe", module_path
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
    main()
