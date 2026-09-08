"""Tests for normalize_prompt_generator_config.

Covers the configuration paths:
  1. prompt_generator disabled                                            -> no-op
  2. MPS enabled but PG disabled                                          -> force-disable MPS
  3. PG enabled + no internal_source                                      -> warn + silently disable (fallback)
  4. PG enabled + internal_source + start_server submodule import fails   -> NotImplementedError
  5. PG enabled + internal_source + start_server lacks entry symbol       -> NotImplementedError
  6. PG enabled + internal_source + import succeeds                       -> only start_server validated
  7. PG+MPS enabled + internal_source + both imports succeed              -> both submodules validated
  8. PG+MPS enabled + internal_source + start_mps import fails            -> NotImplementedError
"""

import unittest
from unittest import mock

from rtp_llm import start_server
from rtp_llm.config.py_config_modules import PyEnvConfigs

START_SERVER_MODULE = "internal_source.rtp_llm.prompt_generator.service.start_server"
START_MPS_MODULE = "internal_source.rtp_llm.prompt_generator.service.start_mps"


def _fake_module(**attrs):
    """Build a stand-in module that only exposes the given attributes.

    Using spec_set + the attribute list makes hasattr(module, "foo") return
    True iff "foo" is in attrs, so we can simulate missing entry symbols.
    """
    return mock.MagicMock(spec_set=list(attrs.keys()), **attrs)


def _import_dispatcher(module_map):
    """Return an import_module side_effect that dispatches by module path.

    Each value is either a module (returned) or an Exception (raised).
    Unexpected module paths raise AssertionError so the test fails loudly.
    """

    def _side_effect(module_path):
        if module_path not in module_map:
            raise AssertionError(f"unexpected import_module call: {module_path}")
        result = module_map[module_path]
        if isinstance(result, BaseException):
            raise result
        return result

    return _side_effect


class NormalizePromptGeneratorConfigTest(unittest.TestCase):
    def setUp(self):
        self.cfg = PyEnvConfigs()

    def test_disabled_is_noop(self):
        """Nothing should be touched when the feature is off."""
        self.cfg.server_config.enable_prompt_generator = False
        self.cfg.server_config.enable_prompt_generator_mps = False
        # Even if internal_source is missing, no warning, no error.
        with mock.patch.object(start_server, "has_internal_source", return_value=False):
            start_server.normalize_prompt_generator_config(self.cfg)
        self.assertFalse(self.cfg.server_config.enable_prompt_generator)

    def test_mps_disabled_when_pg_disabled(self):
        """MPS without prompt_generator must be force-disabled with a warning."""
        self.cfg.server_config.enable_prompt_generator = False
        self.cfg.server_config.enable_prompt_generator_mps = True
        with mock.patch.object(start_server, "has_internal_source", return_value=False):
            start_server.normalize_prompt_generator_config(self.cfg)
        self.assertFalse(self.cfg.server_config.enable_prompt_generator_mps)

    def test_fallback_when_no_internal_source(self):
        """Open-source build (no internal_source dir) silently disables PG."""
        self.cfg.server_config.enable_prompt_generator = True
        self.cfg.server_config.enable_prompt_generator_mps = True
        with mock.patch.object(start_server, "has_internal_source", return_value=False):
            start_server.normalize_prompt_generator_config(self.cfg)
        self.assertFalse(self.cfg.server_config.enable_prompt_generator)
        self.assertFalse(self.cfg.server_config.enable_prompt_generator_mps)

    def test_raises_when_start_server_import_fails(self):
        """internal_source exists but start_server submodule is missing -> hard fail."""
        self.cfg.server_config.enable_prompt_generator = True
        self.cfg.server_config.enable_prompt_generator_mps = False
        with mock.patch.object(
            start_server, "has_internal_source", return_value=True
        ), mock.patch(
            "importlib.import_module",
            side_effect=_import_dispatcher(
                {START_SERVER_MODULE: ImportError("simulated missing start_server")}
            ),
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                start_server.normalize_prompt_generator_config(self.cfg)
        self.assertIn(START_SERVER_MODULE, str(ctx.exception))
        self.assertIn("simulated missing start_server", str(ctx.exception))

    def test_raises_when_start_server_lacks_entry_symbol(self):
        """start_server importable but missing start_prompt_generator -> hard fail."""
        self.cfg.server_config.enable_prompt_generator = True
        self.cfg.server_config.enable_prompt_generator_mps = False
        # Module exists but does NOT expose start_prompt_generator.
        empty_module = _fake_module()
        with mock.patch.object(
            start_server, "has_internal_source", return_value=True
        ), mock.patch(
            "importlib.import_module",
            side_effect=_import_dispatcher({START_SERVER_MODULE: empty_module}),
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                start_server.normalize_prompt_generator_config(self.cfg)
        self.assertIn(START_SERVER_MODULE, str(ctx.exception))
        self.assertIn("start_prompt_generator", str(ctx.exception))

    def test_passes_when_internal_source_and_start_server_ok(self):
        """PG enabled, MPS off -> only start_server submodule is validated."""
        self.cfg.server_config.enable_prompt_generator = True
        self.cfg.server_config.enable_prompt_generator_mps = False
        start_server_mod = _fake_module(start_prompt_generator=mock.MagicMock())
        with mock.patch.object(
            start_server, "has_internal_source", return_value=True
        ), mock.patch(
            "importlib.import_module",
            side_effect=_import_dispatcher({START_SERVER_MODULE: start_server_mod}),
        ) as import_mock:
            start_server.normalize_prompt_generator_config(self.cfg)
        # Only start_server should be imported when MPS is off.
        import_mock.assert_called_once_with(START_SERVER_MODULE)
        self.assertTrue(self.cfg.server_config.enable_prompt_generator)
        self.assertFalse(self.cfg.server_config.enable_prompt_generator_mps)

    def test_passes_when_mps_enabled_and_both_modules_ok(self):
        """PG+MPS enabled -> both start_server and start_mps submodules validated."""
        self.cfg.server_config.enable_prompt_generator = True
        self.cfg.server_config.enable_prompt_generator_mps = True
        start_server_mod = _fake_module(start_prompt_generator=mock.MagicMock())
        start_mps_mod = _fake_module(start_mps=mock.MagicMock())
        with mock.patch.object(
            start_server, "has_internal_source", return_value=True
        ), mock.patch(
            "importlib.import_module",
            side_effect=_import_dispatcher(
                {
                    START_SERVER_MODULE: start_server_mod,
                    START_MPS_MODULE: start_mps_mod,
                }
            ),
        ) as import_mock:
            start_server.normalize_prompt_generator_config(self.cfg)
        self.assertEqual(
            import_mock.call_args_list,
            [mock.call(START_SERVER_MODULE), mock.call(START_MPS_MODULE)],
        )
        self.assertTrue(self.cfg.server_config.enable_prompt_generator)
        self.assertTrue(self.cfg.server_config.enable_prompt_generator_mps)

    def test_raises_when_mps_enabled_and_start_mps_import_fails(self):
        """PG+MPS enabled, start_server OK but start_mps missing -> hard fail."""
        self.cfg.server_config.enable_prompt_generator = True
        self.cfg.server_config.enable_prompt_generator_mps = True
        start_server_mod = _fake_module(start_prompt_generator=mock.MagicMock())
        with mock.patch.object(
            start_server, "has_internal_source", return_value=True
        ), mock.patch(
            "importlib.import_module",
            side_effect=_import_dispatcher(
                {
                    START_SERVER_MODULE: start_server_mod,
                    START_MPS_MODULE: ImportError("simulated missing start_mps"),
                }
            ),
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                start_server.normalize_prompt_generator_config(self.cfg)
        self.assertIn(START_MPS_MODULE, str(ctx.exception))
        self.assertIn("simulated missing start_mps", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
