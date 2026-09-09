"""Public-only startup and reusable SDK boundaries; no hardware required."""

import importlib.abc
import json
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from rtp_llm.platforms import register_backend_hooks
from rtp_llm.utils import backend_registry


class PlatformBootstrapTest(unittest.TestCase):
    def tearDown(self):
        backend_registry.reset_backend_registrations()

    def test_public_manifests_without_internal_source_or_sdk(self):
        code = r"""
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'triton', 'deep_gemm', 'tilelang'}:
            raise AssertionError('eager SDK import: ' + fullname)
        if fullname.split('.')[0] == 'internal_source':
            raise ModuleNotFoundError('absent private extension', name='internal_source')
sys.meta_path.insert(0, Deny())
from rtp_llm.platforms import register_backend_hooks
register_backend_hooks()
import argparse
from rtp_llm.utils.backend_registry import run_backend_registrations
p = argparse.ArgumentParser()
p.add_argument('--moe_strategy', choices=['auto'])
run_backend_registrations('moe_strategy_choices', repeatable=True, parser=p)
assert p.parse_args(['--moe_strategy', 'w8a8_int8_dp_normal_deepgemm']).moe_strategy
from rtp_llm.models.dsv4.adapter import get_registry as get_module_registry
r = get_module_registry()
s = r.implementation('rtp.dsv4.attention', 'ppu.dsv4.attention.fp4_indexer.v1')
assert s.builder.startswith('rtp_llm.platforms.ppu.')
assert s.state_format_id and s.collective_protocol_id and not s.auto_selectable
assert not any(n.startswith('rtp_llm.platforms.ppu.modules.linear.fp8_linear') for n in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_concurrent_and_legacy_bootstrap_registers_each_hook_once(self):
        backend_registry.reset_backend_registrations()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: register_backend_hooks(), range(32)))
        before = {slot: tuple(hooks) for slot, hooks in backend_registry._hooks.items()}
        self.assertTrue(before)
        self.assertTrue(all(len(hooks) == 1 for hooks in before.values()))
        with patch("rtp_llm.device.device_type.get_device_type", return_value=0), patch(
            "rtp_llm.utils.import_util.import_optional_internal_source_entrypoint",
            return_value=False,
        ):
            backend_registry.run_backend_registrations("linear", factory=object())
        register_backend_hooks()
        self.assertEqual(
            before, {s: tuple(h) for s, h in backend_registry._hooks.items()}
        )

    def test_missing_sdk_symbol_fails_before_execution(self):
        from rtp_llm.platforms.ppu.runtime import require_symbol

        with self.assertRaisesRegex(
            RuntimeError, "requires callable json.missing_ppu_op"
        ):
            require_symbol("json", "missing_ppu_op")


if __name__ == "__main__":
    unittest.main()
