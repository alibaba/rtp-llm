"""CPU-only startup contract checks; real PPU pressure tests run separately."""

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

PACKAGE = Path(__file__).resolve().parents[4]
SOURCE = PACKAGE / "platforms" / "ppu" / "models" / "dsv4" / "communication.py"


class PrefillCommunicationTest(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_current_stream_capturing.return_value = False
        self.tensor = object()
        self.events = []
        self.torch.zeros.side_effect = lambda *a, **kw: (
            self.events.append("allocate") or self.tensor
        )
        self.reduce = Mock(side_effect=lambda *a: self.events.append("reduce"))
        self.stream = self.torch.cuda.current_stream.return_value
        self.stream.synchronize.side_effect = lambda: self.events.append("synchronize")
        self.device_type = Mock(return_value="ppu")
        self.group = object()
        replacements = {
            "torch": self.torch,
            "rtp_llm.device.device_type": SimpleNamespace(
                DeviceType=SimpleNamespace(Ppu="ppu"), get_device_type=self.device_type
            ),
            "rtp_llm.models_py.distributed.collective_torch": SimpleNamespace(
                Group=SimpleNamespace(TP=self.group), all_reduce=self.reduce
            ),
            "rtp_llm.ops": SimpleNamespace(
                RoleType=SimpleNamespace(PREFILL="prefill", PDFUSION="pdfusion")
            ),
        }
        with patch.dict(sys.modules, replacements):
            spec = importlib.util.spec_from_file_location(
                "ppu_comm_startup_test", SOURCE
            )
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.cfg = SimpleNamespace(
            role_type="prefill",
            tp_size=4,
            world_size=4,
            local_rank=2,
            prefill_cp_config=SimpleNamespace(prefill_cp_size=1),
        )

    def run_warmup(self, enabled="1"):
        with patch.dict(os.environ, {"DSV4_PPU_TP_COMM_WARMUP": enabled}):
            return self.module.maybe_warmup_ppu_tp_communication(self.cfg)

    def test_default_off_does_not_touch_cuda(self):
        with patch.dict(os.environ):
            os.environ.pop("DSV4_PPU_TP_COMM_WARMUP", None)
            self.assertFalse(self.module.maybe_warmup_ppu_tp_communication(self.cfg))
        self.device_type.assert_not_called()
        self.torch.device.assert_not_called()
        self.reduce.assert_not_called()

    def test_unsupported_scope_is_unchanged(self):
        for key, value in (("role_type", "decode"), ("tp_size", 1), ("world_size", 8)):
            with self.subTest(key=key):
                original = getattr(self.cfg, key)
                setattr(self.cfg, key, value)
                self.assertFalse(self.run_warmup())
                setattr(self.cfg, key, original)
        for cp_size in (0, 4):
            self.cfg.prefill_cp_config.prefill_cp_size = cp_size
            self.assertFalse(self.run_warmup())
        self.cfg.prefill_cp_config.prefill_cp_size = 1
        self.device_type.return_value = "cuda"
        self.assertFalse(self.run_warmup())
        self.torch.zeros.assert_not_called()
        self.reduce.assert_not_called()

    def test_reuses_tp_collective_on_rank_local_device(self):
        self.assertTrue(self.run_warmup())
        self.torch.device.assert_called_once_with("cuda", 2)
        self.torch.zeros.assert_called_once_with(
            128 * 1024, dtype=self.torch.bfloat16, device=self.torch.device.return_value
        )
        self.reduce.assert_called_once_with(self.tensor, self.group)
        self.assertEqual(self.events, ["allocate", "reduce", "synchronize"])
        self.torch.cuda.empty_cache.assert_not_called()
        self.torch.manual_seed.assert_not_called()

    def test_fused_service_prepares_the_same_prefill_communicator(self):
        self.cfg.role_type = "pdfusion"
        self.assertTrue(self.run_warmup())
        self.reduce.assert_called_once_with(self.tensor, self.group)
        self.assertEqual(self.events, ["allocate", "reduce", "synchronize"])

    def test_capture_rejected_before_allocation(self):
        self.torch.cuda.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "before capture"):
            self.run_warmup()
        self.torch.zeros.assert_not_called()
        self.reduce.assert_not_called()

    def test_collective_failure_propagates_without_retry(self):
        original = RuntimeError("PCCL allocation failed")
        self.reduce.side_effect = original
        with self.assertRaises(RuntimeError) as caught:
            self.run_warmup()
        self.assertIs(caught.exception, original)
        self.reduce.assert_called_once()
        self.stream.synchronize.assert_not_called()

    def test_completion_failure_propagates(self):
        original = RuntimeError("stream completion failed")
        self.stream.synchronize.side_effect = original
        with self.assertRaises(RuntimeError) as caught:
            self.run_warmup()
        self.assertIs(caught.exception, original)

    def test_startup_call_precedes_model_engine_allocation(self):
        # Structural integration check, not a substitute for the real service gate.
        package = PACKAGE
        tree = ast.parse((package / "server" / "backend_manager.py").read_text())
        manager = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BackendManager"
        )
        start = next(
            n
            for n in manager.body
            if isinstance(n, ast.FunctionDef) and n.name == "start"
        )
        warm = next(
            n
            for n in ast.walk(start)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "prepare_model_runtime"
        )

        def line(method):
            return next(
                n.lineno
                for n in ast.walk(start)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == method
            )

        self.assertLess(line("update_engine_config_from_model_config"), warm.lineno)
        self.assertLess(warm.lineno, line("from_model_configs"))
        self.assertEqual(ast.unparse(warm.func.value), "get_current_device()")
        self.assertEqual(
            [ast.unparse(arg) for arg in warm.args],
            ["model_config", "engine_config"],
        )


if __name__ == "__main__":
    unittest.main()
