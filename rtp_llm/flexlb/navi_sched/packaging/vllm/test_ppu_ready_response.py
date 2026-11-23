import os

"""Exercise the startup handshake without importing the PPU device runtime."""

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import msgspec


class SchedulerCapabilityTest(unittest.TestCase):
    def test_real_scheduler_capability_matches_forward_admission(self):
        root = Path(os.environ.get("VLLM_SOURCE_ROOT", Path.cwd())).resolve()
        scheduler = ast.parse((root / "vllm/v1/core/sched/scheduler.py").read_text())
        constructor = next(
            node
            for node in ast.walk(scheduler)
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
        )
        constructor = next(
            node
            for node in constructor.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assignments = [
            node
            for node in constructor.body
            if isinstance(node, ast.Assign)
            and any(
                (isinstance(target, ast.Name) and target.id == "supports_mm_inputs")
                or (
                    isinstance(target, ast.Attribute)
                    and target.attr == "supports_mm_inputs"
                )
                for target in node.targets
            )
        ]
        package = Path(importlib.util.find_spec("navi_sched").origin).parent
        observer = ast.parse(
            (package / "adapters/vllm/VLLMEngineObserver.py").read_text()
        )
        admission = next(
            node
            for node in ast.walk(observer)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_enqueue_forward_batch"
        )
        check = next(
            node
            for node in ast.walk(admission)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "unsupported_request_ids"
                for target in node.targets
            )
        )
        # Execute the production constructor fragment and adapter predicate,
        # without loading a model or initializing a device runtime.
        for supported in (False, True):
            for features in (None, [object()]):
                with self.subTest(supported=supported, multimodal=bool(features)):
                    state = SimpleNamespace()
                    namespace = dict(
                        self=state,
                        vllm_config=SimpleNamespace(model_config=None),
                        mm_registry=SimpleNamespace(
                            supports_multimodal_inputs=lambda _: supported
                        ),
                    )
                    exec(
                        compile(
                            ast.Module(body=assignments, type_ignores=[]),
                            "scheduler capability",
                            "exec",
                        ),
                        namespace,
                    )
                    namespace.update(
                        decoded=[(1, SimpleNamespace(mm_features=features))],
                        cancelled_request_ids=set(),
                        scheduler=state,
                    )
                    exec(
                        compile(
                            ast.Module(body=[check], type_ignores=[]),
                            "forward admission",
                            "exec",
                        ),
                        namespace,
                    )
                    self.assertEqual(supported, state.supports_mm_inputs)
                    self.assertEqual(
                        {1} if features and not supported else set(),
                        namespace["unsupported_request_ids"],
                    )


class ReadyResponseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = (
            Path(os.environ.get("VLLM_SOURCE_ROOT", Path.cwd())).resolve()
            / "vllm/v1/engine"
        )
        response = next(
            node
            for node in ast.parse((engine / "__init__.py").read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "EngineCoreReadyResponse"
        )
        apply_response = next(
            node
            for node in ast.walk(ast.parse((engine / "core_client.py").read_text()))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_ready_response"
        )
        namespace = {"dataclass": dataclass, "msgspec": msgspec}
        exec(
            compile(
                ast.Module(body=[response, apply_response], type_ignores=[]),
                str(engine / "core_client.py"),
                "exec",
            ),
            namespace,
        )
        cls.apply_response = staticmethod(namespace["_apply_ready_response"])

    def make_frontend(self):
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=4096),
                cache_config=SimpleNamespace(num_gpu_blocks=None, block_size=16),
            )
        )

    def test_engine_resolved_block_size_reaches_frontend(self):
        for block_size in (16, 128, 512):
            with self.subTest(block_size=block_size):
                frontend = self.make_frontend()
                payload = msgspec.msgpack.encode(
                    dict(
                        max_model_len=2048,
                        num_gpu_blocks=1024,
                        dp_stats_address=None,
                        block_size=block_size,
                    )
                )
                self.apply_response(frontend, payload)
                self.assertEqual(
                    block_size, frontend.vllm_config.cache_config.block_size
                )
                self.assertEqual(1024, frontend.vllm_config.cache_config.num_gpu_blocks)
                self.assertEqual(2048, frontend.vllm_config.model_config.max_model_len)

    def test_old_payload_and_multiple_engines_preserve_existing_behavior(self):
        frontend = self.make_frontend()
        payload = msgspec.msgpack.encode(
            dict(
                max_model_len=2048,
                num_gpu_blocks=1024,
                dp_stats_address=None,
            )
        )
        self.apply_response(frontend, payload)
        self.apply_response(frontend, payload)
        self.assertEqual(16, frontend.vllm_config.cache_config.block_size)
        self.assertEqual(2048, frontend.vllm_config.cache_config.num_gpu_blocks)


if __name__ == "__main__":
    unittest.main()
