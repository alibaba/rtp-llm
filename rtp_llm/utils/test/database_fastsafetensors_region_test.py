import contextlib
import json
import os
import sys
import types
import unittest
from typing import Iterator, List, Tuple
from unittest.mock import Mock, patch

from rtp_llm.utils.database import CkptDatabase


class _FakeCkptFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name


class FastsafetensorsLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.database = object.__new__(CkptDatabase)
        self.database.pretrain_file_list = [
            _FakeCkptFile("b.safetensors"),
            _FakeCkptFile("a.safetensors"),
        ]
        self.loader = Mock(spec=["iterate_weights", "close"])
        self.package = types.ModuleType("fastsafetensors")
        self.package.AutoLoader = Mock(return_value=self.loader)
        self.group = Mock()
        self.group.rank.return_value = 0
        self.package.SingleGroup = Mock(return_value=self.group)
        self.module_patch = patch.dict(sys.modules, fastsafetensors=self.package)
        self.module_patch.start()
        self.distributed_patch = patch(
            "torch.distributed.is_initialized", return_value=False
        )
        self.distributed_patch.start()

    def tearDown(self) -> None:
        self.distributed_patch.stop()
        self.module_patch.stop()

    def _iterator(self, **kwargs):
        return self.database.fastsafetensors_weights_iterator(
            "cuda", use_tqdm_on_load=False, **kwargs
        )

    def test_exhaustion_closes_once_and_forwards_main_options(self) -> None:
        tensor = object()
        self.loader.iterate_weights.return_value = iter([("weight", tensor)])
        stacked = {"stacked": "experts.{expert_id}.weight"}
        local_filter = {"weight"}.__contains__
        iterator = self._iterator(
            stacked_key_config=stacked, local_copyout_filter=local_filter
        )

        self.assertEqual(list(iterator), [("weight", tensor)])
        iterator.close()
        self.loader.close.assert_called_once_with()
        self.package.AutoLoader.assert_called_once_with(
            self.group,
            ["a.safetensors", "b.safetensors"],
            device="cuda:0",
            local_copyout_filter=local_filter,
            stacked_moe_tensors=stacked,
        )

    def test_dual_failure_preserves_iteration_error(self) -> None:
        primary = ValueError("iteration failed")
        self.loader.iterate_weights.side_effect = primary
        self.loader.close.side_effect = RuntimeError("close failed")

        with self.assertLogs(level="WARNING") as logs:
            with self.assertRaises(ValueError) as raised:
                list(self._iterator())

        self.assertIs(raised.exception, primary)
        self.assertIn("preserving active error", "\n".join(logs.output))
        self.assertIn("close failed", "\n".join(logs.output))
        self.loader.close.assert_called_once_with()

    def test_early_generator_close_closes_delegate_and_loader_once(self) -> None:
        delegate_closed = Mock()

        def weights():
            try:
                yield "weight", object()
            finally:
                delegate_closed()

        self.loader.iterate_weights.return_value = weights()
        iterator = self._iterator()
        next(iterator)
        iterator.close()
        iterator.close()

        delegate_closed.assert_called_once_with()
        self.loader.close.assert_called_once_with()


class FastsafetensorsRegionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_fastsafetensors = sys.modules.get("fastsafetensors")
        self._had_fastsafetensors = "fastsafetensors" in sys.modules

    def tearDown(self) -> None:
        if self._had_fastsafetensors:
            sys.modules["fastsafetensors"] = self._saved_fastsafetensors
        else:
            sys.modules.pop("fastsafetensors", None)

    def test_allocation_context_starts_after_auto_loader_init(self) -> None:
        events: List[Tuple[str, bool]] = []
        in_region = False

        def active() -> bool:
            return in_region

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                events.append(("init", active()))

            def iterate_weights(self):
                events.append(("iterate_enter", active()))
                yield "weight", object()
                events.append(("iterate_after_yield", active()))

            def close(self) -> None:
                events.append(("close", active()))

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.__path__ = []
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        @contextlib.contextmanager
        def allocation_context() -> Iterator[None]:
            nonlocal in_region
            events.append(("context_enter", active()))
            in_region = True
            try:
                yield
            finally:
                in_region = False
                events.append(("context_exit", active()))

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        for _key, _tensor in database.fastsafetensors_weights_iterator(
            "cuda",
            use_tqdm_on_load=False,
            allocation_context=allocation_context,
        ):
            events.append(("consumer", active()))

        self.assertEqual(
            events,
            [
                ("init", False),
                ("context_enter", False),
                ("iterate_enter", True),
                ("consumer", True),
                ("iterate_after_yield", True),
                ("context_exit", False),
                ("close", False),
            ],
        )

    def test_default_stacked_experts_are_split_before_delivery(self) -> None:
        observed_split_templates = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                stacked_moe_tensors=None,
            ) -> None:
                observed_split_templates.append(stacked_moe_tensors)

            def iterate_weights(self):
                for expert_id in range(3):
                    yield f"experts.{expert_id}.weight", f"expert-{expert_id}"
                yield "plain", "plain-tensor"

            def close(self) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.__path__ = []
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        result = list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                use_tqdm_on_load=False,
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
            )
        )

        self.assertEqual(
            [key for key, _ in result],
            [
                "experts.0.weight",
                "experts.1.weight",
                "experts.2.weight",
                "plain",
            ],
        )
        self.assertEqual(
            [tensor for _, tensor in result[:3]],
            ["expert-0", "expert-1", "expert-2"],
        )
        self.assertEqual(result[3], ("plain", "plain-tensor"))
        self.assertEqual(
            observed_split_templates,
            [{"stacked": "experts.{expert_id}.weight"}],
        )

    def test_rank_local_copyout_filter_is_forwarded(self) -> None:
        observed_filters = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                stacked_moe_tensors=None,
            ) -> None:
                observed_filters.append(local_copyout_filter)

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.__path__ = []
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        predicate = {"needed"}.__contains__
        list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                use_tqdm_on_load=False,
                local_copyout_filter=predicate,
            )
        )

        self.assertEqual(observed_filters, [predicate])

    def test_force_nogds_overrides_config_json(self) -> None:
        observed_config = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                observed_config.append(
                    json.loads(os.environ["FASTSAFETENSORS_CONFIG_JSON"])
                )

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.__path__ = []
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        with (
            patch.dict(sys.modules, {"fastsafetensors": fake_module}),
            patch.dict(
                os.environ,
                {"FASTSAFETENSORS_CONFIG_JSON": '{"loader":"fuse-shm"}'},
                clear=False,
            ),
        ):
            list(
                database.fastsafetensors_weights_iterator(
                    "cuda", use_tqdm_on_load=False, force_nogds=True
                )
            )

        self.assertEqual(
            observed_config,
            [{"loader": "base", "base": {"copier_type": "nogds"}}],
        )


if __name__ == "__main__":
    unittest.main()
