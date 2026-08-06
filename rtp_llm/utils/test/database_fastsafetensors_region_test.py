import contextlib
import sys
import types
import unittest
from typing import Iterator, List, Tuple

from rtp_llm.utils.database import CkptDatabase


class _FakeCkptFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name


class FastsafetensorsRegionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_fastsafetensors = sys.modules.get("fastsafetensors")
        self._had_fastsafetensors = "fastsafetensors" in sys.modules
        self._saved_parallel_loader = sys.modules.get("fastsafetensors.parallel_loader")
        self._had_parallel_loader = "fastsafetensors.parallel_loader" in sys.modules
        self._saved_model_loader = sys.modules.get("rtp_llm.model_loader")
        self._had_model_loader = "rtp_llm.model_loader" in sys.modules
        self._saved_per_expert = sys.modules.get(
            "rtp_llm.model_loader.per_expert_parallel_loader"
        )
        self._had_per_expert = (
            "rtp_llm.model_loader.per_expert_parallel_loader" in sys.modules
        )

    def tearDown(self) -> None:
        if self._had_fastsafetensors:
            sys.modules["fastsafetensors"] = self._saved_fastsafetensors
        else:
            sys.modules.pop("fastsafetensors", None)
        if self._had_parallel_loader:
            sys.modules["fastsafetensors.parallel_loader"] = self._saved_parallel_loader
        else:
            sys.modules.pop("fastsafetensors.parallel_loader", None)
        if self._had_model_loader:
            sys.modules["rtp_llm.model_loader"] = self._saved_model_loader
        else:
            sys.modules.pop("rtp_llm.model_loader", None)
        if self._had_per_expert:
            sys.modules["rtp_llm.model_loader.per_expert_parallel_loader"] = (
                self._saved_per_expert
            )
        else:
            sys.modules.pop("rtp_llm.model_loader.per_expert_parallel_loader", None)

    def test_allocation_context_starts_after_parallel_loader_init(self) -> None:
        events: List[Tuple[str, bool]] = []
        in_region = False

        def active() -> bool:
            return in_region

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeBackendLoader:
            def close(self) -> None:
                events.append(("close", active()))

        class FakeParallelLoader:
            def __init__(self, **kwargs) -> None:
                events.append(("init", active()))
                self.loader = FakeBackendLoader()

            def iterate_weights(self):
                events.append(("iterate_enter", active()))
                yield "weight", object()
                events.append(("iterate_after_yield", active()))

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.__path__ = []
        fake_module.__version__ = "0.1.19rc5"
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.ParallelLoader = FakeParallelLoader
        sys.modules["fastsafetensors"] = fake_module

        class FakeTimingContext:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self):
                self.elapsed_ms = 0.0
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        fake_parallel_loader = types.ModuleType("fastsafetensors.parallel_loader")
        fake_parallel_loader.TimingContext = FakeTimingContext
        sys.modules["fastsafetensors.parallel_loader"] = fake_parallel_loader

        fake_model_loader = types.ModuleType("rtp_llm.model_loader")
        fake_model_loader.__path__ = []
        fake_per_expert = types.ModuleType(
            "rtp_llm.model_loader.per_expert_parallel_loader"
        )
        fake_per_expert.PerExpertParallelLoader = FakeParallelLoader
        sys.modules["rtp_llm.model_loader"] = fake_model_loader
        sys.modules["rtp_llm.model_loader.per_expert_parallel_loader"] = fake_per_expert

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


if __name__ == "__main__":
    unittest.main()
