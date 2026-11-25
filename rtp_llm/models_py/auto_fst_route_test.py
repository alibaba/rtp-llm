"""CPU routing contracts; simulated accelerator capabilities are not GPU coverage."""

import importlib
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod as Method
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


@register_model("_test_default_fst_route")
class _Owner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(6), requires_grad=False)


def _loader(root, device="cuda:0", **kwargs):
    return NewModelLoader(
        {"model_type": "_test_default_fst_route"},
        NewLoaderConfig(device=device, **kwargs),
        model_path=str(root),
    )


@contextmanager
def _simulated_device(
    backend="Cuda", machine="x86_64", free=8 << 30, *, real_reader=False
):
    # Simulate device capabilities and the wheel interface in policy-only
    # tests; selection, symbol checks, memory arithmetic, Gloo and checkpoint
    # metadata remain real. The separate reader test uses the installed wheel.
    # Import before changing sys.modules; otherwise restoring the wheel fake
    # can also remove this lazy module (leaving a stale parent attribute).
    host_module = (
        importlib.import_module("rtp_llm.models_py.fst_host_backend")
        if backend in ("Ppu", "ROCm")
        else None
    )
    with ExitStack() as stack:
        if not real_reader:

            class ReaderInterface:
                def __init__(self, **kwargs):
                    raise AssertionError("policy selection must not read data")

                def iterate_weights(self, *args, **kwargs):
                    raise AssertionError("policy selection must not read data")

            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "fastsafetensors": SimpleNamespace(
                            ParallelLoader=ReaderInterface, SingleGroup=ReaderInterface
                        )
                    },
                )
            )
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "LOAD_METHOD": "",
                    "PPU_HOME": "/simulated-ppu" if backend == "Ppu" else "",
                    "FASTSAFETENSORS_NOGDS": "0",
                    "FASTSAFETENSORS_USE_SHM": "0",
                },
            )
        )
        stack.enter_context(patch("platform.machine", return_value=machine))
        stack.enter_context(patch.object(torch, "__version__", "2.8.0"))
        stack.enter_context(
            patch.object(torch.version, "hip", "6.4" if backend == "ROCm" else None)
        )
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
        stack.enter_context(
            patch.object(torch.cuda, "mem_get_info", return_value=(free, 8 << 30))
        )
        stack.enter_context(
            patch.object(NewModelLoader, "_cuda_cufile_available", return_value=False)
        )
        probe = None
        if host_module is not None:
            probe = stack.enter_context(
                patch.object(host_module, "fst_host_backend_reason", return_value=None)
            )
        yield probe


def _select(loader, requested=None, paths=None):
    return loader._select_load_method(
        loader.load_config.load_method if requested is None else requested,
        _Owner({}, loader.load_config),
        loader._checkpoint_files() if paths is None else paths,
    )


def _rank_decision(rank, root, init):
    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        loader = _loader(
            root,
            load_method=Method.AUTO,
            parallelism_config=SimpleNamespace(world_size=2, pp_size=1),
        )
        paths = [] if rank == 1 else loader._checkpoint_files()
        with _simulated_device():
            assert _select(loader, paths=paths) == Method.SCRATCH
            case = unittest.TestCase()
            with case.assertRaisesRegex(
                RuntimeError, "rank 1: checkpoint shard selection is empty"
            ):
                _select(loader, Method.FASTSAFETENSORS, paths)
            # A failed vote is not a negative vote: neither AUTO nor forced FST
            # can safely continue after losing contact with a participant.
            with patch.object(dist, "all_gather_object", side_effect=RuntimeError("lost peer")):
                for requested in (Method.AUTO, Method.FASTSAFETENSORS):
                    with case.assertRaisesRegex(RuntimeError, "cannot synchronize") as failed:
                        _select(loader, requested, paths)
                    case.assertEqual(str(failed.exception.__cause__), "lost peer")
    finally:
        dist.destroy_process_group()


class _CheckpointCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.expected = torch.arange(6, dtype=torch.float32)
        save_file({"weight": self.expected}, str(self.root / "model.safetensors"))


class DefaultFstRouteTest(_CheckpointCase):
    def test_legacy_ptuning_bin_overrides_base_checkpoint(self):
        overlay = self.root / "overlay"
        overlay.mkdir()
        torch.save({"state_dict": {"weight": self.expected + 10}},
                   overlay / "pytorch_model.bin", _use_new_zipfile_serialization=False)
        loader = _loader(self.root, device="cpu", load_method=Method.SCRATCH)
        loader.model_config["ptuning_path"] = str(overlay)
        torch.testing.assert_close(loader.load().weight, self.expected + 10)

    def test_load_method_environment_only_supplies_an_omitted_option(self):
        for env in (None, "", "auto", "scratch", "fastsafetensors", "invalid"):
            with self.subTest(env=env), patch.dict(os.environ):
                if env is None:
                    os.environ.pop("LOAD_METHOD", None)
                else:
                    os.environ["LOAD_METHOD"] = env
                if env == "invalid":
                    with self.assertRaisesRegex(ValueError, "LOAD_METHOD"):
                        _loader(self.root)
                else:
                    default = _loader(self.root).load_config.load_method
                    self.assertEqual(default, Method(env) if env else Method.AUTO)
                for explicit in Method:
                    with self.subTest(explicit=explicit):
                        loader = _loader(self.root, load_method=explicit.value)
                        self.assertEqual(loader.load_config.load_method, explicit)

    def test_explicit_auto_keeps_preflight_fallback_with_fst_environment(self):
        with patch.dict(os.environ, {"LOAD_METHOD": "fastsafetensors"}):
            loader = _loader(self.root, "cpu", load_method=Method.AUTO)
            torch.testing.assert_close(loader.load().weight, self.expected)
            self.assertEqual(loader.loaded_method, Method.SCRATCH)

        with _simulated_device("Ppu") as probe, patch.dict(
            os.environ, {"LOAD_METHOD": "fastsafetensors"}
        ):
            probe.return_value = "missing compatible native SHM reader"
            self.assertEqual(
                _select(_loader(self.root, load_method="auto")), Method.SCRATCH
            )
            with self.assertRaisesRegex(
                RuntimeError, "missing compatible native SHM reader"
            ):
                _select(_loader(self.root))

    def test_default_accelerators_enter_real_preflight(self):
        for backend, machine in (
            ("Cuda", "x86_64"),
            ("Cuda", "amd64"),
            ("Cuda", "aarch64"),
            ("Cuda", "arm64"),
            ("ROCm", "x86_64"),
            ("Ppu", "x86_64"),
        ):
            with self.subTest(backend=backend, machine=machine), _simulated_device(
                backend, machine
            ) as host_probe:
                loader = _loader(self.root)
                self.assertEqual(loader.load_config.load_method, Method.AUTO)
                with patch.object(
                    loader, "_fastsafetensors_api", wraps=loader._fastsafetensors_api
                ) as api:
                    self.assertEqual(_select(loader), Method.FASTSAFETENSORS)
                    api.assert_called_once_with()
                if host_probe is not None:
                    host_probe.assert_called_once_with(
                        backend, use_shm=True, use_nogds=False
                    )

    def test_preflight_failures_not_runtime_retry(self):
        loader = _loader(self.root, load_method=Method.AUTO)
        with _simulated_device(), patch.dict(sys.modules, {"fastsafetensors": None}):
            self.assertEqual(_select(loader), Method.SCRATCH)
            with self.assertRaisesRegex(RuntimeError, "fastsafetensors.*unavailable"):
                _select(loader, Method.FASTSAFETENSORS)
        with _simulated_device(free=1):
            self.assertEqual(_select(loader), Method.SCRATCH)
        for backend in ("Ppu", "ROCm"):
            with self.subTest(backend=backend), _simulated_device(
                backend
            ) as host_probe:
                host_probe.return_value = "missing compatible native SHM reader"
                self.assertEqual(_select(loader), Method.SCRATCH)
                with self.assertRaisesRegex(
                    RuntimeError, "missing compatible native SHM reader"
                ):
                    _select(loader, Method.FASTSAFETENSORS)

    def test_remote_rank_preflight_is_collective(self):
        torch.multiprocessing.start_processes(
            _rank_decision,
            args=(str(self.root), (self.root / "rendezvous").as_uri()),
            nprocs=2,
            start_method="fork",
        )

    def test_cpu_default_keeps_scratch(self):
        with patch.dict(os.environ, {"LOAD_METHOD": ""}):
            loader = _loader(self.root, "cpu")
            model = loader.load()
            self.assertEqual(loader.loaded_method, Method.SCRATCH)
            torch.testing.assert_close(model.weight, self.expected)
        with _simulated_device():
            self.assertEqual(
                _select(_loader(self.root, force_cpu_load_weights=True)), Method.SCRATCH
            )


class DefaultFstNativeReaderTest(_CheckpointCase):
    def test_consumer_failure_closes_active_native_reader_before_unwind(self):
        from rtp_llm.models_py import fst_file_plan, fst_range_reader

        for independent in (False, True):
            loader = _loader(self.root, device="cpu", load_method=Method.FASTSAFETENSORS)
            files = loader._checkpoint_files()
            observed = []

            def monitor(iterator):
                try:
                    yield from iterator
                finally:
                    observed.append("closed")

            def reject(owner, weights):
                name, _ = next(weights)
                self.assertEqual(name, "weight")
                raise ValueError("consumer rejected weight")

            owner, method = ((fst_range_reader, "iter_fst_range_tensors") if independent
                             else (loader, "_fastsafetensors_weights_iterator"))
            read = getattr(owner, method)
            with self.subTest(independent=independent), patch.object(
                fst_file_plan, "plan_fst_files",
                return_value=([], files) if independent else (files, []),
            ), patch.object(owner, method, side_effect=lambda *a, **k: monitor(read(*a, **k))), patch.object(
                _Owner, "load_weights", reject
            ):
                try:
                    loader.load()
                except ValueError as exc:
                    self.assertEqual(str(exc), "consumer rejected weight")
                    # Inspect inside the handler: traceback still owns the iterator.
                    self.assertEqual(observed, ["closed"])
                else:
                    self.fail("consumer failure did not propagate")
                self.assertIsNone(loader.loaded_method)

    def test_multifile_readspec_uses_native_batch_order_and_close(self):
        from fastsafetensors import ParallelLoader, SingleGroup
        from fastsafetensors.parallel_loader import TimingContext

        from rtp_llm.models_py.fastsafetensors_read_spec import (
            build_read_spec_parallel_loader,
        )
        from rtp_llm.models_py.read_spec import ReadSpec
        from rtp_llm.models_py.v4_fst_nogds import checked_nogds_loader

        files = []
        for i in range(3):
            path = str(self.root / f"shard-{i}.safetensors")
            save_file({f"w{i}": self.expected + i}, path)
            files.append(path)
        base = checked_nogds_loader(ParallelLoader)
        ordered = hasattr(base, "_take_ordered_batch")
        if ordered:

            class ReversedCompletion(base):
                def _load_single_batch(self, producer_id, loader, batch_id, files):
                    if batch_id == 0 and self.max_concurrent_producers > 1:
                        with self._result_condition:
                            if not self._result_condition.wait_for(
                                lambda: 1 in self._completed_batches, timeout=10
                            ):
                                raise AssertionError(
                                    "batch 1 did not finish before batch 0"
                                )
                    return super()._load_single_batch(
                        producer_id, loader, batch_id, files
                    )

            base = ReversedCompletion
        cls = build_read_spec_parallel_loader(base, TimingContext)
        for q in (0, -1):
            with self.subTest(queue_size=q):
                loader = cls(
                    pg=SingleGroup(),
                    hf_weights_files=files,
                    read_spec_resolver=ReadSpec.full,
                    device="cpu",
                    nogds=True,
                    use_tqdm_on_load=False,
                    queue_size=q,
                    max_concurrent_producers=2 if ordered and q == 0 else 1,
                )
                try:
                    observed = [(k, v.clone()) for k, v in loader.iterate_weights()]
                    self.assertEqual([k for k, _ in observed], ["w0", "w1", "w2"])
                    for i, (_, value) in enumerate(observed):
                        torch.testing.assert_close(value, self.expected + i)
                    if ordered:
                        self.assertEqual(sorted(loader.batch_broadcast_stats), [0, 1, 2])
                finally:
                    loader.close()

    def test_explicit_native_cpu_read(self):
        loader = _loader(self.root, "cpu", load_method=Method.FASTSAFETENSORS)
        model = loader.load()
        self.assertEqual(loader.loaded_method, Method.FASTSAFETENSORS)
        torch.testing.assert_close(model.weight, self.expected)

    def test_failure_after_auto_fst_read_does_not_reopen_scratch(self):
        loader = _loader(self.root, load_method=Method.AUTO)
        original = loader._fastsafetensors_weights_iterator
        observed = []

        def fail_after_real_read(paths, device, *args):
            # Route the simulated accelerator's selected reader to a real
            # native CPU FST load, then inject an error after the first tensor.
            reader = original(paths, "cpu", *args)
            try:
                name, value = next(reader)
                observed.append(name)
                yield name, value
                raise OSError("injected after native read")
            finally:
                reader.close()

        with _simulated_device(real_reader=True), patch.object(
            loader, "_prepare_model_for_fastsafetensors"
        ), patch.object(
            loader,
            "_fastsafetensors_weights_iterator",
            side_effect=fail_after_real_read,
        ), patch.object(
            weight_mapper,
            "get_all_weights",
            side_effect=AssertionError("unexpected scratch reopen"),
        ) as scratch:
            with self.assertRaisesRegex(OSError, "injected after native read"):
                loader.load()
            scratch.assert_not_called()
        self.assertEqual(observed, ["weight"])
        self.assertIsNone(loader.loaded_method)


if __name__ == "__main__":
    unittest.main()
