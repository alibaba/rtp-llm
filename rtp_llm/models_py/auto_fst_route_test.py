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


@register_model("_test_overlay_local_stage")
class _OverlayStage(_Owner):
    def __init__(self, config, load):
        super().__init__(config, load)
        self.bias = torch.nn.Parameter(torch.empty_like(self.weight), requires_grad=False)

    def checkpoint_weight_name_filter(self):
        return lambda name: name in ("weight", "bias")


@register_model("_test_draft_loan_preflight")
class _DraftWithLoan(_Owner):
    def __init__(self, config, load):
        super().__init__(config, load)
        self.head = torch.nn.Linear(2, 1, bias=False).requires_grad_(False)

    def prepare_borrowed_parameters(self, bindings):
        assert set(bindings) == {"head.weight"}


def _stage_loan_failure(rank, root, init):
    from rtp_llm.models_py.borrowed_parameters import BorrowedParameter
    from rtp_llm.models_py.module_base import _mark_loaded

    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=3, timeout=timedelta(seconds=10)
    )
    try:
        tail = dist.new_group([1, 2], timeout=timedelta(seconds=5))
        if rank != 0:
            # Only the last PP stage constructs the draft. Its second rank has
            # a three-column owner, while the draft expects two columns.
            owner = torch.nn.Linear(rank + 1, 1, bias=False).requires_grad_(False)
            _mark_loaded(owner, "weight")
            for method in (Method.AUTO, Method.FASTSAFETENSORS):
                loader = NewModelLoader(
                    {"model_type": "_test_draft_loan_preflight", "is_mtp": True},
                    NewLoaderConfig(device="cpu", load_method=method,
                                    parallelism_config=SimpleNamespace(pp_size=2)),
                    model_path=root,
                    parameter_aliases={"head.weight": BorrowedParameter(owner, "weight")},
                )
                # Supply the real two-rank stage group without initializing
                # the production GPU parallel runtime in this CPU test.
                with patch.object(loader, "_stage_group", return_value=tail), unittest.TestCase().assertRaisesRegex(
                    RuntimeError, "rank 1: ValueError: Borrowed parameter head.weight shape/dtype"
                ):
                    loader.load()
                assert loader.loaded_method is None
        dist.barrier()
    finally:
        dist.destroy_process_group()


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
        from rtp_llm.models_py.fst_file_plan import plan_fst_files
        from rtp_llm.models_py.read_spec import ReadSpec
        local = Path(root) / f"plan-{rank}"
        local.mkdir()
        shard = str(local / "same.safetensors")
        pg = dist.group.WORLD
        case = unittest.TestCase()
        for shape, dtype, key in [((rank + 1,), torch.float32, "w"), ((2,), torch.int32 if rank else torch.float32, "w"),
                                  ((2,), torch.float32, f"w{rank}")]:
            save_file({key: torch.ones(shape, dtype=dtype)}, shard)
            case.assertEqual(plan_fst_files([shard], local, pg, None, None), ([], [shard]))
        save_file({"w": torch.ones(2)}, shard)
        case.assertEqual(plan_fst_files([shard], local, pg, None, None), ([shard], []))
        def inconsistent_placement(name, shape):
            return ReadSpec.full(name, shape).with_targets((rank,), True)
        with case.assertRaisesRegex(ValueError, "Inconsistent expert placement"):
            plan_fst_files([shard], local, pg, None, None, resolver=inconsistent_placement)
        paths = [] if rank == 1 else loader._checkpoint_files()
        # Exercise the real load() preflight, not just the backend vote.
        for method in (Method.AUTO, Method.FASTSAFETENSORS):
            # Rank 1 fails even before there is a model instance or a reader.
            empty = Path(root) / "rank1-empty"
            if rank == 1:
                empty.mkdir(exist_ok=True)
            missing = _loader(empty if rank == 1 else root, "cpu", load_method=method)
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError, "rank 1: FileNotFoundError: No supported checkpoint files"
            ):
                missing.load()
            assert missing.loaded_method is None
            consumer = _loader(root, "cpu", load_method=method)
            failed_import = patch("rtp_llm.models_py.model_loader.get_model_class",
                                  side_effect=ImportError("missing owner backend")) if rank else ExitStack()
            with failed_import, case.assertRaisesRegex(RuntimeError, "rank 1: ImportError: missing owner backend"):
                consumer.load()
            def allocate(*args, **kwargs):
                if rank:
                    raise RuntimeError("rank-local destination exhausted")
            with patch.object(consumer, "_select_load_method", return_value=Method.FASTSAFETENSORS), patch.object(
                consumer, "_prepare_model_for_fastsafetensors", side_effect=allocate
            ), case.assertRaisesRegex(RuntimeError, "rank 1: RuntimeError: rank-local destination exhausted"):
                consumer.load()
            prepare = consumer._prepare_checkpoint
            def different_formats(*args):
                files, expert, meta, ft = prepare(*args)
                return files, expert, meta, object() if rank else ft
            with patch.object(consumer, "_prepare_checkpoint", side_effect=different_formats), case.assertRaisesRegex(
                RuntimeError, "inconsistent checkpoint formats"
            ):
                consumer.load()
            if rank == 1:
                constructor = patch.object(consumer, "_create_checkpoint_model",
                                           side_effect=ValueError("bad rank-local header"))
            else:
                constructor = ExitStack()
            with constructor, unittest.TestCase().assertRaisesRegex(
                RuntimeError, "rank 1: ValueError: bad rank-local header"
            ):
                consumer.load()
            with patch.object(consumer, "_model_checkpoint_name_filter",
                              return_value=lambda key: key == ("weight" if rank == 0 else "absent")):
                with unittest.TestCase().assertRaisesRegex(
                    RuntimeError, "rank 1: ValueError: Checkpoint filter did not match"
                ):
                    consumer.load()
            assert consumer.loaded_method is None
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
    def test_auto_memory_excludes_pure_host_but_not_mixed_shards(self):
        from rtp_llm.models_py.borrowed_parameters import FixedHostCheckpointStorage

        class HostLeaf(FixedHostCheckpointStorage, torch.nn.Module):
            pass

        loader = _loader(self.root, load_method=Method.AUTO)
        model = _Owner({}, loader.load_config)
        model.ple = HostLeaf()
        path = str(self.root / "model.safetensors")
        for mixed in (False, True):
            tensors = {"ple.weight": self.expected}
            if mixed:
                tensors["weight"] = self.expected.clone()
            save_file(tensors, path)
            with self.subTest(mixed=mixed), _simulated_device(free=4096):
                self.assertEqual(loader._select_load_method(Method.AUTO, model, [path]),
                                 Method.SCRATCH if mixed else Method.FASTSAFETENSORS)

    def test_legacy_ptuning_bin_overrides_base_checkpoint(self):
        overlay = self.root / "overlay"
        overlay.mkdir()
        torch.save({"state_dict": {"weight": self.expected + 10}},
                   overlay / "pytorch_model.bin", _use_new_zipfile_serialization=False)
        loader = _loader(self.root, device="cpu", load_method=Method.SCRATCH)
        loader.model_config["ptuning_path"] = str(overlay)
        torch.testing.assert_close(loader.load().weight, self.expected + 10)

    def test_overlay_can_supply_entire_local_stage_but_not_hide_missing_weights(self):
        overlay = self.root / "overlay"
        overlay.mkdir()
        save_file({"other_stage.weight": self.expected}, str(self.root / "model.safetensors"))
        for keys in (("weight", "bias"), ("bias",), ("other_stage.bias",)):
            torch.save({k: self.expected + 5 for k in keys}, overlay / "pytorch_model.bin")
            for method in (Method.SCRATCH, Method.AUTO):
                loader = _loader(self.root, device="cpu", load_method=method)
                loader.model_config["model_type"] = "_test_overlay_local_stage"
                loader.model_config["ptuning_path"] = str(overlay)
                with self.subTest(keys=keys, method=method):
                    if len(keys) == 2:
                        loaded = loader.load()
                        torch.testing.assert_close(loaded.weight, self.expected + 5)
                        torch.testing.assert_close(loaded.bias, self.expected + 5)
                    else:
                        message = ("missing required checkpoint parameters" if keys == ("bias",)
                                   else "filter did not match")
                        with self.assertRaisesRegex((ValueError, RuntimeError), message):
                            loader.load()
                        self.assertIsNone(loader.loaded_method)

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

    def test_failed_draft_binding_only_votes_with_tail_stage(self):
        torch.multiprocessing.start_processes(
            _stage_loan_failure,
            args=(str(self.root), (self.root / "stage-rendezvous").as_uri()),
            nprocs=3,
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
    def test_explicit_fst_with_whole_stage_in_overlay(self):
        save_file({"other_stage.weight": self.expected}, str(self.root / "model.safetensors"))
        overlay = self.root / "overlay"
        overlay.mkdir()
        torch.save({"weight": self.expected, "bias": self.expected + 1}, overlay / "pytorch_model.bin")
        loader = _loader(self.root, device="cpu", load_method=Method.FASTSAFETENSORS)
        loader.model_config.update(model_type="_test_overlay_local_stage", ptuning_path=str(overlay))
        model = loader.load()
        self.assertEqual(loader.loaded_method, Method.FASTSAFETENSORS)
        torch.testing.assert_close(model.weight, self.expected)
        torch.testing.assert_close(model.bias, self.expected + 1)

    def test_native_scatter_applies_residual_slices_and_matches_scratch(self):
        from fastsafetensors import ParallelLoader, SingleGroup
        from fastsafetensors.parallel_loader import TimingContext
        from rtp_llm.models_py.fastsafetensors_read_spec import build_read_spec_parallel_loader
        from rtp_llm.models_py.read_spec import ReadSpec
        from rtp_llm.models_py.v4_fst_nogds import checked_nogds_loader

        path = str(self.root / "model.safetensors")
        value = torch.arange(24).reshape(8, 3)
        save_file({"weight": value}, path)
        cls = build_read_spec_parallel_loader(checked_nogds_loader(ParallelLoader), TimingContext)
        for indices in ((slice(None), slice(None)), (slice(None, None, 2), -1),
                        (slice(0, 0), slice(None)), (2, slice(1, None))):
            def resolve(name, shape):
                return ReadSpec(name, name, tuple(shape), slices=indices, shard_dim=0)
            reader = cls(pg=SingleGroup(), hf_weights_files=[path], read_spec_resolver=resolve,
                         device="cpu", nogds=True, use_tqdm_on_load=False, queue_size=0)
            try:
                with self.subTest(indices=indices):
                    [(name, actual)] = [(n, v.tensor.clone()) for n, v in reader.iterate_weights()]
                    [(key, expected)] = weight_mapper.get_all_weights([path], read_spec_resolver=resolve)
                    self.assertEqual(name, key)
                    torch.testing.assert_close(actual, expected.tensor)
                    torch.testing.assert_close(actual, value[indices])
            finally:
                reader.close()
        for rank in (0, 1):
            spec = ReadSpec("w", "w", (8, 3), slices=(slice(rank*4, rank*4+4, 2), 1),
                            shard_dim=0, shard_count=2, shard_rank=rank)
            torch.testing.assert_close(spec.materialize_scattered(value.chunk(2)[rank]).tensor,
                                       spec.materialize(value).tensor)
            with self.assertRaisesRegex(ValueError, "cross the hinted shard"):
                ReadSpec("w", "w", (8, 3), slices=(slice(None), 1),
                         shard_dim=0, shard_count=2, shard_rank=rank)

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
