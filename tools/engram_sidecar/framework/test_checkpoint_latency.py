"""CPU contracts for the real-table latency harness; no fabricated GPU pass."""
import copy
from contextlib import nullcontext, redirect_stdout
import ctypes
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("checkpoint_latency_under_test", Path(__file__).with_name("checkpoint_latency.py"))
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


class LatencyContractTest(unittest.TestCase):
    def options(self, root):
        return SimpleNamespace(checkpoint=root / "consumer", source_checkpoint=root / "source",
                               bundle=root / "bundle.json", private_inputs=root / "oracle.json",
                               output=root / "new-run", batches="1,2,4,8,16,32,64", layers="first",
                               modes="eager,graph", distributions="hot,random", samples=100,
                               warmup=20, device=0, cpu_slot=0, seed=7,
                               layer_mode="single", memory_headroom_gib=16,
                               discard_private_source_cache=False,
                               source_cache_numa=None,
                               calibration_samples=100, threshold_us=100.0)

    def test_default_screen_and_all_supported_decode_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            args = h.check_options(self.options(Path(directory)))
        self.assertEqual(args.batches, [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(args.modes, ["eager", "graph"])
        self.assertIsNone(args.layers)

    def test_single_default_and_together_cover_actual_model_layers_in_order(self):
        self.assertEqual(h.layer_groups((1, 14), None, "single"), [(1,)])
        self.assertEqual(h.layer_groups((1, 14), [14, 1], "single"), [(14,), (1,)])
        self.assertEqual(h.layer_groups((1, 14), None, "together"), [(1, 14)])
        self.assertEqual(h.layer_groups((1, 14), [14, 1], "together"), [(1, 14)])
        for selected in ([1], [2, 14], [1, 1], []):
            with self.assertRaises(ValueError):
                h.layer_groups((1, 14), selected, "together")

    def test_source_numa_is_explicit_and_requires_private_cache_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.options(Path(directory))
            self.assertIsNone(h.check_options(copy.copy(base)).source_cache_numa)
            for node, advice in ((1, False), (-1, True), (True, True), (1.5, True)):
                args = copy.copy(base)
                args.source_cache_numa, args.discard_private_source_cache = node, advice
                with self.subTest(node=node, advice=advice), self.assertRaises(ValueError):
                    h.check_options(args)
            base.source_cache_numa, base.discard_private_source_cache = 1, True
            self.assertEqual(h.check_options(base).source_cache_numa, 1)

    def test_together_invokes_each_layer_once_with_its_own_ids_and_returns_both(self):
        calls, ids, results = [], {1: object(), 14: object()}, {1: object(), 14: object()}
        def embedding(layer):
            def forward(value, device):
                calls.append((layer, value, device))
                return results[layer]
            return forward
        layers = {1: embedding(1), 14: embedding(14)}
        actual = h.invoke_lookup_chain(layers, (1, 14), ids, "cuda:0")
        self.assertEqual(calls, [(1, ids[1], "cuda:0"), (14, ids[14], "cuda:0")])
        self.assertEqual(actual, (results[1], results[14]))
        self.assertIs(h.invoke_lookup_chain(layers, (1,), ids, "cuda:0"), results[1])

    def test_joint_output_verification_reaches_both_paths_and_layers_and_rejects_missing(self):
        outputs = {"local_raw": (object(), object()), "remote_raw": (object(), object())}
        grouped = h.split_layer_outputs(outputs, (1, 14))
        for position, layer in enumerate((1, 14)):
            for name in outputs:
                self.assertIs(grouped[layer][name], outputs[name][position])
        for value in (object(), (object(),), (object(), object(), object())):
            with self.assertRaises(AssertionError):
                h.split_layer_outputs(dict(outputs, remote_raw=value), (1, 14))
        self.assertIs(h.split_layer_outputs({"local_raw": outputs["local_raw"][0]}, (1,))[1]["local_raw"], outputs["local_raw"][0])

    def test_explicit_layers_and_invalid_lists(self):
        self.assertEqual(h.integer_list("1,14"), [1, 14])
        for value in ("", "1,1", "1,-1", " 1", "1,", "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                h.integer_list(value)
        with self.assertRaises(ValueError):
            h.integer_list("3", allowed=(1, 2, 4))
        for value in ("eager,eager", "graph,other", ""):
            with self.assertRaises(ValueError):
                h.choice_list(value, ("eager", "graph"))

    def test_rejects_invalid_protocol_before_any_cuda_import(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.options(Path(directory))
            for key, value in (("samples", 0), ("samples", 1_000_001), ("warmup", 0),
                               ("device", -1), ("cpu_slot", -1), ("seed", -1),
                               ("calibration_samples", -1), ("threshold_us", float("nan")),
                               ("memory_headroom_gib", 0), ("layer_mode", "invalid"),
                               ("checkpoint", Path("relative"))):
                args = copy.copy(base)
                setattr(args, key, value)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    h.check_options(args)

    def test_refuses_prior_output_instead_of_overwriting_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.options(Path(directory))
            args.output.mkdir()
            with self.assertRaises(FileExistsError):
                h.check_options(args)

    def test_numa_requires_all_resident_pages_on_selected_node(self):
        text = "1000 default file=/dev/shm/rtp_llm_engram_1_abc N0=12 kernelpagesize_kB=64\n"
        self.assertEqual(h.parse_numa_pages(text, 0x1000, 0)["node_pages"], {"0": 12})
        for other in (text.replace("N0=12", "N1=12"), text.replace("N0=12", "N0=11 N1=1"),
                      text.replace("N0=12", "N0=0"), text + text, text.replace("1000", "2000")):
            with self.assertRaises(RuntimeError):
                h.parse_numa_pages(other, 0x1000, 0)

    def test_resolves_current_cgroup_from_mount_root_without_guessing(self):
        mount = "29 23 0:26 /jobs/task /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        self.assertEqual(h.resolve_cgroup_directory("0::/jobs/task/worker\n", mount),
                         (Path("/sys/fs/cgroup/worker"), Path("/sys/fs/cgroup"), False))
        namespaced = "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        self.assertEqual(h.resolve_cgroup_directory("0::/\n", namespaced)[0], Path("/sys/fs/cgroup"))
        for membership, mounts in (("1:memory:/x", mount), ("0::/unrelated", mount), ("0::/jobs/task", mount + mount)):
            with self.assertRaises(RuntimeError):
                h.resolve_cgroup_directory(membership, mounts)

    def test_only_hierarchy_root_may_omit_limit_and_missing_usage_always_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "worker"; child.mkdir()
            (root / "memory.current").write_text("10")
            (child / "memory.current").write_text("5")
            (child / "memory.max").write_text("100")
            values = h.read_cgroup_memory(child, root, True)
            self.assertEqual([item["max_bytes"] for item in values], [100, None])
            self.assertEqual(values[-1]["limit_source"], "root_no_limit_interface")
            with self.assertRaises(FileNotFoundError):
                h.read_cgroup_memory(child, root, False)
            (child / "memory.max").unlink()
            with self.assertRaises(FileNotFoundError):
                h.read_cgroup_memory(child, root, True)
            (root / "memory.current").unlink()
            with self.assertRaises(FileNotFoundError):
                h.read_cgroup_memory(root, root, True)

    def test_memory_kib_fields_accept_numa_prefix_and_reject_missing(self):
        self.assertEqual(h.parse_kib_field("Node 0 MemFree: 100 kB\n", "MemFree"), 102400)
        self.assertEqual(h.parse_kib_field("MemAvailable: 200 kB\n", "MemAvailable"), 204800)
        for text in ("", "MemFree: 12 bytes\n", "MemFree: 1 kB\nMemFree: 2 kB\n"):
            with self.assertRaises(RuntimeError):
                h.parse_kib_field(text, "MemFree")

    def memory(self):
        gib = 1 << 30
        return dict(cgroups=[dict(path="current", current_bytes=94 * gib, max_bytes=384 * gib)],
                    shm_free_bytes=162 * gib, host_available_bytes=400 * gib, numa_free_bytes=350 * gib)

    def test_loading_budget_includes_complete_source_cache_and_reserved_headroom(self):
        gib = 1 << 30
        args = dict(table_bytes=95 * gib, source_bytes=95 * gib, headroom_bytes=16 * gib, before=True)
        self.assertEqual(h.check_memory_budget(self.memory(), **args)["required_marginal_bytes"], 190 * gib)
        snapshot = self.memory()
        snapshot["cgroups"][0]["current_bytes"] = 188 * gib
        with self.assertRaisesRegex(RuntimeError, "cgroup"):
            h.check_memory_budget(snapshot, **args)

        snapshot = self.memory()
        snapshot["cgroups"].append(dict(path="ancestor", current_bytes=350 * gib, max_bytes=384 * gib))
        with self.assertRaisesRegex(RuntimeError, "cgroup"):
            h.check_memory_budget(snapshot, **args)
        for key in ("shm_free_bytes", "host_available_bytes", "numa_free_bytes"):
            snapshot = self.memory(); snapshot[key] = 0
            with self.assertRaises(RuntimeError):
                h.check_memory_budget(snapshot, **args)

    def test_numa_cache_estimate_requires_opt_in_and_never_weakens_post_load_checks(self):
        gib = 1 << 30
        snapshot = self.memory()
        snapshot.update(numa_free_bytes=80*gib, numa_cache_reclaim_estimate_bytes=140*gib)
        args = dict(table_bytes=95*gib, source_bytes=95*gib, headroom_bytes=16*gib, before=True)
        with self.assertRaisesRegex(RuntimeError, "NUMA"):
            h.check_memory_budget(snapshot, **args)
        result = h.check_memory_budget(snapshot, **args, allow_numa_cache_reclaim=True)
        self.assertEqual(result['numa_cache_reclaim_estimate_used_bytes'], 140*gib)
        self.assertFalse(result['memory_reservation_guaranteed'])
        snapshot['host_available_bytes'] = 100*gib
        with self.assertRaisesRegex(RuntimeError, "host"):
            h.check_memory_budget(snapshot, **args, allow_numa_cache_reclaim=True)
        snapshot['host_available_bytes'] = 400*gib
        snapshot['numa_free_bytes'] = 0
        args['before'] = False
        with self.assertRaisesRegex(RuntimeError, "NUMA"):
            h.check_memory_budget(snapshot, **args, allow_numa_cache_reclaim=True)

    def test_postload_budget_checks_actual_headroom_without_readding_loaded_table(self):
        gib = 1 << 30
        snapshot = self.memory()
        snapshot["cgroups"][0]["current_bytes"] = 360 * gib
        result = h.check_memory_budget(snapshot, table_bytes=95*gib, source_bytes=95*gib,
                                       headroom_bytes=16*gib, before=False)
        self.assertEqual(result["required_marginal_bytes"], 0)
        snapshot["cgroups"][0]["current_bytes"] = 370 * gib
        with self.assertRaises(RuntimeError):
            h.check_memory_budget(snapshot, table_bytes=95*gib, source_bytes=95*gib,
                                  headroom_bytes=16*gib, before=False)

    def test_private_file_cache_advice_requires_readonly_single_link_owned_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "table.safetensors"
            path.write_bytes(b"immutable source bytes")
            path.chmod(0o400)
            with patch.object(os, "posix_fadvise", create=True) as advice, patch.object(os, "POSIX_FADV_DONTNEED", 4, create=True):
                record = h.private_source_cache_advice(root, [path])
                fd, offset, size, mode = advice.call_args[0]
                self.assertEqual((offset, size, mode), (0, 0, 4))
                self.assertEqual(record[0]["reclaimed_bytes"], "NOT_ASSUMED")
                with self.assertRaises(OSError):
                    os.fstat(fd)
                advice.reset_mock()
                for mode in (0o600, 0o444, 0o644):
                    path.chmod(mode)
                    with self.assertRaises(ValueError):
                        h.private_source_cache_advice(root, [path])
                advice.assert_not_called()
                path.chmod(0o400)
                os.link(path, root / "shared-hardlink")
                with self.assertRaises(ValueError):
                    h.private_source_cache_advice(root, [path])
                advice.assert_not_called()

    def test_owned_hardlink_advice_is_explicit_and_preserves_bytes_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "table.safetensors"
            path.write_bytes(b"unchanged")
            path.chmod(0o400)
            os.link(path, root / "pinned-alias")
            with patch.object(os, "posix_fadvise", create=True), patch.object(os, "POSIX_FADV_DONTNEED", 4, create=True):
                result = h.private_source_cache_advice(root, [path], allow_hardlinks=True)
            self.assertEqual(result[0]["hardlinks"], 2)
            self.assertEqual(path.read_bytes(), b"unchanged")
            self.assertEqual(path.stat().st_nlink, 2)
            args = self.options(root)
            args.allow_source_hardlinks = True
            with self.assertRaisesRegex(ValueError, "cache discard"):
                h.check_options(args)

    def test_real_numa_metadata_survives_summary_validation_and_json_write(self):
        text = "1000 default file=/dev/shm/rtp_llm_engram_1_abc N1=1546692 kernelpagesize_kB=64\n"
        numa = h.parse_numa_pages(text, 0x1000, 1)
        summary = {"case_metadata": {"baseline_numa": numa}, "model_slo": "NOT_RUN"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            h.write_new_json(path, summary)
            actual = json.loads(path.read_text())
        self.assertEqual(actual, summary)
        self.assertEqual(actual["case_metadata"]["baseline_numa"]["node_pages"], {"1": 1546692})

    def test_selected_row_oracle_chunks_large_batch_without_losing_rows(self):
        chunks = h.chunk_selected_ids(list(range(1536)) + [-1, 7, 7])
        self.assertEqual(list(map(len, chunks)), [512, 512, 512])
        self.assertEqual([row for chunk in chunks for row in chunk], list(range(1536)))
        self.assertEqual(h.chunk_selected_ids([-1]), [])
        for values in ([-2], [True], [1.5]):
            with self.assertRaises(ValueError):
                h.chunk_selected_ids(values)

    def test_independent_source_oracle_preserves_order_duplicates_and_mask(self):
        layout = SimpleNamespace(layer_ids=(1, 14), head_dim=32)
        def selected(source, index, layout, rows):
            self.assertEqual(rows["14"], set())
            values = [{"row_id": row, "fp8_hex": (b"\x38" * 32).hex(), "e8m0_hex": "7f"}
                      for row in rows["1"]]
            return {"1": values, "14": []}, {}
        with patch.object(h.lookup, "read_selected_rows", side_effect=selected):
            result = h.source_rows(Path("/source"), {}, layout, 1, [[7, -1, 7]])
        self.assertEqual(result, [[[1.0] * 32, [0.0] * 32, [1.0] * 32]])

    def test_source_samples_must_match_admitted_remote_generation(self):
        parsed = {"1": {7: (b"\x38" * 32, b"\x7f")}}
        values = {"1": [{"row_id": 7, "fp8_hex": (b"\x38" * 32).hex(), "e8m0_hex": "7f"}]}
        with patch.object(h.lookup, "read_selected_rows", return_value=(values, {"a": "digest"})):
            self.assertEqual(h.source_sample_check(None, None, None, parsed), {"a": "digest"})
        values["1"][0]["e8m0_hex"] = "80"
        with patch.object(h.lookup, "read_selected_rows", return_value=(values, {})), self.assertRaisesRegex(ValueError, "source differs"):
            h.source_sample_check(None, None, None, parsed)

    def records(self, mode="graph", path="remote_raw", pair=0, gpu=5.0):
        return h.make_records(mode=mode, distribution="random", pair_id=pair, path=path,
                              order=["remote_raw", "local_raw"], begin_ns=1000, submit_ns=8000,
                              complete_ns=21000, gpu_us=gpu,
                              before_switches=(2, 3), after_switches=(2, 4))

    def test_wall_is_split_without_calling_sync_gpu_wait(self):
        gpu, wall = self.records()
        self.assertEqual(gpu["duration_us"], 5)
        self.assertEqual(wall["duration_us"], 20)
        self.assertEqual(wall["cpu_submit_call_us"], 7)
        self.assertEqual(wall["cpu_completion_sync_call_us"], 13)
        self.assertNotIn("cpu_graph_replay_call_us", wall)
        self.assertEqual(wall["thread_nivcsw_delta"], 1)

    def test_eager_and_graph_gpu_intervals_are_distinct_metrics(self):
        self.assertEqual(self.records("graph")[0]["clock"], "gpu_graph_event_interval")
        self.assertEqual(self.records("eager")[0]["clock"], "gpu_eager_event_interval")
        for gpu in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.records(gpu=gpu)

    def test_paired_extra_latency_retains_negative_values_and_violations(self):
        records = (self.records(path="remote_raw", pair=0, gpu=180) +
                   self.records(path="local_raw", pair=0, gpu=20) +
                   self.records(path="remote_raw", pair=1, gpu=10) +
                   self.records(path="local_raw", pair=1, gpu=30))
        summary = h.bench.summarize_records(records, h.COMPARISONS, bootstrap_iterations=0)
        delta = next(item for item in summary["paired_deltas"] if item["clock"] == "gpu_graph_event_interval")
        self.assertEqual(delta["paired_delta_stats"]["min_us"], -20)
        self.assertEqual(delta["paired_delta_stats"]["max_us"], 160)
        self.assertEqual(delta["paired_delta_stats"]["over_threshold_count"], 1)
        self.assertEqual(summary["model_slo"], "NOT_RUN")

    def test_result_creation_exclusive_and_capabilities_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            h.write_new_json(path, {"model_slo": "NOT_RUN"})
            self.assertEqual(json.loads(path.read_text())["model_slo"], "NOT_RUN")
            with self.assertRaises(FileExistsError):
                h.write_new_json(path, {})
            with self.assertRaises(ValueError):
                h.write_new_json(Path(directory) / "secret.json", {"fabric_handle": "secret"})

    def test_failed_cleanup_disables_finalizers_and_retains_resources(self):
        resource = SimpleNamespace(_registration_finalizer=SimpleNamespace(atexit=True),
                                   _lease_finalizer=SimpleNamespace(atexit=True))
        count = len(h._FAILED_RESOURCES)
        h.retain_failed(resource)
        self.assertFalse(resource._registration_finalizer.atexit)
        self.assertFalse(resource._lease_finalizer.atexit)
        self.assertIs(h._FAILED_RESOURCES[-1][0], resource)
        del h._FAILED_RESOURCES[count:]

    def test_top_level_failure_never_prints_sensitive_exception_arguments(self):
        secret = "PRIVATE_FABRIC_CAPABILITY_DO_NOT_LOG"
        error = TypeError("incompatible function arguments; Invoked with: b'" + secret + "'")
        error.code = secret
        argv = []
        for name in ("checkpoint", "source-checkpoint", "bundle", "private-inputs", "output"):
            argv += ["--" + name, "/private/example/" + name]
        output = io.StringIO()
        with patch.object(h, "check_options", return_value=object()), patch.object(h, "run", side_effect=error), redirect_stdout(output):
            self.assertEqual(h.main(argv), 1)
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["error_type"], "TypeError")
        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn("Invoked with", output.getvalue())
        self.assertNotIn("message", record)
        self.assertNotIn("code", record)

    def local_cleanup(self, status=0, drain_fail=False):
        calls = []
        def drain(device):
            calls.append("drain")
            if drain_fail:
                raise RuntimeError("GPU drain failed")
        def unregister(pointer):
            calls.append("unregister")
            return (status,)
        def detach():
            calls.append("detach_finalizer")
            return (object(), (), {})
        control = SimpleNamespace(device=0, storage=SimpleNamespace(data_ptr=lambda: 123),
                                  runtime=SimpleNamespace(cudaHostUnregister=unregister),
                                  _registration_finalizer=SimpleNamespace(alive=True, detach=detach),
                                  _lease_finalizer=lambda: calls.append("release_lease"))
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=drain, device=lambda _: nullcontext()))
        return control, torch, calls

    def test_baseline_unregister_requires_drain_and_lease_release_is_last(self):
        control, torch, calls = self.local_cleanup()
        h.close_local(control, torch)
        self.assertEqual(calls, ["drain", "unregister", "detach_finalizer", "release_lease"])

    def test_failed_drain_never_unregisters_pinned_baseline(self):
        control, torch, calls = self.local_cleanup(drain_fail=True)
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            h.close_local(control, torch)
        self.assertEqual(calls, ["drain"])

    def test_failed_unregister_preserves_finalizer_and_lease(self):
        control, torch, calls = self.local_cleanup(status=1)
        with self.assertRaisesRegex(RuntimeError, "Unregister failed"):
            h.close_local(control, torch)
        self.assertEqual(calls, ["drain", "unregister"])

    def alias_inputs(self, *, pointers=(111, 222), delta=0, status=0):
        def tensor(pointer):
            return SimpleNamespace(data_ptr=lambda: pointer)
        runtime = SimpleNamespace(cudaHostGetDevicePointer=Mock(side_effect=[(status, 111), (0, 222)]))
        control = SimpleNamespace(weight=tensor(11), scales=tensor(22), device=0, runtime=runtime)
        embedding = SimpleNamespace(_uva=tuple(tensor(pointer) for pointer in pointers))
        wrapper = Mock(return_value=embedding)
        torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda _: nullcontext(),
                                memory_allocated=Mock(side_effect=[10, 10 + delta])))
        return control, wrapper, torch, embedding

    def test_baseline_views_must_alias_both_registered_host_addresses_without_allocation(self):
        control, wrapper, torch, expected = self.alias_inputs()
        embedding, evidence = h.make_local_embedding(control, wrapper, torch)
        self.assertIs(embedding, expected)
        self.assertEqual(evidence, dict(weight_alias=True, scale_alias=True, cuda_allocated_delta_bytes=0))
        wrapper.assert_called_once_with(control.weight, control.scales, pinned=control)
        self.assertEqual(control.runtime.cudaHostGetDevicePointer.call_args_list[0].args, (11, 0))
        self.assertEqual(control.runtime.cudaHostGetDevicePointer.call_args_list[1].args, (22, 0))
        with tempfile.TemporaryDirectory() as directory:
            h.write_new_json(Path(directory) / "alias.json", evidence)

    def test_baseline_alias_or_allocation_failure_is_fatal(self):
        for options in (dict(pointers=(111, 223)), dict(pointers=(112, 222)), dict(delta=1), dict(delta=-1), dict(status=1)):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                h.make_local_embedding(*self.alias_inputs(**options)[:3])


class SourceCacheContractTest(unittest.TestCase):
    def setUp(self):
        self.retained = len(h._FAILED_RESOURCES)

    def tearDown(self):
        del h._FAILED_RESOURCES[self.retained:]

    def snapshot(self):
        gib = 1 << 30
        return dict(cgroups=[dict(path="current", current_bytes=100*gib, max_bytes=384*gib)],
                    host_available_bytes=300*gib, numa_free_bytes=120*gib,
                    source_numa_free_bytes=120*gib, shm_free_bytes=120*gib)

    def budget_args(self, phase="before_prefault"):
        gib = 1 << 30
        return dict(table_bytes=95*gib, source_bytes=95*gib, headroom_bytes=16*gib, phase=phase)

    def test_split_admission_requires_full_global_charge_but_each_numa_only_its_allocation(self):
        result = h.check_source_cache_budget(self.snapshot(), **self.budget_args())
        self.assertEqual(result["future_table_bytes"], 95 << 30)
        self.assertEqual(result["future_source_bytes"], 95 << 30)
        self.assertEqual(result["transient_touch_bytes"], 8 << 20)
        # 120 GiB per NUMA can admit 95 GiB each; 190 GiB on one NUMA cannot.
        with self.assertRaises(RuntimeError):
            h.check_memory_budget(self.snapshot(), **{k: v for k, v in self.budget_args().items() if k != "phase"}, before=True)
        for field in ("host_available_bytes", "numa_free_bytes", "source_numa_free_bytes", "shm_free_bytes"):
            snapshot = self.snapshot(); snapshot[field] = 0
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                h.check_source_cache_budget(snapshot, **self.budget_args())
        snapshot = self.snapshot()
        snapshot["cgroups"].append(dict(path="ancestor", current_bytes=200 << 30, max_bytes=384 << 30))
        with self.assertRaisesRegex(RuntimeError, "cgroup"):
            h.check_source_cache_budget(snapshot, **self.budget_args())

    def test_transient_margin_is_real_and_all_sources_are_charged_together(self):
        snapshot = self.snapshot()
        required = (95 + 95 + 16) * (1 << 30) + h.SOURCE_TOUCH_BYTES
        snapshot["host_available_bytes"] = required - 1
        with self.assertRaisesRegex(RuntimeError, "host"):
            h.check_source_cache_budget(snapshot, **self.budget_args())
        snapshot["host_available_bytes"] += 1
        h.check_source_cache_budget(snapshot, **self.budget_args())
        args = self.budget_args(); args["source_bytes"] *= 2
        with self.assertRaises(RuntimeError):
            h.check_source_cache_budget(self.snapshot(), **args)

    def test_resident_source_is_not_charged_again_but_actual_remaining_memory_still_checked(self):
        snapshot = self.snapshot()
        snapshot["cgroups"][0]["current_bytes"] = 220 << 30
        snapshot["source_numa_free_bytes"] = 20 << 30
        evidence = h.check_source_cache_budget(snapshot, **self.budget_args("prefaulted"))
        self.assertEqual(evidence["future_source_bytes"], 0)
        self.assertEqual(evidence["future_table_bytes"], 95 << 30)
        snapshot["cgroups"][0]["current_bytes"] = 370 << 30
        with self.assertRaises(RuntimeError):
            h.check_source_cache_budget(snapshot, **self.budget_args("after_load"))
        snapshot["cgroups"][0]["current_bytes"] = 350 << 30
        snapshot["numa_free_bytes"] = 20 << 30
        result = h.check_source_cache_budget(snapshot, **self.budget_args("after_load"))
        self.assertEqual((result["future_table_bytes"], result["future_source_bytes"]), (0, 0))

    def test_complete_mapping_counts_partial_tail_and_greater_than_4gib_without_truncation(self):
        self.assertEqual(h.rounded_mapping_bytes((1 << 32) + 1, 65536), (1 << 32) + 65536)
        text = "1000 default file=/source/shard N1=65537 kernelpagesize_kB=64\n"
        result = h.parse_complete_numa_pages(text, 0x1000, 1, (1 << 32) + 1, 65536)
        self.assertEqual(result["resident_pages"], 65537)
        self.assertEqual(result["node_pages"], {"1": 65537})
        for size, page in ((0, 4096), (-1, 4096), (True, 4096), (1, 0), (1, 3), (sys.maxsize, 4096), (sys.maxsize + 1, 4096)):
            with self.subTest(size=size, page=page), self.assertRaises(ValueError):
                h.rounded_mapping_bytes(size, page)

    def test_complete_mapping_refuses_missing_tail_mixed_nodes_or_incompatible_page_size(self):
        text = "1000 default file=/source/shard N1=3 kernelpagesize_kB=4\n"
        h.parse_complete_numa_pages(text, 0x1000, 1, 8193, 4096)
        for bad in (text.replace("N1=3", "N1=2"), text.replace("N1=3", "N1=4"),
                    text.replace("N1=3", "N1=2 N0=1"), text.replace("N1=3", "N0=3"),
                    text.replace("N1=3", ""), text.replace("kB=4", "kB=64"),
                    text.replace("1000", "2000"), text + text, text.replace("file=", "anon=")):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                h.parse_complete_numa_pages(bad, 0x1000, 1, 8193, 4096)

    def private_file(self, directory, name="shard", size=8193):
        root = Path(directory).resolve(); root.chmod(0o700)
        path = root / name
        path.write_bytes(b"a" * (size - 1) + b"z"); path.chmod(0o400)
        return root, path

    def test_real_readonly_mmap_reads_every_chunk_and_releases_fd_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.private_file(directory, size=h.SOURCE_TOUCH_BYTES + 129)
            mapping = h.ReadOnlySourceMapping(root, path)
            fd = mapping.fd
            try:
                self.assertEqual(mapping.touch(), path.stat().st_size)
                self.assertEqual(ctypes.string_at(mapping.address + mapping.size - 1, 1), b"z")
                self.assertEqual(mapping.libc.mmap.restype, ctypes.c_void_p)
                self.assertGreaterEqual(mapping.resident_bytes, mapping.size)
            finally:
                mapping.close()
            self.assertIsNone(mapping.address)
            with self.assertRaises(OSError):
                os.fstat(fd)
            mapping.close()
            with self.assertRaises(RuntimeError):
                mapping.touch()

    def test_private_mmap_rejects_writable_shared_symlink_and_nonprivate_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.private_file(directory)
            for mode in (0o600, 0o444):
                path.chmod(mode)
                with self.assertRaises(ValueError):
                    h.ReadOnlySourceMapping(root, path)
            path.chmod(0o400)
            os.link(path, root / "hardlink")
            with self.assertRaises(ValueError):
                h.ReadOnlySourceMapping(root, path)
            (root / "hardlink").unlink()
            (root / "symlink").symlink_to(path)
            with self.assertRaises(RuntimeError):
                h.ReadOnlySourceMapping(root, root / "symlink")
            root.chmod(0o755)
            with self.assertRaises(ValueError):
                h.ReadOnlySourceMapping(root, path)

    def test_mmap_failed_uses_errno_and_closes_fd_without_unmapping_sentinel(self):
        libc = SimpleNamespace(mmap=Mock(return_value=ctypes.c_void_p(-1).value), munmap=Mock())
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.private_file(directory)
            with patch.object(ctypes, "CDLL", return_value=libc), patch.object(ctypes, "get_errno", return_value=errno.ENOMEM):
                with self.assertRaises(OSError) as caught:
                    h.ReadOnlySourceMapping(root, path)
            self.assertEqual(caught.exception.errno, errno.ENOMEM)
            fd = libc.mmap.call_args.args[4]
            with self.assertRaises(OSError):
                os.fstat(fd)
            libc.munmap.assert_not_called()

    def test_failed_munmap_keeps_address_for_retry_and_closes_fd(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.private_file(directory)
            mapping = h.ReadOnlySourceMapping(root, path)
            original, address, fd = mapping.libc, mapping.address, mapping.fd
            mapping.libc = SimpleNamespace(munmap=Mock(return_value=-1))
            with patch.object(ctypes, "get_errno", return_value=errno.EINVAL), self.assertRaises(OSError):
                mapping.close()
            self.assertEqual(mapping.address, address)
            self.assertIsNone(mapping.fd)
            with self.assertRaises(OSError):
                os.fstat(fd)
            mapping.libc = original
            mapping.close()

    def test_cpu_selection_uses_original_affinity_intersection(self):
        with patch.object(Path, "read_text", return_value="64-67,70\n"):
            self.assertEqual(h.cpu_on_node(1, {0, 66, 70}), 66)
            with self.assertRaises(RuntimeError):
                h.cpu_on_node(1, {0})

    def fake_batch(self):
        batch = object.__new__(h.SourceCacheBatch)
        batch.node, batch.allowed_cpus = 1, {0, 64}
        batch.mappings = [SimpleNamespace(size=4097, resident_bytes=8192, touch=Mock(return_value=4097), close=Mock()),
                          SimpleNamespace(size=8193, resident_bytes=12288, touch=Mock(return_value=8193), close=Mock())]
        return batch

    def test_batch_accumulates_all_sources_and_restores_local_cpu_after_complete_prefault(self):
        batch = self.fake_batch()
        self.assertEqual(batch.resident_bytes, 20480)
        with patch.object(os, "sched_getaffinity", return_value={0}, create=True), \
             patch.object(os, "sched_setaffinity", create=True) as bind, \
             patch.object(h, "cpu_on_node", return_value=64) as select:
            self.assertEqual(batch.prefault(), 64)
            self.assertEqual([call.args for call in bind.call_args_list], [(0, {64}), (0, {0})])
            select.assert_called_once_with(1, {0, 64})
        for mapping in batch.mappings:
            mapping.touch.assert_called_once_with()

    def test_partial_touch_and_read_exception_both_restore_prior_affinity(self):
        for failure in (None, OSError("read failed")):
            batch = self.fake_batch()
            if failure is None:
                batch.mappings[0].touch.return_value = 1
            else:
                batch.mappings[0].touch.side_effect = failure
            with patch.object(os, "sched_getaffinity", return_value={0}, create=True), \
                 patch.object(os, "sched_setaffinity", create=True) as bind, \
                 patch.object(h, "cpu_on_node", return_value=64), self.assertRaises((RuntimeError, OSError)):
                batch.prefault()
            self.assertEqual(bind.call_args.args, (0, {0}))
            batch.mappings[1].touch.assert_not_called()

    def test_failed_affinity_restore_is_fatal_and_never_returns_a_successful_prefault(self):
        batch = self.fake_batch()
        with patch.object(os, "sched_getaffinity", return_value={0}, create=True), \
             patch.object(os, "sched_setaffinity", side_effect=[None, OSError("restore failed")], create=True) as bind, \
             patch.object(h, "cpu_on_node", return_value=64), self.assertRaisesRegex(OSError, "restore failed"):
            batch.prefault()
        self.assertEqual(bind.call_args.args, (0, {0}))

    def construction(self, *, failure=None):
        events = []
        control = SimpleNamespace(storage=SimpleNamespace(data_ptr=lambda: 0x2000),
                    _registration_finalizer=SimpleNamespace(atexit=True),
                    _lease_finalizer=SimpleNamespace(atexit=True))
        cache = SimpleNamespace(resident_bytes=20480)
        cache.prefault = lambda: events.append("prefault") or 64
        def verify():
            events.append("verify")
            if failure == "verify_before":
                raise RuntimeError("source placement incomplete")
            if failure == "verify_after" and events.count("verify") == 2:
                raise RuntimeError("post-copy source moved")
            return [dict(node_pages={"1": 5})]
        def close():
            events.append("close_source")
            if failure == "close":
                raise RuntimeError("source close failed")
        def construct():
            events.append("main_construct")
            self.assertNotIn("close_source", events)
            if failure == "construct":
                raise RuntimeError("main construction failed")
            return control
        cache.verify, cache.close = verify, close
        if failure == "affinity_restore":
            cache.prefault = Mock(side_effect=OSError("restore failed"))
        def budget(snapshot, **kwargs):
            events.append(kwargs["phase"])
            self.assertEqual(kwargs["source_bytes"], 20480)
            return {"phase": kwargs["phase"]}
        context = dict(source=Path("/private/source"), files=[Path("/private/source/shard")], source_node=1,
                       gpu_node=0, allowed_cpus={0, 64}, table_bytes=4097, headroom_bytes=16 << 30, construct=construct)
        return control, cache, budget, events, context

    def execute_construction(self, values):
        control, cache, budget, events, context = values
        with patch.object(h, "SourceCacheBatch", return_value=cache), \
             patch.object(h, "memory_snapshot", return_value=self.snapshot()), \
             patch.object(h, "check_source_cache_budget", side_effect=budget), \
             patch.object(h, "parse_complete_numa_pages", return_value={"node_pages": {"0": 2}}), \
             patch.object(Path, "read_text", return_value="unused"):
            return h.construct_with_source_cache(**context)

    def test_source_maps_survive_main_construction_with_complete_before_and_after_checks(self):
        values = self.construction()
        control, evidence = self.execute_construction(values)
        self.assertIs(control, values[0])
        self.assertEqual(values[3], ["before_prefault", "prefault", "verify", "prefaulted", "main_construct",
                                     "verify", "after_load", "close_source"])
        self.assertFalse(evidence["eviction_prevented"])
        with tempfile.TemporaryDirectory() as directory:
            h.write_new_json(Path(directory) / "source-cache.json", evidence)

    def test_postconstruction_verification_or_cleanup_failure_retains_registered_control(self):
        for failure in ("verify_after", "close"):
            values = self.construction(failure=failure)
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                self.execute_construction(values)
            self.assertEqual(values[3][-1], "close_source")
            self.assertFalse(values[0]._registration_finalizer.atexit)
            self.assertTrue(any(values[0] is item for group in h._FAILED_RESOURCES for item in group))

    def test_construct_failure_closes_only_source_maps_and_same_node_is_rejected_before_mapping(self):
        values = self.construction(failure="construct")
        with self.assertRaises(RuntimeError):
            self.execute_construction(values)
        self.assertEqual(values[3][-1], "close_source")
        self.assertTrue(values[0]._registration_finalizer.atexit)
        values[-1]["source_node"] = 0
        with patch.object(h, "SourceCacheBatch") as mapping, self.assertRaises(ValueError):
            h.construct_with_source_cache(**values[-1])
        mapping.assert_not_called()

    def test_failed_prefault_or_source_validation_never_calls_main_constructor(self):
        for failure in ("affinity_restore", "verify_before"):
            values = self.construction(failure=failure)
            with self.subTest(failure=failure), self.assertRaises((RuntimeError, OSError)):
                self.execute_construction(values)
            self.assertNotIn("main_construct", values[3])
            self.assertEqual(values[3][-1], "close_source")

    def test_mapping_identity_is_rechecked_after_read_and_changed_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.private_file(directory)
            mapping = h.ReadOnlySourceMapping(root, path)
            try:
                mapping.touch()
                page_kib = mapping.page_size // 1024
                text = (f"{mapping.address:x} default file=/source/shard N1={mapping.resident_bytes // mapping.page_size} "
                        f"kernelpagesize_kB={page_kib}\n")
                self.assertEqual(mapping.verify(text, 1)["file_bytes"], 8193)
                path.chmod(0o600)  # Even unchanged bytes are no longer the admitted immutable source.
                # Model filesystems whose ctime resolution misses the chmod.
                with patch("tools.engram_sidecar.framework.stage_checkpoint._unchanged"), self.assertRaises(RuntimeError):
                    mapping.verify(text, 1)
            finally:
                mapping.close()

    def test_partial_batch_creation_failure_closes_previous_maps(self):
        previous = SimpleNamespace(close=Mock())
        with patch.object(h, "ReadOnlySourceMapping", side_effect=[previous, OSError("second open failed")]), \
             self.assertRaisesRegex(OSError, "second open failed"):
            h.SourceCacheBatch(Path("/private/source"), [Path("first"), Path("second")], 1, {0, 64})
        previous.close.assert_called_once_with()

    def test_source_batch_cleanup_attempts_all_mappings_and_retains_failed_mapping(self):
        batch = self.fake_batch()
        batch.mappings[1].close.side_effect = OSError("munmap failed")
        with self.assertRaises(RuntimeError):
            batch.close()
        for mapping in batch.mappings:
            mapping.close.assert_called_once_with()
        self.assertIs(h._FAILED_RESOURCES[-1][0], batch)


class DiagnosticContractTest(unittest.TestCase):
    def test_default_has_no_declaration_or_cuda_nvtx_access(self):
        args = SimpleNamespace()
        self.assertEqual(h.diagnostic_declaration(args), {})
        self.assertIsNone(h.create_diagnostic_trace(args, object()))

    def test_capture_options_require_markers_and_one_case_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            original = LatencyContractTest().options(Path(directory))
            original.diagnostic_nvtx = original.diagnostic_cuda_profiler = True
            original.layer_mode, original.layers = "together", "1,14"
            original.batches, original.modes, original.distributions = "1", "graph", "random"
            self.assertEqual(h.check_options(copy.copy(original)).layers, [1, 14])
            for changes in (dict(diagnostic_nvtx=False), dict(batches="1,2"),
                            dict(modes="eager,graph"), dict(distributions="hot,random"),
                            dict(layer_mode="single")):
                args = copy.copy(original)
                for key, value in changes.items():
                    setattr(args, key, value)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    h.check_options(args)
            original.diagnostic_cuda_profiler = False
            original.batches, original.modes, original.distributions = "1,2", "eager,graph", "hot,random"
            self.assertEqual(h.check_options(original).batches, [1, 2])

    def test_marker_uses_only_allowlisted_geometry_and_rejects_arbitrary_content(self):
        secret = "opaque-private-pointer-row-contents"
        metadata = dict(layer_ids_measured=[1, 14], batch=1, mode="graph", secret=secret)
        label = h.diagnostic_marker(metadata, "random", 1509, "remote_raw", 0)
        self.assertEqual(label, "engram_lookup.layers=1,14.M=1.mode=graph.ids=random.pair=1509.path=remote_raw.order=0")
        self.assertNotIn(secret, label)
        for changes in (dict(layer_ids_measured=[secret]), dict(batch=True), dict(mode=secret)):
            with self.assertRaises(ValueError):
                h.diagnostic_marker(dict(metadata, **changes), "random", 0, "local_raw", 1)
        for values in ((secret, 0, "local_raw", 0), ("hot", -1, "local_raw", 0),
                       ("hot", 0, secret, 0), ("hot", 0, "local_raw", True)):
            with self.assertRaises(ValueError):
                h.diagnostic_marker(metadata, *values)

    def fake_case(self, root, *, diagnostic=True, profiler=True, mode="graph",
                  fail_invoke=None, fail_verify=False, stop_status=0, fail_pop=False):
        calls, events, guards = [], [], []
        state = dict(active=False, verified=0, clock=0)
        class Stream:
            def wait_stream(self, other):
                calls.append("wait_stream")
            def synchronize(self):
                calls.append("stream_sync")
        stream = Stream()
        class Event:
            def __init__(self, **kwargs):
                self.index = len(events)
                events.append(self)
            def record(self, stream=None):
                calls.append("event_record")
            def synchronize(self):
                calls.append("event_sync")
            def elapsed_time(self, other):
                calls.append("event_elapsed")
                return .001
        class Guard:
            @classmethod
            def capture(cls, forward, storages, warmup):
                calls.append("capture")
                result = cls()
                result.output = forward()
                guards.append(result)
                return result
            def replay(self):
                calls.append("replay")
                if state["active"] and fail_invoke is not None:
                    raise fail_invoke
                return self.output
            def close(self):
                calls.append("graph_close")
        def push(label):
            calls.append(("push", label))
            state["active"] = True
        def pop():
            calls.append("pop")
            state["active"] = False
            if fail_pop:
                raise RuntimeError("private-nvtx-detail")
        nvtx = SimpleNamespace(range_push=push, range_pop=pop)
        def start():
            calls.append("profiler_start")
            return (0,)
        def stop():
            calls.append("profiler_stop")
            return (stop_status,)
        runtime = SimpleNamespace(cudaProfilerStart=start, cudaProfilerStop=stop)
        trace = h.DiagnosticTrace(nvtx, runtime if profiler else None) if diagnostic else None
        cuda = SimpleNamespace(Stream=lambda **_: stream, current_stream=lambda _: stream,
                               stream=lambda _: nullcontext(), Event=Event, nvtx=nvtx)
        torch = SimpleNamespace(cuda=cuda, inference_mode=nullcontext)
        expected = {name: (object(), object()) for name in ("local_raw", "remote_raw")}
        def path(name):
            def forward():
                calls.append("lookup")
                if state["active"] and fail_invoke is not None:
                    raise fail_invoke
                return expected[name]
            return forward
        def update(pair, distribution):
            calls.append(("update", pair, distribution))
        def verify(pair, distribution, outputs):
            calls.append(("verify", pair, distribution))
            state["verified"] += 1
            for name, value in outputs.items():
                self.assertIs(value, expected[name])
            if fail_verify and state["verified"] == 3:
                raise AssertionError("oracle failure")
        def clock():
            calls.append("clock")
            state["clock"] += 1000
            return state["clock"]
        def switches():
            calls.append("switches")
            return (0, 0)
        args = SimpleNamespace(device=0, l2_disturb=False, warmup=1, samples=2,
                               distributions=["random"], seed=7, threshold_us=100.0,
                               calibration_samples=0, diagnostic_nvtx=diagnostic,
                               diagnostic_cuda_profiler=diagnostic and profiler)
        metadata = dict(mode=mode, batch=1, layer_ids_measured=[1, 14])
        kwargs = dict(paths={name: path(name) for name in expected}, storage=object(),
                      update=update, verify=verify, args=args, metadata=metadata, output=root)
        def execute():
            with patch.dict(sys.modules, {"torch": torch,
                    "rtp_llm.models_py.modules.dsv41.engram_storage": SimpleNamespace(CapturedEngramGraph=Guard)}), \
                 patch.object(h, "create_diagnostic_trace", return_value=trace), \
                 patch.object(h.bench, "_l2_bytes", return_value=128), \
                 patch.object(h.bench, "_thread_switches_reader", return_value=switches), \
                 patch.object(h.time, "perf_counter_ns", side_effect=clock), \
                 patch.object(h, "retain_failed"):
                return h.benchmark_case(**kwargs)
        return execute, calls, trace

    def test_actual_case_keeps_graph_tuple_oracles_outside_profile_and_clocks_inside_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execute, calls, trace = self.fake_case(root)
            summary = execute()
            records = [json.loads(s) for s in (root / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(calls.count("profiler_start"), 1)
        self.assertEqual(calls.count("profiler_stop"), 1)
        start, stop = calls.index("profiler_start"), calls.index("profiler_stop")
        self.assertTrue(all(calls.index(item) < start for item in ("capture", ("verify", 0, "random"), ("verify", 1, "random"))))
        self.assertGreater(max(i for i, c in enumerate(calls) if isinstance(c, tuple) and c[0] == "verify"), stop)
        self.assertGreater(calls.index("graph_close"), stop)
        labels = []
        for i, item in enumerate(calls):
            if isinstance(item, tuple) and item[0] == "push":
                labels.append(item[1])
                self.assertEqual(calls[i-1], "stream_sync")
                self.assertEqual(calls[i+1:i+9], ["switches", "clock", "replay", "clock", "event_sync", "clock", "switches", "pop"])
        self.assertEqual([r["diagnostic_nvtx_range"] for r in records], [v for label in labels for v in (label, label)])
        self.assertTrue(all(r["measurement_status"] == "PROFILE_ONLY" for r in records))
        self.assertEqual(summary["measurement_status"], "PROFILE_ONLY")
        self.assertTrue(all(c["threshold_observation"] == "PROFILE_ONLY" for c in summary["paired_deltas"]))
        self.assertEqual((trace.pushes, trace.pops, trace.stop_status), (4, 4, "SUCCESS"))

    def test_default_case_retains_existing_fields_and_makes_no_diagnostic_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execute, calls, trace = self.fake_case(root, diagnostic=False)
            summary = execute()
            records = [json.loads(s) for s in (root / "samples.jsonl").read_text().splitlines()]
        self.assertIsNone(trace)
        self.assertNotIn("measurement_status", summary)
        self.assertNotIn("diagnostics", summary)
        self.assertTrue(all(set(r) == set(LatencyContractTest().records()[0]) for r in records))
        self.assertFalse(any(isinstance(c, tuple) and c[0] == "push" for c in calls))
        self.assertNotIn("profiler_start", calls)
        self.assertNotIn("profiler_stop", calls)

    def test_nvtx_only_eager_has_markers_without_profiler_api(self):
        with tempfile.TemporaryDirectory() as directory:
            execute, calls, trace = self.fake_case(Path(directory), profiler=False, mode="eager")
            summary = execute()
        self.assertEqual(summary["measurement_status"], "PROFILE_ONLY")
        self.assertNotIn("capture", calls)
        self.assertNotIn("profiler_start", calls)
        self.assertEqual((trace.start_status, trace.stop_status), ("NOT_REQUESTED", "NOT_REQUESTED"))
        for i, item in enumerate(calls):
            if isinstance(item, tuple) and item[0] == "push":
                self.assertEqual(calls[i+1:i+11], ["switches", "clock", "event_record", "lookup", "event_record", "clock", "event_sync", "clock", "switches", "pop"])

    def test_failed_invoke_preserves_primary_when_nvtx_and_profiler_cleanup_fail(self):
        primary = ValueError("private-original-exception")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execute, calls, trace = self.fake_case(root, fail_invoke=primary, fail_pop=True, stop_status=1)
            with self.assertRaises(ValueError) as caught:
                execute()
            self.assertIs(caught.exception, primary)
            self.assertFalse((root / "summary.json").exists())
        self.assertEqual(calls.count("pop"), 1)
        self.assertEqual(calls.count("profiler_stop"), 1)
        self.assertEqual(trace.cleanup_errors, ["NVTX_POP_FAILED", "PROFILER_STOP_FAILED"])
        self.assertNotIn("private", json.dumps(trace.evidence()))

    def test_normal_stop_nonzero_is_fatal_before_oracle_and_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execute, calls, trace = self.fake_case(root, stop_status=1)
            with self.assertRaisesRegex(RuntimeError, "cudaProfilerStop"):
                execute()
            self.assertFalse((root / "summary.json").exists())
        self.assertEqual(calls.count("profiler_stop"), 1)
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == "verify" for c in calls), 2)
        self.assertEqual(trace.stop_status, "FAILED")

    def test_normal_pop_failure_still_stops_and_is_not_masked_by_stop_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execute, calls, trace = self.fake_case(root, fail_pop=True, stop_status=1)
            with self.assertRaisesRegex(RuntimeError, "NVTX pop failed"):
                execute()
            self.assertFalse((root / "summary.json").exists())
        self.assertEqual(calls.count("profiler_stop"), 1)
        self.assertEqual(trace.cleanup_errors, ["NVTX_POP_FAILED", "PROFILER_STOP_FAILED"])

    def test_postmeasurement_oracle_failure_occurs_after_stop_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execute, calls, trace = self.fake_case(root, fail_verify=True)
            with self.assertRaisesRegex(AssertionError, "oracle failure"):
                execute()
            self.assertFalse((root / "summary.json").exists())
        self.assertEqual(calls.count("profiler_stop"), 1)
        self.assertEqual(trace.stop_status, "SUCCESS")

    def test_partial_start_failure_attempts_stop_and_never_exposes_api_arguments(self):
        runtime = SimpleNamespace(cudaProfilerStart=Mock(side_effect=TypeError("Invoked with: PRIVATE_HANDLE")),
                                  cudaProfilerStop=Mock(return_value=(0,)))
        trace = h.DiagnosticTrace(Mock(), runtime)
        with self.assertRaisesRegex(RuntimeError, "cudaProfilerStart") as caught:
            trace.start()
        trace.close(caught.exception)
        self.assertNotIn("PRIVATE_HANDLE", str(caught.exception))
        runtime.cudaProfilerStop.assert_called_once_with()
        self.assertEqual((trace.start_status, trace.stop_status), ("FAILED", "SUCCESS"))

    def test_profiler_requires_explicit_success_status_and_preserves_interrupt(self):
        for result in (None, 0, (), (1,), (0, 0)):
            runtime = SimpleNamespace(cudaProfilerStart=Mock(return_value=(0,)),
                                      cudaProfilerStop=Mock(return_value=result))
            trace = h.DiagnosticTrace(Mock(), runtime)
            trace.start()
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                trace.stop()
            self.assertEqual(trace.stop_status, "FAILED")
        interrupt = KeyboardInterrupt()
        runtime = SimpleNamespace(cudaProfilerStart=Mock(side_effect=interrupt),
                                  cudaProfilerStop=Mock(return_value=(0,)))
        trace = h.DiagnosticTrace(Mock(), runtime)
        with self.assertRaises(KeyboardInterrupt) as caught:
            trace.start()
        self.assertIs(caught.exception, interrupt)
        trace.close(interrupt)
        runtime.cudaProfilerStop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
