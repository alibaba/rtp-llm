"""Maintained checkpoint-to-NormalEngine CEP4PP2 token and lifecycle smoke."""

import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
import unittest
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_engine_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import restore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_loader_worker_sm120 import (
    load_configuration,
    save_checkpoint,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_collapse_reference import (
    reference_collapse,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import (
    fp32_mhc_prenorm,
    reference_mhc_post,
)
from rtp_llm.models_py.modules.dsv4.hc import delayed as delayed_hc
from rtp_llm.models_py.modules.dsv4.hc import v41_prenorm
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_engine_sm120 import (
    tensor_hash,
    verify_outputs,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_execution_sm120 import (
    raw_snapshot,
    world_check,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    clean_worker,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_pp_loader_sm120 import make_pp_loader
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import wide_fixture
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract
from rtp_llm.models_py.modules.dsv4.v41_engram_host import plan_stage_engram_tables
from rtp_llm.ops import NcclCommConfig
from rtp_llm.ops.compute_ops import PyModelInitResources
from rtp_llm.tools.dsv41_pp_smoke import checkpoint_identities
from rtp_llm.utils.model_weight import W

MHC_MODES = {"collapse": "ordered_fp32", "post": "ordered_fp32", "prenorm": "fp32"}


@contextmanager
def pp_mhc_equations(choices, diagnostic):
    with ExitStack() as stack:
        for component, module, method, equation in (
            ("collapse", delayed_hc, "collapse_delayed", reference_collapse),
            ("post", delayed_hc, "post_delayed", reference_mhc_post),
            ("prenorm", v41_prenorm, "prenorm", fp32_mhc_prenorm),
        ):
            if f"mhc_{component}_mode" not in choices:
                continue
            if diagnostic:
                if component == "post":
                    module, method = delayed_hc.DelayedHCUnit, "_post_impl"
                stack.enter_context(patch.object(module, method, equation))
            else:
                stack.enter_context(
                    patch.object(
                        module,
                        method,
                        side_effect=AssertionError(
                            f"Selected PP mHC {component} reached default dispatch"
                        ),
                    )
                )
        yield


def save_tensors(path, values):
    with path.open("xb") as stream:
        torch.save(values, stream)


def require_exact_tree(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert expected.shape == actual.shape and expected.dtype == actual.dtype
        torch.testing.assert_close(
            expected.contiguous().view(torch.uint8),
            actual.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        return 1
    assert type(expected) is type(actual)
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        return sum(
            require_exact_tree(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        return sum(
            require_exact_tree(left, right) for left, right in zip(expected, actual)
        )
    assert expected == actual, (expected, actual)
    return 0


def run_rank(
    rank,
    rendezvous,
    checkpoint,
    directory,
    prompts,
    full,
    first_ids,
    mhc_choices=None,
    diagnostic_mhc=False,
    artifact_directory=None,
    max_seq_len=256,
):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    mhc_choices = dict(mhc_choices or {})
    if "mhc_prenorm_mode" in mhc_choices:
        torch.set_float32_matmul_precision("highest")
    artifact_directory = (
        Path(artifact_directory) if artifact_directory is not None else None
    )
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(192 * 2**30 if full else 2**20),
            "DSV41_ENGRAM_CACHE_DIR": directory,
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "RTP_V41_BATCHED_PRODUCER": "0",
            "DSV4_PREFILL_FAST_PATH": "1",
            "FAKE_BALANCE_EXPERT": "0",
            "RTP_LLM_STREAM_ASYNC": "0",
            "RTP_LLM_DEVICE_INPUT": "0",
            "MOEDBG": "0",
        }
    )
    timeout = timedelta(seconds=3600 if full else 180)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=8,
        timeout=timeout,
    )
    # Stage one must not enter native WORLD scheduling during stage-zero
    # Engram staging. Neither initialization nor main-thread controls may
    # compete with the native background scheduler's WORLD collectives.
    startup = dist.new_group(backend="gloo", timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timeout
        return new_group(*args, **kwargs)

    worker = engine = None
    with pp_mhc_equations(mhc_choices, diagnostic_mhc), patch.dict(
        sys.modules, {"librtp_compute_ops": ops}
    ), patch.object(dist, "new_group", side_effect=bounded_group):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), 0
            )
            cp_group = collective._get_group(collective.Group.TP)
            with torch.inference_mode():
                config, kv_config = load_configuration(
                    Path(checkpoint), max_seq_len=max_seq_len
                )
                owner_choices = {} if diagnostic_mhc else mhc_choices
                config.deepseek_v41_config.update(owner_choices)
                parallelism, hw, loader = make_pp_loader(config, kv_config, rank)
                loaded = loader.load_weights(f"cuda:{rank}")
                head = loaded.global_weights[W.lm_head] if rank >= 4 else torch.empty(0)
                if rank >= 4:
                    assert head.size(0) * 4 == config.vocab_size
                worker = DeepSeekV41Model(
                    config, parallelism, loaded, None, 2, py_hw_kernel_config=hw
                )
                del loaded, loader
                identities = checkpoint_identities(Path(checkpoint))
                contract = V41L20Contract(
                    **identities, max_sequence_length=max_seq_len, **owner_choices
                )
                initialize = worker.initialize
                original_execute = worker._execute_native_pp_engine
                initialization_calls = []
                records = []
                expected_outputs = {}
                phase = [0]
                previous_stage_hashes = {}

                def checked_initialize(resource):
                    assert isinstance(resource, PyModelInitResources)
                    assert resource.kv_cache is not None and not worker._materialized
                    initialization_calls.append(worker._materialized)
                    assert initialize(resource)
                    assert len(worker.v4.layers) == (21 if rank < 4 else 19)
                    for layer in worker.v4.layers:
                        for unit in (layer.attn_hc, layer.ffn_hc):
                            for component in MHC_MODES:
                                assert getattr(
                                    unit, f"{component}_mode"
                                ) == owner_choices.get(f"mhc_{component}_mode")
                    if artifact_directory is not None:
                        # Initialize unused pages and padding for complete byte comparisons.
                        for layer_id in range(resource.kv_cache.layer_count):
                            for group in resource.kv_cache.get_layer_cache_groups(
                                layer_id
                            ):
                                group.kv_cache_base.view(torch.uint8).fill_(37)
                        print(
                            "DSV41_PP_MHC_OWNER "
                            + json.dumps(
                                {
                                    "rank": rank,
                                    "stage": rank // 4,
                                    "layers": len(worker.v4.layers),
                                    "choices": mhc_choices,
                                    "owner_choices": owner_choices,
                                    "diagnostic_mhc": diagnostic_mhc,
                                    "default_dispatch_guarded": not diagnostic_mhc,
                                    "contract_fingerprint": contract.fingerprint.hex(),
                                    "identities": identities,
                                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                    "initial_cache_byte": 37,
                                }
                            ),
                            flush=True,
                        )
                    worker._configure_native_pp_engine(contract)
                    dist.barrier(group=startup)
                    return True

                def execute(dispatch, transfer_round, tensors=None):
                    try:
                        output = original_execute(dispatch, transfer_round, tensors)
                        clean_worker(worker)
                        value = (
                            output.pp_intermediates["hidden_states"]
                            if rank < 4
                            else output.hidden_states
                        )
                        assert torch.isfinite(value).all()
                        hashes = {"hidden": tensor_hash(value.cpu())}
                        if rank < 4:
                            hashes["pre_mix"] = tensor_hash(
                                output.pp_intermediates["v41_prev_pre_mix"].cpu()
                            )
                        else:
                            # Compare native delivered results with the actual
                            # maintained stage output and native LM-head GEMM.
                            # This checks delivery/argmax, not reference math.
                            restored = restore(value, dispatch.cp, cp_group)
                            lengths = [span.count for span in dispatch.spans]
                            indexes = torch.tensor(
                                [
                                    sum(lengths[: row + 1]) - 1
                                    for row in range(len(lengths))
                                ],
                                device="cuda",
                                dtype=torch.long,
                            )
                            local = ops.native_lm_logits(restored[indexes], head)
                            shards = [torch.empty_like(local) for _ in range(4)]
                            dist.all_gather(shards, local, group=cp_group)
                            logits = torch.cat(shards, 1)
                            hashes["logits"] = tensor_hash(logits.cpu())
                            if rank == 4:
                                for row, span in enumerate(dispatch.spans):
                                    begin = sum(lengths[:row])
                                    end = begin + lengths[row]
                                    selected = logits[row : row + 1].cpu()
                                    expected_outputs[span.identity.request_id] = {
                                        "token": int(selected.argmax(-1).item()),
                                        "hidden": restored[end - 1 : end].cpu(),
                                        "logits": selected,
                                        "all_hidden": restored[begin:end].cpu(),
                                    }
                        if artifact_directory is not None:
                            frame = output.pp_intermediates if rank < 4 else tensors
                            assert frame[
                                "v41_cache_fingerprint"
                            ].cpu().tolist() == list(contract.fingerprint)
                            snapshot = {
                                "rank": rank,
                                "stage": rank // 4,
                                "phase": phase[0],
                                "round": transfer_round,
                                "spans": [
                                    span.wire_values() for span in dispatch.spans
                                ],
                                "hidden": value.cpu().clone(),
                                "cache": raw_snapshot(worker.kv_cache),
                                "frame": {
                                    key: tensor.cpu().clone()
                                    for key, tensor in frame.items()
                                    if key != "v41_cache_fingerprint"
                                },
                            }
                            if rank >= 4:
                                snapshot["logits"] = logits.cpu().clone()
                            save_tensors(
                                artifact_directory / f"rank{rank}-phase{phase[0]}.pt",
                                snapshot,
                            )
                        if phase[0] == 0:
                            previous_stage_hashes.update(hashes)
                        if phase[0] == 1:
                            assert hashes == previous_stage_hashes
                        record = {
                            "rank": rank,
                            "stage": rank // 4,
                            "phase": phase[0],
                            "round": transfer_round,
                            "spans": [span.wire_values() for span in dispatch.spans],
                            "contract_fingerprint": contract.fingerprint.hex(),
                            "max_sequence_length": contract.max_sequence_length,
                            "identities": identities,
                            "hashes": hashes,
                            "full_prompt": phase[0] < 2,
                            "maintained_arithmetic": not diagnostic_mhc,
                            "mhc_choices": mhc_choices,
                            "diagnostic_mhc": diagnostic_mhc,
                        }
                        records.append(record)
                        print(
                            "DSV41_PP_CHECKPOINT_EXECUTION " + json.dumps(record),
                            flush=True,
                        )
                        return output
                    except BaseException:
                        traceback.print_exc()
                        raise

                worker.initialize = checked_initialize
                worker._execute_native_pp_engine = execute
                engine = ops.NativeEngine(
                    worker, config, parallelism, head, pp_chunk_size=256
                )
                assert initialization_calls == [False]
                assert worker._materialized and not hasattr(worker, "weight")
                table_paths = [table._path for table in worker._engram_tables.values()]
                previous_outputs = None
                for step in range(4):
                    phase[0] = step
                    full_outputs = step < 2
                    selected_prompts = prompts if full_outputs else [prompts[step - 2]]
                    expected_outputs.clear()
                    dist.barrier(group=control)
                    ids = (
                        engine.enqueue(selected_prompts, full_outputs=full_outputs)
                        if rank == 0
                        else None
                    )
                    box = [ids]
                    dist.broadcast_object_list(box, src=0, group=control)
                    ids = box[0]
                    result = engine.outputs() if rank == 0 else None
                    dist.barrier(group=control)
                    expected_box = [expected_outputs if rank == 4 else None]
                    dist.broadcast_object_list(expected_box, src=4, group=control)

                    def verify_delivered():
                        assert len(records) == step + 1
                        assert records[-1]["phase"] == step
                        assert [span[0] for span in records[-1]["spans"]] == ids
                        if rank != 0:
                            return
                        assert result["steps"] == [1] * len(selected_prompts)
                        actual = result["outputs"]
                        verify_outputs(
                            actual, expected_box[0], ids, selected_prompts, full_outputs
                        )
                        if full:
                            wanted = (
                                first_ids if full_outputs else [first_ids[step - 2]]
                            )
                            assert [row["tokens"][-1] for row in actual] == wanted
                        if step == 1:
                            for older, newer in zip(previous_outputs, actual):
                                assert older["tokens"] == newer["tokens"]
                                for key in ("hidden", "logits", "all_hidden"):
                                    torch.testing.assert_close(
                                        older[key], newer[key], rtol=0, atol=0
                                    )

                    world_check(
                        verify_delivered, "checkpoint PP native results", control
                    )
                    if rank == 0:
                        actual = result["outputs"]
                        if artifact_directory is not None:
                            save_tensors(
                                artifact_directory / f"outputs-phase{step}.pt", actual
                            )
                        if step == 0:
                            previous_outputs = actual
                        print(
                            "DSV41_PP_CHECKPOINT_OUTPUT "
                            + json.dumps(
                                {
                                    "phase": step,
                                    "full_prompt": full_outputs,
                                    "full_nas": full,
                                    "request_ids": ids,
                                    "prompts": selected_prompts,
                                    "tokens": [row["tokens"][-1] for row in actual],
                                    "hidden_sha256": [
                                        tensor_hash(row["hidden"]) for row in actual
                                    ],
                                    "logits_sha256": [
                                        tensor_hash(row["logits"]) for row in actual
                                    ],
                                    "all_hidden_sha256": [
                                        (
                                            tensor_hash(row["all_hidden"])
                                            if full_outputs
                                            else None
                                        )
                                        for row in actual
                                    ],
                                    "mhc_choices": mhc_choices,
                                    "diagnostic_mhc": diagnostic_mhc,
                                    "native_argmax_exact": True,
                                    "native_delivery_exact": True,
                                }
                            ),
                            flush=True,
                        )
                dist.barrier(group=control)
                stopped = engine.stop()
                engine = None

                def verify_stopped():
                    assert stopped["shutdown"]
                    assert stopped["active_requests"] == stopped["identity_count"] == 0
                    assert stopped["retained_slots"] == 0
                    assert stopped["free_before"] == stopped["free_after"]
                    assert stopped["cache_blocks"] == 8
                    assert (
                        worker._native_pp_receiver is None
                        or not worker._native_pp_receiver._requests
                    )
                    clean_worker(worker)
                    assert len(records) == 4

                world_check(
                    verify_stopped, "checkpoint PP production shutdown", control
                )
                del worker.initialize, worker._execute_native_pp_engine
                worker.close()
                dist.barrier(group=control)

                def verify_closed():
                    assert all(not path.exists() for path in table_paths)
                    if rank == 0:
                        assert not list(Path(directory).glob("*.lease.*"))
                        assert not list(Path(directory).glob("*.json"))

                world_check(verify_closed, "checkpoint PP Engram cleanup", control)
                print(
                    "DSV41_PP_CHECKPOINT_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "mhc_choices": mhc_choices,
                            "diagnostic_mhc": diagnostic_mhc,
                            "cold_initializations": len(initialization_calls),
                            "executions": len(records),
                            "receiver_empty": True,
                            "engram_cleanup": True,
                            **stopped,
                        }
                    ),
                    flush=True,
                )
                worker = None
                del initialize, original_execute, head
                gc.collect()
                Dsv4SharedRuntimeBufferStore._reset_for_test()
        except BaseException:
            print(f"DSV41_PP_CHECKPOINT_FAILURE rank={rank}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A unilateral stop could wait forever on peers still scheduling.
            # mp.spawn owns and terminates only this test's failed peers.
            os._exit(1)
        finally:
            if engine is not None:
                engine.stop()
            if worker is not None:
                worker.close()
            collective.destroy_distributed_environment()


class V41PPCheckpointEngineSM120Test(unittest.TestCase):
    def run_checkpoint(
        self,
        root,
        directory,
        prompts,
        *,
        full,
        first_ids=None,
        mhc_choices=None,
        diagnostic_mhc=False,
        artifact_directory=None,
        max_seq_len=256,
    ):
        self.assertEqual(torch.cuda.device_count(), 8)
        rendezvous = Path(directory).resolve() / "world.rendezvous"
        self.assertFalse(rendezvous.exists())
        mp.spawn(
            run_rank,
            args=(
                rendezvous.as_uri(),
                str(root),
                str(directory),
                prompts,
                full,
                first_ids,
                mhc_choices,
                diagnostic_mhc,
                str(artifact_directory) if artifact_directory is not None else None,
                max_seq_len,
            ),
            nprocs=8,
            join=True,
        )
        # FileStore removal is best effort; all ranks have exited before cleanup.
        rendezvous.unlink(missing_ok=True)
        self.assertFalse(rendezvous.exists())
        print(
            "DSV41_PP_CHECKPOINT_RENDEZVOUS "
            + json.dumps(
                {
                    "choices": mhc_choices or {},
                    "diagnostic_mhc": diagnostic_mhc,
                    "scheme": "file",
                    "removed": True,
                }
            ),
            flush=True,
        )

    def run_mhc_case(self, *components, prompts=None, label_suffix="", max_seq_len=256):
        choices = {f"mhc_{key}_mode": MHC_MODES[key] for key in components}
        prompts = prompts or [[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]]
        label = ("-".join(components) or "default") + label_suffix
        root = Path(os.environ["DSV41_TEST_PP_MHC_ROOT"]).resolve() / label
        root.mkdir(parents=True)
        checkpoint_root = root / "checkpoint"
        checkpoint_root.mkdir()
        torch.manual_seed(41701)
        args, weights = wide_fixture()
        save_checkpoint(checkpoint_root, args, weights)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        for kind in ("reference", "maintained"):
            directory = root / kind
            directory.mkdir()
            cache = directory / "tables"
            cache.mkdir()
            self.run_checkpoint(
                checkpoint_root,
                cache,
                prompts,
                full=False,
                mhc_choices=choices,
                diagnostic_mhc=kind == "reference",
                artifact_directory=directory,
                max_seq_len=max_seq_len,
            )
            self.assertTrue(
                all(path.name.endswith(".lock") for path in cache.iterdir())
            )
        tensors = cache_pools = 0
        for rank in range(8):
            for phase in range(4):
                filename = f"rank{rank}-phase{phase}.pt"
                reference = torch.load(
                    root / "reference" / filename, map_location="cpu", weights_only=True
                )
                actual = torch.load(
                    root / "maintained" / filename,
                    map_location="cpu",
                    weights_only=True,
                )
                tensors += require_exact_tree(reference, actual)
                cache_pools += len(actual["cache"])
        for phase in range(4):
            filename = f"outputs-phase{phase}.pt"
            reference = torch.load(
                root / "reference" / filename, map_location="cpu", weights_only=True
            )
            actual = torch.load(
                root / "maintained" / filename, map_location="cpu", weights_only=True
            )
            tensors += require_exact_tree(reference, actual)
        print(
            "DSV41_PP_MHC_COMPARISON "
            + json.dumps(
                {
                    "policy": label,
                    "choices": choices,
                    "prompts": prompts,
                    "max_sequence_length": max_seq_len,
                    "rank_phase_checks_exact": 32,
                    "delivered_phase_checks_exact": 4,
                    "tensors_exact": tensors,
                    "cache_pools_exact": cache_pools,
                    "full_nas": False,
                    "standalone_decode_role": False,
                }
            ),
            flush=True,
        )

    def run_boundary_case(self, boundary, selected):
        lengths = [127, 129] if boundary == 128 else [255, 256]
        prompts = [
            [(3 + request * 17 + token) % 127 for token in range(length)]
            for request, length in enumerate(lengths)
        ]
        components = tuple(MHC_MODES) if selected else ()
        self.run_mhc_case(
            *components,
            prompts=prompts,
            label_suffix=f"-boundary{boundary}",
            max_seq_len=512,
        )

    def test_default_mhc_window_boundary(self):
        self.run_boundary_case(128, False)

    def test_composed_mhc_window_boundary(self):
        self.run_boundary_case(128, True)

    def test_default_mhc_page_boundary(self):
        self.run_boundary_case(256, False)

    def test_composed_mhc_page_boundary(self):
        self.run_boundary_case(256, True)

    def test_bounded_default_mhc_equations(self):
        self.run_mhc_case()

    def test_bounded_collapse_mhc_equations(self):
        self.run_mhc_case("collapse")

    def test_bounded_post_mhc_equations(self):
        self.run_mhc_case("post")

    def test_bounded_prenorm_mhc_equations(self):
        self.run_mhc_case("prenorm")

    def test_bounded_collapse_post_mhc_equations(self):
        self.run_mhc_case("collapse", "post")

    def test_bounded_collapse_prenorm_mhc_equations(self):
        self.run_mhc_case("collapse", "prenorm")

    def test_bounded_post_prenorm_mhc_equations(self):
        self.run_mhc_case("post", "prenorm")

    def test_bounded_collapse_post_prenorm_mhc_equations(self):
        self.run_mhc_case("collapse", "post", "prenorm")

    def test_bounded_named_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(41701)
            args, weights = wide_fixture()
            save_checkpoint(root, args, weights)
            del weights
            gc.collect()
            torch.cuda.empty_cache()
            cache = root / "cache"
            cache.mkdir()
            self.run_checkpoint(
                root, cache, [[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]], full=False
            )

    @unittest.skipUnless(
        os.environ.get("DSV41_RUN_FULL_PP_ENGINE") == "1",
        "full CEP4PP2 checkpoint requires explicit host/device admission",
    )
    def test_full_nas_cep4pp2_prefill(self):
        from transformers import AutoTokenizer

        root = Path(os.environ["DSV41_TEST_CHECKPOINT"]).resolve(strict=True)
        storage = Path(os.environ["DSV41_ENGRAM_CACHE_DIR"]).resolve(strict=True)
        config, _ = load_configuration(root)
        identity = hashlib.sha256(
            json.dumps(
                config.deepseek_v41_config, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        sources = plan_stage_engram_tables(
            SimpleNamespace(text=config.deepseek_v41_config), root, [1, 14], identity
        )
        total = sum(source.nbytes for source in sources.values())
        self.assertEqual(total, 202758032400)
        reserve_gib = int(os.environ.get("DSV41_TEST_STORAGE_RESERVE_GIB", "32"))
        self.assertGreaterEqual(reserve_gib, 32)
        free_storage = shutil.disk_usage(storage).free
        self.assertGreater(free_storage, total + reserve_gib * 2**30)
        memory = {
            key: int(value.split()[0]) * 1024
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
        }
        self.assertGreater(memory["MemAvailable"], 2 * total + 128 * 2**30)
        limits = dict(
            line.split()
            for line in Path("/sys/fs/cgroup/memory/memory.stat")
            .read_text()
            .splitlines()
        )
        self.assertGreaterEqual(
            int(limits["hierarchical_memory_limit"]), memory["MemTotal"]
        )
        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
        prompts = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in ("The capital of France is", "2 + 2 =")
        ]
        self.assertEqual(
            prompts, [[671, 6102, 294, 8760, 344], [20, 940, 223, 20, 438]]
        )
        print(
            "DSV41_PP_CHECKPOINT_ADMISSION "
            + json.dumps(
                {
                    "full_nas": True,
                    "engram_bytes": total,
                    "available_ram": memory["MemAvailable"],
                    "free_storage": free_storage,
                    "reserve_gib": reserve_gib,
                    "fresh_staging": True,
                    "prompts": prompts,
                }
            ),
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="v41-full-pp-engine-", dir=storage
        ) as directory:
            self.run_checkpoint(
                root, directory, prompts, full=True, first_ids=[11111, 223]
            )


if __name__ == "__main__":
    unittest.main()
