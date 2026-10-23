"""Native V4.1 CEP4PP2 cold-vs-warm TTFT (enqueue-to-one-token wall), no HTTP.

This is the *native* (models_py) counterpart of the serving-path reference TTFT
benchmark: it drives the in-process NativeEngine directly, so it measures the
V4.1 native PP path that the serving stack cannot currently reach.

With ``V41_TEST_REUSE_CACHE=1`` the test engine retains KV prefix blocks across
enqueues. The same 32768-token prompt is enqueued repeatedly at concurrency 1:

* run 0 is **cold** — the prefix cache is empty, so the full 32K context is
  computed;
* runs 1..N are **warm** — the whole 32K prefix is already cached, so only the
  trailing token must be produced.

The reported warm TTFT is the median of the warm runs; the parity gate
(final token == 11111) must hold on warm runs exactly as on cold ones, proving
that prefix reuse does not perturb the result.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_engine_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_loader_worker_sm120 import (
    load_configuration,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_pp_loader_sm120 import make_pp_loader
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract
from rtp_llm.models_py.modules.dsv4.v41_engram_host import plan_stage_engram_tables
from rtp_llm.ops import NcclCommConfig
from rtp_llm.ops.compute_ops import PyModelInitResources
from rtp_llm.tools.dsv41_pp_smoke import checkpoint_identities
from rtp_llm.utils.model_weight import W

MAX_SEQ_LEN = 33024
INPUT_TOKENS = 32768
PP_CHUNK_SIZE = 32768  # monolithic: the qualified, shipping configuration


def run_rank(rank, checkpoint, table_directory, run_directory, prompts, expected_ids):
    run = Path(run_directory)
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        dict(
            V41_TEST_REUSE_CACHE="1",  # the whole point: retain prefixes across enqueues
            NCCL_P2P_DISABLE="1",
            NCCL_IB_DISABLE="1",
            DSV41_ENGRAM_HOST_BYTES=str(192 * 2**30),
            DSV41_ENGRAM_CACHE_DIR=table_directory,
            DSV4_PP_EP_ENABLE="1",
            DSV4_PP_EP_BACKEND="fork_nccl_mxfp8",
            DSV4_MOE_STRATEGY="auto",
            DSV4_MOE_CHUNK_TOKENS="8192",
            RTP_V41_BATCHED_PRODUCER="0",
            DSV4_PREFILL_FAST_PATH="1",
            FAKE_BALANCE_EXPERT="0",
            RTP_LLM_STREAM_ASYNC="0",
            RTP_LLM_DEVICE_INPUT="0",
            MOEDBG="0",
            RTP_LLM_PP_OBJ_LOG="0",
            RTP_LLM_LOG_FASTGEN="0",
            DSV4_FWD_PROFILE="0",
            DSV4_RECORD_FUNCTION_RANGES="0",
            CUDA_LAUNCH_BLOCKING="0",
            GEN_TIMELINE_NUM_STEPS="0",
        )
    )
    os.environ.pop("RTP_LLM_LOG_TPSYNC", None)
    timeout = timedelta(seconds=3600)
    dist.init_process_group(
        "nccl",
        init_method=(Path(table_directory) / "world.rendezvous").as_uri(),
        rank=rank,
        world_size=8,
        timeout=timeout,
    )
    startup = dist.new_group(backend="gloo", timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timeout
        return new_group(*args, **kwargs)

    worker = engine = None
    setup = dict(rank=rank, stage=rank // 4, checkpoint=checkpoint)
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), 0
            )
            with torch.inference_mode():
                config, kv_config = load_configuration(
                    Path(checkpoint), max_seq_len=MAX_SEQ_LEN
                )
                parallelism, hw, loader = make_pp_loader(config, kv_config, rank)
                loaded = loader.load_weights(f"cuda:{rank}")
                head = loaded.global_weights[W.lm_head] if rank >= 4 else torch.empty(0)
                worker = DeepSeekV41Model(
                    config, parallelism, loaded, None, 2, py_hw_kernel_config=hw
                )
                del loaded, loader
                contract = V41L20Contract(
                    **checkpoint_identities(Path(checkpoint)),
                    max_sequence_length=MAX_SEQ_LEN,
                )

                original_initialize = worker.initialize

                def checked_initialize(resource):
                    assert isinstance(resource, PyModelInitResources)
                    assert original_initialize(resource)
                    assert len(worker.v4.layers) == (21 if rank < 4 else 19)
                    worker._configure_native_pp_engine(contract)
                    dist.barrier(group=startup)
                    return True

                worker.initialize = checked_initialize
                engine = ops.NativeEngine(
                    worker, config, parallelism, head, pp_chunk_size=PP_CHUNK_SIZE
                )
                torch.cuda.synchronize()
                (run / f"setup-rank{rank}.json").write_text(json.dumps(setup) + "\n")

                # One prompt is enqueued repeatedly at concurrency 1. Run 0 is cold
                # (empty prefix cache); runs 1..N are warm (full 32K prefix hit).
                warmups, measured = 1, 8
                cold_ms, warm_samples, outputs = None, [], []
                total = warmups + measured
                for step in range(total):
                    prompt = prompts[0]
                    dist.barrier(group=control)
                    if rank == 0:
                        started = time.perf_counter()
                        ids = engine.enqueue([prompt], return_diagnostics=False)
                        result = engine.outputs(timeout_ms=300000)
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        row = result["outputs"][0]
                        token = row["tokens"][-1]
                        # Warm runs must return the identical next token as cold:
                        # prefix reuse must not perturb the result.
                        assert token == expected_ids[0], (step, token, expected_ids[0])
                        if step < warmups:
                            cold_ms = elapsed_ms
                        else:
                            warm_samples.append(elapsed_ms)
                            outputs.append(token)
                        print(
                            "DSV41_WARM_TTFT_SAMPLE "
                            + json.dumps(
                                dict(
                                    iteration=step,
                                    phase="cold" if step < warmups else "warm",
                                    latency_ms=elapsed_ms,
                                    first_token_id=token,
                                    steps=result["steps"],
                                )
                            ),
                            flush=True,
                        )
                    dist.barrier(group=control)
                if rank == 0:
                    report = dict(
                        topology="CP4/EP4 per PP stage, PP2, world8, TP attention1",
                        input_tokens=INPUT_TOKENS,
                        output_tokens=1,
                        batch_size=1,
                        concurrency=1,
                        prefix_reuse=True,
                        reuse_cache=True,
                        pp_chunk_size=PP_CHUNK_SIZE,
                        timing="native_enqueue_to_one_token_wall",
                        loading_and_initialization_excluded=True,
                        cold_ttft_ms=cold_ms,
                        warm_ttft_median_ms=float(np.median(warm_samples)),
                        warm_ttft_min_ms=float(min(warm_samples)),
                        warm_ttft_p90_ms=float(np.percentile(warm_samples, 90)),
                        warm_latencies_ms=warm_samples,
                        warm_speedup_vs_cold=(
                            cold_ms / float(np.median(warm_samples))
                            if warm_samples and np.median(warm_samples) > 0
                            else None
                        ),
                        generated_token_ids=outputs,
                    )
                    (run / "metrics.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                dist.barrier(group=control)
                stopped = engine.stop()
                engine = None
                assert stopped["shutdown"]
                worker.close()
                dist.barrier(group=control)
                worker = None
                Dsv4SharedRuntimeBufferStore._reset_for_test()
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        finally:
            if engine is not None:
                engine.stop()
            if worker is not None:
                worker.close()
            collective.destroy_distributed_environment()


class V41PPWarmTtftSM120Test(unittest.TestCase):
    def test_warm_ttft_full_nas(self):
        from transformers import AutoTokenizer

        root = Path(os.environ["DSV41_TEST_CHECKPOINT"]).resolve(strict=True)
        storage = Path(os.environ["DSV41_ENGRAM_CACHE_DIR"]).resolve(strict=True)
        run = Path(os.environ["DSV41_WARM_TTFT_RUN"]).resolve()
        run.mkdir(parents=True, exist_ok=True)
        config, _ = load_configuration(root, max_seq_len=MAX_SEQ_LEN)
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
        self.assertGreater(shutil.disk_usage(storage).free, total + 32 * 2**30)
        available = (
            int(
                next(
                    line
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                ).split()[1]
            )
            * 1024
        )
        self.assertGreater(available, 2 * total + 128 * 2**30)

        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
        suffix = tokenizer.encode("The capital of France is", add_special_tokens=False)
        self.assertEqual(suffix, [671, 6102, 294, 8760, 344])
        filler = tokenizer.encode(
            "This document records an inference context validation. ",
            add_special_tokens=False,
        )
        prompt = (filler * ((INPUT_TOKENS + len(filler) - 1) // len(filler)))[
            : INPUT_TOKENS - len(suffix)
        ] + suffix
        self.assertEqual(len(prompt), INPUT_TOKENS)
        (run / "inputs.json").write_text(json.dumps(dict(prompt=prompt)) + "\n")
        with tempfile.TemporaryDirectory(
            prefix="v41-warm-ttft-", dir=storage
        ) as tables:
            mp.spawn(
                run_rank,
                args=(str(root), tables, str(run), [prompt], [11111]),
                nprocs=8,
                join=True,
            )
        report = json.loads((run / "metrics.json").read_text())
        # The whole point: reusing the cached prefix must be faster than recomputing.
        self.assertLess(report["warm_ttft_median_ms"], report["cold_ttft_ms"])
        print("DSV41_WARM_TTFT_COMPLETE " + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    unittest.main()
