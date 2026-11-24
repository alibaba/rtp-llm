"""Capture Qwen4 M11 DP8/EP8 prefill, decode, prefix-hit, and MTP timelines."""

import argparse
import json
import logging
import os
import shlex
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

WORLD_SIZE = 8
PAGE_SIZE = 128
PREFILL_PREFIX_LENGTH = 14336  # 14K cached + 2K new fits the paged Indexer workspace.


def _wait_for_traces(
    directory: Path,
    tag: str,
    phase: str,
    steps: int,
    prefix_length: int,
    alignment_steps: int = 0,
    decode_lookahead: bool = False,
) -> None:
    deadline = time.monotonic() + 180
    paths = []
    while time.monotonic() < deadline:
        paths = sorted(directory.glob(f"{tag}_wr*_*.json"))
        if len(paths) == WORLD_SIZE and all(
            any(path.name.startswith(f"{tag}_wr{rank}_") for path in paths)
            for rank in range(WORLD_SIZE)
        ):
            try:
                for path in paths:
                    trace = json.loads(path.read_text())
                    if phase == "mtp":
                        _write_mtp_phase_traces(trace, path, steps)
                        continue
                    _check_trace(
                        trace["traceEvents"],
                        path,
                        phase,
                        steps,
                        prefix_length,
                        alignment_steps,
                        decode_lookahead,
                    )
            except json.JSONDecodeError:
                pass  # Kineto exports asynchronously after the request returns.
            else:
                return
        time.sleep(2)
    raise TimeoutError(f"Incomplete {phase} timelines in {directory}: {paths}")


def _write_mtp_phase_traces(trace, path: Path, decode_steps: int) -> None:
    """Keep the original capture and export views with the original timestamps."""
    events = trace["traceEvents"]
    cpu_scopes = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") in {"cpu_op", "user_annotation"}
    ]
    names = [event["name"] for event in cpu_scopes]
    prefill = "executor.mtp.prefill_step("
    if any(
        names.count(prefill + model + "_model_forward)") != 1
        for model in ("target", "draft")
    ):
        raise RuntimeError(f"Expected one target and draft prefill in {path}")
    decode_scopes = [
        event["ts"]
        for event in cpu_scopes
        if event["name"].startswith("executor.mtp.decode_step(decode_stream_size=")
    ]
    if not decode_scopes:
        raise RuntimeError(f"Missing MTP decode scope in {path}")
    first_decode = min(decode_scopes)
    boundary = next(
        (
            event["ts"]
            for event in cpu_scopes
            if event["name"] == "engine.normal.step_work"
            and event["ts"] <= first_decode < event["ts"] + event["dur"]
        ),
        first_decode,
    )
    launch_phases = {
        event["args"]["correlation"]: event["ts"] >= boundary
        for event in events
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "correlation" in event.get("args", {})
    }
    views = ([], [])
    for event in events:
        if event.get("ph") == "M" or "ts" not in event:
            for view in views:
                view.append(event)
            continue
        if event.get("cat") == "Trace":
            continue
        decode = event["ts"] >= boundary
        if event.get("cat") in {"kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset"}:
            decode = launch_phases.get(event.get("args", {}).get("correlation"), decode)
        views[decode].append(event)

    graphs = sum(
        event.get("cat") == "cuda_runtime"
        and event.get("ph") == "X"
        and event.get("name") == "cudaGraphLaunch"
        for event in views[1]
    )
    if graphs != 4 * decode_steps or any(
        event.get("name", "").startswith(prefill) for event in views[1]
    ):
        raise RuntimeError(f"Unexpected MTP decode trace in {path}: {graphs=}")
    for phase, view in zip(("prefill", "decode"), views):
        kernels = Counter(
            event["name"]
            for event in view
            if event.get("cat") in {"kernel", "gpu_kernel"}
        )
        if not kernels:
            raise RuntimeError(f"No {phase} GPU kernels in {path}")
        if phase == "prefill" and (
            "_hc_prefill_inject_norm" not in kernels
            or ("_wr0_" in path.name and "_ple_conv_prefill_tiled" not in kernels)
        ):
            raise RuntimeError(f"Missing fused GR or PLE prefill in {path}")
        if phase == "prefill":
            # The target has 15 QSA layers and the draft has one. Each scores four windows.
            expected = {
                "_prepare_fused_prefill": 15,
                "indexer_score_unified_sm100": 64,
                "compress_top_k_sm100": 64,
                "_qsa_qtoken_prepare": 16,
                "decode_gen_kernel": 16,
                "_qtoken_output_gate": 16,
            }
            actual = {
                name: sum(count for kernel, count in kernels.items() if name in kernel)
                for name in expected
            }
            if actual != expected:
                raise RuntimeError(
                    f"Unexpected MTP prefill QSA kernels in {path}: {actual}"
                )
        name = path.name.replace("-mtp-", f"-mtp-{phase}-", 1)
        path.with_name(name).write_text(json.dumps({**trace, "traceEvents": view}))


def _check_trace(
    events,
    path: Path,
    phase: str,
    steps: int,
    prefix_length: int = 0,
    alignment_steps: int = 0,
    decode_lookahead: bool = False,
) -> None:
    launches = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime"
        and event.get("ph") == "X"
        and event.get("name") == "cudaGraphLaunch"
    ]
    graphs = len(launches)
    if decode_lookahead:
        # Kineto may stop while the last asynchronously submitted graph is
        # still running. Validate the complete predecessors and their launches.
        launches.sort(key=lambda event: event["ts"])
        groups = {event["args"]["correlation"]: [] for event in launches}
        for event in events:
            cid = event.get("args", {}).get("correlation")
            if cid in groups and event.get("cat") in {
                "kernel",
                "gpu_kernel",
                "gpu_memcpy",
                "gpu_memset",
            }:
                groups[cid].append(event)
        ahead = 0
        for pair, (current, following) in enumerate(zip(launches, launches[1:])):
            kernels = groups[current["args"]["correlation"]]
            _check_qtoken(kernels, path, phase, 1)
            previous_gpu_end = max(event["ts"] + event["dur"] for event in kernels)
            launch_delay = following["ts"] + following["dur"] - previous_gpu_end
            if pair == 0 and launch_delay >= 0:
                raise RuntimeError(
                    f"First page crossing in {path} submitted {launch_delay:.1f} us after graph 1 ended"
                )
            ahead += launch_delay < 0
        if graphs != steps or ahead < 2:
            raise RuntimeError(
                f"Expected at least two ahead-of-time replays in {path}, got {ahead} of {graphs} launches"
            )
        return
    kernel_names = [
        event.get("name", "")
        for event in events
        if event.get("cat") in {"kernel", "gpu_kernel"}
    ]
    prep = kernel_names.count(f"_prepare_fused_{phase}")
    if phase == "decode":
        score_kernel, topk_kernel, windows = "qsa_score_paged", "qsa_topk_decode", 1
    else:
        score_kernel, topk_kernel, windows = (
            "indexer_score_unified_sm100",
            "compress_top_k_sm100",
            1 if prefix_length else 4,
        )
    score = sum(score_kernel in name for name in kernel_names)
    topk = sum(topk_kernel in name for name in kernel_names)
    graph_count_ok = graphs == steps if phase == "decode" else graphs <= alignment_steps
    if (
        not graph_count_ok
        or prep != 15 * steps
        or score != 15 * steps * windows
        or topk != score
    ):
        raise RuntimeError(
            f"Unexpected {phase} trace in {path}: {graphs=} {prep=} {score=} {topk=}"
        )
    _check_qtoken(events, path, phase, steps)


def _check_qtoken(events, path, phase, steps):
    """Separate the real forward from any DP alignment steps in the raw trace."""
    phase_by_stream = {}
    conversions = launches = gates = 0
    for event in sorted(events, key=lambda event: event.get("ts", 0)):
        if event.get("cat") not in {"kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        stream = event.get("args", {}).get("stream")
        name = event.get("name", "")
        if name in {"_prepare_fused_prefill", "_prepare_fused_decode"}:
            phase_by_stream[stream] = name.removeprefix("_prepare_fused_")
        if phase_by_stream.get(stream) == phase:
            conversions += name == "_qsa_qtoken_prepare"
            launches += "decode_gen_kernel" in name
            gates += name == "_qtoken_output_gate"
    if (conversions, launches, gates) != (15 * steps,) * 3:
        raise RuntimeError(
            f"Unexpected QToken G8 launches in {path}: {conversions=} {launches=} {gates=}"
        )


def _capture(
    phase: str,
    port: int,
    checkpoint: Path,
    output: Path,
    prefix_length: int = 0,
    short_prefix: bool = False,
    alignment_steps: int = 0,
    decode_lookahead: bool = False,
    decode_new_tokens: Optional[int] = None,
) -> None:
    import grpc
    import requests

    from rtp_llm.config.py_config_modules import ServerConfig
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import StartProfileRequestPB
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
    from rtp_llm.test.perf_test.dataclass import ResponseInfo, analyze_results
    from rtp_llm.test.perf_test.test_util import create_query

    length, batch, new_tokens, steps = {
        "decode": (32768, 32, 16, 3),
        "prefill": (16384, 8, 1, 1),
        "mtp": (16381, WORLD_SIZE, 64, 6),
    }[phase]
    if decode_lookahead:
        steps = 5
        if decode_new_tokens is not None:
            new_tokens = decode_new_tokens
    if short_prefix:
        length = prefix_length + 8
        batch = 8 * WORLD_SIZE
    tag = f"qwen4-m11-dp8ep8-{phase}-b{batch}-s{length}"
    metric_name = phase
    if prefix_length:
        tag += f"-prefix{prefix_length}"
        metric_name += "_prefix_cache"

    backends = [ServerConfig() for _ in range(WORLD_SIZE)]
    for rank, backend in enumerate(backends):
        backend.start_port = port
        backend.set_local_rank(rank)

    def post(path, body, timeout):
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"http://127.0.0.1:{port}{path}", json=body, timeout=timeout
            )
            response.raise_for_status()
            return response.json()

    def set_batch_size(size):
        scheduler = post(
            "/update_scheduler_info",
            {"batch_size": size, "mode": phase},
            60,
        )
        if scheduler.get("status") != "ok":
            raise RuntimeError(f"Could not select {phase} scheduler mode: {scheduler}")

    if phase != "mtp":
        set_batch_size(batch // WORLD_SIZE)

    def infer(prompt, input_length, reuse, *, cache_enabled=False, rank=None):
        config = {
            "max_new_tokens": new_tokens,
            "min_new_tokens": new_tokens,
            "force_sp_accept": phase != "mtp",
            "reuse_cache": cache_enabled,
        }
        if rank is not None:
            backend = backends[rank]
            config["role_addrs"] = [
                {
                    "role": "PDFUSION",
                    "ip": "127.0.0.1",
                    "http_port": backend.http_port,
                    "grpc_port": backend.rpc_server_port,
                }
            ]
        result = post(
            "/", {"prompt": prompt, "top_k": 1, "generate_config": config}, 600
        )
        response = ResponseInfo(result)
        aux_info = result.get("aux_info", {})
        actual_reuse = aux_info.get("reuse_len")
        if (
            not response.success
            or (response.input_len, response.output_len) != (input_length, new_tokens)
            or actual_reuse != reuse
        ):
            raise RuntimeError(
                f"Unexpected {tag} response on rank {rank}: success={response.success}, "
                f"input_tokens={response.input_len}, output_tokens={response.output_len}, "
                f"reuse_len={actual_reuse}, expected_reuse_len={reuse}"
            )
        if phase == "mtp" and aux_info.get("speculative_draft_rounds", 0) < 2:
            raise RuntimeError(f"{tag} did not exercise multiple MTP draft rounds")
        return response

    def arm(rank):
        address = f"127.0.0.1:{backends[rank].rpc_server_port}"
        with grpc.insecure_channel(address) as channel:
            RpcServiceStub(channel).StartProfile(
                StartProfileRequestPB(
                    trace_name=tag,
                    start_step=0,
                    num_steps=steps + alignment_steps + (phase == "mtp"),
                    enable_all_rank=True,
                ),
                timeout=30,
            )

    cases = []
    if prefix_length:
        from rtp_llm.test.perf_test.cache_grid_runner import PrefixPromptFactory
        from rtp_llm.test.perf_test.test_util import _load_tokenizer

        prompts = PrefixPromptFactory(_load_tokenizer(str(checkpoint)))
        for case in range(3):
            target, prefix, _ = prompts.make_case(case, length, prefix_length)
            seed = prompts.make_seed(case, prefix, prefix_length, PAGE_SIZE)
            cases.append((seed, target))
    else:
        query = create_query(tokenizer_path=str(checkpoint), input_len_list=[length])[
            length
        ]
        cases = [(None, query)] * 3

    def prepare_batch(pool, case):
        seed, target = cases[case]
        if seed is None:
            return lambda rank: infer(
                target,
                length,
                0,
                rank=rank if phase == "mtp" or alignment_steps else None,
            )
        # Seed the matching prefix on each DP rank before profiling. A fresh
        # case keeps the previous continuation from becoming a longer hit.
        if short_prefix:
            set_batch_size(1)
        list(
            pool.map(
                lambda rank: infer(
                    seed, prefix_length + PAGE_SIZE, 0, cache_enabled=True, rank=rank
                ),
                range(WORLD_SIZE if short_prefix else batch),
            )
        )
        if short_prefix:
            set_batch_size(batch // WORLD_SIZE)
        return lambda rank: infer(
            target,
            length,
            prefix_length,
            cache_enabled=True,
            rank=rank % WORLD_SIZE if short_prefix else rank,
        )

    with ThreadPoolExecutor(max_workers=batch) as pool:
        if phase != "mtp":
            list(pool.map(prepare_batch(pool, 0), range(batch)))  # JIT warmup
            metrics = analyze_results(
                list(pool.map(prepare_batch(pool, 1), range(batch)))
            )
        profiled_batch = prepare_batch(pool, 2)
        # TP1 has no TP peers to broadcast to. Arm each DP backend directly.
        with ThreadPoolExecutor(max_workers=WORLD_SIZE) as profile_pool:
            list(profile_pool.map(arm, range(WORLD_SIZE)))
        profiled = list(pool.map(profiled_batch, range(batch)))
        if phase == "mtp":
            metrics = analyze_results(profiled)
    _wait_for_traces(
        output / "timelines",
        tag,
        phase,
        steps,
        prefix_length,
        alignment_steps,
        decode_lookahead,
    )
    (output / f"{metric_name}_metrics.json").write_text(
        json.dumps({**asdict(metrics), "reuse_len": prefix_length}, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prefix_cache",
        action="store_true",
        help="Also profile a 14K prefix hit in 16K prefill.",
    )
    mode.add_argument(
        "--short_prefix",
        action="store_true",
        help="Profile cold prefill and 8 requests per rank with 8 new tokens over a 32K prefix.",
    )
    mode.add_argument(
        "--mtp", action="store_true", help="Profile MTP prefill and decode (default)."
    )
    mode.add_argument(
        "--decode_lookahead",
        action="store_true",
        help="Validate DP8/EP8 decode submitting the next graph during the previous replay.",
    )
    parser.add_argument(
        "--decode_new_tokens",
        type=int,
        help="Override the fixed output length for --decode_lookahead.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    )
    args = parser.parse_args()
    mtp = not (args.prefix_cache or args.short_prefix or args.decode_lookahead)
    if args.output_dir is None:
        parser.error("--output_dir is required when invoking this script directly")
    if args.decode_new_tokens is not None and (
        not args.decode_lookahead or args.decode_new_tokens < 5
    ):
        parser.error(
            "--decode_new_tokens requires --decode_lookahead and at least 5 tokens"
        )
    checkpoint = args.checkpoint_path.resolve(strict=True)
    output = args.output_dir.resolve()
    (output / "timelines").mkdir(parents=True, exist_ok=False)
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = str(output)
    os.environ["MAGA_SERVER_WORK_DIR"] = str(output)
    os.environ.pop("ENABLE_FP32_LM_HEAD", None)
    logging.basicConfig(level=logging.INFO)

    from rtp_llm.test.perf_test.qwen4_m11_ple import shared_ple
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager

    server_args = """
        --role_type PDFUSION --frontend_server_count 1
        --world_size 8 --local_world_size 8 --world_rank 0 --tp_size 1 --dp_size 8 --ep_size 8 --pp_size 1
        --load_method scratch --act_type BF16 --moe_strategy mega_moe --use_deepep_moe 0
        --vit_separation 2 --fake_balance_expert 0
        --prefill_chunk_size 0 --seq_size_per_block 128 --kernel_seq_size_per_block 128
        --enable_cuda_graph 1 --warm_up 0 --startup_health_check_timeout 3000
        --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0
    """
    if args.short_prefix:
        server_args += " --max_context_batch_size 8 --concurrency_limit 64"
    else:
        server_args += " --max_context_batch_size 1 --concurrency_limit 32"

    if mtp:
        # Match the draft's token capacity so both models share the MegaMoE buffer.
        server_args += """
            --max_seq_len 16464 --max_batch_tokens_size 16464 --kv_cache_mem_mb 4352 --linear_pool_blocks 8
            --decode_capture_config 1 --use_batch_decode_scheduler 0
            --sp_type mtp --sp_model_type qwen4_exp_mtp --sp_act_type BF16 --gen_num_per_cycle 3
        """
        server_args += f" --sp_checkpoint_path {shlex.quote(str(checkpoint))}"
    elif args.short_prefix:
        server_args += """
            --max_seq_len 33024 --max_batch_tokens_size 33024 --kv_cache_mem_mb 16384 --linear_pool_blocks 16
            --decode_capture_config 4 --use_batch_decode_scheduler 1
        """
    else:
        server_args += f"""
            --max_seq_len {32768 + (args.decode_new_tokens or 16)} --max_batch_tokens_size 16384
            --kv_cache_mem_mb 16384 --linear_pool_blocks 16
            --decode_capture_config 4 --use_batch_decode_scheduler 1
        """
    # Limit recurrent checkpoints to the tail. Each seed ends one page after
    # the shared prefix, so the matching GDN boundary is retained.
    reuse_cache = args.prefix_cache or args.short_prefix or mtp
    server_args += f" --reuse_cache {int(reuse_cache)}"
    if reuse_cache:
        server_args += " --reuse_cache_tail_blocks 4"
    with shared_ple(checkpoint, output) as manifest:
        server_env = {
            "BATCH_DECODE_SCHEDULER_WARMUP_TYPE": "0",
            "FT_SERVER_TEST": "1",
            "GEN_TIMELINE_SYNC": "0",
            "RTP_BATCH_INVARIANT": "0",
            "RTP_LLM_STREAM_ASYNC": "1",
            "RTP_LLM_DEVICE_INPUT": "1",
            "RTP_LLM_DROP_BROAD_SYNC": "1",
            "RTP_LLM_TEST_MTP_DECODE_ONLY": str(int(mtp)),
            "OMP_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Override online-quantization settings inherited from the shell.
            "QUANTIZATION": "",
            "SP_QUANTIZATION": "",
            "QWEN4_GDN_BACKEND": "flashinfer",
            "QWEN4_PLE_SHARED_TABLE": str(manifest),
            "QWEN4_QSA_FUSED_INDEXER_PREP": "1",
            "QWEN4_QSA_INDEXER_BACKEND": "flashinfer",
            "QWEN4_QSA_INDEXER_CHUNK_SIZE": "4096",
            "TORCH_CUDA_PROFILER_DIR": str(output / "timelines"),
        }
        if mtp and os.environ.get("QWEN4_PROFILE_VERIFY_GRAPH_EVENT") == "1":
            server_env["QWEN4_PROFILE_VERIFY_GRAPH_EVENT"] = "1"
        (output / "launch.json").write_text(
            json.dumps(
                {"server_args": shlex.split(server_args), "server_env": server_env},
                indent=2,
            )
            + "\n"
        )
        server = MagaServerManager(smoke_args_str=server_args, env_args=server_env)
        try:
            if not server.start_server(
                model_path=str(checkpoint),
                model_type="qwen4_exp",
                tokenizer_path=str(checkpoint),
                timeout=3000,
            ):
                raise RuntimeError(f"Qwen4 did not start. See {server.log_file_path}")
            phases = ("mtp",) if mtp else ("decode",)
            if not mtp and not args.decode_lookahead:
                phases += ("prefill",)
            # The DP engine can run short decode forwards to keep ranks aligned.
            # Retain those steps in the raw trace and check the one real prefill.
            alignment_steps = 3 if args.short_prefix else 0
            for phase in phases:
                _capture(
                    phase,
                    server.port,
                    checkpoint,
                    output,
                    alignment_steps=alignment_steps if phase == "prefill" else 0,
                    decode_lookahead=args.decode_lookahead,
                    decode_new_tokens=args.decode_new_tokens,
                )
            if args.prefix_cache or args.short_prefix:
                _capture(
                    "prefill",
                    server.port,
                    checkpoint,
                    output,
                    32768 if args.short_prefix else PREFILL_PREFIX_LENGTH,
                    short_prefix=args.short_prefix,
                    alignment_steps=alignment_steps,
                )
        finally:
            server.stop_server()


if __name__ == "__main__":
    main()
