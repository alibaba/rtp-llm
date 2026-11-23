"""Capture Qwen4 M11 DP8/EP8 prefill, decode, prefix-hit, and MTP timelines."""

import argparse
import json
import logging
import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

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
                    events = json.loads(path.read_text())["traceEvents"]
                    _check_trace(
                        events,
                        path,
                        phase,
                        steps,
                        prefix_length,
                        alignment_steps,
                    )
            except json.JSONDecodeError:
                pass  # Kineto exports asynchronously after the request returns.
            else:
                return
        time.sleep(2)
    raise TimeoutError(f"Incomplete {phase} timelines in {directory}: {paths}")


def _check_trace(
    events,
    path: Path,
    phase: str,
    steps: int,
    prefix_length: int = 0,
    alignment_steps: int = 0,
) -> None:
    graphs = sum(
        event.get("cat") == "cuda_runtime"
        and event.get("ph") == "X"
        and event.get("name") == "cudaGraphLaunch"
        for event in events
    )
    kernel_names = [
        event.get("name", "")
        for event in events
        if event.get("cat") in {"kernel", "gpu_kernel"}
    ]
    if phase == "mtp":
        if (
            graphs < steps
            or "_hc_prefill_inject_norm" not in kernel_names
            or ("_wr0_" in path.name and "_ple_conv_prefill_tiled" not in kernel_names)
        ):
            raise RuntimeError(
                f"Missing MTP graph replay, fused GR, or fused PLE prefill in {path}"
            )
        return
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
    previous_by_stream = {}
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
            if name == "_qsa_qtoken_prepare":
                conversions += 1
                previous = previous_by_stream.get(stream, "")
                if "deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl" not in previous:
                    raise RuntimeError(
                        f"Unexpected operation before QToken prepare in {path}: {previous}"
                    )
            launches += "decode_gen_kernel" in name
            gates += name == "_qtoken_output_gate"
        previous_by_stream[stream] = name
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
        "mtp": (4608, 1, 20, 4),
    }[phase]
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
                    num_steps=steps + alignment_steps,
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
        list(pool.map(prepare_batch(pool, 0), range(batch)))  # JIT warmup
        metrics = analyze_results(list(pool.map(prepare_batch(pool, 1), range(batch))))
        profiled_batch = prepare_batch(pool, 2)
        # TP1 has no TP peers to broadcast to. Arm each DP backend directly.
        with ThreadPoolExecutor(max_workers=WORLD_SIZE) as profile_pool:
            list(profile_pool.map(arm, range(WORLD_SIZE)))
        list(pool.map(profiled_batch, range(batch)))
    _wait_for_traces(
        output / "timelines",
        tag,
        phase,
        steps,
        prefix_length,
        alignment_steps,
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
        "--mtp", action="store_true", help="Profile DP8/EP8 MTP with CUDA Graph."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    )
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output_dir is required when invoking this script directly")
    checkpoint = args.checkpoint_path.resolve(strict=True)
    output = args.output_dir.resolve()
    (output / "timelines").mkdir(parents=True, exist_ok=False)
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = str(output)
    os.environ["MAGA_SERVER_WORK_DIR"] = str(output)
    logging.basicConfig(level=logging.INFO)

    from rtp_llm.test.perf_test.qwen4_m11_ple import shared_ple
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager

    server_args = """
        --role_type PDFUSION --frontend_server_count 1
        --world_size 8 --local_world_size 8 --world_rank 0 --tp_size 1 --dp_size 8 --ep_size 8 --pp_size 1
        --load_method scratch --act_type BF16 --quantization MXFP8 --moe_strategy mega_moe_se --use_deepep_moe 0
        --vit_separation 2 --enable_fp32_lm_head 0 --fake_balance_expert 0
        --prefill_chunk_size 0 --seq_size_per_block 128 --kernel_seq_size_per_block 128
        --enable_cuda_graph 1 --warm_up 0 --startup_health_check_timeout 3000
        --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0
    """
    if args.short_prefix:
        server_args += " --max_context_batch_size 8 --concurrency_limit 64"
    else:
        server_args += " --max_context_batch_size 1 --concurrency_limit 32"

    if args.mtp:
        server_args += """
            --max_seq_len 8192 --max_batch_tokens_size 8192 --kv_cache_mem_mb 8192 --linear_pool_blocks 8
            --decode_capture_config 1 --use_batch_decode_scheduler 0
            --sp_type mtp --sp_model_type qwen4_exp_mtp --sp_act_type BF16 --sp_quantization MXFP8 --gen_num_per_cycle 3
        """
        server_args += f" --sp_checkpoint_path {shlex.quote(str(checkpoint))}"
    elif args.short_prefix:
        server_args += """
            --max_seq_len 33024 --max_batch_tokens_size 33024 --kv_cache_mem_mb 16384 --linear_pool_blocks 16
            --decode_capture_config 4 --use_batch_decode_scheduler 1
        """
    else:
        server_args += """
            --max_seq_len 32784 --max_batch_tokens_size 16384 --kv_cache_mem_mb 16384 --linear_pool_blocks 16
            --decode_capture_config 4 --use_batch_decode_scheduler 1
        """
    # Limit recurrent checkpoints to the tail. Each seed ends one page after
    # the shared prefix, so the matching GDN boundary is retained.
    reuse_cache = args.prefix_cache or args.short_prefix or args.mtp
    server_args += f" --reuse_cache {int(reuse_cache)}"
    if reuse_cache:
        server_args += " --reuse_cache_tail_blocks 4"
    with shared_ple(checkpoint, output) as manifest:
        server_env = {
            "BATCH_DECODE_SCHEDULER_WARMUP_TYPE": "0",
            "FT_SERVER_TEST": "1",
            "GEN_TIMELINE_SYNC": "0" if args.short_prefix else "1",
            "OMP_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "QWEN4_GDN_BACKEND": "flashinfer",
            "QWEN4_PLE_SHARED_TABLE": str(manifest),
            "QWEN4_QSA_FUSED_INDEXER_PREP": "1",
            "QWEN4_QSA_INDEXER_BACKEND": "flashinfer",
            "QWEN4_QSA_INDEXER_CHUNK_SIZE": "4096",
            "TORCH_CUDA_PROFILER_DIR": str(output / "timelines"),
        }
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
            phases = ("mtp",) if args.mtp else ("decode", "prefill")
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
