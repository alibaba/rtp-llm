load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test", "smoke_test")

"""Community smoke suites.

Adapted-model smoke cases live here before promotion into a platform suite
(suites_cuda13/sm120/sm100/sm8x/h20_oss). Entries carry the `manual` tag so
they do not join default wildcard expansions; promote a case only after it is
stably green on a pool AND its checkpoint has a CI-reachable staged location.
Cases must exercise the model's real request schema, never a generic chat
prompt stand-in."""

STAGED_D1_3B = "/mnt/fuse/smoke-test/hf/LiquidAI--d1-3B--051bcc464b01b9f92942b364d9586b0ef5912432"

def community_suites():
    native.test_suite(
        name = "smoke_community_lfm2_vl",
        tests = [
            # One server startup covers the whole multimodal case set
            # (adapt-model one-case convention): text greedy goldens vs the HF
            # reference, decode CUDA graph + async stream scheduling, bf16
            # lm_head, and a vision E2E query.
            smoke_test(
                name = "lfm2_vl_all",
                task_info = "data/model/community/lfm2_vl/q_r_all.json",
                smoke_args = "--act_type BF16 --tp_size 1 --world_size 1 --use_local 1 --enable_cuda_graph 1 --warm_up 0 --max_seq_len 2048 --seq_size_per_block 2048 --concurrency_limit 8 --enable_fp32_lm_head false",
                envs = [
                    "TRITON_AUTOTUNE_CACHE_MODE=cached",
                    "RTP_LLM_STREAM_ASYNC=1",
                ],
                data = ["data/model/qwen_vl/1.jpeg"],
                gpu_type = ["H20_CU13"],
                tags = ["manual"],
            ),
            # Reuse-cache three scenarios (2026-10-14 rule): shared-prefix
            # fork hit, multimodal partial-prefix hit (one image cached, one
            # new), and mixed hit/miss in one concurrent batch. Asserts
            # aux_info reuse_len plus token-identity vs the no-cache baseline.
            custom_smoke_test(
                name = "lfm2_reuse_cache",
                deps = ["//rtp_llm:transformers"],
                data = [
                    "//rtp_llm:sdk",
                    "data/model/qwen_vl/1.jpeg",
                ],
                main = "lfm2_reuse_cache_test.py",
                smoke_args = "--act_type BF16 --tp_size 1 --world_size 1 --use_local 1 --warm_up 0 --max_seq_len 16384 --seq_size_per_block 2048 --concurrency_limit 4 --enable_cuda_graph 1 --enable_fp32_lm_head false --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0",
                gpu_type = ["H20_CU13"],
                tags = ["manual"],
            ),
        ],
    )

def community_measurement_suites():
    """Arch-variant reuse entries for first validation on sm103/sm12x.

    The reuse scenarios are token-identity self-checks (no cross-arch
    goldens needed), so they are the first target on a new architecture;
    golden cases for those pools follow after their first green run."""
    # sm103 (B300): trtllm-gen kernel lookup fails at numTokensPerPage=2048;
    # use the standard page size 128 without CUDA graph for validation (the
    # block-crossing corruption is specific to graph+async; eager mode is safe).
    reuse_args = "--act_type BF16 --tp_size 1 --world_size 1 --use_local 1 --warm_up 0 --max_seq_len 16384 --seq_size_per_block 64 --concurrency_limit 4 --enable_cuda_graph 0 --enable_fp32_lm_head false --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0"
    for name, gpu in (
        ("lfm2_reuse_cache_sm103", "SM103_CU13"),
        ("lfm2_reuse_cache_sm12x", "RTX_5000_PRO_CU13"),
    ):
        custom_smoke_test(
            name = name,
            main = "lfm2_reuse_cache_test.py",
            deps = ["//rtp_llm:transformers"],
            data = [
                "//rtp_llm:sdk",
                "data/model/qwen_vl/1.jpeg",
            ],
            smoke_args = reuse_args,
            gpu_type = [gpu],
            tags = ["manual"],
        )
