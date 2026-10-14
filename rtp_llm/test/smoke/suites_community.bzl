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

def community_vl_sm120_suites():
    """First VL-on-sm120 validation for the can_use_flash_attn gate.

    The gate now admits sm12x with flash-attn >= 2.8.4 (commit 7d413a87cf);
    its real consumers are the qwen_vl-family ViTs, which no sm120-pool
    smoke covers (merge-gate sm12x suites are dense/MoE/BERT only). This
    case closes that gap: Qwen3-VL-4B, single RTX PRO GPU, the stock
    separated llm/vit roles so the ViT server runs the flash_attn path on
    real images. Golden comparison against the recorded query_result stays
    exact - a token diff here is the numeric-divergence signal the
    round-6 review asked to surface, not noise to silence.
    """
    native.test_suite(
        name = "smoke_community_vl_sm120",
        tests = [
            smoke_test(
                name = "qwen3_vl_sm120",
                task_info = "data/model/qwen_vl/q_r_3.json",
                smoke_args = {
                    "llm": "--act_type BF16 --use_local 1 --tp_size 1 --reuse_cache 1",
                    "vit": "--act_type BF16 --use_local 1 --use_local_preprocess 1",
                },
                gpu_type = ["RTX_5000_PRO_CU13"],
                tags = ["manual"],
                data = native.glob(["data/model/llava/*.jpg"]),
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
    base_args = "--act_type BF16 --tp_size 1 --world_size 1 --use_local 1 --warm_up 0 --max_seq_len 16384 --concurrency_limit 4 --enable_fp32_lm_head false --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0"
    for name, gpu, page, graph in (
        ("lfm2_reuse_cache_sm103", "SM103_CU13", "64", "1"),
        ("lfm2_reuse_cache_sm12x", "RTX_5000_PRO_CU13", "2048", "1"),
    ):
        custom_smoke_test(
            name = name,
            main = "lfm2_reuse_cache_test.py",
            deps = ["//rtp_llm:transformers"],
            data = [
                "//rtp_llm:sdk",
                "data/model/qwen_vl/1.jpeg",
            ],
            smoke_args = base_args + " --seq_size_per_block " + page + " --enable_cuda_graph " + graph,
            gpu_type = [gpu],
            tags = ["manual"],
        )
