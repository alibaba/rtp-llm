"""PP topology correctness and feature smoke suites.

smoke_pp runs the core topology and feature cases. smoke_pp_optional retains
additional cache backends and feature combinations. qwen3_pd_pp_test remains
the manual --cases / PP_TEST_CASES entry; default suite members fix their cases.
Platform selection supplies GPU resources only.
"""

load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

_COMMON_ARGS = "--act_type BF16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10"
# Qwen3.5 keeps recurrent state per 2048-token block; attention kernels use 64-token pages.
_DENSE_ARGS = " --model_warm_up 0 --ssm_state_dtype fp32 --kv_cache_mem_mb 8192 --kernel_seq_size_per_block 64"

def _pp_test(name, gpus, cases = [], dense = False, max_seq_len = 4096, extra_args = "", gpu_type = []):
    custom_smoke_test(
        name = name,
        main = "pp_test.py",
        args = ["--cases=" + ",".join(cases)] if cases else [],
        smoke_args = "%s --world_size %d --max_seq_len %d --seq_size_per_block %d%s%s" % (
            _COMMON_ARGS,
            gpus,
            max_seq_len,
            2048 if dense else 16,
            _DENSE_ARGS if dense else "",
            extra_args,
        ),
        gpu_type = gpu_type,
    )

def pp_suites(gpu_type):
    _pp_test("qwen3_pd_pp_test", 8, gpu_type = gpu_type)
    _pp_test("qwen3_pd_pp_topology_test", 6, ["sym", "asym", "conv", "cp_full"], gpu_type = gpu_type)
    _pp_test("qwen3_pp_memory_cache_test", 2, ["pdfusion_pp2_memory_cache"], gpu_type = gpu_type)
    _pp_test("qwen3_pp_disk_cache_test", 2, ["pdfusion_pp2_disk_cache"], gpu_type = gpu_type)
    _pp_test("qwen3_pp_device_to_memory_cache_test", 2, ["pdfusion_pp2_device_to_memory_cache"], gpu_type = gpu_type)
    _pp_test("qwen3_pd_pp_cp_memory_cache_test", 6, ["pd_pp2_cp2_memory_cache"], gpu_type = gpu_type)
    _pp_test("qwen35_pp_mtp_memory_cache_test", 2, ["pdfusion_pp2_mtp_memory_cache"], dense = True, max_seq_len = 8192, gpu_type = gpu_type)
    _pp_test("qwen35_dense_mtp_pp_test", 4, ["pdfusion_mtp_regression"], dense = True, gpu_type = gpu_type)
    _pp_test("qwen3_pp_depth_test", 4, ["pdfusion_pp4_tp1"], gpu_type = gpu_type)
    # Set FFN_SP=1 so both PP=1 and PP=2 runs use FFN_TP=2.
    _pp_test("qwen3_pp_ffn_tp_test", 4, ["pdfusion_pp2_tp2_ffn_tp2"], extra_args = " --ffn_sp_size 1", gpu_type = gpu_type)
    _pp_test("qwen3_pp_tp_dp_test", 8, ["pdfusion_pp2_tp2_dp2"], extra_args = " --ffn_sp_size 1", gpu_type = gpu_type)
    _pp_test("qwen35_pp_shutdown_mtp_test", 8, ["fake_pp2_tp2_dp2_mtp"], dense = True, gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_tp2_test", 4, ["multi_task_prompt_pp2_tp2"], gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_dp2_test", 4, ["multi_task_prompt_pp2_dp2"], gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_pd_test", 4, ["multi_task_prompt_pp2_pd"], gpu_type = gpu_type)
    _pp_test("qwen35_multi_task_prompt_pp2_mtp_test", 2, ["multi_task_prompt_pp2_mtp"], dense = True, max_seq_len = 8192, gpu_type = gpu_type)
    _pp_test("qwen35_multi_task_prompt_pp2_pd_mtp_test", 4, ["multi_task_prompt_pp2_pd_mtp"], dense = True, max_seq_len = 8192, gpu_type = gpu_type)
    _pp_test("qwen35_pd_mtp_pp_test", 4, ["sym_mtp1", "sym_mtp3", "pp2_pp1_mtp3"], dense = True, gpu_type = gpu_type)
    _pp_test("qwen3_pd_pp_alltoall_test", 6, ["cp_full_alltoall"], extra_args = " --use_ub_comm 0", gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_cp2_test", 8, ["multi_task_prompt_pp2_cp2"], gpu_type = gpu_type)

    native.test_suite(
        name = "smoke_pp_topology",
        tests = [
            # PP4 exercises a middle stage's receive/compute/send path.
            ":qwen3_pp_depth_test",
            # One matched PP1 comparison covers TP, DP routing and full FFN TP.
            ":qwen3_pp_tp_dp_test",
            # Three TP slice plans and CP's full-KV transfer use distinct paths.
            ":qwen3_pd_pp_topology_test",
        ],
        tags = ["manual"],
    )

    native.test_suite(
        name = "smoke_pp_features",
        tests = [
            ":qwen3_pp_memory_cache_test",
            ":qwen35_dense_mtp_pp_test",
            ":qwen35_pd_mtp_pp_test",
            ":qwen35_pp_shutdown_mtp_test",
            ":qwen3_multi_task_prompt_pp2_tp2_test",
        ],
        tags = ["manual"],
    )

    native.test_suite(
        name = "smoke_pp",
        tests = [":smoke_pp_topology", ":smoke_pp_features"],
        tags = ["manual"],
    )

    native.test_suite(
        name = "smoke_pp_optional",
        tests = [
            ":qwen3_pp_ffn_tp_test",
            ":qwen3_pp_disk_cache_test",
            ":qwen3_pp_device_to_memory_cache_test",
            ":qwen3_pd_pp_cp_memory_cache_test",
            ":qwen35_pp_mtp_memory_cache_test",
            ":qwen3_multi_task_prompt_pp2_dp2_test",
            ":qwen3_multi_task_prompt_pp2_pd_test",
            ":qwen35_multi_task_prompt_pp2_mtp_test",
            ":qwen35_multi_task_prompt_pp2_pd_mtp_test",
            ":qwen3_pd_pp_alltoall_test",
            ":qwen3_multi_task_prompt_pp2_cp2_test",
        ],
        tags = ["manual"],
    )

    native.test_suite(
        name = "pp_topology_test",
        tags = ["manual"],
        tests = [":smoke_pp_topology"],
    )
    native.test_suite(
        name = "qwen3_multi_task_prompt_pp_test",
        tests = [":qwen3_multi_task_prompt_pp2_tp2_test"],
        tags = ["manual"],
    )
