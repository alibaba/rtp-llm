"""PP smoke declarations. Platform selection supplies GPU resources only.

The PD entry keeps --cases / PP_TEST_CASES selection; it does not run the whole
case catalogue. CP scenarios remain separate optional targets.
"""

load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

_COMMON_ARGS = "--act_type BF16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10"
_DENSE_ARGS = " --model_warm_up 0 --ssm_state_dtype fp32 --kv_cache_mem_mb 8192"

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
    _pp_test("qwen35_dense_mtp_pp_test", 4, ["pdfusion_mtp_regression"], dense = True, gpu_type = gpu_type)
    _pp_test("qwen3_pp_depth_test", 4, ["pdfusion_pp4_tp1"], gpu_type = gpu_type)
    _pp_test("qwen35_pp_shutdown_mtp_test", 8, ["fake_pp2_tp2_dp2_mtp"], dense = True, gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_tp2_test", 4, ["multi_task_prompt_pp2_tp2"], gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_dp2_test", 4, ["multi_task_prompt_pp2_dp2"], gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_pd_test", 4, ["multi_task_prompt_pp2_pd"], gpu_type = gpu_type)
    _pp_test("qwen35_multi_task_prompt_pp2_mtp_test", 2, ["multi_task_prompt_pp2_mtp"], dense = True, max_seq_len = 8192, gpu_type = gpu_type)
    _pp_test("qwen35_multi_task_prompt_pp2_pd_mtp_test", 4, ["multi_task_prompt_pp2_pd_mtp"], dense = True, max_seq_len = 8192, gpu_type = gpu_type)
    _pp_test("qwen35_pd_mtp_pp_test", 4, ["sym_mtp1", "sym_mtp3"], dense = True, gpu_type = gpu_type)
    _pp_test("qwen3_pd_pp_alltoall_test", 6, ["cp_full_alltoall"], extra_args = " --use_ub_comm 0", gpu_type = gpu_type)
    _pp_test("qwen3_multi_task_prompt_pp2_cp2_test", 8, ["multi_task_prompt_pp2_cp2"], gpu_type = gpu_type)

    native.test_suite(
        name = "smoke_pp",
        tests = [
            ":qwen3_pd_pp_test",
            ":qwen35_dense_mtp_pp_test",
            ":qwen3_pp_depth_test",
            ":qwen35_pp_shutdown_mtp_test",
            ":qwen3_multi_task_prompt_pp2_tp2_test",
            ":qwen3_multi_task_prompt_pp2_dp2_test",
            ":qwen3_multi_task_prompt_pp2_pd_test",
            ":qwen35_multi_task_prompt_pp2_mtp_test",
            ":qwen35_multi_task_prompt_pp2_pd_mtp_test",
        ],
        tags = ["manual"],
    )

    # /** Preserve existing entry names when consolidating PP declarations. */
    native.test_suite(
        name = "pp_topology_test",
        tags = ["manual"],
        tests = [":qwen3_pd_pp_test"],
    )
    native.test_suite(
        name = "qwen3_multi_task_prompt_pp_test",
        tests = [":qwen3_multi_task_prompt_pp2_tp2_test"],
        tags = ["manual"],
    )
