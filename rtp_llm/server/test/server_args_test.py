from unittest import TestCase, main
import os
import sys
import importlib

class ServerArgsDefaultTest(TestCase):
    def setUp(self):
        os.environ.clear()
        sys.argv = ["prog"]  # 不传任何参数

    def test_default_args_env(self):
        import rtp_llm.server.server_args
        import importlib
        importlib.reload(rtp_llm.server.server_args)
        rtp_llm.server.server_args.setup_args()
        env = os.environ

        # 1. Parallelism and Distributed Setup Configuration
        self.assertIsNone(env.get("TP_SIZE"))           # 有默认值
        self.assertIsNone(env.get("EP_SIZE"))               # 无默认值
        self.assertIsNone(env.get("DP_SIZE"))               # 无默认值
        self.assertIsNone(env.get("WORLD_SIZE"))            # 无默认值
        self.assertIsNone(env.get("WORLD_RANK"))            # 无默认值
        self.assertIsNone(env.get("LOCAL_WORLD_SIZE"))      # 无默认值

        # 2. Concurrency 控制
        self.assertEqual(env.get("CONCURRENCY_WITH_BLOCK"), "0")  # 默认False->"0"
        self.assertEqual(env.get("CONCURRENCY_LIMIT"),"32")

        # 3. FMHA
        self.assertEqual(env.get("ENABLE_FMHA"), "1")       # 默认True->"1"
        self.assertEqual(env.get("ENABLE_TRT_FMHA"), "1")   # 默认False->"0"
        self.assertEqual(env.get("ENABLE_PAGED_TRT_FMHA"), "1")
        self.assertEqual(env.get("ENABLE_OPENSOURCE_FMHA"), "1")
        self.assertEqual(env.get("ENABLE_PAGED_OPEN_SOURCE_FMHA"), "1")
        self.assertEqual(env.get("ENABLE_TRTV1_FMHA"), "1")
        self.assertEqual(env.get("FMHA_PERF_INSTRUMENT"), "0")
        self.assertEqual(env.get("FMHA_SHOW_PARAMS"), "0")
        self.assertEqual(env.get("DISABLE_FLASH_INFER"), "0")
        self.assertEqual(env.get("ENABLE_XQA"), "1")

        # 4. KV Cache 相关配置
        self.assertEqual(env.get("REUSE_CACHE"), "0")
        self.assertIsNone(env.get("MULTI_TASK_PROMPT"))
        self.assertIsNone(env.get("MULTI_TASK_PROMPT_STR"))

        # 5. Profiling、Debugging、Logging
        self.assertEqual(env.get("FT_NVTX"), "0")
        self.assertEqual(env.get("PY_INFERENCE_LOG_RESPONSE"), "0")
        self.assertEqual(env.get("RTP_LLM_TRACE_MEMORY"), "0")
        self.assertEqual(env.get("RTP_LLM_TRACE_MALLOC_STACK"), "0")
        self.assertEqual(env.get("ENABLE_DEVICE_PERF"), "0")
        self.assertEqual(env.get("FT_CORE_DUMP_ON_EXCEPTION"), "0")
        self.assertIsNone(env.get("FT_ALOG_CONF_PATH"))
        self.assertEqual(env.get("LOG_LEVEL"), "INFO")
        self.assertEqual(env.get("LOG_LEVEL"), "INFO")
        self.assertEqual(env.get("GEN_TIMELINE_SYNC"),"0")

        # 6. 硬件/Kernel 特定优化
        self.assertIsNone(env.get("DEEP_GEMM_NUM_SM"))
        self.assertEqual(env.get("ARM_GEMM_USE_KAI"), "0")
        self.assertEqual(env.get("ENABLE_STABLE_SCATTER_ADD"), "0")
        self.assertEqual(env.get("ENABLE_MULTI_BLOCK_MODE"), "1")
        self.assertEqual(env.get("ROCM_HIPBLASLT_CONFIG"),"gemm_config.csv")
        self.assertEqual(env.get("FT_DISABLE_CUSTOM_AR"), "1")

        # 7. 采样
        self.assertEqual(env.get("MAX_BATCH_SIZE"), "0")
        self.assertEqual(env.get("ENABLE_FLASHINFER_SAMPLE_KERNEL"), "1")

        # 8. 设备和资源管理
        self.assertEqual(env.get("DEVICE_RESERVE_MEMORY_BYTES"),"0")
        self.assertEqual(env.get("HOST_RESERVE_MEMORY_BYTES"),"4294967296")
        self.assertEqual(env.get("OVERLAP_MATH_SM_COUNT"),"0")
        self.assertEqual(env.get("OVERLAP_COMM_TYPE"),"0")
        self.assertEqual(env.get("M_SPLIT"),"0")
        self.assertEqual(env.get("ENABLE_COMM_OVERLAP"), "1")
        self.assertEqual(env.get("ENABLE_LAYER_MICRO_BATCH"), "0")
        self.assertEqual(env.get("NOT_USE_DEFAULT_STREAM"), "0")

        # 9. MOE 专家并行
        self.assertEqual(env.get("USE_DEEPEP_MOE"), "0")
        self.assertEqual(env.get("USE_DEEPEP_INTERNODE"), "0")
        self.assertEqual(env.get("USE_DEEPEP_LOW_LATENCY"), "1")
        self.assertEqual(env.get("USE_DEEPEP_P2P_LOW_LATENCY"), "0")
        self.assertEqual(env.get("DEEP_EP_NUM_SM"),"0")
        self.assertEqual(env.get("FAKE_BALANCE_EXPERT"), "0")
        self.assertEqual(env.get("EPLB_CONTROL_STEP"),"100")
        self.assertEqual(env.get("EPLB_TEST_MODE"), "0")
        self.assertEqual(env.get("EPLB_BALANCE_LAYER_PER_STEP"),"1")

        # 10. 模型特定配置
        self.assertEqual(env.get("MAX_LORA_MODEL_SIZE"),"-1")

        # 11. 投机采样配置
        self.assertEqual(env.get("SP_MODEL_TYPE"),"")
        self.assertEqual(env.get("SP_TYPE"),"")
        self.assertEqual(env.get("SP_MIN_TOKEN_MATCH"),"2")
        self.assertEqual(env.get("SP_MAX_TOKEN_MATCH"),"2")
        self.assertEqual(env.get("TREE_DECODE_CONFIG"),"")
        self.assertEqual(env.get("GEN_NUM_PER_CIRCLE"),"1")

        # 12. RPC 与服务发现配置
        self.assertEqual(env.get("USE_LOCAL"), "0")
        self.assertIsNone(env.get("REMOTE_RPC_SERVER_IP"))
        self.assertIsNone(env.get("RTP_LLM_DECODE_CM2_CONFIG"))
        self.assertIsNone(env.get("REMOTE_VIT_SERVER_IP"))
        self.assertIsNone(env.get("RTP_LLM_MULTIMODAL_PART_CM2_CONFIG"))

        # 13. Cache Store 配置
        self.assertEqual(env.get("CACHE_STORE_RDMA_MODE"), "0")
        self.assertEqual(env.get("WRR_AVAILABLE_RATIO"),"80")
        self.assertEqual(env.get("RANK_FACTOR"),"0")

        # 14. 调度器配置
        self.assertEqual(env.get("USE_BATCH_DECODE_SCHEDULER"), "0")

        # 15. FIFO 调度器配置
        self.assertEqual(env.get("MAX_CONTEXT_BATCH_SIZE"), "1")
        self.assertEqual(env.get("SCHEDULER_RESERVE_RESOURCE_RATIO"), "5")
        self.assertEqual(env.get("ENABLE_FAST_GEN"), "0")
        self.assertIsNone(env.get("FAST_GEN_MAX_CONTEXT_LEN"))
        self.assertEqual(env.get("ENABLE_PARTIAL_FALLBACK"), "0")

        # 16. BatchDecode 调度器配置
        self.assertEqual(env.get("BATCH_DECODE_SCHEDULER_BATCH_SIZE"), "1")

        # 17. Miscellaneous 配置
        self.assertEqual(env.get("LOAD_BALANCE"), "0")
        self.assertEqual(env.get("STEP_RECORDS_TIME_RANGE"), "60000000")
        self.assertEqual(env.get("STEP_RECORDS_MAX_SIZE"), "1000")

class ServerArgsSetTest(TestCase):
    def setUp(self):
        self._environ_backup = os.environ.copy()
        self._argv_backup = sys.argv.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._environ_backup)
        sys.argv = self._argv_backup

    def test_all_args_set_env(self):
        sys.argv = [
            "prog",
            # 1. Parallelism and Distributed Setup Configuration
            "--tp_size", "3",
            "--ep_size", "5",
            "--dp_size", "4",
            "--world_size", "12",
            "--world_rank", "2",
            "--local_world_size", "6",

            # 2. Concurrency 控制
            "--concurrency_with_block", "True",
            "--concurrency_limit", "64",

            # 3. FMHA
            "--enable_fmha", "False",
            "--enable_trt_fmha", "False",
            "--enable_paged_trt_fmha", "False",
            "--enable_open_source_fmha", "False",
            "--enable_paged_open_source_fmha", "False",
            "--enable_trtv1_fmha", "False",
            "--fmha_perf_instrument", "True",
            "--fmha_show_params", "True",
            "--disable_flash_infer", "True",
            "--enable_xqa", "False",

            # 4. KV Cache 相关配置
            "--reuse_cache", "True",
            "--multi_task_prompt", "/tmp/another_prompt.json",
            "--multi_task_prompt_str", '{"task": "another"}',

            # 5. Profiling、Debugging、Logging
            "--ft_nvtx", "True",
            "--py_inference_log_response", "True",
            "--rtp_llm_trace_memory", "True",
            "--rtp_llm_trace_malloc_stack", "True",
            "--enable_device_perf", "True",
            "--ft_core_dump_on_exception", "True",
            "--ft_alog_conf_path", "/tmp/another_log.conf",
            "--log_level", "ERROR",
            "--gen_timeline_sync", "True",

            # 6. 硬件/Kernel 特定优化
            "--deep_gemm_num_sm", "16",
            "--arm_gemm_use_kai", "True",
            "--enable_stable_scatter_add", "True",
            "--enable_multi_block_mode", "False",
            "--rocm_hipblaslt_config", "another_gemm_config.csv",
            "--ft_disable_custom_ar", "False",

            # 7. 采样
            "--max_batch_size", "128",
            "--enable_flashinfer_sample_kernel", "False",

            # 8. 设备和资源管理
            "--device_reserve_memory_bytes", "4096000",
            "--host_reserve_memory_bytes", "8192000",
            "--overlap_math_sm_count", "3",
            "--overlap_comm_type", "2",
            "--m_split", "8",
            "--enable_comm_overlap", "False",
            "--enable_layer_micro_batch", "2",
            "--not_use_default_stream", "True",

            # 9. MOE 专家并行
            "--use_deepep_moe", "True",
            "--use_deepep_internode", "True",
            "--use_deepep_low_latency", "False",
            "--use_deepep_p2p_low_latency", "True",
            "--deep_ep_num_sm", "7",
            "--fake_balance_expert", "True",
            "--eplb_control_step", "300",
            "--eplb_test_mode", "True",
            "--eplb_balance_layer_per_step", "5",

            # 10. 模型特定配置
            "--max_lora_model_size", "2048",

            # 11. 投机采样配置
            "--sp_model_type", "deepseek-v3-mtp",
            "--sp_type", "mtp",
            "--sp_min_token_match", "5",
            "--sp_max_token_match", "7",
            "--tree_decode_config", "/tmp/another_tree.json",
            "--gen_num_per_cycle", "8",

            # 12. RPC 与服务发现配置
            "--use_local", "True",
            "--remote_rpc_server_ip", "192.168.1.100:9000",
            "--rtp_llm_decode_cm2_config", '{"cm2": "decode2"}',
            "--remote_vit_server_ip", "192.168.1.101:9001",
            "--rtp_llm_multimodal_part_cm2_config", '{"cm2": "multi2"}',

            # 13. Cache Store 配置
            "--cache_store_rdma_mode", "True",
            "--wrr_available_ratio", "95",
            "--rank_factor", "1",

            # 14. 调度器配置
            "--use_batch_decode_scheduler", "True",

            # 15. FIFO 调度器配置
            "--max_context_batch_size", "16",
            "--scheduler_reserve_resource_ratio", "20",
            "--enable_fast_gen", "True",
            "--fast_gen_context_budget", "256",
            "--enable_partial_fallback", "True",

            # 16. BatchDecode 调度器配置
            "--batch_decode_scheduler_batch_size", "32",

            # 17. Miscellaneous 配置
            "--load_balance", "True",
            "--step_records_time_range", "240000000",
            "--step_records_max_size", "4000"
        ]

        # 重新加载 server_args 并执行 setup_args
        import rtp_llm.server.server_args
        importlib.reload(rtp_llm.server.server_args)
        rtp_llm.server.server_args.setup_args()

        # 断言所有环境变量都被正确设置
        env = os.environ
        # Parallelism and Distributed Setup Configuration
        self.assertEqual(env["TP_SIZE"], "3")
        self.assertEqual(env["EP_SIZE"], "5")
        self.assertEqual(env["DP_SIZE"], "4")
        self.assertEqual(env["WORLD_SIZE"], "12")
        self.assertEqual(env["WORLD_RANK"], "2")
        self.assertEqual(env["LOCAL_WORLD_SIZE"], "6")

        # Concurrency 控制
        self.assertEqual(env["CONCURRENCY_WITH_BLOCK"], "1")
        self.assertEqual(env["CONCURRENCY_LIMIT"], "64")

        # FMHA
        self.assertEqual(env["ENABLE_FMHA"], "0")
        self.assertEqual(env["ENABLE_TRT_FMHA"], "0")
        self.assertEqual(env["ENABLE_PAGED_TRT_FMHA"], "0")
        self.assertEqual(env["ENABLE_OPENSOURCE_FMHA"], "0")
        self.assertEqual(env["ENABLE_PAGED_OPEN_SOURCE_FMHA"], "0")
        self.assertEqual(env["ENABLE_TRTV1_FMHA"], "0")
        self.assertEqual(env["FMHA_PERF_INSTRUMENT"], "1")
        self.assertEqual(env["FMHA_SHOW_PARAMS"], "1")
        self.assertEqual(env["DISABLE_FLASH_INFER"], "1")
        self.assertEqual(env["ENABLE_XQA"], "0")

        # KV Cache 相关配置
        self.assertEqual(env["REUSE_CACHE"], "1")
        self.assertEqual(env["MULTI_TASK_PROMPT"], "/tmp/another_prompt.json")
        self.assertEqual(env["MULTI_TASK_PROMPT_STR"], '{"task": "another"}')

        # Profiling、Debugging、Logging
        self.assertEqual(env["FT_NVTX"], "1")
        self.assertEqual(env["PY_INFERENCE_LOG_RESPONSE"], "1")
        self.assertEqual(env["RTP_LLM_TRACE_MEMORY"], "1")
        self.assertEqual(env["RTP_LLM_TRACE_MALLOC_STACK"], "1")
        self.assertEqual(env["ENABLE_DEVICE_PERF"], "1")
        self.assertEqual(env["FT_CORE_DUMP_ON_EXCEPTION"], "1")
        self.assertEqual(env["FT_ALOG_CONF_PATH"], "/tmp/another_log.conf")
        self.assertEqual(env["LOG_LEVEL"], "ERROR")
        self.assertEqual(env.get("GEN_TIMELINE_SYNC"),"1")

        # 硬件/Kernel 特定优化
        self.assertEqual(env["DEEP_GEMM_NUM_SM"], "16")
        self.assertEqual(env["ARM_GEMM_USE_KAI"], "1")
        self.assertEqual(env["ENABLE_STABLE_SCATTER_ADD"], "1")
        self.assertEqual(env["ENABLE_MULTI_BLOCK_MODE"], "0")
        self.assertEqual(env["ROCM_HIPBLASLT_CONFIG"], "another_gemm_config.csv")
        self.assertEqual(env["FT_DISABLE_CUSTOM_AR"], "0")

        # 采样
        self.assertEqual(env["MAX_BATCH_SIZE"], "128")
        self.assertEqual(env["ENABLE_FLASHINFER_SAMPLE_KERNEL"], "0")

        # 设备和资源管理
        self.assertEqual(env["DEVICE_RESERVE_MEMORY_BYTES"], "4096000")
        self.assertEqual(env["HOST_RESERVE_MEMORY_BYTES"], "8192000")
        self.assertEqual(env["OVERLAP_MATH_SM_COUNT"], "3")
        self.assertEqual(env["OVERLAP_COMM_TYPE"], "2")
        self.assertEqual(env["M_SPLIT"], "8")
        self.assertEqual(env["ENABLE_COMM_OVERLAP"], "0")
        self.assertEqual(env["ENABLE_LAYER_MICRO_BATCH"], "2")
        self.assertEqual(env["NOT_USE_DEFAULT_STREAM"], "1")

        # MOE 专家并行
        self.assertEqual(env["USE_DEEPEP_MOE"], "1")
        self.assertEqual(env["USE_DEEPEP_INTERNODE"], "1")
        self.assertEqual(env["USE_DEEPEP_LOW_LATENCY"], "0")
        self.assertEqual(env["USE_DEEPEP_P2P_LOW_LATENCY"], "1")
        self.assertEqual(env["DEEP_EP_NUM_SM"], "7")
        self.assertEqual(env["FAKE_BALANCE_EXPERT"], "1")
        self.assertEqual(env["EPLB_CONTROL_STEP"], "300")
        self.assertEqual(env["EPLB_TEST_MODE"], "1")
        self.assertEqual(env["EPLB_BALANCE_LAYER_PER_STEP"], "5")

        # 模型特定配置
        self.assertEqual(env["MAX_LORA_MODEL_SIZE"], "2048")

        # 投机采样配置
        self.assertEqual(env["SP_MODEL_TYPE"], "deepseek-v3-mtp")
        self.assertEqual(env["SP_TYPE"], "mtp")
        self.assertEqual(env["SP_MIN_TOKEN_MATCH"], "5")
        self.assertEqual(env["SP_MAX_TOKEN_MATCH"], "7")
        self.assertEqual(env["TREE_DECODE_CONFIG"], "/tmp/another_tree.json")
        self.assertEqual(env["GEN_NUM_PER_CIRCLE"], "8")

        # RPC 与服务发现配置
        self.assertEqual(env["USE_LOCAL"], "1")
        self.assertEqual(env["REMOTE_RPC_SERVER_IP"], "192.168.1.100:9000")
        self.assertEqual(env["RTP_LLM_DECODE_CM2_CONFIG"], '{"cm2": "decode2"}')
        self.assertEqual(env["REMOTE_VIT_SERVER_IP"], "192.168.1.101:9001")
        self.assertEqual(env["RTP_LLM_MULTIMODAL_PART_CM2_CONFIG"], '{"cm2": "multi2"}')

        # Cache Store 配置
        self.assertEqual(env["CACHE_STORE_RDMA_MODE"], "1")
        self.assertEqual(env["WRR_AVAILABLE_RATIO"], "95")
        self.assertEqual(env["RANK_FACTOR"], "1")

        # 调度器配置
        self.assertEqual(env["USE_BATCH_DECODE_SCHEDULER"], "1")

        # FIFO 调度器配置
        self.assertEqual(env["MAX_CONTEXT_BATCH_SIZE"], "16")
        self.assertEqual(env["SCHEDULER_RESERVE_RESOURCE_RATIO"], "20")
        self.assertEqual(env["ENABLE_FAST_GEN"], "1")
        self.assertEqual(env["FAST_GEN_MAX_CONTEXT_LEN"], "256")
        self.assertEqual(env["ENABLE_PARTIAL_FALLBACK"], "1")

        # BatchDecode 调度器配置
        self.assertEqual(env["BATCH_DECODE_SCHEDULER_BATCH_SIZE"], "32")

        # Miscellaneous 配置
        self.assertEqual(env["LOAD_BALANCE"], "1")
        self.assertEqual(env["STEP_RECORDS_TIME_RANGE"], "240000000")
        self.assertEqual(env["STEP_RECORDS_MAX_SIZE"], "4000")

if __name__ == '__main__':
    main()
