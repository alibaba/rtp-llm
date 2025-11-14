import importlib
import os
import sys
from unittest import TestCase, main


class ServerArgsDefaultTest(TestCase):
    def setUp(self):
        # 清空环境变量和命令行参数，确保测试环境纯净
        os.environ.clear()
        sys.argv = ["prog"]

    def test_default_args_env(self):
        # 动态导入 server_args 以应用纯净的环境
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        rtp_llm.server.server_args.server_args.setup_args()
        env = os.environ

        # 1. Parallelism and Distributed Setup Configuration
        self.assertIsNone(env.get("TP_SIZE"))  # 有默认值
        self.assertIsNone(env.get("EP_SIZE"))  # 无默认值
        self.assertIsNone(env.get("DP_SIZE"))  # 无默认值
        self.assertIsNone(env.get("WORLD_SIZE"))  # 无默认值
        self.assertIsNone(env.get("WORLD_RANK"))  # 无默认值
        self.assertIsNone(env.get("LOCAL_WORLD_SIZE"))  # 无默认值
        self.assertEqual(env.get("FFN_SP_SIZE"), "1")

        # 2. Concurrency 控制
        self.assertEqual(env.get("CONCURRENCY_WITH_BLOCK"), "0")  # 默认False->"0"
        self.assertEqual(env.get("CONCURRENCY_LIMIT"), "32")

        # 3. FMHA
        self.assertEqual(env.get("ENABLE_FMHA"), "1")  # 默认True->"1"
        self.assertEqual(env.get("ENABLE_TRT_FMHA"), "1")  # 默认False->"0"
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
        self.assertEqual(env.get("INT8_KV_CACHE"), "0")
        self.assertEqual(env.get("KV_CACHE_MEM_MB"), "-1")
        # self.assertIsNone(env.get("SEQ_SIZE_PER_BLOCK"))
        self.assertEqual(env.get("TEST_BLOCK_NUM"), "0")
        self.assertEqual(env["ENABLE_3FS"], "0")
        self.assertEqual(env["THREEFS_MATCH_TIMEOUT_MS"], "1000")
        self.assertEqual(env.get("THREEFS_RPC_GET_CACHE_TIMEOUT_MS"), "3000")
        self.assertEqual(env.get("THREEFS_RPC_PUT_CACHE_TIMEOUT_MS"), "3000")
        self.assertEqual(env.get("THREEFS_READ_TIMEOUT_MS"), "1000")
        self.assertEqual(env.get("THREEFS_WRITE_TIMEOUT_MS"), "2000")
        self.assertEqual(env.get("MAX_BLOCK_SIZE_PER_ITEM"), "16")
        self.assertEqual(env.get("THREEFS_READ_IOV_SIZE"), "4294967296")
        self.assertEqual(env.get("THREEFS_WRITE_IOV_SIZE"), "4294967296")
        self.assertEqual(env.get("MEMORY_BLOCK_CACHE_SIZE_MB"), "0")
        self.assertEqual(env.get("MEMORY_BLOCK_CACHE_SYNC_TIMEOUT_MS"), "10000")

        # 5. Profiling、Debugging、Logging
        self.assertEqual(env.get("RTP_LLM_TRACE_MEMORY"), "0")
        self.assertEqual(env.get("RTP_LLM_TRACE_MALLOC_STACK"), "0")
        self.assertEqual(env.get("ENABLE_DEVICE_PERF"), "0")
        self.assertEqual(env.get("FT_CORE_DUMP_ON_EXCEPTION"), "0")
        # self.assertIsNone(env.get("FT_ALOG_CONF_PATH"))
        self.assertEqual(env.get("LOG_LEVEL"), "INFO")
        self.assertEqual(env.get("GEN_TIMELINE_SYNC"), "0")
        # TORCH_CUDA_PROFILER_DIR has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("TORCH_CUDA_PROFILER_DIR"))
        # self.assertIsNone(env.get("NCCL_DEBUG_FILE"))
        self.assertEqual(env.get("DEBUG_LOAD_SERVER"), "0")
        self.assertEqual(env.get("HACK_LAYER_NUM"), "0")
        self.assertIsNone(env.get("DEBUG_START_FAKE_PROCESS"))
        self.assertIsNone(env.get("DG_PRINT_REG_REUSE"))
        self.assertIsNone(env.get("DISABLE_DPC_RANDOM"))
        self.assertEqual(env.get("QWEN_AGENT_DEBUG"), "0")

        # 6. 硬件/Kernel 特定优化
        self.assertIsNone(env.get("DEEP_GEMM_NUM_SM"))
        self.assertEqual(env.get("ARM_GEMM_USE_KAI"), "0")
        self.assertEqual(env.get("ENABLE_STABLE_SCATTER_ADD"), "0")
        self.assertEqual(env.get("ENABLE_MULTI_BLOCK_MODE"), "1")
        self.assertEqual(env.get("ROCM_HIPBLASLT_CONFIG"), "gemm_config.csv")
        self.assertEqual(env.get("USE_SWIZZLEA"), "0")
        # self.assertIsNone(env.get("FT_DISABLE_CUSTOM_AR"))
        self.assertEqual(env.get("ENABLE_CUDA_GRAPH"), "0")
        self.assertEqual(env.get("ENABLE_CUDA_GRAPH_DEBUG_MODE"), "0")
        self.assertEqual(env.get("USE_AITER_PA"), "1")
        self.assertEqual(env.get("USE_ASM_PA"), "1")
        self.assertEqual(env.get("ENABLE_NATIVE_CUDA_GRAPH"), "0")
        self.assertEqual(env.get("NUM_NATIVE_CUDA_GRAPH"), "200")

        # 7. 采样
        self.assertEqual(env.get("MAX_BATCH_SIZE"), "0")
        self.assertEqual(env.get("ENABLE_FLASHINFER_SAMPLE_KERNEL"), "1")

        # 8. 设备和资源管理
        self.assertEqual(env.get("DEVICE_RESERVE_MEMORY_BYTES"), "0")
        self.assertEqual(env.get("HOST_RESERVE_MEMORY_BYTES"), "4294967296")
        self.assertEqual(env.get("OVERLAP_MATH_SM_COUNT"), "0")
        self.assertEqual(env.get("OVERLAP_COMM_TYPE"), "0")
        self.assertEqual(env.get("M_SPLIT"), "0")
        # self.assertIsNone(env.get("ENABLE_COMM_OVERLAP"))
        self.assertEqual(env.get("ENABLE_LAYER_MICRO_BATCH"), "0")
        self.assertEqual(env.get("NOT_USE_DEFAULT_STREAM"), "0")
        self.assertEqual(env.get("RESERVER_RUNTIME_MEM_MB"), "1024")
        # SPECIFY_GPU_ARCH has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("SPECIFY_GPU_ARCH"))
        self.assertIsNone(env.get("ACEXT_GEMM_CONFIG_DIR"))
        self.assertEqual(env.get("DEVICE_RESERVE_MEMORY_BYTES"), "0")

        # 9. MOE 专家并行
        self.assertEqual(env.get("USE_DEEPEP_MOE"), "0")
        self.assertEqual(env.get("USE_DEEPEP_INTERNODE"), "0")
        self.assertEqual(env.get("USE_DEEPEP_LOW_LATENCY"), "1")
        self.assertEqual(env.get("USE_DEEPEP_P2P_LOW_LATENCY"), "0")
        self.assertEqual(env.get("DEEP_EP_NUM_SM"), "0")
        self.assertEqual(env.get("FAKE_BALANCE_EXPERT"), "0")
        self.assertEqual(env.get("EPLB_CONTROL_STEP"), "100")
        self.assertEqual(env.get("EPLB_TEST_MODE"), "0")
        self.assertEqual(env.get("EPLB_BALANCE_LAYER_PER_STEP"), "1")
        self.assertEqual(env.get("EPLB_MODE"), "NONE")
        self.assertEqual(env.get("EPLB_UPDATE_TIME"), "5000")
        self.assertEqual(env.get("REDUNDANT_EXPERT"), "0")
        self.assertEqual(env.get("HACK_EP_SINGLE_ENTRY"), "0")
        self.assertEqual(env.get("BALANCE_METHOD"), "mix")
        self.assertEqual(env.get("EPLB_FORCE_REPACK"), "0")
        self.assertEqual(env.get("EPLB_STATS_WINDOW_SIZE"), "10")
        self.assertEqual(env.get("RTP_LLM_MAX_MOE_NORMAL_MASKED_TOKEN_NUM"), "1024")

        # 10. 模型特定配置
        self.assertEqual(env.get("MAX_LORA_MODEL_SIZE"), "-1")

        # 11. 投机采样配置
        # SP_MODEL_TYPE and SP_TYPE have default="", so they won't be set unless explicitly provided
        self.assertIsNone(env.get("SP_MODEL_TYPE"))
        self.assertIsNone(env.get("SP_TYPE"))
        self.assertEqual(env.get("SP_MIN_TOKEN_MATCH"), "2")
        self.assertEqual(env.get("SP_MAX_TOKEN_MATCH"), "2")
        # TREE_DECODE_CONFIG has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("TREE_DECODE_CONFIG"))
        self.assertEqual(env.get("GEN_NUM_PER_CIRCLE"), "1")
        self.assertIsNone(env.get("SP_ACT_TYPE"))
        self.assertIsNone(env.get("SP_QUANTIZATION"))
        self.assertIsNone(env.get("SP_CHECKPOINT_PATH"))
        self.assertEqual(env.get("FORCE_STREAM_SAMPLE"), "0")
        self.assertEqual(env.get("FORCE_SCORE_CONTEXT_ATTENTION"), "1")

        # 12. RPC 与服务发现配置
        self.assertEqual(env.get("USE_LOCAL"), "0")
        self.assertIsNone(env.get("REMOTE_RPC_SERVER_IP"))
        self.assertIsNone(env.get("RTP_LLM_DECODE_CM2_CONFIG"))
        self.assertIsNone(env.get("REMOTE_VIT_SERVER_IP"))
        self.assertIsNone(env.get("RTP_LLM_MULTIMODAL_PART_CM2_CONFIG"))

        # 13. Cache Store 配置
        self.assertEqual(env.get("CACHE_STORE_RDMA_MODE"), "0")
        self.assertEqual(env.get("WRR_AVAILABLE_RATIO"), "80")
        self.assertEqual(env.get("RANK_FACTOR"), "0")

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

        # 17. Gang Configuration
        self.assertEqual(env.get("FAKE_GANG_ENV"), "0")
        self.assertEqual(env.get("GANG_ANNOCATION_PATH"), "/etc/podinfo/annotations")
        self.assertIsNone(env.get("GANG_CONFIG_STRING"))
        # ZONE_NAME has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("ZONE_NAME"))
        self.assertIsNone(env.get("DISTRIBUTE_CONFIG_FILE"))
        self.assertEqual(env.get("DIST_BARRIER_TIMEOUT"), "45")
        self.assertEqual(env.get("GANG_SLEEP_TIME"), "10")
        self.assertEqual(env.get("GANG_TIMEOUT_MIN"), "30")

        # 18. Vit Configuration
        self.assertEqual(env.get("VIT_SEPARATION"), "0")
        self.assertEqual(env.get("VIT_TRT"), "0")
        self.assertEqual(env.get("TRT_CACHE_ENABLED"), "0")
        self.assertEqual(
            env.get("TRT_CACHE_PATH"), os.path.join(os.getcwd(), "trt_cache")
        )
        # DOWNLOAD_HEADERS has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("DOWNLOAD_HEADERS"))
        self.assertEqual(env.get("MM_CACHE_ITEM_NUM"), "10")
        self.assertEqual(env.get("URL_CACHE_ITEM_NUM"), "100")

        # 19. Server Configuration
        self.assertEqual(env.get("FRONTEND_SERVER_COUNT"), "4")
        self.assertEqual(env.get("START_PORT"), "8088")
        self.assertEqual(env.get("TIMEOUT_KEEP_ALIVE"), "5")
        self.assertEqual(env.get("FRONTEND_SERVER_ID"), "0")

        # 20. Generate Configuration
        self.assertEqual(env.get("THINK_END_TAG"), "</think>\n\n")
        self.assertEqual(env.get("THINK_END_TOKEN_ID"), "-1")
        self.assertEqual(env.get("THINK_MODE"), "0")
        self.assertEqual(env.get("FORCE_STOP_WORDS"), "0")
        self.assertIsNone(env.get("STOP_WORDS_LIST"))
        self.assertIsNone(env.get("STOP_WORDS_STR"))
        self.assertEqual(env.get("THINK_START_TAG"), "<think>\\n")
        self.assertIsNone(env.get("GENERATION_CONFIG_PATH"))

        # 21. Quantization Configuration
        self.assertEqual(env.get("INT8_MODE"), "0")
        self.assertIsNone(env.get("QUANTIZATION"))

        # 22. Sparse Configuration (deprecated)

        # 23. Engine Configuration
        self.assertEqual(env.get("WARM_UP"), "1")
        self.assertEqual(env.get("WARM_UP_WITH_LOSS"), "0")
        # MAX_SEQ_LEN is in ModelConfig, not server_args, so it won't be set here
        self.assertIsNone(env.get("MAX_SEQ_LEN"))

        # 24. Embedding Configuration
        self.assertEqual(env.get("EMBEDDING_MODEL"), "0")
        self.assertIsNone(env.get("EXTRA_INPUT_IN_MM_EMBEDDING"))

        # 25. Worker Configuration
        self.assertEqual(env.get("WORKER_INFO_PORT_NUM"), "7")

        # 26. Model Configuration
        self.assertIsNone(env.get("EXTRA_DATA_PATH"))
        self.assertIsNone(env.get("LOCAL_EXTRA_DATA_PATH"))
        self.assertIsNone(env.get("TOKENIZER_PATH"))
        self.assertEqual(env.get("ACT_TYPE"), "FP16")
        self.assertEqual(env.get("USE_FLOAT32"), "0")
        self.assertEqual(env.get("MLA_OPS_TYPE"), "AUTO")
        self.assertIsNone(env.get("WEIGHT_TYPE"))
        self.assertIsNone(env.get("TASK_TYPE"))
        self.assertIsNone(env.get("MODEL_TYPE"))
        self.assertIsNone(env.get("CHECKPOINT_PATH"))
        self.assertIsNone(env.get("OSS_ENDPOINT"))
        self.assertIsNone(env.get("PTUNING_PATH"))
        self.assertEqual(env.get("DASHSCOPE_API_KEY"), "EMPTY")
        self.assertIsNone(env.get("DASHSCOPE_HTTP_URL"))
        self.assertIsNone(env.get("DASHSCOPE_WEBSOCKET_URL"))
        self.assertEqual(env.get("OPENAI_API_KEY"), "EMPTY")
        self.assertEqual(env.get("JSON_MODEL_OVERRIDE_ARGS"), "{}")

        # 27. Lora Configuration
        self.assertEqual(env.get("LORA_INFO"), "{}")
        self.assertEqual(env.get("MERGE_LORA"), "1")

        # 28. Load Configuration
        # PHY2LOG_PATH has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("PHY2LOG_PATH"))
        self.assertEqual(env.get("CONVERTER_NUM_PER_GPU"), "4")
        self.assertEqual(env.get("TOKENIZERS_PARALLELISM"), "0")
        self.assertEqual(env.get("LOAD_CKPT_NUM_PROCESS"), "0")

        # 29. Render Configuration
        self.assertIsNone(env.get("MODEL_TEMPLATE_TYPE"))
        self.assertEqual(env.get("DEFAULT_CHAT_TEMPLATE_KEY"), "default")
        self.assertEqual(env.get("DEFAULT_TOOL_USE_TEMPLATE_KEY"), "tool_use")
        # LLAVA_CHAT_TEMPLATE has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("LLAVA_CHAT_TEMPLATE"))

        # 30. Miscellaneous Configuration
        self.assertEqual(env.get("DISABLE_PDL"), "1")


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
            "--tp_size",
            "3",
            "--ep_size",
            "5",
            "--dp_size",
            "4",
            "--world_size",
            "12",
            "--world_rank",
            "2",
            "--local_world_size",
            "6",
            "--ffn_sp_size",
            "2",
            # 2. Concurrency 控制
            "--concurrency_with_block",
            "True",
            "--concurrency_limit",
            "64",
            # 3. FMHA
            "--enable_fmha",
            "False",
            "--enable_trt_fmha",
            "False",
            "--enable_paged_trt_fmha",
            "False",
            "--enable_open_source_fmha",
            "False",
            "--enable_paged_open_source_fmha",
            "False",
            "--enable_trtv1_fmha",
            "False",
            "--fmha_perf_instrument",
            "True",
            "--fmha_show_params",
            "True",
            "--disable_flash_infer",
            "True",
            "--enable_xqa",
            "False",
            # 4. KV Cache 相关配置
            "--reuse_cache",
            "True",
            "--multi_task_prompt",
            "/tmp/another_prompt.json",
            "--multi_task_prompt_str",
            '{"task": "another"}',
            "--int8_kv_cache",
            "1",
            "--kv_cache_mem_mb",
            "2048",
            "--seq_size_per_block",
            "64",
            "--test_block_num",
            "128",
            "--enable_3fs",
            "True",
            "--threefs_match_timeout_ms",
            "5000",
            "--threefs_rpc_get_cache_timeout_ms",
            "5000",
            "--threefs_rpc_put_cache_timeout_ms",
            "5000",
            "--threefs_read_timeout_ms",
            "5000",
            "--threefs_write_timeout_ms",
            "5000",
            "--threefs_read_iov_size",
            "1073741824",
            "--threefs_write_iov_size",
            "1073741824",
            "--memory_block_cache_size_mb",
            "10",
            "--memory_block_cache_sync_timeout_ms",
            "5000",
            # 5. Profiling、Debugging、Logging
            "--trace_memory",
            "True",
            "--trace_malloc_stack",
            "True",
            "--enable_device_perf",
            "True",
            "--ft_core_dump_on_exception",
            "True",
            "--ft_alog_conf_path",
            "/tmp/another_log.conf",
            "--log_level",
            "ERROR",
            "--gen_timeline_sync",
            "True",
            "--torch_cuda_profiler_dir",
            "/path/to/dir",
            # Note: log_path and log_file_backup_count are in ProfilingDebugLoggingConfig
            # They are not command-line arguments, but are set via environment variables
            "--nccl_debug_file",
            "/tmp/nccl.log",
            "--debug_load_server",
            "True",
            "--hack_layer_num",
            "2",
            "--debug_start_fake_process",
            "True",
            "--dg_print_reg_reuse",
            "True",
            "--qwen_agent_debug",
            "1",
            "--disable_dpc_random",
            "True",
            # 6. 硬件/Kernel 特定优化
            "--deep_gemm_num_sm",
            "16",
            "--arm_gemm_use_kai",
            "True",
            "--enable_stable_scatter_add",
            "True",
            "--enable_multi_block_mode",
            "False",
            "--rocm_hipblaslt_config",
            "another_gemm_config.csv",
            "--use_swizzleA",
            "False",
            "--ft_disable_custom_ar",
            "False",
            "--enable_cuda_graph",
            "True",
            "--enable_cuda_graph_debug_mode",
            "True",
            "--use_aiter_pa",
            "False",
            "--use_asm_pa",
            "False",
            "--enable_native_cuda_graph",
            "True",
            "--num_native_cuda_graph",
            "100",
            # 7. 采样
            "--max_batch_size",
            "128",
            "--enable_flashinfer_sample_kernel",
            "False",
            # 8. 设备和资源管理
            "--device_reserve_memory_bytes",
            "4096000",
            "--host_reserve_memory_bytes",
            "8192000",
            "--overlap_math_sm_count",
            "3",
            "--overlap_comm_type",
            "2",
            "--m_split",
            "8",
            "--enable_comm_overlap",
            "False",
            "--enable_layer_micro_batch",
            "2",
            "--not_use_default_stream",
            "True",
            "--reserver_runtime_mem_mb",
            "256",
            "--specify_gpu_arch",
            "sm_90",
            "--acext_gemm_config_dir",
            "/path/to/acext",
            # 9. MOE 专家并行
            "--use_deepep_moe",
            "True",
            "--use_deepep_internode",
            "True",
            "--use_deepep_low_latency",
            "False",
            "--use_deepep_p2p_low_latency",
            "True",
            "--deep_ep_num_sm",
            "7",
            "--fake_balance_expert",
            "True",
            "--eplb_control_step",
            "300",
            "--eplb_test_mode",
            "True",
            "--eplb_balance_layer_per_step",
            "5",
            "--eplb_mode",
            "FULL",
            "--eplb_update_time",
            "9999",
            "--redundant_expert",
            "2",
            "--balance_method",
            "greedy",
            "--eplb_force_repack",
            "1",
            "--eplb_stats_window_size",
            "20",
            "--max_moe_normal_masked_token_num",
            "512",
            # 10. 模型特定配置
            "--max_lora_model_size",
            "2048",
            # 11. 投机采样配置
            "--sp_model_type",
            "deepseek-v3-mtp",
            "--sp_type",
            "mtp",
            "--sp_min_token_match",
            "5",
            "--sp_max_token_match",
            "7",
            "--tree_decode_config",
            "/tmp/another_tree.json",
            "--gen_num_per_cycle",
            "8",
            "--sp_act_type",
            "FP8",
            "--sp_quantization",
            "int8",
            "--sp_checkpoint_path",
            "/path/to/sp_ckpt",
            "--force_stream_sample",
            "True",
            "--force_score_context_attention",
            "False",
            # 12. RPC 与服务发现配置
            "--use_local",
            "True",
            "--remote_rpc_server_ip",
            "192.168.1.100:9000",
            "--decode_cm2_config",
            '{"cm2": "decode2"}',
            "--remote_vit_server_ip",
            "192.168.1.101:9001",
            "--multimodal_part_cm2_config",
            '{"cm2": "multi2"}',
            # 13. Cache Store 配置
            "--cache_store_rdma_mode",
            "True",
            "--wrr_available_ratio",
            "95",
            "--rank_factor",
            "1",
            "--cache_store_thread_count",
            "8",
            "--cache_store_rdma_connect_timeout_ms",
            "500",
            "--cache_store_rdma_qp_count_per_connection",
            "8",
            # 14. 调度器配置
            "--use_batch_decode_scheduler",
            "True",
            # 15. FIFO 调度器配置
            "--max_context_batch_size",
            "16",
            "--scheduler_reserve_resource_ratio",
            "20",
            "--enable_fast_gen",
            "True",
            "--fast_gen_context_budget",
            "256",
            "--enable_partial_fallback",
            "True",
            # 16. BatchDecode 调度器配置
            "--batch_decode_scheduler_batch_size",
            "32",
            # 17. Gang Configuration
            "--fake_gang_env",
            "True",
            "--gang_annocation_path",
            "/tmp/annocations",
            "--gang_config_string",
            "test_gang_string",
            "--zone_name",
            "test_zone",
            "--distribute_config_file",
            "/tmp/dist.conf",
            "--dist_barrier_timeout",
            "90",
            "--gang_sleep_time",
            "20",
            "--gang_timeout_min",
            "60",
            # 18. Vit Configuration
            "--vit_separation",
            "1",
            "--vit_trt",
            "1",
            "--trt_cache_enabled",
            "1",
            "--trt_cache_path",
            "/tmp/trt_cache",
            "--download_headers",
            '{"X-Test": "True"}',
            "--mm_cache_item_num",
            "20",
            "--url_cache_item_num",
            "120",
            # 19. Server Configuration
            "--frontend_server_count",
            "8",
            "--start_port",
            "9099",
            "--timeout_keep_alive",
            "10",
            "--frontend_server_id",
            "2",
            # 20. Generate Configuration
            "--think_end_tag",
            "end_think",
            "--think_end_token_id",
            "12345",
            "--think_mode",
            "1",
            "--force_stop_words",
            "True",
            "--stop_words_list",
            "54321,54322",
            "--stop_words_str",
            "stop,word",
            "--think_start_tag",
            "start_think",
            "--generation_config_path",
            "/path/to/gen.json",
            # 21. Quantization Configuration
            "--int8_mode",
            "1",
            "--quantization",
            "w8a8",
            # 22. Sparse Configuration (deprecated)
            # 23. Engine Configuration
            "--warm_up",
            "0",
            "--warm_up_with_loss",
            "1",
            # Note: max_seq_len is in ModelConfig, not ModelArgs
            # It will be set when ModelConfig is created from model_args
            # 24. Embedding Configuration
            "--embedding_model",
            "1",
            "--extra_input_in_mm_embedding",
            "INDEX",
            # 25. Worker Configuration
            "--worker_info_port_num",
            "10",
            # 26. Model Configuration
            "--extra_data_path",
            "/path/to/extra",
            "--local_extra_data_path",
            "/local/path/to/extra",
            "--tokenizer_path",
            "/path/to/tokenizer",
            "--act_type",
            "BF16",
            "--use_float32",
            "True",
            "--mla_ops_type",
            "CUSTOM",
            "--weight_type",
            "FP16",
            "--task_type",
            "generation",
            "--model_type",
            "qwen",
            "--checkpoint_path",
            "/path/to/checkpoint",
            "--oss_endpoint",
            "test.oss.endpoint",
            "--ptuning_path",
            "/path/to/ptuning",
            "--dashscope_api_key",
            "test_key",
            "--dashscope_http_url",
            "http://test.url",
            "--dashscope_websocket_url",
            "ws://test.url",
            "--openai_api_key",
            "test_openai_key",
            "--json_model_override_args",
            '{"rope_scaling":{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"beta_slow":1.0,"beta_fast":1.0,"mscale":1.0,"extrapolation_factor":1.0}}',
            # 27. Lora Configuration
            "--lora_info",
            '{"lora1": "/path/to/lora1"}',
            "--merge_lora",
            "False",
            # 28. Load Configuration
            "--phy2log_path",
            "/path/to/pylog",
            "--converter_num_per_gpu",
            "8",
            "--tokenizers_parallelism",
            "True",
            "--load_ckpt_num_process",
            "4",
            # 29. Render Configuration
            "--model_template_type",
            "qwen",
            "--default_chat_template_key",
            "custom_chat",
            "--default_tool_use_template_key",
            "custom_tool",
            "--llava_chat_template",
            "llava_template_string",
            # 30. Miscellaneous Configuration
            "--disable_pdl",
            "True",
            # 31. PD-Separation Configuration
            "--prefill_retry_times",
            "2",
            "--decode_entrance",
            "1",
            # 32 jit
            "--remote_jit_dir",
            "/home/admin/jit_dir",
        ]

        # 重新加载 server_args 并执行 setup_args
        import rtp_llm.server.server_args.server_args

        importlib.reload(rtp_llm.server.server_args.server_args)
        rtp_llm.server.server_args.server_args.setup_args()
        env = os.environ

        # 1. Parallelism and Distributed Setup Configuration
        self.assertEqual(env["TP_SIZE"], "3")
        self.assertEqual(env["EP_SIZE"], "5")
        self.assertEqual(env["DP_SIZE"], "4")
        self.assertEqual(env["WORLD_SIZE"], "12")
        self.assertEqual(env["WORLD_RANK"], "2")
        self.assertEqual(env["LOCAL_WORLD_SIZE"], "6")
        self.assertEqual(env["FFN_SP_SIZE"], "2")

        # 2. Concurrency 控制
        self.assertEqual(env["CONCURRENCY_WITH_BLOCK"], "1")
        self.assertEqual(env["CONCURRENCY_LIMIT"], "64")

        # 3. FMHA
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

        # 4. KV Cache 相关配置
        self.assertEqual(env["REUSE_CACHE"], "1")
        self.assertEqual(env["MULTI_TASK_PROMPT"], "/tmp/another_prompt.json")
        self.assertEqual(env["MULTI_TASK_PROMPT_STR"], '{"task": "another"}')
        self.assertEqual(env["INT8_KV_CACHE"], "1")
        self.assertEqual(env["KV_CACHE_MEM_MB"], "2048")
        self.assertEqual(env["SEQ_SIZE_PER_BLOCK"], "64")
        self.assertEqual(env["TEST_BLOCK_NUM"], "128")
        self.assertEqual(env["ENABLE_3FS"], "1")
        self.assertEqual(env.get("THREEFS_MATCH_TIMEOUT_MS"), "5000")
        self.assertEqual(env.get("THREEFS_RPC_GET_CACHE_TIMEOUT_MS"), "5000")
        self.assertEqual(env.get("THREEFS_RPC_PUT_CACHE_TIMEOUT_MS"), "5000")
        self.assertEqual(env.get("THREEFS_READ_TIMEOUT_MS"), "5000")
        self.assertEqual(env.get("THREEFS_WRITE_TIMEOUT_MS"), "5000")
        self.assertEqual(env.get("THREEFS_READ_IOV_SIZE"), "1073741824")
        self.assertEqual(env.get("THREEFS_WRITE_IOV_SIZE"), "1073741824")
        self.assertEqual(env.get("MEMORY_BLOCK_CACHE_SIZE_MB"), "10")
        self.assertEqual(env.get("MEMORY_BLOCK_CACHE_SYNC_TIMEOUT_MS"), "5000")

        # 5. Profiling、Debugging、Logging
        self.assertEqual(env["RTP_LLM_TRACE_MEMORY"], "1")
        self.assertEqual(env["RTP_LLM_TRACE_MALLOC_STACK"], "1")
        self.assertEqual(env["ENABLE_DEVICE_PERF"], "1")
        self.assertEqual(env["FT_CORE_DUMP_ON_EXCEPTION"], "1")
        self.assertEqual(env["FT_ALOG_CONF_PATH"], "/tmp/another_log.conf")
        self.assertEqual(env["LOG_LEVEL"], "ERROR")
        self.assertEqual(env["GEN_TIMELINE_SYNC"], "1")
        self.assertEqual(env["TORCH_CUDA_PROFILER_DIR"], "/path/to/dir")
        self.assertEqual(env["NCCL_DEBUG_FILE"], "/tmp/nccl.log")
        self.assertEqual(env["DEBUG_LOAD_SERVER"], "1")
        self.assertEqual(env["HACK_LAYER_NUM"], "2")
        self.assertEqual(env["DEBUG_START_FAKE_PROCESS"], "1")
        self.assertEqual(env["DG_PRINT_REG_REUSE"], "1")
        self.assertEqual(env["QWEN_AGENT_DEBUG"], "1")
        self.assertEqual(env["DISABLE_DPC_RANDOM"], "1")

        # 6. 硬件/Kernel 特定优化
        self.assertEqual(env["DEEP_GEMM_NUM_SM"], "16")
        self.assertEqual(env["ARM_GEMM_USE_KAI"], "1")
        self.assertEqual(env["ENABLE_STABLE_SCATTER_ADD"], "1")
        self.assertEqual(env["ENABLE_MULTI_BLOCK_MODE"], "0")
        self.assertEqual(env["ROCM_HIPBLASLT_CONFIG"], "another_gemm_config.csv")
        self.assertEqual(env["USE_SWIZZLEA"], "0")
        self.assertEqual(env["FT_DISABLE_CUSTOM_AR"], "0")
        self.assertEqual(env.get("ENABLE_CUDA_GRAPH"), "1")
        self.assertEqual(env.get("ENABLE_CUDA_GRAPH_DEBUG_MODE"), "1")
        self.assertEqual(env.get("USE_AITER_PA"), "0")
        self.assertEqual(env.get("USE_ASM_PA"), "0")
        self.assertEqual(env.get("ENABLE_NATIVE_CUDA_GRAPH"), "1")
        self.assertEqual(env.get("NUM_NATIVE_CUDA_GRAPH"), "100")

        # 7. 采样
        self.assertEqual(env["MAX_BATCH_SIZE"], "128")
        self.assertEqual(env["ENABLE_FLASHINFER_SAMPLE_KERNEL"], "0")

        # 8. 设备和资源管理
        self.assertEqual(env["DEVICE_RESERVE_MEMORY_BYTES"], "4096000")
        self.assertEqual(env["HOST_RESERVE_MEMORY_BYTES"], "8192000")
        self.assertEqual(env["OVERLAP_MATH_SM_COUNT"], "3")
        self.assertEqual(env["OVERLAP_COMM_TYPE"], "2")
        self.assertEqual(env["M_SPLIT"], "8")
        self.assertEqual(env["ENABLE_COMM_OVERLAP"], "0")
        self.assertEqual(env["ENABLE_LAYER_MICRO_BATCH"], "2")
        self.assertEqual(env["NOT_USE_DEFAULT_STREAM"], "1")
        self.assertEqual(env["RESERVER_RUNTIME_MEM_MB"], "256")
        self.assertEqual(env["SPECIFY_GPU_ARCH"], "sm_90")
        self.assertEqual(env["ACEXT_GEMM_CONFIG_DIR"], "/path/to/acext")

        # 9. MOE 专家并行
        self.assertEqual(env["USE_DEEPEP_MOE"], "1")
        self.assertEqual(env["USE_DEEPEP_INTERNODE"], "1")
        self.assertEqual(env["USE_DEEPEP_LOW_LATENCY"], "0")
        self.assertEqual(env["USE_DEEPEP_P2P_LOW_LATENCY"], "1")
        self.assertEqual(env["DEEP_EP_NUM_SM"], "7")
        self.assertEqual(env["FAKE_BALANCE_EXPERT"], "1")
        self.assertEqual(env["EPLB_CONTROL_STEP"], "300")
        self.assertEqual(env["EPLB_TEST_MODE"], "1")
        self.assertEqual(env["EPLB_BALANCE_LAYER_PER_STEP"], "5")
        self.assertEqual(env["EPLB_MODE"], "FULL")
        self.assertEqual(env["EPLB_UPDATE_TIME"], "9999")
        self.assertEqual(env["REDUNDANT_EXPERT"], "2")
        self.assertEqual(env["HACK_EP_SINGLE_ENTRY"], "1")
        self.assertEqual(env["BALANCE_METHOD"], "greedy")
        self.assertEqual(env["EPLB_FORCE_REPACK"], "1")
        self.assertEqual(env["EPLB_STATS_WINDOW_SIZE"], "20")
        self.assertEqual(env.get("RTP_LLM_MAX_MOE_NORMAL_MASKED_TOKEN_NUM"), "512")

        # 10. 模型特定配置
        self.assertEqual(env["MAX_LORA_MODEL_SIZE"], "2048")

        # 11. 投机采样配置
        self.assertEqual(env["SP_MODEL_TYPE"], "deepseek-v3-mtp")
        self.assertEqual(env["SP_TYPE"], "mtp")
        self.assertEqual(env["SP_MIN_TOKEN_MATCH"], "5")
        self.assertEqual(env["SP_MAX_TOKEN_MATCH"], "7")
        self.assertEqual(env["TREE_DECODE_CONFIG"], "/tmp/another_tree.json")
        self.assertEqual(env["GEN_NUM_PER_CIRCLE"], "8")
        self.assertEqual(env["SP_ACT_TYPE"], "FP8")
        self.assertEqual(env["SP_QUANTIZATION"], "int8")
        self.assertEqual(env["SP_CHECKPOINT_PATH"], "/path/to/sp_ckpt")
        self.assertEqual(env["FORCE_STREAM_SAMPLE"], "1")
        self.assertEqual(env["FORCE_SCORE_CONTEXT_ATTENTION"], "0")

        # 12. RPC 与服务发现配置
        self.assertEqual(env["USE_LOCAL"], "1")
        self.assertEqual(env["REMOTE_RPC_SERVER_IP"], "192.168.1.100:9000")
        self.assertEqual(env["RTP_LLM_DECODE_CM2_CONFIG"], '{"cm2": "decode2"}')
        self.assertEqual(env["REMOTE_VIT_SERVER_IP"], "192.168.1.101:9001")
        self.assertEqual(env["RTP_LLM_MULTIMODAL_PART_CM2_CONFIG"], '{"cm2": "multi2"}')

        # 13. Cache Store 配置
        self.assertEqual(env["CACHE_STORE_RDMA_MODE"], "1")
        self.assertEqual(env["WRR_AVAILABLE_RATIO"], "95")
        self.assertEqual(env["RANK_FACTOR"], "1")
        self.assertEqual(env["CACHE_STORE_THREAD_COUNT"], "8")
        self.assertEqual(env["CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS"], "500")
        self.assertEqual(env["CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION"], "8")

        # 14. 调度器配置
        self.assertEqual(env["USE_BATCH_DECODE_SCHEDULER"], "1")

        # 15. FIFO 调度器配置
        self.assertEqual(env["MAX_CONTEXT_BATCH_SIZE"], "16")
        self.assertEqual(env["SCHEDULER_RESERVE_RESOURCE_RATIO"], "20")
        self.assertEqual(env["ENABLE_FAST_GEN"], "1")
        self.assertEqual(env["FAST_GEN_MAX_CONTEXT_LEN"], "256")
        self.assertEqual(env["ENABLE_PARTIAL_FALLBACK"], "1")

        # 16. BatchDecode 调度器配置
        self.assertEqual(env["BATCH_DECODE_SCHEDULER_BATCH_SIZE"], "32")

        # 17. Gang Configuration
        self.assertEqual(env["FAKE_GANG_ENV"], "1")
        self.assertEqual(env["GANG_ANNOCATION_PATH"], "/tmp/annocations")
        self.assertEqual(env["GANG_CONFIG_STRING"], "test_gang_string")
        self.assertEqual(env["ZONE_NAME"], "test_zone")
        self.assertEqual(env["DISTRIBUTE_CONFIG_FILE"], "/tmp/dist.conf")
        self.assertEqual(env["DIST_BARRIER_TIMEOUT"], "90")
        self.assertEqual(env["GANG_SLEEP_TIME"], "20")
        self.assertEqual(env["GANG_TIMEOUT_MIN"], "60")

        # 18. Vit Configuration
        self.assertEqual(env["VIT_SEPARATION"], "1")
        self.assertEqual(env["VIT_TRT"], "1")
        self.assertEqual(env["TRT_CACHE_ENABLED"], "1")
        self.assertEqual(env["TRT_CACHE_PATH"], "/tmp/trt_cache")
        self.assertEqual(env["DOWNLOAD_HEADERS"], '{"X-Test": "True"}')
        self.assertEqual(env["MM_CACHE_ITEM_NUM"], "20")
        self.assertEqual(env["URL_CACHE_ITEM_NUM"], "120")
        # 19. Server Configuration
        self.assertEqual(env["FRONTEND_SERVER_COUNT"], "8")
        self.assertEqual(env["START_PORT"], "9099")
        self.assertEqual(env["TIMEOUT_KEEP_ALIVE"], "10")
        self.assertEqual(env["FRONTEND_SERVER_ID"], "2")

        # 20. Generate Configuration
        self.assertEqual(env["THINK_END_TAG"], "end_think")
        self.assertEqual(env["THINK_END_TOKEN_ID"], "12345")
        self.assertEqual(env["THINK_MODE"], "1")
        self.assertEqual(env["FORCE_STOP_WORDS"], "1")
        self.assertEqual(env["STOP_WORDS_LIST"], "54321,54322")
        self.assertEqual(env["STOP_WORDS_STR"], "stop,word")
        self.assertEqual(env["THINK_START_TAG"], "start_think")
        self.assertEqual(env["GENERATION_CONFIG_PATH"], "/path/to/gen.json")

        # 21. Quantization Configuration
        self.assertEqual(env["INT8_MODE"], "1")
        self.assertEqual(env["QUANTIZATION"], "w8a8")

        # 22. Sparse Configuration (deprecated)

        # 23. Engine Configuration
        self.assertEqual(env["WARM_UP"], "0")
        self.assertEqual(env["WARM_UP_WITH_LOSS"], "1")
        # MAX_SEQ_LEN is in ModelConfig, not server_args, so it won't be set here
        # Note: The test sets --max_seq_len 8192, but it's handled by ModelConfig, not server_args

        # 24. Embedding Configuration
        self.assertEqual(env["EMBEDDING_MODEL"], "1")
        self.assertEqual(env["EXTRA_INPUT_IN_MM_EMBEDDING"], "INDEX")

        # 25. Worker Configuration
        self.assertEqual(env["WORKER_INFO_PORT_NUM"], "10")

        # 26. Model Configuration
        self.assertEqual(env["EXTRA_DATA_PATH"], "/path/to/extra")
        self.assertEqual(env["LOCAL_EXTRA_DATA_PATH"], "/local/path/to/extra")
        self.assertEqual(env["TOKENIZER_PATH"], "/path/to/tokenizer")
        self.assertEqual(env["ACT_TYPE"], "BF16")
        self.assertEqual(env["USE_FLOAT32"], "1")
        self.assertEqual(env["MLA_OPS_TYPE"], "CUSTOM")
        self.assertEqual(env["WEIGHT_TYPE"], "FP16")
        self.assertEqual(env["TASK_TYPE"], "generation")
        self.assertEqual(env["MODEL_TYPE"], "qwen")
        self.assertEqual(env["CHECKPOINT_PATH"], "/path/to/checkpoint")
        self.assertEqual(env["OSS_ENDPOINT"], "test.oss.endpoint")
        self.assertEqual(env["PTUNING_PATH"], "/path/to/ptuning")
        self.assertEqual(env["DASHSCOPE_API_KEY"], "test_key")
        self.assertEqual(env["DASHSCOPE_HTTP_URL"], "http://test.url")
        self.assertEqual(env["DASHSCOPE_WEBSOCKET_URL"], "ws://test.url")
        self.assertEqual(env["OPENAI_API_KEY"], "test_openai_key")
        self.assertEqual(env["JSON_MODEL_OVERRIDE_ARGS"],
                         '{"rope_scaling":{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"beta_slow":1.0,"beta_fast":1.0,"mscale":1.0,"extrapolation_factor":1.0}}')

        # 27. Lora Configuration
        self.assertEqual(env["LORA_INFO"], '{"lora1": "/path/to/lora1"}')
        self.assertEqual(env["MERGE_LORA"], "0")

        # 28. Load Configuration
        self.assertEqual(env["PHY2LOG_PATH"], "/path/to/pylog")
        self.assertEqual(env["CONVERTER_NUM_PER_GPU"], "8")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "1")
        self.assertEqual(env["LOAD_CKPT_NUM_PROCESS"], "4")

        # 29. Render Configuration
        self.assertEqual(env["MODEL_TEMPLATE_TYPE"], "qwen")
        self.assertEqual(env["DEFAULT_CHAT_TEMPLATE_KEY"], "custom_chat")
        self.assertEqual(env["DEFAULT_TOOL_USE_TEMPLATE_KEY"], "custom_tool")
        self.assertEqual(env["LLAVA_CHAT_TEMPLATE"], "llava_template_string")

        # 30. Miscellaneous Configuration
        self.assertEqual(env["DISABLE_PDL"], "1")
        # AUX_STRING has default="", so it won't be set unless explicitly provided
        self.assertIsNone(env.get("AUX_STRING"))

        # 31. PD-Separation Configuration
        self.assertEqual(env["PREFILL_RETRY_TIMES"], "2")
        self.assertEqual(env["DECODE_ENTRANCE"], "1")

        # 32. jit
        self.assertEqual(env["REMOTE_JIT_DIR"], "/home/admin/jit_dir")


class ServerArgsPyEnvConfigsTest(TestCase):
    """Test that environment variables and command line arguments are correctly set to py_env_configs structure."""
    
    def setUp(self):
        self._environ_backup = os.environ.copy()
        self._argv_backup = sys.argv.copy()
        os.environ.clear()
    
    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._environ_backup)
        sys.argv = self._argv_backup
    
    def test_env_vars_set_to_py_env_configs(self):
        """Test that environment variables are correctly set to py_env_configs."""
        # Set environment variables
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["TP_SIZE"] = "4"
        os.environ["DP_SIZE"] = "2"
        os.environ["WORLD_SIZE"] = "8"
        os.environ["CONCURRENCY_LIMIT"] = "64"
        os.environ["MAX_CONTEXT_BATCH_SIZE"] = "32"
        os.environ["WARM_UP"] = "1"
        os.environ["MAX_SEQ_LEN"] = "4096"
        
        sys.argv = ["prog"]
        
        # Import and setup args
        import rtp_llm.server.server_args.server_args
        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        
        # Verify model_args
        self.assertEqual(py_env_configs.model_args.model_type, "qwen")
        self.assertEqual(py_env_configs.model_args.ckpt_path, "/path/to/checkpoint")
        self.assertEqual(py_env_configs.model_args.act_type, "BF16")
        
        # Verify parallelism_config
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 4)
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 2)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 8)
        
        # Verify concurrency_config
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)
        
        # Verify fifo_scheduler_config
        self.assertEqual(py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size, 32)
        
        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, True)  # bool in C++
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args
    
    def test_cmd_args_set_to_py_env_configs(self):
        """Test that command line arguments are correctly set to py_env_configs."""
        sys.argv = [
            "prog",
            "--model_type", "llama",
            "--checkpoint_path", "/path/to/llama/checkpoint",
            "--act_type", "FP16",
            "--tp_size", "8",
            "--dp_size", "4",
            "--world_size", "32",
            "--concurrency_limit", "128",
            "--max_context_batch_size", "64",
            "--warm_up", "0",
            # Note: max_seq_len is in ModelConfig, not ModelArgs
            # It will be set when ModelConfig is created from model_args
        ]
        
        # Import and setup args
        import rtp_llm.server.server_args.server_args
        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        
        # Verify model_args
        self.assertEqual(py_env_configs.model_args.model_type, "llama")
        self.assertEqual(py_env_configs.model_args.ckpt_path, "/path/to/llama/checkpoint")
        self.assertEqual(py_env_configs.model_args.act_type, "FP16")
        
        # Verify parallelism_config
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 8)
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 4)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 32)
        
        # Verify concurrency_config
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 128)
        
        # Verify fifo_scheduler_config
        self.assertEqual(py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size, 64)
        
        # Verify runtime_config (warm_up is now in RuntimeConfig)
        self.assertEqual(py_env_configs.runtime_config.warm_up, False)  # bool in C++
        # Note: max_seq_len is in ModelConfig, not RuntimeConfig or EngineConfig
        # It will be set when ModelConfig is created from model_args
    
    def test_cmd_args_override_env_vars(self):
        """Test that command line arguments override environment variables."""
        # Set environment variables
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/qwen/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["TP_SIZE"] = "4"
        os.environ["CONCURRENCY_LIMIT"] = "32"
        
        # Set command line arguments (should override env vars)
        sys.argv = [
            "prog",
            "--model_type", "llama",
            "--checkpoint_path", "/path/to/llama/checkpoint",
            "--act_type", "FP16",
            "--tp_size", "8",
            "--concurrency_limit", "64",
        ]
        
        # Import and setup args
        import rtp_llm.server.server_args.server_args
        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        
        # Verify that command line arguments override environment variables
        self.assertEqual(py_env_configs.model_args.model_type, "llama")  # Overridden
        self.assertEqual(py_env_configs.model_args.ckpt_path, "/path/to/llama/checkpoint")  # Overridden
        self.assertEqual(py_env_configs.model_args.act_type, "FP16")  # Overridden
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 8)  # Overridden
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)  # Overridden
    
    def test_mixed_env_and_cmd_args(self):
        """Test mixed environment variables and command line arguments."""
        # Set some environment variables
        os.environ["MODEL_TYPE"] = "qwen"
        os.environ["CHECKPOINT_PATH"] = "/path/to/qwen/checkpoint"
        os.environ["ACT_TYPE"] = "BF16"
        os.environ["DP_SIZE"] = "2"
        os.environ["WORLD_SIZE"] = "8"
        
        # Set some command line arguments
        sys.argv = [
            "prog",
            "--tp_size", "4",
            "--concurrency_limit", "64",
            "--max_context_batch_size", "32",
        ]
        
        # Import and setup args
        import rtp_llm.server.server_args.server_args
        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        
        # Verify values from environment variables
        self.assertEqual(py_env_configs.model_args.model_type, "qwen")
        self.assertEqual(py_env_configs.model_args.ckpt_path, "/path/to/qwen/checkpoint")
        self.assertEqual(py_env_configs.model_args.act_type, "BF16")
        self.assertEqual(py_env_configs.parallelism_config.dp_size, 2)
        self.assertEqual(py_env_configs.parallelism_config.world_size, 8)
        
        # Verify values from command line arguments
        self.assertEqual(py_env_configs.parallelism_config.tp_size, 4)
        self.assertEqual(py_env_configs.concurrency_config.concurrency_limit, 64)
        self.assertEqual(py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size, 32)
    
    def test_batch_decode_scheduler_config(self):
        """Test that batch_decode_scheduler_config is correctly set."""
        sys.argv = [
            "prog",
            "--use_batch_decode_scheduler", "1",
            "--batch_decode_scheduler_batch_size", "16",
            "--batch_decode_scheduler_warmup_type", "1",
        ]
        
        # Import and setup args
        import rtp_llm.server.server_args.server_args
        importlib.reload(rtp_llm.server.server_args.server_args)
        py_env_configs = rtp_llm.server.server_args.server_args.setup_args()
        
        # Verify batch_decode_scheduler_config
        self.assertEqual(py_env_configs.runtime_config.use_batch_decode_scheduler, True)
        self.assertEqual(py_env_configs.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size, 16)
        self.assertEqual(py_env_configs.runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type, 1)
        
        # Verify it's also set in the C++ binding object
        runtime_config = py_env_configs.runtime_config
        self.assertEqual(runtime_config.use_batch_decode_scheduler, True)
        self.assertEqual(runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size, 16)
        self.assertEqual(runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type, 1)


if __name__ == "__main__":
    main()
