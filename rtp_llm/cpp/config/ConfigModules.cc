#include "rtp_llm/cpp/config/ConfigModules.h"
#include "autil/EnvUtil.h"
#include <sstream>
#include <algorithm>
#include <string>
#include <cctype>

namespace rtp_llm {

std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
    return result;
}

bool bool_from_env_for_test(std::string env_name, bool default_value) {
    const char* val = getenv(env_name.c_str());
    if (!val) {
        return default_value;
    }
    std::string lower = to_lower(val);
    return lower == "1" || lower == "on" || lower == "true";
}

// ParallelismConfig
void ParallelismConfig::update_from_env_for_test() {
    tp_size          = autil::EnvUtil::getEnv("TP_SIZE", 1);
    ep_size          = autil::EnvUtil::getEnv("EP_SIZE", 1);
    dp_size          = autil::EnvUtil::getEnv("DP_SIZE", 1);
    pp_size          = autil::EnvUtil::getEnv("PP_SIZE", 1);
    world_size       = autil::EnvUtil::getEnv("WORLD_SIZE", 1);
    world_rank       = autil::EnvUtil::getEnv("WORLD_RANK", 0);
    local_world_size = autil::EnvUtil::getEnv("LOCAL_WORLD_SIZE", 1);
    ffn_sp_size      = autil::EnvUtil::getEnv("FFN_SP_SIZE", 1);
    ffn_tp_size      = ffn_sp_size;
    tp_rank          = world_rank % tp_size;
    ep_rank          = (world_rank / tp_size) % ep_size;
    dp_rank          = world_rank / (tp_size * ep_size);
}

// ParallelismConfig
std::string ParallelismConfig::to_string() const {
    std::ostringstream oss;
    oss << "tp_size: " << tp_size << "\n"
        << "ep_size: " << ep_size << "\n"
        << "dp_size: " << dp_size << "\n"
        << "world_size: " << world_size << "\n"
        << "world_rank: " << world_rank << "\n"
        << "pp_size: " << pp_size << "\n"
        << "local_world_size: " << local_world_size << "\n"
        << "ffn_sp_size: " << ffn_sp_size << "\n"
        << "tp_rank: " << tp_rank << "\n"
        << "ep_rank: " << ep_rank << "\n"
        << "dp_rank: " << dp_rank << "\n"
        << "ffn_tp_size: " << ffn_tp_size << "\n"
        << "ffn_tp_rank: " << ffn_tp_rank << "\n"
        << "enable_sp: " << enable_sp << "\n"
        << "nccl_ip: " << nccl_ip << "\n"
        << "use_all_gather: " << use_all_gather << "\n"
        << "tp_nccl_port: " << tp_nccl_port << "\n"
        << "dp_tp_nccl_port: " << dp_tp_nccl_port << "\n"
        << "ffn_tp_nccl_port: " << ffn_tp_nccl_port << "\n"
        << "http_port: " << http_port << "\n"
        << "model_rpc_port: " << model_rpc_port;
    return oss.str();
}

// ConcurrencyConfig
void ConcurrencyConfig::update_from_env_for_test() {
    concurrency_with_block = bool_from_env_for_test("CONCURRENCY_WITH_BLOCK", false);
    concurrency_limit      = autil::EnvUtil::getEnv("CONCURRENCY_LIMIT", 32);
}

std::string ConcurrencyConfig::to_string() const {
    std::ostringstream oss;
    oss << "concurrency_with_block: " << concurrency_with_block << "\n"
        << "concurrency_limit: " << concurrency_limit;
    return oss.str();
}

// FMHAConfig
void FMHAConfig::update_from_env_for_test() {
    enable_fmha                   = bool_from_env_for_test("ENABLE_FMHA", true);
    enable_trt_fmha               = bool_from_env_for_test("ENABLE_TRT_FMHA", true);
    enable_paged_trt_fmha         = bool_from_env_for_test("ENABLE_PAGED_TRT_FMHA", true);
    enable_open_source_fmha       = bool_from_env_for_test("ENABLE_OPENSOURCE_FMHA", true);
    enable_paged_open_source_fmha = bool_from_env_for_test("ENABLE_PAGED_OPEN_SOURCE_FMHA", true);
    enable_trtv1_fmha             = bool_from_env_for_test("ENABLE_TRTV1_FMHA", true);
    fmha_perf_instrument          = bool_from_env_for_test("FMHA_PERF_INSTRUMENT", false);
    fmha_show_params              = bool_from_env_for_test("FMHA_SHOW_PARAMS", false);
    disable_flash_infer           = bool_from_env_for_test("DISABLE_FLASH_INFER", false);
    enable_xqa                    = bool_from_env_for_test("ENABLE_XQA", true);
}

std::string FMHAConfig::to_string() const {
    std::ostringstream oss;
    oss << "enable_fmha: " << enable_fmha << "\n"
        << "enable_trt_fmha: " << enable_trt_fmha << "\n"
        << "enable_paged_trt_fmha: " << enable_paged_trt_fmha << "\n"
        << "enable_open_source_fmha: " << enable_open_source_fmha << "\n"
        << "enable_paged_open_source_fmha: " << enable_paged_open_source_fmha << "\n"
        << "enable_trtv1_fmha: " << enable_trtv1_fmha << "\n"
        << "fmha_perf_instrument: " << fmha_perf_instrument << "\n"
        << "fmha_show_params: " << fmha_show_params << "\n"
        << "disable_flash_infer: " << disable_flash_infer << "\n"
        << "enable_xqa: " << enable_xqa << "\n";
    return oss.str();
}

// KVCacheConfig
void KVCacheConfig::update_from_env_for_test() {
    reuse_cache                        = bool_from_env_for_test("REUSE_CACHE", false);
    multi_task_prompt                  = autil::EnvUtil::getEnv("MULTI_TASK_PROMPT", "");
    multi_task_prompt_str              = autil::EnvUtil::getEnv("MULTI_TASK_PROMPT_STR", "");
    enable_3fs                         = bool_from_env_for_test("ENABLE_3FS", false);
    match_timeout_ms                   = autil::EnvUtil::getEnv("MATCH_TIMEOUT_MS", 1000);
    rpc_get_cache_timeout_ms           = autil::EnvUtil::getEnv("RPC_GET_CACHE_TIMEOUT_MS", 2000);
    rpc_put_cache_timeout_ms           = autil::EnvUtil::getEnv("RPC_PUT_CACHE_TIMEOUT_MS", 2000);
    threefs_read_timeout_ms            = autil::EnvUtil::getEnv("THREEFS_READ_TIMEOUT_MS", 1000);
    threefs_write_timeout_ms           = autil::EnvUtil::getEnv("THREEFS_WRITE_TIMEOUT_MS", 2000);
    max_block_size_per_item            = autil::EnvUtil::getEnv("MAX_BLOCK_SIZE_PER_ITEM", 16);
    threefs_read_iov_size              = autil::EnvUtil::getEnv("THREEFS_READ_IOV_SIZE", 1LL << 32);   // 4GB
    threefs_write_iov_size             = autil::EnvUtil::getEnv("THREEFS_WRITE_IOV_SIZE", 1LL << 32);  // 4GB
    memory_block_cache_size_mb         = autil::EnvUtil::getEnv("MEMORY_BLOCK_CACHE_SIZE_MB", 0);
    memory_block_cache_sync_timeout_ms = autil::EnvUtil::getEnv("MEMORY_BLOCK_CACHE_SYNC_TIMEOUT_MS", 10000);
}

void KVCacheConfig::insertMultiTaskPromptTokens(std::string task_id, std::vector<int64_t> tokens_id) {
    std::vector<int> new_tokens_id;  // to convert tokens of type int64_t to type int32_t
    for (auto token_id : tokens_id) {
        new_tokens_id.push_back(token_id);
    }
    multi_task_prompt_tokens[task_id] = new_tokens_id;
}

std::string KVCacheConfig::to_string() const {
    std::ostringstream oss;
    oss << "reuse_cache: " << reuse_cache << "\n"
        << "multi_task_prompt: " << multi_task_prompt << "\n"
        << "multi_task_prompt_str: " << multi_task_prompt_str << "\n"
        << "multi_task_prompt_tokens: " << (multi_task_prompt_tokens.empty() ? "empty" : "non-empty") << "\n"
        << "enable_3fs: " << enable_3fs << "\n"
        << "match_timeout_ms: " << match_timeout_ms << "\n"
        << "rpc_get_cache_timeout_ms: " << rpc_get_cache_timeout_ms << "\n"
        << "rpc_put_cache_timeout_ms: " << rpc_put_cache_timeout_ms << "\n"
        << "threefs_read_timeout_ms: " << threefs_read_timeout_ms << "\n"
        << "threefs_write_timeout_ms: " << threefs_write_timeout_ms << "\n"
        << "max_block_size_per_item: " << max_block_size_per_item << "\n"
        << "threefs_read_iov_size: " << threefs_read_iov_size << "\n"
        << "threefs_write_iov_size: " << threefs_write_iov_size << "\n"
        << "memory_block_cache_size_mb: " << memory_block_cache_size_mb << "\n"
        << "memory_block_cache_sync_timeout_ms: " << memory_block_cache_sync_timeout_ms;
    return oss.str();
}

// ProfilingDebugLoggingConfig
void ProfilingDebugLoggingConfig::update_from_env_for_test() {
    trace_memory              = bool_from_env_for_test("RTP_LLM_TRACE_MEMORY", false);
    trace_malloc_stack        = bool_from_env_for_test("RTP_LLM_TRACE_MALLOC_STACK", false);
    enable_device_perf        = bool_from_env_for_test("ENABLE_DEVICE_PERF", false);
    ft_core_dump_on_exception = bool_from_env_for_test("FT_CORE_DUMP_ON_EXCEPTION", false);
    ft_alog_conf_path         = autil::EnvUtil::getEnv("FT_ALOG_CONF_PATH", "");
    log_level                 = autil::EnvUtil::getEnv("LOG_LEVEL", "INFO");
    gen_timeline_sync         = bool_from_env_for_test("GEN_TIMELINE_SYNC", false);
    torch_cuda_profiler_dir   = autil::EnvUtil::getEnv("TORCH_CUDA_PROFILER_DIR", "");
    log_path                  = autil::EnvUtil::getEnv("LOG_PATH", "logs");
    log_file_backup_count     = autil::EnvUtil::getEnv("LOG_FILE_BACKUP_COUNT", 16);
    nccl_debug_file           = autil::EnvUtil::getEnv("NCCL_DEBUG_FILE", "");
    debug_load_server         = bool_from_env_for_test("DEBUG_LOAD_SERVER", false);
    hack_layer_num            = autil::EnvUtil::getEnv("HACK_LAYER_NUM", 0);
    debug_start_fake_process  = bool_from_env_for_test("DEBUG_START_FAKE_PROCESS", false);
    dg_print_reg_reuse        = bool_from_env_for_test("DG_PRINT_REG_REUSE", false);
    qwen_agent_debug          = bool_from_env_for_test("QWEN_AGENT_DEBUG", false);
    disable_dpc_random        = bool_from_env_for_test("DISABLE_DPC_RANDOM", false);
    check_nan                 = bool_from_env_for_test("CHECK_NAN", false);
}

std::string ProfilingDebugLoggingConfig::to_string() const {
    std::ostringstream oss;
    oss << "trace_memory: " << trace_memory << "\n"
        << "trace_malloc_stack: " << trace_malloc_stack << "\n"
        << "enable_device_perf: " << enable_device_perf << "\n"
        << "ft_core_dump_on_exception: " << ft_core_dump_on_exception << "\n"
        << "ft_alog_conf_path: " << ft_alog_conf_path << "\n"
        << "log_level: " << log_level << "\n"
        << "gen_timeline_sync: " << gen_timeline_sync << "\n"
        << "torch_cuda_profiler_dir" << torch_cuda_profiler_dir << "\n"
        << "log_path: " << log_path << "\n"
        << "log_file_backup_count: " << log_file_backup_count << "\n"
        << "nccl_debug_file: " << nccl_debug_file << "\n"
        << "debug_load_server: " << debug_load_server << "\n"
        << "hack_layer_num: " << hack_layer_num << "\n"
        << "debug_start_fake_process: " << debug_start_fake_process << "\n"
        << "dg_print_reg_reuse: " << dg_print_reg_reuse << "\n"
        << "qwen_agent_debug" << qwen_agent_debug << "\n"
        << "disable_dpc_random" << disable_dpc_random << "\n"
        << "check_nan" << check_nan << "\n";
    return oss.str();
}

// HWKernelConfig
void HWKernelConfig::update_from_env_for_test() {
    deep_gemm_num_sm             = autil::EnvUtil::getEnv("DEEP_GEMM_NUM_SM", -1);
    arm_gemm_use_kai             = bool_from_env_for_test("ARM_GEMM_USE_KAI", false);
    enable_stable_scatter_add    = bool_from_env_for_test("ENABLE_STABLE_SCATTER_ADD", false);
    enable_multi_block_mode      = bool_from_env_for_test("ENABLE_MULTI_BLOCK_MODE", true);
    ft_disable_custom_ar         = bool_from_env_for_test("FT_DISABLE_CUSTOM_AR", true);
    rocm_hipblaslt_config        = autil::EnvUtil::getEnv("ROCM_HIPBLASLT_CONFIG", "gemm_config.csv");
    use_swizzleA                 = bool_from_env_for_test("USE_SWIZZLEA", false);
    enable_cuda_graph            = bool_from_env_for_test("ENABLE_CUDA_GRAPH", false);
    enable_cuda_graph_debug_mode = bool_from_env_for_test("ENABLE_CUDA_GRAPH_DEBUG_MODE", false);
    use_aiter_pa                 = bool_from_env_for_test("USE_AITER_PA", true);
    use_asm_pa                   = bool_from_env_for_test("USE_ASM_PA", true);
    enable_native_cuda_graph     = bool_from_env_for_test("ENABLE_NATIVE_CUDA_GRAPH", false);
    num_native_cuda_graph        = autil::EnvUtil::getEnv("NUM_NATIVE_CUDA_GRAPH", 200);
}

std::string HWKernelConfig::to_string() const {
    std::ostringstream oss;
    oss << "deep_gemm_num_sm: " << deep_gemm_num_sm << "\n"
        << "arm_gemm_use_kai: " << arm_gemm_use_kai << "\n"
        << "enable_stable_scatter_add: " << enable_stable_scatter_add << "\n"
        << "enable_multi_block_mode: " << enable_multi_block_mode << "\n"
        << "ft_disable_custom_ar: " << ft_disable_custom_ar << "\n"
        << "rocm_hipblaslt_config: " << rocm_hipblaslt_config << "\n"
        << "use_swizzleA: " << use_swizzleA << "\n"
        << "enable_cuda_graph: " << enable_cuda_graph << "\n"
        << "enable_cuda_graph_debug_mode: " << enable_cuda_graph_debug_mode << "\n"
        << "use_aiter_pa: " << use_aiter_pa << "\n"
        << "use_asm_pa: " << use_asm_pa << "\n"
        << "enable_native_cuda_graph: " << enable_native_cuda_graph << "\n"
        << "num_native_cuda_graph: " << num_native_cuda_graph;
    return oss.str();
}

// DeviceResourceConfig
void DeviceResourceConfig::update_from_env_for_test() {
    device_reserve_memory_bytes = autil::EnvUtil::getEnv("DEVICE_RESERVE_MEMORY_BYTES", 0);
    host_reserve_memory_bytes   = autil::EnvUtil::getEnv("HOST_RESERVE_MEMORY_BYTES", 4LL * 1024 * 1024 * 1024);
    overlap_math_sm_count       = autil::EnvUtil::getEnv("OVERLAP_MATH_SM_COUNT", 0);
    overlap_comm_type           = autil::EnvUtil::getEnv("OVERLAP_COMM_TYPE", 0);
    m_split                     = autil::EnvUtil::getEnv("M_SPLIT", 0);
    enable_comm_overlap         = bool_from_env_for_test("ENABLE_COMM_OVERLAP", true);
    enable_layer_micro_batch    = autil::EnvUtil::getEnv("ENABLE_LAYER_MICRO_BATCH", 0);
    not_use_default_stream      = bool_from_env_for_test("NOT_USE_DEFAULT_STREAM", false);
}

std::string DeviceResourceConfig::to_string() const {
    std::ostringstream oss;
    oss << "device_reserve_memory_bytes: " << device_reserve_memory_bytes << "\n"
        << "host_reserve_memory_bytes: " << host_reserve_memory_bytes << "\n"
        << "overlap_math_sm_count: " << overlap_math_sm_count << "\n"
        << "overlap_comm_type: " << overlap_comm_type << "\n"
        << "m_split: " << m_split << "\n"
        << "enable_comm_overlap: " << enable_comm_overlap << "\n"
        << "enable_layer_micro_batch: " << enable_layer_micro_batch;
    return oss.str();
}

// MoeConfig
void MoeConfig::update_from_env_for_test() {
    use_deepep_moe                  = bool_from_env_for_test("USE_DEEPEP_MOE", false);
    use_deepep_internode            = bool_from_env_for_test("USE_DEEPEP_INTERNODE", false);
    use_deepep_low_latency          = bool_from_env_for_test("USE_DEEPEP_LOW_LATENCY", true);
    use_deepep_p2p_low_latency      = bool_from_env_for_test("USE_DEEPEP_P2P_LOW_LATENCY", false);
    fake_balance_expert             = bool_from_env_for_test("FAKE_BALANCE_EXPERT", false);
    eplb_control_step               = autil::EnvUtil::getEnv("EPLB_CONTROL_STEP", 100);
    eplb_test_mode                  = bool_from_env_for_test("EPLB_TEST_MODE", false);
    hack_moe_expert                 = bool_from_env_for_test("HACK_MOE_EXPERT", false);
    eplb_balance_layer_per_step     = autil::EnvUtil::getEnv("EPLB_BALANCE_LAYER_PER_STEP", 1);
    deep_ep_num_sm                  = autil::EnvUtil::getEnv("DEEP_EP_NUM_SM", 0);
    max_moe_normal_masked_token_num = autil::EnvUtil::getEnv("RTP_LLM_MAX_MOE_NORMAL_MASKED_TOKEN_NUM", 1024);
}

std::string MoeConfig::to_string() const {
    std::ostringstream oss;
    oss << "use_deepep_moe: " << use_deepep_moe << "\n"
        << "use_deepep_internode: " << use_deepep_internode << "\n"
        << "use_deepep_low_latency: " << use_deepep_low_latency << "\n"
        << "use_deepep_p2p_low_latency: " << use_deepep_p2p_low_latency << "\n"
        << "fake_balance_expert: " << fake_balance_expert << "\n"
        << "eplb_control_step: " << eplb_control_step << "\n"
        << "eplb_test_mode: " << eplb_test_mode << "\n"
        << "hack_moe_expert: " << hack_moe_expert << "\n"
        << "eplb_balance_layer_per_step: " << eplb_balance_layer_per_step << "\n"
        << "deep_ep_num_sm: " << deep_ep_num_sm << "\n"
        << "max_moe_normal_masked_token_num: " << max_moe_normal_masked_token_num;
    return oss.str();
}

// ModelSpecificConfig
void ModelSpecificConfig::update_from_env_for_test() {
    max_lora_model_size = autil::EnvUtil::getEnv("MAX_LORA_MODEL_SIZE", -1);
    load_python_model   = bool_from_env_for_test("LOAD_PYTHON_MODEL", false);
}

std::string ModelSpecificConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_lora_model_size: " << max_lora_model_size << "\n";
    oss << "load_python_model:" << load_python_model << "\n";
    return oss.str();
}

// SpeculativeExecutionConfig
void SpeculativeExecutionConfig::update_from_env_for_test() {
    sp_model_type                 = autil::EnvUtil::getEnv("SP_MODEL_TYPE", "");
    sp_type                       = autil::EnvUtil::getEnv("SP_TYPE", "");
    sp_min_token_match            = autil::EnvUtil::getEnv("SP_MIN_TOKEN_MATCH", 2);
    sp_max_token_match            = autil::EnvUtil::getEnv("SP_MAX_TOKEN_MATCH", 2);
    tree_decode_config            = autil::EnvUtil::getEnv("TREE_DECODE_CONFIG", "");
    gen_num_per_cycle             = autil::EnvUtil::getEnv("GEN_NUM_PER_CIRCLE", 1);
    force_stream_sample           = autil::EnvUtil::getEnv("FORCE_STREAM_SAMPLE", false);
    force_score_context_attention = autil::EnvUtil::getEnv("FORCE_SCORE_CONTEXT_ATTENTION", true);
}

std::string SpeculativeExecutionConfig::to_string() const {
    std::ostringstream oss;
    oss << "sp_model_type: " << sp_model_type << "\n"
        << "sp_type: " << sp_type << "\n"
        << "sp_min_token_match: " << sp_min_token_match << "\n"
        << "sp_max_token_match: " << sp_max_token_match << "\n"
        << "tree_decode_config: " << tree_decode_config << "\n"
        << "gen_num_per_cycle: " << gen_num_per_cycle << "\n"
        << "force_stream_sample: " << force_stream_sample << "\n"
        << "force_score_context_attention: " << force_score_context_attention;
    return oss.str();
}

}

// CacheStoreConfig
void CacheStoreConfig::update_from_env_for_test() {
    cache_store_rdma_mode        = bool_from_env_for_test("CACHE_STORE_RDMA_MODE", false);
    wrr_available_ratio          = autil::EnvUtil::getEnv("WRR_AVAILABLE_RATIO", 80);
    rank_factor                  = autil::EnvUtil::getEnv("RANK_FACTOR", 0);
    thread_count                 = autil::EnvUtil::getEnv("CACHE_STORE_THREAD_COUNT", 16);
    rdma_connect_timeout_ms      = autil::EnvUtil::getEnv("CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS", 250);
    rdma_qp_count_per_connection = autil::EnvUtil::getEnv("CACHE_STORE_RDMA_CONNECTION_COUNT_PER_CONNECTION", 2);
    messager_io_thread_count     = autil::EnvUtil::getEnv("MESSAGER_IO_THREAD_COUNT", 2);
    messager_worker_thread_count = autil::EnvUtil::getEnv("MESSAGER_WORKER_THREAD_COUNT", 16);
}

// CacheStoreConfig
std::string CacheStoreConfig::to_string() const {
    std::ostringstream oss;
    oss << "cache_store_rdma_mode: " << cache_store_rdma_mode << "\n"
        << "wrr_available_ratio: " << wrr_available_ratio << "\n"
        << "rank_factor: " << rank_factor << "\n"
        << "thread_count: " << thread_count << "\n"
        << "rdma_connect_timeout_ms: " << rdma_connect_timeout_ms << "\n"
        << "rdma_qp_count_per_connection: " << rdma_qp_count_per_connection << "\n"
        << "messager_io_thread_count: " << messager_io_thread_count << "\n"
        << "messager_worker_thread_count: " << messager_worker_thread_count << "\n";
    return oss.str();
}

// MiscellaneousConfig
void MiscellaneousConfig::update_from_env_for_test() {
    disable_pdl = bool_from_env_for_test("DISABLE_PDL", true);
    aux_string  = autil::EnvUtil::getEnv("AUX_STRING", "");
}

std::string MiscellaneousConfig::to_string() const {
    std::ostringstream oss;
    oss << "disable_pdl" << disable_pdl << "\n"
        << "aux_string: " << aux_string << "\n";
    return oss.str();
}

// RuntimeConfig
void RuntimeConfig::update_from_env_for_test() {
    max_generate_batch_size = autil::EnvUtil::getEnv("MAX_GENERATE_BATCH_SIZE", 1);
    max_context_batch_size  = autil::EnvUtil::getEnv("MAX_CONTEXT_BATCH_SIZE", 1);
    gen_num_per_circle      = autil::EnvUtil::getEnv("GEN_NUM_PER_CIRCLE", 1);
    pre_allocate_op_mem     = bool_from_env_for_test("PRE_ALLOCATE_OP_MEM", true);
    max_block_size_per_item = autil::EnvUtil::getEnv("MAX_BLOCK_SIZE_PER_ITEM", 16);
    max_batch_tokens_size   = autil::EnvUtil::getEnv("MAX_BATCH_TOKENS_SIZE", 0);
    block_nums               = autil::EnvUtil::getEnv("BLOCK_NUMS", 0);
    scheduler_reserve_resource_ratio = autil::EnvUtil::getEnv("SCHEDULER_RESERVE_RESOURCE_RATIO", 5);
    reserve_runtime_mem_mb           = autil::EnvUtil::getEnv("RESERVE_RUNTIME_MEM_MB", 0);
    kv_cache_mem_mb                  = autil::EnvUtil::getEnv("KV_CACHE_MEM_MB", 0);
    reuse_cache                      = bool_from_env_for_test("REUSE_CACHE", false);
    enable_partial_fallback          = bool_from_env_for_test("ENABLE_PARTIAL_FALLBACK", false);
    enable_fast_gen                  = bool_from_env_for_test("ENABLE_FAST_GEN", false);
    warm_up                          = bool_from_env_for_test("WARM_UP", false);
    warm_up_with_loss                = bool_from_env_for_test("WARM_UP_WITH_LOSS", false);
    fast_gen_max_context_len         = autil::EnvUtil::getEnv("FAST_GEN_MAX_CONTEXT_LEN", 0);
    fast_gen_context_budget          = autil::EnvUtil::getEnv("FAST_GEN_MAX_CONTEXT_LEN", -1);
    vit_separation                    = autil::EnvUtil::getEnv("VIT_SEPARATION", 0);
    enable_speculative_decoding      = bool_from_env_for_test("ENABLE_SPECULATIVE_DECODING", false);
    
    // Scheduler configuration (merged from SchedulerConfig, BatchDecodeSchedulerConfig, FIFOSchedulerConfig)
    use_batch_decode_scheduler = bool_from_env_for_test("USE_BATCH_DECODE_SCHEDULER", false);
    batch_decode_scheduler_batch_size = autil::EnvUtil::getEnv("BATCH_DECODE_SCHEDULER_BATCH_SIZE", 1);
    // model_name, worker_addrs, worker_grpc_addrs are typically set programmatically, not from env
}

std::string RuntimeConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_generate_batch_size: " << max_generate_batch_size << "\n"
        << "max_context_batch_size: " << max_context_batch_size << "\n"
        << "gen_num_per_circle: " << gen_num_per_circle << "\n"
        << "pre_allocate_op_mem: " << pre_allocate_op_mem << "\n"
        << "max_block_size_per_item: " << max_block_size_per_item << "\n"
        << "max_batch_tokens_size: " << max_batch_tokens_size << "\n"
        << "block_nums: " << block_nums << "\n"
        << "scheduler_reserve_resource_ratio: " << scheduler_reserve_resource_ratio << "\n"
        << "reserve_runtime_mem_mb: " << reserve_runtime_mem_mb << "\n"
        << "kv_cache_mem_mb: " << kv_cache_mem_mb << "\n"
        << "reuse_cache: " << reuse_cache << "\n"
        << "enable_partial_fallback: " << enable_partial_fallback << "\n"
        << "enable_fast_gen: " << enable_fast_gen << "\n"
        << "warm_up: " << warm_up << "\n"
        << "warm_up_with_loss: " << warm_up_with_loss << "\n"
        << "fast_gen_max_context_len: " << fast_gen_max_context_len << "\n"
        << "fast_gen_context_budget: " << fast_gen_context_budget << "\n"
        << "use_batch_decode_scheduler: " << use_batch_decode_scheduler << "\n"
        << "use_gather_batch_scheduler: " << use_gather_batch_scheduler << "\n"
        << "batch_decode_scheduler_batch_size: " << batch_decode_scheduler_batch_size << "\n"
        << "batch_decode_scheduler_warmup_type: " << batch_decode_scheduler_warmup_type << "\n"
        << "vit_separation: " << vit_separation << "\n"
        << "enable_speculative_decoding: " << enable_speculative_decoding << "\n"
        << "model_name: " << model_name << "\n"
        << "worker_addrs: [";
    for (size_t i = 0; i < worker_addrs.size(); ++i) {
        oss << worker_addrs[i];
        if (i < worker_addrs.size() - 1) oss << ", ";
    }
    oss << "]\n"
        << "worker_grpc_addrs: [";
    for (size_t i = 0; i < worker_grpc_addrs.size(); ++i) {
        oss << worker_grpc_addrs[i];
        if (i < worker_grpc_addrs.size() - 1) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

// EPLBConfig
void EPLBConfig::update_from_env_for_test() {
    enable_eplb      = bool_from_env_for_test("ENABLE_EPLB", false);
    phy_exp_num      = autil::EnvUtil::getEnv("PHY_EXP_NUM", 0);
    eplb_update_time = autil::EnvUtil::getEnv("EPLB_UPDATE_TIME", 5000);
    redundant_expert = autil::EnvUtil::getEnv("REDUNDANT_EXPERT", 0);
    hack_ep_single_entry = autil::EnvUtil::getEnv("HACK_EP_SINGLE_ENTRY", 0);
    balance_method = autil::EnvUtil::getEnv("BALANCE_METHOD", "mix");
    eplb_force_repack = autil::EnvUtil::getEnv("EPLB_FORCE_REPACK", 0);
    eplb_stats_window_size = autil::EnvUtil::getEnv("EPLB_STATS_WINDOW_SIZE", 10);
    // eplb_mode and py_eplb are typically set programmatically, not from env
}

std::string EPLBConfig::to_string() const {
    std::ostringstream oss;
    oss << "enable_eplb: " << enable_eplb << "\n"
        << "phy_exp_num: " << phy_exp_num << "\n"
        << "eplb_update_time: " << eplb_update_time << "\n"
        << "eplb_mode: " << (int)eplb_mode << "\n"
        << "py_eplb: " << (py_eplb.is_none() ? "None" : "<pybind11::object>") << "\n"
        << "redundant_expert: " << redundant_expert << "\n"
        << "hack_ep_single_entry: " << hack_ep_single_entry << "\n"
        << "balance_method: " << balance_method << "\n"
        << "eplb_force_repack: " << eplb_force_repack << "\n"
        << "eplb_stats_window_size: " << eplb_stats_window_size;
    return oss.str();
}

// ArpcConfig
std::string ArpcConfig::to_string() const {
    std::ostringstream oss;
    oss << "threadNum: " << threadNum << "\n"
        << "queueNum: " << queueNum << "\n"
        << "ioThreadNum: " << ioThreadNum;
    return oss.str();
}

// FfnDisAggregateConfig
void FfnDisAggregateConfig::update_from_env_for_test() {
    enable_ffn_disaggregate = bool_from_env_for_test("ENABLE_FFN_DISAGGREGATE", false);
    attention_tp_size       = autil::EnvUtil::getEnv("ATTENTION_TP_SIZE", 1);
    attention_dp_size       = autil::EnvUtil::getEnv("ATTENTION_DP_SIZE", 1);
    ffn_tp_size             = autil::EnvUtil::getEnv("FFN_TP_SIZE", 1);
    ffn_dp_size             = autil::EnvUtil::getEnv("FFN_DP_SIZE", 1);
    is_ffn_rank             = bool_from_env_for_test("IS_FFN_RANK", false);
}

std::string FfnDisAggregateConfig::to_string() const {
    std::ostringstream oss;
    oss << "enable_ffn_disaggregate: " << enable_ffn_disaggregate << "\n";
    if (enable_ffn_disaggregate) {
        oss << "attention_tp_size: " << attention_tp_size << " attention_dp_size: " << attention_dp_size << "\n"
            << "ffn_tp_size: " << ffn_tp_size << " ffn_dp_size: " << ffn_dp_size << "\n"
            << "is_ffn_rank: " << is_ffn_rank;
    }
    return oss.str();
}

// PDSepConfig
void PDSepConfig::update_from_env_for_test() {
    // role_type is typically set programmatically based on vit_separation, not from env
    cache_store_rdma_mode           = bool_from_env_for_test("CACHE_STORE_RDMA_MODE", true);
    cache_store_listen_port         = autil::EnvUtil::getEnv("CACHE_STORE_LISTEN_PORT", 0);
    cache_store_connect_port        = autil::EnvUtil::getEnv("CACHE_STORE_CONNECT_PORT", 0);
    cache_store_rdma_listen_port    = autil::EnvUtil::getEnv("CACHE_STORE_RDMA_LISTEN_PORT", 0);
    cache_store_rdma_connect_port   = autil::EnvUtil::getEnv("CACHE_STORE_RDMA_CONNECT_PORT", 0);
    remote_rpc_server_port          = autil::EnvUtil::getEnv("REMOTE_RPC_SERVER_PORT", 0);
    prefill_retry_times             = autil::EnvUtil::getEnv("PREFILL_RETRY_TIMES", 0);
    prefill_retry_timeout_ms        = autil::EnvUtil::getEnv("PREFILL_RETRY_TIMEOUT_MS", 20);
    prefill_max_wait_timeout_ms     = autil::EnvUtil::getEnv("PREFILL_MAX_WAIT_TIMEOUT_MS", 600 * 1000);
    decode_retry_times              = autil::EnvUtil::getEnv("DECODE_RETRY_TIMES", 100);
    decode_retry_timeout_ms         = autil::EnvUtil::getEnv("DECODE_RETRY_TIMEOUT_MS", 100);
    decode_polling_kv_cache_step_ms = autil::EnvUtil::getEnv("DECODE_POLLING_KV_CACHE_STEP_MS", 30);
    decode_polling_call_prefill_ms  = autil::EnvUtil::getEnv("DECODE_POLLING_CALL_PREFILL_MS", 30);
    rdma_connect_retry_times        = autil::EnvUtil::getEnv("RDMA_CONNECT_RETRY_TIMES", 0);
    load_cache_timeout_ms           = autil::EnvUtil::getEnv("LOAD_CACHE_TIMEOUT_MS", 5000);
    max_rpc_timeout_ms              = autil::EnvUtil::getEnv("MAX_RPC_TIMEOUT_MS", 0);
    worker_port_offset              = autil::EnvUtil::getEnv("WORKER_PORT_OFFSET", 0);
    decode_entrance                 = bool_from_env_for_test("DECODE_ENTRANCE", false);
}

std::string PDSepConfig::to_string() const {
    std::ostringstream oss;
    oss << "role_type: " << (int)role_type << "\n"
        << "cache_store_rdma_mode: " << cache_store_rdma_mode << "\n"
        << "cache_store_listen_port: " << cache_store_listen_port << "\n"
        << "cache_store_connect_port: " << cache_store_connect_port << "\n"
        << "cache_store_rdma_listen_port: " << cache_store_rdma_listen_port << "\n"
        << "cache_store_rdma_connect_port: " << cache_store_rdma_connect_port << "\n"
        << "remote_rpc_server_port: " << remote_rpc_server_port << "\n"
        << "prefill_retry_times: " << prefill_retry_times << "\n"
        << "prefill_retry_timeout_ms: " << prefill_retry_timeout_ms << "\n"
        << "prefill_max_wait_timeout_ms: " << prefill_max_wait_timeout_ms << "\n"
        << "decode_retry_times: " << decode_retry_times << "\n"
        << "decode_retry_timeout_ms: " << decode_retry_timeout_ms << "\n"
        << "decode_polling_kv_cache_step_ms: " << decode_polling_kv_cache_step_ms << "\n"
        << "decode_polling_call_prefill_ms: " << decode_polling_call_prefill_ms << "\n"
        << "rdma_connect_retry_times: " << rdma_connect_retry_times << "\n"
        << "load_cache_timeout_ms: " << load_cache_timeout_ms << "\n"
        << "max_rpc_timeout_ms: " << max_rpc_timeout_ms << "\n"
        << "worker_port_offset: " << worker_port_offset << "\n"
        << "decode_entrance: " << decode_entrance;
    return oss.str();
}

}  // namespace rtp_llm
