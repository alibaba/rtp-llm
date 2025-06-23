#include "rtp_llm/cpp/th_op/ConfigModules.h"
#include "pybind11/cast.h"
#include "autil/EnvUtil.h"
#include <sstream>
#include <algorithm>
#include <string>
#include <cctype>

namespace rtp_llm {

int StaticConfig::user_deep_gemm_num_sm = -1;
bool StaticConfig::user_arm_gemm_use_kai = false;
bool StaticConfig::user_ft_core_dump_on_exception = false;

std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

bool bool_from_env_for_test(std::string env_name, bool default_value){
    const char* val = getenv(env_name.c_str());
    if(!val) {
        return default_value;
    }
    std::string lower = to_lower(val);
    return lower == "1" || lower == "on" || lower == "true";
}

// ParallelismDistributedConfig
void ParallelismDistributedConfig::update_from_env_for_test(){
    tp_size = autil::EnvUtil::getEnv("TP_SIZE", 1);
    ep_size = autil::EnvUtil::getEnv("EP_SIZE", 1);
    dp_size = autil::EnvUtil::getEnv("DP_SIZE", 1);
    pp_size = autil::EnvUtil::getEnv("PP_SIZE", 1);
    world_size = autil::EnvUtil::getEnv("WORLD_SIZE", 1);
    world_rank = autil::EnvUtil::getEnv("WORLD_RANK", 0);
    local_world_size = autil::EnvUtil::getEnv("LOCAL_WORLD_SIZE", 1);
}

void register_parallelism_distributed_config(pybind11::module& m) {
    pybind11::class_<ParallelismDistributedConfig>(m, "ParallelismDistributedConfig")
        .def(pybind11::init<
            int, int, int, int, int, int, int
        >(),
        pybind11::arg("tp_size") = 1,
        pybind11::arg("ep_size") = 1,
        pybind11::arg("dp_size") = 1,
        pybind11::arg("pp_size") = 1,
        pybind11::arg("world_size") = 1,
        pybind11::arg("world_rank") = 0,
        pybind11::arg("local_world_size") = 1
        )
        .def("to_string", &ParallelismDistributedConfig::to_string)
        .def("update_from_env_for_test", &ParallelismDistributedConfig::update_from_env_for_test)
        .def_readwrite("tp_size", &ParallelismDistributedConfig::tp_size)
        .def_readwrite("ep_size", &ParallelismDistributedConfig::ep_size)
        .def_readwrite("dp_size", &ParallelismDistributedConfig::dp_size)
        .def_readwrite("pp_size", &ParallelismDistributedConfig::pp_size)
        .def_readwrite("world_size", &ParallelismDistributedConfig::world_size)
        .def_readwrite("world_rank", &ParallelismDistributedConfig::world_rank)
        .def_readwrite("local_world_size", &ParallelismDistributedConfig::local_world_size);
}

void register_arpc_config(pybind11::module& m) {
    pybind11::class_<ArpcConfig>(m, "ArpcConfig")
        .def(pybind11::init<
            int, int, int
        >(),
        pybind11::arg("threadNum") = 10,
        pybind11::arg("queueNum") = 50,
        pybind11::arg("ioThreadNum") = 2
        )
        .def("to_string", &ArpcConfig::to_string)
        .def_readwrite("threadNum", &ArpcConfig::threadNum)
        .def_readwrite("queueNum", &ArpcConfig::queueNum)
        .def_readwrite("ioThreadNum", &ArpcConfig::ioThreadNum);
}


// ConcurrencyConfig
void ConcurrencyConfig::update_from_env_for_test() {
    concurrency_with_block = bool_from_env_for_test("CONCURRENCY_WITH_BLOCK", false);
    concurrency_limit = autil::EnvUtil::getEnv("CONCURRENCY_LIMIT", 32);
}

void register_concurrency_config(pybind11::module& m) {
    pybind11::class_<ConcurrencyConfig>(m, "ConcurrencyConfig")
        .def(pybind11::init<
            bool, int
        >(),
        pybind11::arg("concurrency_with_block") = false,
        pybind11::arg("concurrency_limit") = 32
        )
        .def("to_string", &ConcurrencyConfig::to_string)
        .def("update_from_env_for_test", &ConcurrencyConfig::update_from_env_for_test)
        .def_readwrite("concurrency_with_block", &ConcurrencyConfig::concurrency_with_block)
        .def_readwrite("concurrency_limit", &ConcurrencyConfig::concurrency_limit);
}

// FMHAConfig
void FMHAConfig::update_from_env_for_test(){
    enable_fmha = bool_from_env_for_test("ENABLE_FMHA", true);
    enable_trt_fmha = bool_from_env_for_test("ENABLE_TRT_FMHA", true);
    enable_paged_trt_fmha = bool_from_env_for_test("ENABLE_PAGED_TRT_FMHA", true);
    enable_open_source_fmha = bool_from_env_for_test("ENABLE_OPENSOURCE_FMHA", true);
    enable_paged_open_source_fmha = bool_from_env_for_test("ENABLE_PAGED_OPEN_SOURCE_FMHA", true);
    enable_trtv1_fmha = bool_from_env_for_test("ENABLE_TRTV1_FMHA", true);
    fmha_perf_instrument = bool_from_env_for_test("FMHA_PERF_INSTRUMENT", false);
    fmha_show_params = bool_from_env_for_test("FMHA_SHOW_PARAMS", false);
    disable_flash_infer = bool_from_env_for_test("DISABLE_FLASH_INFER", false);
    enable_xqa = bool_from_env_for_test("ENABLE_XQA", true);
}

void register_fmha_config(pybind11::module& m) {
    pybind11::class_<FMHAConfig>(m, "FMHAConfig")
        .def(pybind11::init<
            bool, bool, bool, bool, bool, bool,
            bool, bool, bool, bool
        >(),
        pybind11::arg("enable_fmha") = true,
        pybind11::arg("enable_trt_fmha") = true,
        pybind11::arg("enable_paged_trt_fmha") = true,
        pybind11::arg("enable_open_source_fmha") = true,
        pybind11::arg("enable_paged_open_source_fmha") = true,
        pybind11::arg("enable_trtv1_fmha") = true,
        pybind11::arg("fmha_perf_instrument") = false,
        pybind11::arg("fmha_show_params") = false,
        pybind11::arg("disable_flash_infer") = false,
        pybind11::arg("enable_xqa") = true
        )
        .def("to_string", &FMHAConfig::to_string)
        .def("update_from_env_for_test", &FMHAConfig::update_from_env_for_test)
        .def_readwrite("enable_fmha", &FMHAConfig::enable_fmha)
        .def_readwrite("enable_trt_fmha", &FMHAConfig::enable_trt_fmha)
        .def_readwrite("enable_paged_trt_fmha", &FMHAConfig::enable_paged_trt_fmha)
        .def_readwrite("enable_open_source_fmha", &FMHAConfig::enable_open_source_fmha)
        .def_readwrite("enable_paged_open_source_fmha", &FMHAConfig::enable_paged_open_source_fmha)
        .def_readwrite("enable_trtv1_fmha", &FMHAConfig::enable_trtv1_fmha)
        .def_readwrite("fmha_perf_instrument", &FMHAConfig::fmha_perf_instrument)
        .def_readwrite("fmha_show_params", &FMHAConfig::fmha_show_params)
        .def_readwrite("disable_flash_infer", &FMHAConfig::disable_flash_infer)
        .def_readwrite("enable_xqa", &FMHAConfig::enable_xqa);
}

// KVCacheConfig
void KVCacheConfig::update_from_env_for_test(){
    reuse_cache = bool_from_env_for_test("REUSE_CACHE", false);
    multi_task_prompt = autil::EnvUtil::getEnv("MULTI_TASK_PROMPT", "");
    multi_task_prompt_str = autil::EnvUtil::getEnv("MULTI_TASK_PROMPT_STR", "");
}

void register_kvcache_config(pybind11::module& m) {
    pybind11::class_<KVCacheConfig>(m, "KVCacheConfig")
        .def(pybind11::init<
            bool, std::string, std::string
        >(),
        pybind11::arg("reuse_cache") = false,
        pybind11::arg("multi_task_prompt") = "",
        pybind11::arg("multi_task_prompt_str") = ""
        )
        .def("to_string", &KVCacheConfig::to_string)
        .def("update_from_env_for_test", &KVCacheConfig::update_from_env_for_test)
        .def_readwrite("reuse_cache", &KVCacheConfig::reuse_cache)
        .def_readwrite("multi_task_prompt", &KVCacheConfig::multi_task_prompt)
        .def_readwrite("multi_task_prompt_str", &KVCacheConfig::multi_task_prompt_str);
}

// ProfilingDebugLoggingConfig
void ProfilingDebugLoggingConfig::update_from_env_for_test(){
    ft_nvtx = bool_from_env_for_test("FT_NVTX", false);
    py_inference_log_response = bool_from_env_for_test("PY_INFERENCE_LOG_RESPONSE", false);
    rtp_llm_trace_memory = bool_from_env_for_test("RTP_LLM_TRACE_MEMORY", false);
    rtp_llm_trace_malloc_stack = bool_from_env_for_test("RTP_LLM_TRACE_MALLOC_STACK", false);
    enable_device_perf = bool_from_env_for_test("ENABLE_DEVICE_PERF", false);
    ft_core_dump_on_exception = bool_from_env_for_test("FT_CORE_DUMP_ON_EXCEPTION", false);
    ft_alog_conf_path = autil::EnvUtil::getEnv("FT_ALOG_CONF_PATH", "");
    log_level = autil::EnvUtil::getEnv("LOG_LEVEL", "INFO");
    gen_timeline_sync = bool_from_env_for_test("GEN_TIMELINE_SYNC", false);
}

void register_profiling_debug_logging_config(pybind11::module& m) {
    pybind11::class_<ProfilingDebugLoggingConfig>(m, "ProfilingDebugLoggingConfig")
        .def(pybind11::init<
            bool, bool, bool, bool, bool, bool, std::string, std::string, bool
        >(),
        pybind11::arg("ft_nvtx") = false,
        pybind11::arg("py_inference_log_response") = false,
        pybind11::arg("rtp_llm_trace_memory") = false,
        pybind11::arg("rtp_llm_trace_malloc_stack") = false,
        pybind11::arg("enable_device_perf") = false,
        pybind11::arg("ft_core_dump_on_exception") = false,
        pybind11::arg("ft_alog_conf_path") = "",
        pybind11::arg("log_level") = "INFO",
        pybind11::arg("gen_timeline_sync") = false
        )
        .def("to_string", &ProfilingDebugLoggingConfig::to_string)
        .def("update_from_env_for_test", &ProfilingDebugLoggingConfig::update_from_env_for_test)
        .def_readwrite("ft_nvtx", &ProfilingDebugLoggingConfig::ft_nvtx)
        .def_readwrite("py_inference_log_response", &ProfilingDebugLoggingConfig::py_inference_log_response)
        .def_readwrite("rtp_llm_trace_memory", &ProfilingDebugLoggingConfig::rtp_llm_trace_memory)
        .def_readwrite("rtp_llm_trace_malloc_stack", &ProfilingDebugLoggingConfig::rtp_llm_trace_malloc_stack)
        .def_readwrite("enable_device_perf", &ProfilingDebugLoggingConfig::enable_device_perf)
        .def_readwrite("ft_core_dump_on_exception", &ProfilingDebugLoggingConfig::ft_core_dump_on_exception)
        .def_readwrite("ft_alog_conf_path", &ProfilingDebugLoggingConfig::ft_alog_conf_path)
        .def_readwrite("log_level", &ProfilingDebugLoggingConfig::log_level)
        .def_readwrite("gen_timeline_sync", &ProfilingDebugLoggingConfig::gen_timeline_sync);
}

// HWKernelConfig
void HWKernelConfig::update_from_env_for_test(){
    deep_gemm_num_sm = autil::EnvUtil::getEnv("DEEP_GEMM_NUM_SM", -1);
    arm_gemm_use_kai = bool_from_env_for_test("ARM_GEMM_USE_KAI", false);
    enable_stable_scatter_add = bool_from_env_for_test("ENABLE_STABLE_SCATTER_ADD", false);
    enable_multi_block_mode = bool_from_env_for_test("ENABLE_MULTI_BLOCK_MODE", true);
    ft_disable_custom_ar = bool_from_env_for_test("FT_DISABLE_CUSTOM_AR", true);
    rocm_hipblaslt_config = autil::EnvUtil::getEnv("ROCM_HIPBLASLT_CONFIG", "gemm_config.csv");
}

void register_hwkernel_config(pybind11::module& m) {
    pybind11::class_<HWKernelConfig>(m, "HWKernelConfig")
        .def(pybind11::init<
            int, bool, bool, bool, bool, std::string
        >(),
        pybind11::arg("deep_gemm_num_sm") = -1,
        pybind11::arg("arm_gemm_use_kai") = false,
        pybind11::arg("enable_stable_scatter_add") = false,
        pybind11::arg("enable_multi_block_mode") = true,
        pybind11::arg("ft_disable_custom_ar") = true,
        pybind11::arg("rocm_hipblaslt_config") = "gemm_config.csv"
        )
        .def("to_string", &HWKernelConfig::to_string)
        .def("update_from_env_for_test", &HWKernelConfig::update_from_env_for_test)
        .def_readwrite("deep_gemm_num_sm", &HWKernelConfig::deep_gemm_num_sm)
        .def_readwrite("arm_gemm_use_kai", &HWKernelConfig::arm_gemm_use_kai)
        .def_readwrite("enable_stable_scatter_add", &HWKernelConfig::enable_stable_scatter_add)
        .def_readwrite("enable_multi_block_mode", &HWKernelConfig::enable_multi_block_mode)
        .def_readwrite("ft_disable_custom_ar", &HWKernelConfig::ft_disable_custom_ar)
        .def_readwrite("rocm_hipblaslt_config", &HWKernelConfig::rocm_hipblaslt_config);
}

// DeviceResourceConfig
void DeviceResourceConfig::update_from_env_for_test(){
    device_reserve_memory_bytes = autil::EnvUtil::getEnv("DEVICE_RESERVE_MEMORY_BYTES", 0);
    host_reserve_memory_bytes = autil::EnvUtil::getEnv("HOST_RESERVE_MEMORY_BYTES", 4LL * 1024 * 1024 * 1024);
    overlap_math_sm_count = autil::EnvUtil::getEnv("OVERLAP_MATH_SM_COUNT", 0);
    overlap_comm_type = autil::EnvUtil::getEnv("OVERLAP_COMM_TYPE", 0);
    m_split = autil::EnvUtil::getEnv("M_SPLIT", 0);
    enable_comm_overlap = bool_from_env_for_test("ENABLE_COMM_OVERLAP", true);
    enable_layer_micro_batch = autil::EnvUtil::getEnv("ENABLE_LAYER_MICRO_BATCH", 0);
    not_use_default_stream = bool_from_env_for_test("NOT_USE_DEFAULT_STREAM", false);
}

void register_device_resource_config(pybind11::module& m) {
    pybind11::class_<DeviceResourceConfig>(m, "DeviceResourceConfig")
        .def(pybind11::init<
            int64_t, int64_t, int, int, int, bool, int, bool
        >(),
        pybind11::arg("device_reserve_memory_bytes") = 0,
        pybind11::arg("host_reserve_memory_bytes") = 4LL * 1024 * 1024 * 1024,
        pybind11::arg("overlap_math_sm_count") = 0,
        pybind11::arg("overlap_comm_type") = 0,
        pybind11::arg("m_split") = 0,
        pybind11::arg("enable_comm_overlap") = true,
        pybind11::arg("enable_layer_micro_batch") = 0,
        pybind11::arg("not_use_default_stream") = false
        )
        .def("to_string", &DeviceResourceConfig::to_string)
        .def("update_from_env_for_test", &DeviceResourceConfig::update_from_env_for_test)
        .def_readwrite("device_reserve_memory_bytes", &DeviceResourceConfig::device_reserve_memory_bytes)
        .def_readwrite("host_reserve_memory_bytes", &DeviceResourceConfig::host_reserve_memory_bytes)
        .def_readwrite("overlap_math_sm_count", &DeviceResourceConfig::overlap_math_sm_count)
        .def_readwrite("overlap_comm_type", &DeviceResourceConfig::overlap_comm_type)
        .def_readwrite("m_split", &DeviceResourceConfig::m_split)
        .def_readwrite("enable_comm_overlap", &DeviceResourceConfig::enable_comm_overlap)
        .def_readwrite("enable_layer_micro_batch", &DeviceResourceConfig::enable_layer_micro_batch)
        .def_readwrite("not_use_default_stream", &DeviceResourceConfig::not_use_default_stream);
}

// SamplerConfig
void SamplerConfig::update_from_env_for_test(){
    max_batch_size = autil::EnvUtil::getEnv("MAX_BATCH_SIZE", 0);
    enable_flashinfer_sample_kernel = bool_from_env_for_test("ENABLE_FLASHINFER_SAMPLE_KERNEL", true);
}

void register_sampler_config(pybind11::module& m) {
    pybind11::class_<SamplerConfig>(m, "SamplerConfig")
        .def(pybind11::init<
            int64_t, bool
        >(),
        pybind11::arg("max_batch_size") = 0,
        pybind11::arg("enable_flashinfer_sample_kernel") = true
        )
        .def("to_string", &SamplerConfig::to_string)
        .def("update_from_env_for_test", &SamplerConfig::update_from_env_for_test)
        .def_readwrite("max_batch_size", &SamplerConfig::max_batch_size)
        .def_readwrite("enable_flashinfer_sample_kernel", &SamplerConfig::enable_flashinfer_sample_kernel);
}

// MoeConfig
void MoeConfig::update_from_env_for_test() {
    use_deepep_moe = bool_from_env_for_test("USE_DEEPEP_MOE", false);
    use_deepep_internode = bool_from_env_for_test("USE_DEEPEP_INTERNODE", false);
    use_deepep_low_latency = bool_from_env_for_test("USE_DEEPEP_LOW_LATENCY", true);
    fake_balance_expert = bool_from_env_for_test("FAKE_BALANCE_EXPERT", false);
    eplb_control_step = autil::EnvUtil::getEnv("EPLB_CONTROL_STEP", 100);
    eplb_test_mode = bool_from_env_for_test("EPLB_TEST_MODE", false);
    hack_moe_expert = bool_from_env_for_test("HACK_MOE_EXPERT", false);
    eplb_balance_layer_per_step = autil::EnvUtil::getEnv("EPLB_BALANCE_LAYER_PER_STEP", 1);
    deep_ep_num_sm = autil::EnvUtil::getEnv("DEEP_EP_NUM_SM", 0);
}

void register_moe_config(pybind11::module& m) {
    pybind11::class_<MoeConfig>(m, "MoeConfig")
        .def(pybind11::init<
            bool, bool, bool, bool, bool, int, bool, bool, int, int
        >(),
        pybind11::arg("use_deepep_moe") = false,
        pybind11::arg("use_deepep_internode") = false,
        pybind11::arg("use_deepep_low_latency") = true,
        pybind11::arg("use_deepep_p2p_low_latency") = false,
        pybind11::arg("fake_balance_expert") = false,
        pybind11::arg("eplb_control_step") = 100,
        pybind11::arg("eplb_test_mode") = false,
        pybind11::arg("hack_moe_expert") = false,
        pybind11::arg("eplb_balance_layer_per_step") = 1,
        pybind11::arg("deep_ep_num_sm") = 0
        )
        .def("to_string", &MoeConfig::to_string)
        .def("update_from_env_for_test", &MoeConfig::update_from_env_for_test)
        .def_readwrite("use_deepep_moe", &MoeConfig::use_deepep_moe)
        .def_readwrite("use_deepep_internode", &MoeConfig::use_deepep_internode)
        .def_readwrite("use_deepep_low_latency", &MoeConfig::use_deepep_low_latency)
        .def_readwrite("use_deepep_p2p_low_latency", &MoeConfig::use_deepep_p2p_low_latency)
        .def_readwrite("fake_balance_expert", &MoeConfig::fake_balance_expert)
        .def_readwrite("eplb_control_step", &MoeConfig::eplb_control_step)
        .def_readwrite("eplb_test_mode", &MoeConfig::eplb_test_mode)
        .def_readwrite("hack_moe_expert", &MoeConfig::hack_moe_expert)
        .def_readwrite("eplb_balance_layer_per_step", &MoeConfig::eplb_balance_layer_per_step)
        .def_readwrite("deep_ep_num_sm", &MoeConfig::deep_ep_num_sm);
}

// ModelSpecificConfig
void ModelSpecificConfig::update_from_env_for_test(){
    max_lora_model_size = autil::EnvUtil::getEnv("MAX_LORA_MODEL_SIZE", -1);
}

void register_model_specific_config(pybind11::module& m) {
    pybind11::class_<ModelSpecificConfig>(m, "ModelSpecificConfig")
        .def(pybind11::init<
            int64_t
        >(),
        pybind11::arg("max_lora_model_size") = -1
        )
        .def("to_string", &ModelSpecificConfig::to_string)
        .def("update_from_env_for_test", &ModelSpecificConfig::update_from_env_for_test)
        .def_readwrite("max_lora_model_size", &ModelSpecificConfig::max_lora_model_size);
}

// SpeculativeExecutionConfig
void SpeculativeExecutionConfig::update_from_env_for_test(){
    sp_model_type = autil::EnvUtil::getEnv("SP_MODEL_TYPE", "");
    sp_type = autil::EnvUtil::getEnv("SP_TYPE", "");
    sp_min_token_match = autil::EnvUtil::getEnv("SP_MIN_TOKEN_MATCH", 2);
    sp_max_token_match = autil::EnvUtil::getEnv("SP_MAX_TOKEN_MATCH", 2);
    tree_decode_config = autil::EnvUtil::getEnv("TREE_DECODE_CONFIG", "");
    gen_num_per_cycle = autil::EnvUtil::getEnv("GEN_NUM_PER_CIRCLE", 1);
    force_stream_sample = autil::EnvUtil::getEnv("FORCE_STREAM_SAMPLE", false);
    force_score_context_attention = autil::EnvUtil::getEnv("FORCE_SCORE_CONTEXT_ATTENTION", true);
}

void register_speculative_execution_config(pybind11::module& m) {
    pybind11::class_<SpeculativeExecutionConfig>(m, "SpeculativeExecutionConfig")
        .def(pybind11::init<
            std::string, std::string, int64_t, int64_t, std::string, int, bool, bool
        >(),
        pybind11::arg("sp_model_type") = "",
        pybind11::arg("sp_type") = "",
        pybind11::arg("sp_min_token_match") = 2,
        pybind11::arg("sp_max_token_match") = 2,
        pybind11::arg("tree_decode_config") = "",
        pybind11::arg("gen_num_per_cycle") = 1,
        pybind11::arg("force_stream_sample") = false,
        pybind11::arg("force_score_context_attention") = true
        )
        .def("to_string", &SpeculativeExecutionConfig::to_string)
        .def("update_from_env_for_test", &SpeculativeExecutionConfig::update_from_env_for_test)
        .def_readwrite("sp_model_type", &SpeculativeExecutionConfig::sp_model_type)
        .def_readwrite("sp_type", &SpeculativeExecutionConfig::sp_type)
        .def_readwrite("sp_min_token_match", &SpeculativeExecutionConfig::sp_min_token_match)
        .def_readwrite("sp_max_token_match", &SpeculativeExecutionConfig::sp_max_token_match)
        .def_readwrite("tree_decode_config", &SpeculativeExecutionConfig::tree_decode_config)
        .def_readwrite("gen_num_per_cycle", &SpeculativeExecutionConfig::gen_num_per_cycle)
        .def_readwrite("force_stream_sample", &SpeculativeExecutionConfig::force_stream_sample)
        .def_readwrite("force_score_context_attention", &SpeculativeExecutionConfig::force_score_context_attention);
}

// ServiceDiscoveryConfig
void ServiceDiscoveryConfig::update_from_env_for_test(){
    use_local = bool_from_env_for_test("USE_LOCAL", false);
    remote_rpc_server_ip = autil::EnvUtil::getEnv("REMOTE_RPC_SERVER_IP", "");
    rtp_llm_decode_cm2_config = autil::EnvUtil::getEnv("RTP_LLM_DECODE_CM2_CONFIG", "");
    remote_vit_server_ip = autil::EnvUtil::getEnv("REMOTE_VIT_SERVER_IP", "");
    rtp_llm_multimodal_part_cm2_config = autil::EnvUtil::getEnv("RTP_LLM_MULTIMODAL_PART_CM2_CONFIG", "");
}

void register_service_discovery_config(pybind11::module& m) {
    pybind11::class_<ServiceDiscoveryConfig>(m, "ServiceDiscoveryConfig")
        .def(pybind11::init<
            bool, std::string, std::string, std::string, std::string
        >(),
        pybind11::arg("use_local") = false,
        pybind11::arg("remote_rpc_server_ip") = "",
        pybind11::arg("rtp_llm_decode_cm2_config") = "",
        pybind11::arg("remote_vit_server_ip") = "",
        pybind11::arg("rtp_llm_multimodal_part_cm2_config") = ""
        )
        .def("to_string", &ServiceDiscoveryConfig::to_string)
        .def("update_from_env_for_test", &ServiceDiscoveryConfig::update_from_env_for_test)
        .def_readwrite("use_local", &ServiceDiscoveryConfig::use_local)
        .def_readwrite("remote_rpc_server_ip", &ServiceDiscoveryConfig::remote_rpc_server_ip)
        .def_readwrite("rtp_llm_decode_cm2_config", &ServiceDiscoveryConfig::rtp_llm_decode_cm2_config)
        .def_readwrite("remote_vit_server_ip", &ServiceDiscoveryConfig::remote_vit_server_ip)
        .def_readwrite("rtp_llm_multimodal_part_cm2_config", &ServiceDiscoveryConfig::rtp_llm_multimodal_part_cm2_config);
}

// CacheStoreConfig
void CacheStoreConfig::update_from_env_for_test(){
    cache_store_rdma_mode = bool_from_env_for_test("CACHE_STORE_RDMA_MODE", false);
    wrr_available_ratio = autil::EnvUtil::getEnv("WRR_AVAILABLE_RATIO", 80);
    rank_factor = autil::EnvUtil::getEnv("RANK_FACTOR", 0);
}

void register_cache_store_config(pybind11::module& m) {
    pybind11::class_<CacheStoreConfig>(m, "CacheStoreConfig")
        .def(pybind11::init<
            bool, int, int
        >(),
        pybind11::arg("cache_store_rdma_mode") = false,
        pybind11::arg("wrr_available_ratio") = 80,
        pybind11::arg("rank_factor") = 0
        )
        .def("to_string", &CacheStoreConfig::to_string)
        .def("update_from_env_for_test", &CacheStoreConfig::update_from_env_for_test)
        .def_readwrite("cache_store_rdma_mode", &CacheStoreConfig::cache_store_rdma_mode)
        .def_readwrite("wrr_available_ratio", &CacheStoreConfig::wrr_available_ratio)
        .def_readwrite("rank_factor", &CacheStoreConfig::rank_factor);
}

// SchedulerConfig
void SchedulerConfig::update_from_env_for_test(){
    use_batch_decode_scheduler = bool_from_env_for_test("USE_BATCH_DECODE_SCHEDULER", false);
}

void register_scheduler_config(pybind11::module& m) {
    pybind11::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(pybind11::init<
            bool
        >(),
        pybind11::arg("use_batch_decode_scheduler") = false
        )
        .def("to_string", &SchedulerConfig::to_string)
        .def("update_from_env_for_test", &SchedulerConfig::update_from_env_for_test)
        .def_readwrite("use_batch_decode_scheduler", &SchedulerConfig::use_batch_decode_scheduler);
}

// BatchDecodeSchedulerConfig
void BatchDecodeSchedulerConfig::update_from_env_for_test(){
    batch_decode_scheduler_batch_size = autil::EnvUtil::getEnv("BATCH_DECODE_SCHEDULER_BATCH_SIZE", 1);
}

void register_batch_decode_scheduler_config(pybind11::module& m) {
    pybind11::class_<BatchDecodeSchedulerConfig>(m, "BatchDecodeSchedulerConfig")
        .def(pybind11::init<
            int64_t
        >(),
        pybind11::arg("batch_decode_scheduler_batch_size") = 1
        )
        .def("to_string", &BatchDecodeSchedulerConfig::to_string)
        .def("update_from_env_for_test", &BatchDecodeSchedulerConfig::update_from_env_for_test)
        .def_readwrite("batch_decode_scheduler_batch_size", &BatchDecodeSchedulerConfig::batch_decode_scheduler_batch_size);
}

// FIFOSchedulerConfig
void FIFOSchedulerConfig::update_from_env_for_test(){
    max_context_batch_size = autil::EnvUtil::getEnv("MAX_CONTEXT_BATCH_SIZE", 1);
    scheduler_reserve_resource_ratio = autil::EnvUtil::getEnv("SCHEDULER_RESERVE_RESOURCE_RATIO", 5);
    enable_fast_gen = bool_from_env_for_test("ENABLE_FAST_GEN", false);
    enable_partial_fallback = bool_from_env_for_test("ENABLE_PARTIAL_FALLBACK", false);
    fast_gen_context_budget = autil::EnvUtil::getEnv("FAST_GEN_MAX_CONTEXT_LEN", -1);
}

void register_fifo_scheduler_config(pybind11::module& m) {
    pybind11::class_<FIFOSchedulerConfig>(m, "FIFOSchedulerConfig")
        .def(pybind11::init<
            int64_t, int, bool, bool, int64_t
        >(),
        pybind11::arg("max_context_batch_size") = 1,
        pybind11::arg("scheduler_reserve_resource_ratio") = 5,
        pybind11::arg("enable_fast_gen") = false,
        pybind11::arg("enable_partial_fallback") = false,
        pybind11::arg("fast_gen_context_budget") = -1
        )
        .def("to_string", &FIFOSchedulerConfig::to_string)
        .def("update_from_env_for_test", &FIFOSchedulerConfig::update_from_env_for_test)
        .def_readwrite("max_context_batch_size", &FIFOSchedulerConfig::max_context_batch_size)
        .def_readwrite("scheduler_reserve_resource_ratio", &FIFOSchedulerConfig::scheduler_reserve_resource_ratio)
        .def_readwrite("enable_fast_gen", &FIFOSchedulerConfig::enable_fast_gen)
        .def_readwrite("enable_partial_fallback", &FIFOSchedulerConfig::enable_partial_fallback)
        .def_readwrite("fast_gen_context_budget", &FIFOSchedulerConfig::fast_gen_context_budget);
}

// MiscellaneousConfig
void MiscellaneousConfig::update_from_env_for_test(){
    load_balance = autil::EnvUtil::getEnv("LOAD_BALANCE", 0);
    step_records_time_range = autil::EnvUtil::getEnv("STEP_RECORDS_TIME_RANGE", 60 * 1000 * 1000);
    step_records_max_size = autil::EnvUtil::getEnv("STEP_RECORDS_MAX_SIZE", 1000);
}

void register_misc_config(pybind11::module& m) {
    pybind11::class_<MiscellaneousConfig>(m, "MiscellaneousConfig")
        .def(pybind11::init<
            int, int, int
        >(),
        pybind11::arg("load_balance") = 0,
        pybind11::arg("step_records_time_range") = 60 * 1000 * 1000,
        pybind11::arg("step_records_max_size") = 1000
        )
        .def("to_string", &MiscellaneousConfig::to_string)
        .def("update_from_env_for_test", &MiscellaneousConfig::update_from_env_for_test)
        .def_readwrite("load_balance", &MiscellaneousConfig::load_balance)
        .def_readwrite("step_records_time_range", &MiscellaneousConfig::step_records_time_range)
        .def_readwrite("step_records_max_size", &MiscellaneousConfig::step_records_max_size);
}

// ParallelismDistributedConfig
inline std::string ParallelismDistributedConfig::to_string() const {
    std::ostringstream oss;
    oss << "tp_size: " << tp_size << "\n"
        << "ep_size: " << ep_size << "\n"
        << "dp_size: " << dp_size << "\n"
        << "world_size: " << world_size << "\n"
        << "world_rank: " << world_rank << "\n"
        << "pp_size: " << pp_size << "\n"
        << "local_world_size: " << local_world_size;
    return oss.str();
}

// ConcurrencyConfig
inline std::string ConcurrencyConfig::to_string() const {
    std::ostringstream oss;
    oss << "concurrency_with_block: " << concurrency_with_block << "\n"
        << "concurrency_limit: " << concurrency_limit;
    return oss.str();
}

// FMHAConfig
inline std::string FMHAConfig::to_string() const {
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
        << "enable_xqa: "<<enable_xqa <<"\n";
    return oss.str();
}

// KVCacheConfig
inline std::string KVCacheConfig::to_string() const {
    std::ostringstream oss;
    oss << "reuse_cache: " << reuse_cache << "\n"
        << "multi_task_prompt: " << multi_task_prompt << "\n"
        << "multi_task_prompt_str: " << multi_task_prompt_str << "\n";
    return oss.str();
}

// ProfilingDebugLoggingConfig
inline std::string ProfilingDebugLoggingConfig::to_string() const {
    std::ostringstream oss;
    oss << "ft_nvtx: " << ft_nvtx << "\n"
        << "py_inference_log_response: " << py_inference_log_response << "\n"
        << "rtp_llm_trace_memory: " << rtp_llm_trace_memory << "\n"
        << "rtp_llm_trace_malloc_stack: " << rtp_llm_trace_malloc_stack << "\n"
        << "enable_device_perf: " << enable_device_perf << "\n"
        << "ft_core_dump_on_exception: " << ft_core_dump_on_exception << "\n"
        << "ft_alog_conf_path: " << ft_alog_conf_path << "\n"
        << "log_level: " << log_level << "\n"
        << "gen_timeline_sync: "<< gen_timeline_sync;
    return oss.str();
}

// HWKernelConfig
inline std::string HWKernelConfig::to_string() const {
    std::ostringstream oss;
    oss << "deep_gemm_num_sm: " << deep_gemm_num_sm << "\n"
        << "arm_gemm_use_kai: " << arm_gemm_use_kai << "\n"
        << "enable_stable_scatter_add: " << enable_stable_scatter_add << "\n"
        << "enable_multi_block_mode: " << enable_multi_block_mode << "\n"
        << "ft_disable_custom_ar: " << ft_disable_custom_ar << "\n"
        << "rocm_hipblaslt_config: " << rocm_hipblaslt_config;
    return oss.str();
}

// DeviceResourceConfig
inline std::string DeviceResourceConfig::to_string() const {
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

inline std::string SamplerConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_batch_size: " << max_batch_size << "\n"
        << "enable_flashinfer_sample_kernel: " << enable_flashinfer_sample_kernel;
    return oss.str();
}

// MoeConfig
inline std::string MoeConfig::to_string() const {
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
        << "deep_ep_num_sm: " << deep_ep_num_sm;
    return oss.str();
}

// ModelSpecificConfig
inline std::string ModelSpecificConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_lora_model_size: " << max_lora_model_size << "\n";
    return oss.str();
}

// SpeculativeExecutionConfig
inline std::string SpeculativeExecutionConfig::to_string() const {
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

// ServiceDiscoveryConfig
inline std::string ServiceDiscoveryConfig::to_string() const {
    std::ostringstream oss;
    oss << "use_local: " << use_local << "\n"
        << "remote_rpc_server_ip: " << remote_rpc_server_ip << "\n"
        << "rtp_llm_decode_cm2_config: " << rtp_llm_decode_cm2_config << "\n"
        << "remote_vit_server_ip: " << remote_vit_server_ip << "\n"
        << "rtp_llm_multimodal_part_cm2_config: " << rtp_llm_multimodal_part_cm2_config;
    return oss.str();
}

// CacheStoreConfig
inline std::string CacheStoreConfig::to_string() const {
    std::ostringstream oss;
    oss << "cache_store_rdma_mode: " << cache_store_rdma_mode << "\n"
        << "wrr_available_ratio: " << wrr_available_ratio << "\n"
        << "rank_factor: " << rank_factor;
    return oss.str();
}

// SchedulerConfig
inline std::string SchedulerConfig::to_string() const {
    std::ostringstream oss;
    oss << "use_batch_decode_scheduler: " << use_batch_decode_scheduler << "\n";
    return oss.str();
}

// BatchDecodeSchedulerConfig
inline std::string BatchDecodeSchedulerConfig::to_string() const {
    std::ostringstream oss;
    oss << "batch_decode_scheduler_batch_size: " << batch_decode_scheduler_batch_size << "\n";
    return oss.str();
}

// FIFOSchedulerConfig
inline std::string FIFOSchedulerConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_context_batch_size: " << max_context_batch_size << "\n"
        << "scheduler_reserve_resource_ratio: " << scheduler_reserve_resource_ratio << "\n"
        << "enable_fast_gen: " << enable_fast_gen << "\n"
        << "enable_partial_fallback: " << enable_partial_fallback << "\n"
        << "fast_gen_context_budget: " << fast_gen_context_budget;
    return oss.str();
}

// MiscellaneousConfig
inline std::string MiscellaneousConfig::to_string() const {
    std::ostringstream oss;
    oss << "load_balance: " << load_balance << "\n"
        << "step_records_time_range: " << step_records_time_range << "\n"
        << "step_records_max_size: " << step_records_max_size << "\n";
    return oss.str();
}

// ArpcConfig
inline std::string ArpcConfig::to_string() const {
    std::ostringstream oss;
    oss << "threadNum: " << threadNum << "\n"
        << "queueNum: " << queueNum << "\n"
        << "ioThreadNum: " << ioThreadNum;
    return oss.str();
}

}
