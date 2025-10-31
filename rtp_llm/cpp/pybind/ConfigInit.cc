#include <torch/library.h>
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/pybind/common/blockUtil.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {

void registerFMHAType(py::module m) {
    py::enum_<FMHAType>(m, "FMHAType")
        .value("NONE", FMHAType::NONE)
        .value("PAGED_TRT_V2", FMHAType::PAGED_TRT_V2)
        .value("TRT_V2", FMHAType::TRT_V2)
        .value("PAGED_OPEN_SOURCE", FMHAType::PAGED_OPEN_SOURCE)
        .value("OPEN_SOURCE", FMHAType::OPEN_SOURCE)
        .value("TRT_V1", FMHAType::TRT_V1)
        .value("FLASH_INFER", FMHAType::FLASH_INFER)
        .value("XQA", FMHAType::XQA)
        .value("AITER_PREFILL", FMHAType::AITER_PREFILL)
        .value("AITER_DECODE", FMHAType::AITER_DECODE);
}

void register_parallelism_distributed_config(pybind11::module& m) {
    pybind11::class_<ParallelismDistributedConfig>(m, "ParallelismDistributedConfig")
        .def(pybind11::init<int, int, int, int, int, int, int, int, bool>(),
             pybind11::arg("tp_size")          = 1,
             pybind11::arg("ep_size")          = 1,
             pybind11::arg("dp_size")          = 1,
             pybind11::arg("pp_size")          = 1,
             pybind11::arg("world_size")       = 1,
             pybind11::arg("world_rank")       = 0,
             pybind11::arg("local_world_size") = 1,
             pybind11::arg("ffn_sp_size")      = 1,
             pybind11::arg("use_all_gather")   = true)
        .def("to_string", &ParallelismDistributedConfig::to_string)
        .def("update_from_env", &ParallelismDistributedConfig::update_from_env_for_test)
        .def_readwrite("tp_size", &ParallelismDistributedConfig::tp_size)
        .def_readwrite("ep_size", &ParallelismDistributedConfig::ep_size)
        .def_readwrite("dp_size", &ParallelismDistributedConfig::dp_size)
        .def_readwrite("pp_size", &ParallelismDistributedConfig::pp_size)
        .def_readwrite("world_size", &ParallelismDistributedConfig::world_size)
        .def_readwrite("world_rank", &ParallelismDistributedConfig::world_rank)
        .def_readwrite("local_world_size", &ParallelismDistributedConfig::local_world_size)
        .def_readwrite("ffn_sp_size", &ParallelismDistributedConfig::ffn_sp_size)
        .def_readwrite("use_all_gather", &ParallelismDistributedConfig::use_all_gather);
}

void register_arpc_config(pybind11::module& m) {
    pybind11::class_<ArpcConfig>(m, "ArpcConfig")
        .def(pybind11::init<int, int, int>(),
             pybind11::arg("threadNum")   = 10,
             pybind11::arg("queueNum")    = 50,
             pybind11::arg("ioThreadNum") = 2)
        .def("to_string", &ArpcConfig::to_string)
        .def_readwrite("threadNum", &ArpcConfig::threadNum)
        .def_readwrite("queueNum", &ArpcConfig::queueNum)
        .def_readwrite("ioThreadNum", &ArpcConfig::ioThreadNum);
}

void register_grpc_config(pybind11::module& m) {
    pybind11::class_<GrpcConfig>(m, "GrpcConfig")
        .def(pybind11::init<>())  // Default constructor
        .def(pybind11::init<const std::string&>(),
             pybind11::arg("json_str"))  // JSON string constructor
        .def("to_string", &GrpcConfig::to_string)
        .def("update_from_env", &GrpcConfig::update_from_env_for_test)
        .def("from_json", &GrpcConfig::from_json, "Initialize from JSON string")
        .def("get_client_config", &GrpcConfig::get_client_config)
        .def("get_server_config", &GrpcConfig::get_server_config);
}

void register_ffn_disaggregate_config(pybind11::module& m) {
    pybind11::class_<FfnDisAggregateConfig>(m, "FfnDisAggregateConfig")
        .def(pybind11::init<bool, int, int, int, int, bool>(),
             pybind11::arg("enable_ffn_disaggregate") = false,
             pybind11::arg("attention_tp_size")       = 1,
             pybind11::arg("attention_dp_size")       = 1,
             pybind11::arg("ffn_tp_size")             = 1,
             pybind11::arg("ffn_dp_size")             = 1,
             pybind11::arg("is_ffn_rank")             = false)
        .def("to_string", &FfnDisAggregateConfig::to_string)
        .def("update_from_env", &FfnDisAggregateConfig::update_from_env_for_test)
        .def("is_ffn_service", &FfnDisAggregateConfig::is_ffn_service)
        .def_readwrite("enable_ffn_disaggregate", &FfnDisAggregateConfig::enable_ffn_disaggregate)
        .def_readwrite("attention_tp_size", &FfnDisAggregateConfig::attention_tp_size)
        .def_readwrite("attention_dp_size", &FfnDisAggregateConfig::attention_dp_size)
        .def_readwrite("ffn_tp_size", &FfnDisAggregateConfig::ffn_tp_size)
        .def_readwrite("ffn_dp_size", &FfnDisAggregateConfig::ffn_dp_size)
        .def_readwrite("is_ffn_rank", &FfnDisAggregateConfig::is_ffn_rank);
}

// ConcurrencyConfig
void register_concurrency_config(pybind11::module& m) {
    pybind11::class_<ConcurrencyConfig>(m, "ConcurrencyConfig")
        .def(pybind11::init<bool, int>(),
             pybind11::arg("concurrency_with_block") = false,
             pybind11::arg("concurrency_limit")      = 32)
        .def("to_string", &ConcurrencyConfig::to_string)
        .def("update_from_env", &ConcurrencyConfig::update_from_env_for_test)
        .def_readwrite("concurrency_with_block", &ConcurrencyConfig::concurrency_with_block)
        .def_readwrite("concurrency_limit", &ConcurrencyConfig::concurrency_limit);
}

// FMHAConfig
void register_fmha_config(pybind11::module& m) {
    pybind11::class_<FMHAConfig>(m, "FMHAConfig")
        .def(pybind11::init<bool, bool, bool, bool, bool, bool, bool, bool, bool, bool>(),
             pybind11::arg("enable_fmha")                   = true,
             pybind11::arg("enable_trt_fmha")               = true,
             pybind11::arg("enable_paged_trt_fmha")         = true,
             pybind11::arg("enable_open_source_fmha")       = true,
             pybind11::arg("enable_paged_open_source_fmha") = true,
             pybind11::arg("enable_trtv1_fmha")             = true,
             pybind11::arg("fmha_perf_instrument")          = false,
             pybind11::arg("fmha_show_params")              = false,
             pybind11::arg("disable_flash_infer")           = false,
             pybind11::arg("enable_xqa")                    = true)
        .def("to_string", &FMHAConfig::to_string)
        .def("update_from_env", &FMHAConfig::update_from_env_for_test)
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
void register_kvcache_config(pybind11::module& m) {
    pybind11::class_<KVCacheConfig>(m, "KVCacheConfig")
        .def(pybind11::
                 init<bool, std::string, std::string, bool, int, int, int, int, int, int, int64_t, int64_t, int, int>(),
             pybind11::arg("reuse_cache")                        = false,
             pybind11::arg("multi_task_prompt")                  = "",
             pybind11::arg("multi_task_prompt_str")              = "",
             pybind11::arg("enable_3fs")                         = false,
             pybind11::arg("match_timeout_ms")                   = 1000,
             pybind11::arg("rpc_get_cache_timeout_ms")           = 2000,
             pybind11::arg("rpc_put_cache_timeout_ms")           = 2000,
             pybind11::arg("threefs_read_timeout_ms")            = 1000,
             pybind11::arg("threefs_write_timeout_ms")           = 2000,
             pybind11::arg("max_block_size_per_item")            = 16,
             pybind11::arg("threefs_read_iov_size")              = 1LL << 32,
             pybind11::arg("threefs_write_iov_size")             = 1LL << 32,
             pybind11::arg("memory_block_cache_size_mb")         = 0,
             pybind11::arg("memory_block_cache_sync_timeout_ms") = 10000)
        .def("to_string", &KVCacheConfig::to_string)
        .def("update_from_env", &KVCacheConfig::update_from_env_for_test)
        .def_readwrite("reuse_cache", &KVCacheConfig::reuse_cache)
        .def_readwrite("multi_task_prompt", &KVCacheConfig::multi_task_prompt)
        .def_readwrite("multi_task_prompt_str", &KVCacheConfig::multi_task_prompt_str)
        .def_readwrite("enable_3fs", &KVCacheConfig::enable_3fs)
        .def_readwrite("match_timeout_ms", &KVCacheConfig::match_timeout_ms)
        .def_readwrite("rpc_get_cache_timeout_ms", &KVCacheConfig::rpc_get_cache_timeout_ms)
        .def_readwrite("rpc_put_cache_timeout_ms", &KVCacheConfig::rpc_put_cache_timeout_ms)
        .def_readwrite("threefs_read_timeout_ms", &KVCacheConfig::threefs_read_timeout_ms)
        .def_readwrite("threefs_write_timeout_ms", &KVCacheConfig::threefs_write_timeout_ms)
        .def_readwrite("max_block_size_per_item", &KVCacheConfig::max_block_size_per_item)
        .def_readwrite("threefs_read_iov_size", &KVCacheConfig::threefs_read_iov_size)
        .def_readwrite("threefs_write_iov_size", &KVCacheConfig::threefs_write_iov_size)
        .def_readwrite("memory_block_cache_size_mb", &KVCacheConfig::memory_block_cache_size_mb)
        .def_readwrite("memory_block_cache_sync_timeout_ms", &KVCacheConfig::memory_block_cache_sync_timeout_ms);
}

// ProfilingDebugLoggingConfig
void register_profiling_debug_logging_config(pybind11::module& m) {
    pybind11::class_<ProfilingDebugLoggingConfig>(m, "ProfilingDebugLoggingConfig")
        .def(pybind11::init<bool,
                            bool,
                            bool,
                            bool,
                            std::string,
                            std::string,
                            bool,
                            std::string,
                            std::string,
                            int,
                            std::string,
                            bool,
                            int,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool>(),
             pybind11::arg("trace_memory")              = false,
             pybind11::arg("trace_malloc_stack")        = false,
             pybind11::arg("enable_device_perf")        = false,
             pybind11::arg("ft_core_dump_on_exception") = false,
             pybind11::arg("ft_alog_conf_path")         = "",
             pybind11::arg("log_level")                 = "INFO",
             pybind11::arg("gen_timeline_sync")         = false,
             pybind11::arg("torch_cuda_profiler_dir")   = "",
             pybind11::arg("log_path")                  = "logs",
             pybind11::arg("log_file_backup_count")     = 16,
             pybind11::arg("nccl_debug_file")           = "",
             pybind11::arg("debug_load_server")         = false,
             pybind11::arg("hack_layer_num")            = 0,
             pybind11::arg("debug_start_fake_process")  = false,
             pybind11::arg("dg_print_reg_reuse")        = false,
             pybind11::arg("qwen_agent_debug")          = false,
             pybind11::arg("disable_dpc_random")        = false,
             pybind11::arg("enable_detail_log")         = false,
             pybind11::arg("check_nan")                 = false)
        .def("to_string", &ProfilingDebugLoggingConfig::to_string)
        .def("update_from_env", &ProfilingDebugLoggingConfig::update_from_env_for_test)
        .def_readwrite("trace_memory", &ProfilingDebugLoggingConfig::trace_memory)
        .def_readwrite("trace_malloc_stack", &ProfilingDebugLoggingConfig::trace_malloc_stack)
        .def_readwrite("enable_device_perf", &ProfilingDebugLoggingConfig::enable_device_perf)
        .def_readwrite("ft_core_dump_on_exception", &ProfilingDebugLoggingConfig::ft_core_dump_on_exception)
        .def_readwrite("ft_alog_conf_path", &ProfilingDebugLoggingConfig::ft_alog_conf_path)
        .def_readwrite("log_level", &ProfilingDebugLoggingConfig::log_level)
        .def_readwrite("gen_timeline_sync", &ProfilingDebugLoggingConfig::gen_timeline_sync)
        .def_readwrite("torch_cuda_profiler_dir", &ProfilingDebugLoggingConfig::torch_cuda_profiler_dir)
        .def_readwrite("log_path", &ProfilingDebugLoggingConfig::log_path)
        .def_readwrite("log_file_backup_count", &ProfilingDebugLoggingConfig::log_file_backup_count)
        .def_readwrite("nccl_debug_file", &ProfilingDebugLoggingConfig::nccl_debug_file)
        .def_readwrite("debug_load_server", &ProfilingDebugLoggingConfig::debug_load_server)
        .def_readwrite("hack_layer_num", &ProfilingDebugLoggingConfig::hack_layer_num)
        .def_readwrite("debug_start_fake_process", &ProfilingDebugLoggingConfig::debug_start_fake_process)
        .def_readwrite("dg_print_reg_reuse", &ProfilingDebugLoggingConfig::dg_print_reg_reuse)
        .def_readwrite("qwen_agent_debug", &ProfilingDebugLoggingConfig::qwen_agent_debug)
        .def_readwrite("disable_dpc_random", &ProfilingDebugLoggingConfig::disable_dpc_random)
        .def_readwrite("enable_detail_log", &ProfilingDebugLoggingConfig::enable_detail_log)
        .def_readwrite("check_nan", &ProfilingDebugLoggingConfig::check_nan);
}

void register_hwkernel_config(pybind11::module& m) {
    pybind11::class_<HWKernelConfig>(m, "HWKernelConfig")
        .def(pybind11::init<int,
                            bool,
                            bool,
                            bool,
                            bool,
                            std::string,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool,
                            int,
                            std::vector<int>,
                            std::vector<int>>(),
             pybind11::arg("deep_gemm_num_sm")             = -1,
             pybind11::arg("arm_gemm_use_kai")             = false,
             pybind11::arg("enable_stable_scatter_add")    = false,
             pybind11::arg("enable_multi_block_mode")      = true,
             pybind11::arg("ft_disable_custom_ar")         = true,
             pybind11::arg("rocm_hipblaslt_config")        = "gemm_config.csv",
             pybind11::arg("use_swizzleA")                 = false,
             pybind11::arg("enable_cuda_graph")            = false,
             pybind11::arg("enable_cuda_graph_debug_mode") = false,
             pybind11::arg("use_aiter_pa")                 = true,
             pybind11::arg("use_asm_pa")                   = true,
             pybind11::arg("enable_native_cuda_graph")     = false,
             pybind11::arg("num_native_cuda_graph")        = 200,
             pybind11::arg("prefill_capture_seq_lens")     = std::vector<int>(),
             pybind11::arg("decode_capture_batch_sizes")   = std::vector<int>())
        .def("to_string", &HWKernelConfig::to_string)
        .def("update_from_env", &HWKernelConfig::update_from_env_for_test)
        .def_readwrite("deep_gemm_num_sm", &HWKernelConfig::deep_gemm_num_sm)
        .def_readwrite("arm_gemm_use_kai", &HWKernelConfig::arm_gemm_use_kai)
        .def_readwrite("enable_stable_scatter_add", &HWKernelConfig::enable_stable_scatter_add)
        .def_readwrite("enable_multi_block_mode", &HWKernelConfig::enable_multi_block_mode)
        .def_readwrite("ft_disable_custom_ar", &HWKernelConfig::ft_disable_custom_ar)
        .def_readwrite("rocm_hipblaslt_config", &HWKernelConfig::rocm_hipblaslt_config)
        .def_readwrite("use_swizzleA", &HWKernelConfig::use_swizzleA)
        .def_readwrite("enable_cuda_graph", &HWKernelConfig::enable_cuda_graph)
        .def_readwrite("enable_cuda_graph_debug_mode", &HWKernelConfig::enable_cuda_graph_debug_mode)
        .def_readwrite("use_aiter_pa", &HWKernelConfig::use_aiter_pa)
        .def_readwrite("use_asm_pa", &HWKernelConfig::use_asm_pa)
        .def_readwrite("enable_native_cuda_graph", &HWKernelConfig::enable_native_cuda_graph)
        .def_readwrite("num_native_cuda_graph", &HWKernelConfig::num_native_cuda_graph)
        .def_readwrite("prefill_capture_seq_lens", &HWKernelConfig::prefill_capture_seq_lens)
        .def_readwrite("decode_capture_batch_sizes", &HWKernelConfig::decode_capture_batch_sizes);
}

// DeviceResourceConfig
void register_device_resource_config(pybind11::module& m) {
    pybind11::class_<DeviceResourceConfig>(m, "DeviceResourceConfig")
        .def(pybind11::init<int64_t, int64_t, int, int, int, bool, int, bool>(),
             pybind11::arg("device_reserve_memory_bytes") = -1073741824,
             pybind11::arg("host_reserve_memory_bytes")   = 4LL * 1024 * 1024 * 1024,
             pybind11::arg("overlap_math_sm_count")       = 0,
             pybind11::arg("overlap_comm_type")           = 0,
             pybind11::arg("m_split")                     = 0,
             pybind11::arg("enable_comm_overlap")         = true,
             pybind11::arg("enable_layer_micro_batch")    = 0,
             pybind11::arg("not_use_default_stream")      = false)
        .def("to_string", &DeviceResourceConfig::to_string)
        .def("update_from_env", &DeviceResourceConfig::update_from_env_for_test)
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
void register_sampler_config(pybind11::module& m) {
    pybind11::class_<SamplerConfig>(m, "SamplerConfig")
        .def(pybind11::init<int64_t, bool>(),
             pybind11::arg("max_batch_size")                  = 0,
             pybind11::arg("enable_flashinfer_sample_kernel") = true)
        .def("to_string", &SamplerConfig::to_string)
        .def("update_from_env", &SamplerConfig::update_from_env_for_test)
        .def_readwrite("max_batch_size", &SamplerConfig::max_batch_size)
        .def_readwrite("enable_flashinfer_sample_kernel", &SamplerConfig::enable_flashinfer_sample_kernel);
}

// MoeConfig
void register_moe_config(pybind11::module& m) {
    pybind11::class_<MoeConfig>(m, "MoeConfig")
        .def(pybind11::init<bool, bool, bool, bool, bool, int, bool, bool, int, int, int>(),
             pybind11::arg("use_deepep_moe")                  = false,
             pybind11::arg("use_deepep_internode")            = false,
             pybind11::arg("use_deepep_low_latency")          = true,
             pybind11::arg("use_deepep_p2p_low_latency")      = false,
             pybind11::arg("fake_balance_expert")             = false,
             pybind11::arg("eplb_control_step")               = 100,
             pybind11::arg("eplb_test_mode")                  = false,
             pybind11::arg("hack_moe_expert")                 = false,
             pybind11::arg("eplb_balance_layer_per_step")     = 1,
             pybind11::arg("deep_ep_num_sm")                  = 0,
             pybind11::arg("max_moe_normal_masked_token_num") = 1024)
        .def("to_string", &MoeConfig::to_string)
        .def("update_from_env", &MoeConfig::update_from_env_for_test)
        .def_readwrite("use_deepep_moe", &MoeConfig::use_deepep_moe)
        .def_readwrite("use_deepep_internode", &MoeConfig::use_deepep_internode)
        .def_readwrite("use_deepep_low_latency", &MoeConfig::use_deepep_low_latency)
        .def_readwrite("use_deepep_p2p_low_latency", &MoeConfig::use_deepep_p2p_low_latency)
        .def_readwrite("fake_balance_expert", &MoeConfig::fake_balance_expert)
        .def_readwrite("eplb_control_step", &MoeConfig::eplb_control_step)
        .def_readwrite("eplb_test_mode", &MoeConfig::eplb_test_mode)
        .def_readwrite("hack_moe_expert", &MoeConfig::hack_moe_expert)
        .def_readwrite("eplb_balance_layer_per_step", &MoeConfig::eplb_balance_layer_per_step)
        .def_readwrite("deep_ep_num_sm", &MoeConfig::deep_ep_num_sm)
        .def_readwrite("max_moe_normal_masked_token_num", &MoeConfig::max_moe_normal_masked_token_num);
}

// ModelSpecificConfig
void register_model_specific_config(pybind11::module& m) {
    pybind11::class_<ModelSpecificConfig>(m, "ModelSpecificConfig")
        .def(pybind11::init<int64_t, bool>(),
             pybind11::arg("max_lora_model_size") = -1,
             pybind11::arg("load_python_model")   = false)
        .def("to_string", &ModelSpecificConfig::to_string)
        .def("update_from_env", &ModelSpecificConfig::update_from_env_for_test)
        .def_readwrite("max_lora_model_size", &ModelSpecificConfig::max_lora_model_size)
        .def_readwrite("load_python_model", &ModelSpecificConfig::load_python_model);
}

// SpeculativeExecutionConfig
void register_speculative_execution_config(pybind11::module& m) {
    pybind11::class_<SpeculativeExecutionConfig>(m, "SpeculativeExecutionConfig")
        .def(pybind11::init<std::string, std::string, int64_t, int64_t, std::string, int, bool, bool>(),
             pybind11::arg("sp_model_type")                 = "",
             pybind11::arg("sp_type")                       = "",
             pybind11::arg("sp_min_token_match")            = 2,
             pybind11::arg("sp_max_token_match")            = 2,
             pybind11::arg("tree_decode_config")            = "",
             pybind11::arg("gen_num_per_cycle")             = 1,
             pybind11::arg("force_stream_sample")           = false,
             pybind11::arg("force_score_context_attention") = true)
        .def("to_string", &SpeculativeExecutionConfig::to_string)
        .def("update_from_env", &SpeculativeExecutionConfig::update_from_env_for_test)
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
void register_service_discovery_config(pybind11::module& m) {
    pybind11::class_<ServiceDiscoveryConfig>(m, "ServiceDiscoveryConfig")
        .def(pybind11::init<bool, std::string, std::string, std::string, std::string>(),
             pybind11::arg("use_local")                  = false,
             pybind11::arg("remote_rpc_server_ip")       = "",
             pybind11::arg("decode_cm2_config")          = "",
             pybind11::arg("remote_vit_server_ip")       = "",
             pybind11::arg("multimodal_part_cm2_config") = "")
        .def("to_string", &ServiceDiscoveryConfig::to_string)
        .def("update_from_env", &ServiceDiscoveryConfig::update_from_env_for_test)
        .def_readwrite("use_local", &ServiceDiscoveryConfig::use_local)
        .def_readwrite("remote_rpc_server_ip", &ServiceDiscoveryConfig::remote_rpc_server_ip)
        .def_readwrite("decode_cm2_config", &ServiceDiscoveryConfig::decode_cm2_config)
        .def_readwrite("remote_vit_server_ip", &ServiceDiscoveryConfig::remote_vit_server_ip)
        .def_readwrite("multimodal_part_cm2_config", &ServiceDiscoveryConfig::multimodal_part_cm2_config);
}

void register_cache_store_config(pybind11::module& m) {
    pybind11::class_<CacheStoreConfig>(m, "CacheStoreConfig")
        .def(pybind11::init<bool, int, int, int, int, int, int, int>(),
             pybind11::arg("cache_store_rdma_mode")        = false,
             pybind11::arg("wrr_available_ratio")          = 80,
             pybind11::arg("rank_factor")                  = 0,
             pybind11::arg("thread_count")                 = 16,
             pybind11::arg("rdma_connect_timeout_ms")      = 250,
             pybind11::arg("rdma_qp_count_per_connection") = 2,
             pybind11::arg("messager_io_thread_count")     = 2,
             pybind11::arg("messager_worker_thread_count") = 16)
        .def("to_string", &CacheStoreConfig::to_string)
        .def("update_from_env", &CacheStoreConfig::update_from_env_for_test)
        .def_readwrite("cache_store_rdma_mode", &CacheStoreConfig::cache_store_rdma_mode)
        .def_readwrite("wrr_available_ratio", &CacheStoreConfig::wrr_available_ratio)
        .def_readwrite("rank_factor", &CacheStoreConfig::rank_factor)
        .def_readwrite("thread_count", &CacheStoreConfig::thread_count)
        .def_readwrite("rdma_connect_timeout_ms", &CacheStoreConfig::rdma_connect_timeout_ms)
        .def_readwrite("rdma_qp_count_per_connection", &CacheStoreConfig::rdma_qp_count_per_connection)
        .def_readwrite("messager_io_thread_count", &CacheStoreConfig::messager_io_thread_count)
        .def_readwrite("messager_worker_thread_count", &CacheStoreConfig::messager_worker_thread_count);
}

// SchedulerConfig
void register_scheduler_config(pybind11::module& m) {
    pybind11::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(pybind11::init<bool, bool>(),
             pybind11::arg("use_batch_decode_scheduler") = false,
             pybind11::arg("use_gather_batch_scheduler") = false)
        .def("to_string", &SchedulerConfig::to_string)
        .def("update_from_env", &SchedulerConfig::update_from_env_for_test)
        .def_readwrite("use_batch_decode_scheduler", &SchedulerConfig::use_batch_decode_scheduler)
        .def_readwrite("use_gather_batch_scheduler", &SchedulerConfig::use_gather_batch_scheduler);
}

// BatchDecodeSchedulerConfig
void register_batch_decode_scheduler_config(pybind11::module& m) {
    pybind11::class_<BatchDecodeSchedulerConfig>(m, "BatchDecodeSchedulerConfig")
        .def(pybind11::init<int64_t, int64_t>(),
             pybind11::arg("batch_decode_scheduler_batch_size")  = 1,
             pybind11::arg("batch_decode_scheduler_warmup_type") = 0)
        .def("to_string", &BatchDecodeSchedulerConfig::to_string)
        .def("update_from_env", &BatchDecodeSchedulerConfig::update_from_env_for_test)
        .def_readwrite("batch_decode_scheduler_batch_size",
                       &BatchDecodeSchedulerConfig::batch_decode_scheduler_batch_size);
}

// FIFOSchedulerConfig
void register_fifo_scheduler_config(pybind11::module& m) {
    pybind11::class_<FIFOSchedulerConfig>(m, "FIFOSchedulerConfig")
        .def(pybind11::init<int64_t, int, bool, bool, int64_t>(),
             pybind11::arg("max_context_batch_size")           = 1,
             pybind11::arg("scheduler_reserve_resource_ratio") = 5,
             pybind11::arg("enable_fast_gen")                  = false,
             pybind11::arg("enable_partial_fallback")          = false,
             pybind11::arg("fast_gen_context_budget")          = -1)
        .def("to_string", &FIFOSchedulerConfig::to_string)
        .def("update_from_env", &FIFOSchedulerConfig::update_from_env_for_test)
        .def_readwrite("max_context_batch_size", &FIFOSchedulerConfig::max_context_batch_size)
        .def_readwrite("scheduler_reserve_resource_ratio", &FIFOSchedulerConfig::scheduler_reserve_resource_ratio)
        .def_readwrite("enable_fast_gen", &FIFOSchedulerConfig::enable_fast_gen)
        .def_readwrite("enable_partial_fallback", &FIFOSchedulerConfig::enable_partial_fallback)
        .def_readwrite("fast_gen_context_budget", &FIFOSchedulerConfig::fast_gen_context_budget);
}

// MiscellaneousConfig
void register_misc_config(pybind11::module& m) {
    pybind11::class_<MiscellaneousConfig>(m, "MiscellaneousConfig")
        .def(pybind11::init<bool, std::string>(), pybind11::arg("disable_pdl") = true, pybind11::arg("aux_string") = "")
        .def("to_string", &MiscellaneousConfig::to_string)
        .def("update_from_env", &MiscellaneousConfig::update_from_env_for_test)
        .def_readwrite("disable_pdl", &MiscellaneousConfig::disable_pdl)
        .def_readwrite("aux_string", &MiscellaneousConfig::aux_string);
}

void registerGptInitParameter(py::module m) {
    py::enum_<MlaOpsType>(m, "MlaOpsType")
        .value("AUTO", MlaOpsType::AUTO)
        .value("MHA", MlaOpsType::MHA)
        .value("FLASH_INFER", MlaOpsType::FLASH_INFER)
        .value("FLASH_MLA", MlaOpsType::FLASH_MLA);

    py::enum_<EplbMode>(m, "EplbMode")
        .value("NONE", EplbMode::NONE)
        .value("STATS", EplbMode::STATS)
        .value("EPLB", EplbMode::EPLB)
        .value("ALL", EplbMode::ALL);

    pybind11::class_<EplbConfig>(m, "EplbConfig")
        .def(pybind11::init<>())
        .def_readwrite("mode", &EplbConfig::mode)
        .def_readwrite("update_time", &EplbConfig::update_time)
        .def("__eq__", &EplbConfig::operator==)
        .def("__ne__", &EplbConfig::operator!=)
        .def("__str__", &EplbConfig::toString);

#define DEF_PROPERTY(name) .def_readwrite(#name, &RoleSpecialTokens::name##_)

#define REGISTER_PROPERTYS                                                                                             \
    DEF_PROPERTY(token_ids)                                                                                            \
    DEF_PROPERTY(eos_token_ids)

    pybind11::class_<RoleSpecialTokens>(m, "RoleSpecialTokens").def(pybind11::init<>()) REGISTER_PROPERTYS;

#undef DEF_PROPERTY
#undef REGISTER_PROPERTYS

#define DEF_PROPERTY(name) .def_readwrite(#name, &SpecialTokens::name##_)

#define REGISTER_PROPERTYS                                                                                             \
    DEF_PROPERTY(bos_token_id)                                                                                         \
    DEF_PROPERTY(eos_token_id)                                                                                         \
    DEF_PROPERTY(decoder_start_token_id)                                                                               \
    DEF_PROPERTY(user)                                                                                                 \
    DEF_PROPERTY(assistant)                                                                                            \
    DEF_PROPERTY(system)                                                                                               \
    DEF_PROPERTY(stop_words_id_list)                                                                                   \
    DEF_PROPERTY(stop_words_str_list)                                                                                  \
    DEF_PROPERTY(pad_token_id)

    pybind11::class_<SpecialTokens>(m, "SpecialTokens").def(pybind11::init<>()) REGISTER_PROPERTYS;

#undef DEF_PROPERTY
#undef REGISTER_PROPERTYS

    pybind11::class_<QuantAlgo>(m, "QuantAlgo")
        .def(pybind11::init<>())  // quant_pre_scales
        .def("setQuantAlgo", &QuantAlgo::setQuantAlgo, py::arg("quant_method"), py::arg("bits"), py::arg("group_size"))
        .def("isWeightOnlyPerCol", &QuantAlgo::isWeightOnlyPerCol)
        .def("isGptq", &QuantAlgo::isGptq)
        .def("isAwq", &QuantAlgo::isAwq)
        .def("isSmoothQuant", &QuantAlgo::isSmoothQuant)
        .def("isOmniQuant", &QuantAlgo::isOmniQuant)
        .def("isPerTensorQuant", &QuantAlgo::isPerTensorQuant)
        .def("isFp8", &QuantAlgo::isFp8)
        .def("isFp8PTPC", &QuantAlgo::isFp8PTPC)
        .def("isQuant", &QuantAlgo::isQuant)
        .def("isGroupwise", &QuantAlgo::isGroupwise)
        .def("getGroupSize", &QuantAlgo::getGroupSize)
        .def("getWeightBits", &QuantAlgo::getWeightBits)
        .def("getActivationBits", &QuantAlgo::getActivationBits)
        .def(py::pickle(
            [](const QuantAlgo& quant_algo) {
                return py::make_tuple(int(quant_algo.getQuantMethod()),
                                      int(quant_algo.getWeightBits()),
                                      int(quant_algo.getGroupSize()),
                                      int(quant_algo.getActivationBits()));
            },
            [](py::tuple t) { return QuantAlgo(QuantMethod(t[0].cast<int>()), t[1].cast<int>(), t[2].cast<int>()); }));

    pybind11::enum_<RoleType>(m, "RoleType")
        .value("PDFUSION", RoleType::PDFUSION)
        .value("PREFILL", RoleType::PREFILL)
        .value("DECODE", RoleType::DECODE)
        .value("VIT", RoleType::VIT)
        .value("FRONTEND", RoleType::FRONTEND)
        .def("__str__", [](RoleType role) {
            switch (role) {
                case RoleType::PDFUSION:
                    return "pd_fusion";
                case RoleType::PREFILL:
                    return "prefill";
                case RoleType::DECODE:
                    return "decode";
                case RoleType::VIT:
                    return "vit";
                case RoleType::FRONTEND:
                    return "FRONTEND";
                default:
                    return "invalid";
            }
        });

#define DEF_PROPERTY(name, member) .def_readwrite(#name, &GptInitParameter::member)

#define REGISTER_PROPERTYS                                                                                             \
    DEF_PROPERTY(head_num, head_num_)                                                                                  \
    DEF_PROPERTY(head_num_kv, head_num_kv_)                                                                            \
    DEF_PROPERTY(size_per_head, size_per_head_)                                                                        \
    DEF_PROPERTY(max_seq_len, max_seq_len_)                                                                            \
    DEF_PROPERTY(max_batch_tokens_size, max_batch_tokens_size_)                                                        \
    DEF_PROPERTY(vocab_size, vocab_size_)                                                                              \
    DEF_PROPERTY(input_vocab_size, input_vocab_size_)                                                                  \
    DEF_PROPERTY(hidden_size, hidden_size_)                                                                            \
    DEF_PROPERTY(type_vocab_size, type_vocab_size_)                                                                    \
    DEF_PROPERTY(embedding_size, embedding_size_)                                                                      \
    DEF_PROPERTY(gen_num_per_circle, gen_num_per_circle_)                                                              \
    DEF_PROPERTY(inter_size, inter_size_)                                                                              \
    DEF_PROPERTY(inter_padding_size, inter_padding_size_)                                                              \
    DEF_PROPERTY(moe_inter_padding_size, moe_inter_padding_size_)                                                      \
    DEF_PROPERTY(is_sparse_head, is_sparse_head_)                                                                      \
    DEF_PROPERTY(layer_head_num, layer_head_num_)                                                                      \
    DEF_PROPERTY(layer_head_num_kv, layer_head_num_kv_)                                                                \
    DEF_PROPERTY(layer_inter_size, layer_inter_size_)                                                                  \
    DEF_PROPERTY(layer_inter_padding_size, layer_inter_padding_size_)                                                  \
    DEF_PROPERTY(num_layers, num_layers_)                                                                              \
    DEF_PROPERTY(layer_num, num_layers_)                                                                               \
    DEF_PROPERTY(num_valid_layer, num_valid_layer_)                                                                    \
    DEF_PROPERTY(expert_num, expert_num_)                                                                              \
    DEF_PROPERTY(moe_k, moe_k_)                                                                                        \
    DEF_PROPERTY(moe_normalize_expert_scale, moe_normalize_expert_scale_)                                              \
    DEF_PROPERTY(moe_style, moe_style_)                                                                                \
    DEF_PROPERTY(moe_layer_index, moe_layer_index_)                                                                    \
    DEF_PROPERTY(scoring_func, scoring_func_)                                                                          \
    DEF_PROPERTY(layernorm_eps, layernorm_eps_)                                                                        \
    /* In python, the following types use strings for branch condition */                                              \
    /* Everytime type changes, corresponding set type function should  */                                              \
    /* be called.                                                      */                                              \
    DEF_PROPERTY(layernorm_type, layernorm_type_str_)                                                                  \
    DEF_PROPERTY(norm_type, norm_type_str_)                                                                            \
    DEF_PROPERTY(activation_type, activation_type_str_)                                                                \
    DEF_PROPERTY(rotary_embedding_dim, rotary_embedding_dim_)                                                          \
    DEF_PROPERTY(kv_cache_data_type, kv_cache_data_type_str_)                                                          \
    DEF_PROPERTY(rotary_embedding_style, rotary_embedding_style_)                                                      \
    DEF_PROPERTY(position_ids_style, position_ids_style_)                                                              \
    DEF_PROPERTY(position_id_len_factor, position_id_len_factor_)                                                      \
    DEF_PROPERTY(rotary_embedding_base, rotary_embedding_base_)                                                        \
    DEF_PROPERTY(rotary_embedding_scale, rotary_embedding_scale_)                                                      \
    DEF_PROPERTY(org_embedding_max_pos, org_embedding_max_pos_)                                                        \
    DEF_PROPERTY(rotary_factor1, rotary_factor1_)                                                                      \
    DEF_PROPERTY(rotary_factor2, rotary_factor2_)                                                                      \
    DEF_PROPERTY(partial_rotary_factor, partial_rotary_factor_)                                                        \
    DEF_PROPERTY(mrope_section, mrope_section_)                                                                        \
    DEF_PROPERTY(input_embedding_scalar, input_embedding_scalar_)                                                      \
    DEF_PROPERTY(residual_scalar, residual_scalar_)                                                                    \
    DEF_PROPERTY(use_norm_input_residual, use_norm_input_residual_)                                                    \
    DEF_PROPERTY(use_norm_attn_out_residual, use_norm_attn_out_residual_)                                              \
    DEF_PROPERTY(data_type, data_type_str_)                                                                            \
    DEF_PROPERTY(has_positional_encoding, has_positional_encoding_)                                                    \
    DEF_PROPERTY(has_pre_decoder_layernorm, has_pre_decoder_layernorm_)                                                \
    DEF_PROPERTY(has_post_decoder_layernorm, has_post_decoder_layernorm_)                                              \
    DEF_PROPERTY(has_moe_norm, has_moe_norm_)                                                                          \
    DEF_PROPERTY(logit_scale, logit_scale_)                                                                            \
    DEF_PROPERTY(has_lm_head, has_lm_head_)                                                                            \
    DEF_PROPERTY(use_attention_linear_bias, use_attention_linear_bias_)                                                \
    DEF_PROPERTY(use_fp32_to_compute_logit, use_fp32_to_compute_logit_)                                                \
    DEF_PROPERTY(add_bias_linear, add_bias_linear_)                                                                    \
    DEF_PROPERTY(tokenizer_path, tokenizer_path_)                                                                      \
    DEF_PROPERTY(ckpt_path, ckpt_path_)                                                                                \
    DEF_PROPERTY(pre_seq_len, pre_seq_len_)                                                                            \
    DEF_PROPERTY(prefix_projection, prefix_projection_)                                                                \
    DEF_PROPERTY(using_hf_sampling, using_hf_sampling_)                                                                \
    DEF_PROPERTY(max_generate_batch_size, max_generate_batch_size_)                                                    \
    DEF_PROPERTY(max_context_batch_size, max_context_batch_size_)                                                      \
    DEF_PROPERTY(special_tokens, special_tokens_)                                                                      \
    DEF_PROPERTY(quant_algo, quant_algo_)                                                                              \
    DEF_PROPERTY(use_logn_attn, use_logn_attn_)                                                                        \
    DEF_PROPERTY(q_scaling, q_scaling_)                                                                                \
    DEF_PROPERTY(qk_norm, qk_norm_)                                                                                    \
    DEF_PROPERTY(use_cross_attn, use_cross_attn_)                                                                      \
    DEF_PROPERTY(cross_attn_input_len, cross_attn_input_len_)                                                          \
    DEF_PROPERTY(is_multimodal, is_multimodal_)                                                                        \
    DEF_PROPERTY(mm_sep_tokens, mm_sep_tokens_)                                                                        \
    DEF_PROPERTY(include_sep_tokens, include_sep_tokens_)                                                              \
    DEF_PROPERTY(mm_position_ids_style, mm_position_ids_style_)                                                        \
    DEF_PROPERTY(pre_allocate_op_mem, pre_allocate_op_mem_)                                                            \
    DEF_PROPERTY(seq_size_per_block, seq_size_per_block_)                                                              \
    DEF_PROPERTY(max_block_size_per_item, max_block_size_per_item_)                                                    \
    DEF_PROPERTY(block_nums, block_nums_)                                                                              \
    DEF_PROPERTY(kv_cache_mem_mb, kv_cache_mem_mb_)                                                                    \
    DEF_PROPERTY(reserve_runtime_mem_mb, reserve_runtime_mem_mb_)                                                      \
    DEF_PROPERTY(reuse_cache, reuse_cache_)                                                                            \
    DEF_PROPERTY(enable_partial_fallback, enable_partial_fallback_)                                                    \
    DEF_PROPERTY(enable_fast_gen, enable_fast_gen_)                                                                    \
    DEF_PROPERTY(warm_up, warm_up_)                                                                                    \
    DEF_PROPERTY(warm_up_with_loss, warm_up_with_loss_)                                                                \
    DEF_PROPERTY(engine_async_worker_count, engine_async_worker_count_)                                                \
    DEF_PROPERTY(fast_gen_max_context_len, fast_gen_max_context_len_)                                                  \
    DEF_PROPERTY(is_causal, is_causal_)                                                                                \
    DEF_PROPERTY(nccl_ip, nccl_ip_)                                                                                    \
    DEF_PROPERTY(tp_nccl_port, tp_nccl_port_)                                                                          \
    DEF_PROPERTY(dp_tp_nccl_port, dp_tp_nccl_port_)                                                                    \
    DEF_PROPERTY(ffn_tp_nccl_port, ffn_tp_nccl_port_)                                                                  \
    DEF_PROPERTY(model_rpc_port, model_rpc_port_)                                                                      \
    DEF_PROPERTY(embedding_rpc_port, embedding_rpc_port_)                                                              \
    DEF_PROPERTY(http_port, http_port_)                                                                                \
    DEF_PROPERTY(tp_size, tp_size_)                                                                                    \
    DEF_PROPERTY(tp_rank, tp_rank_)                                                                                    \
    DEF_PROPERTY(dp_size, dp_size_)                                                                                    \
    DEF_PROPERTY(dp_rank, dp_rank_)                                                                                    \
    DEF_PROPERTY(ffn_tp_size, ffn_tp_size_)                                                                            \
    DEF_PROPERTY(ffn_tp_rank, ffn_tp_rank_)                                                                            \
    DEF_PROPERTY(enable_sp, enable_sp_)                                                                                \
    DEF_PROPERTY(world_size, world_size_)                                                                              \
    DEF_PROPERTY(use_all_gather, use_all_gather_)                                                                      \
    DEF_PROPERTY(cache_store_listen_port, cache_store_listen_port_)                                                    \
    DEF_PROPERTY(cache_store_connect_port, cache_store_connect_port_)                                                  \
    DEF_PROPERTY(cache_store_rdma_connect_port, cache_store_rdma_connect_port_)                                        \
    DEF_PROPERTY(cache_store_rdma_listen_port, cache_store_rdma_listen_port_)                                          \
    DEF_PROPERTY(worker_port_offset, worker_port_offset_)                                                              \
    DEF_PROPERTY(worker_addrs, worker_addrs_)                                                                          \
    DEF_PROPERTY(worker_grpc_addrs, worker_grpc_addrs_)                                                                \
    DEF_PROPERTY(remote_rpc_server_port, remote_rpc_server_port_)                                                      \
    DEF_PROPERTY(role_type, role_type_)                                                                                \
    DEF_PROPERTY(cache_store_rdma_mode, cache_store_rdma_mode_)                                                        \
    DEF_PROPERTY(prefill_retry_times, prefill_retry_times_)                                                            \
    DEF_PROPERTY(prefill_retry_timeout_ms, prefill_retry_timeout_ms_)                                                  \
    DEF_PROPERTY(prefill_max_wait_timeout_ms, prefill_max_wait_timeout_ms_)                                            \
    DEF_PROPERTY(decode_retry_times, decode_retry_times_)                                                              \
    DEF_PROPERTY(decode_retry_timeout_ms, decode_retry_timeout_ms_)                                                    \
    DEF_PROPERTY(decode_retry_interval_ms, decode_retry_interval_ms_)                                                  \
    DEF_PROPERTY(decode_polling_kv_cache_step_ms, decode_polling_kv_cache_step_ms_)                                    \
    DEF_PROPERTY(decode_polling_call_prefill_ms, decode_polling_call_prefill_ms_)                                      \
    DEF_PROPERTY(rdma_connect_retry_times, rdma_connect_retry_times_)                                                  \
    DEF_PROPERTY(decode_entrance, decode_entrance_)                                                                    \
    DEF_PROPERTY(load_cache_timeout_ms, load_cache_timeout_ms_)                                                        \
    DEF_PROPERTY(max_rpc_timeout_ms, max_rpc_timeout_ms_)                                                              \
    DEF_PROPERTY(ep_size, ep_size_)                                                                                    \
    DEF_PROPERTY(ep_rank, ep_rank_)                                                                                    \
    DEF_PROPERTY(use_kvcache, use_kvcache_)                                                                            \
    DEF_PROPERTY(local_rank, local_rank_)                                                                              \
    DEF_PROPERTY(rotary_embedding_mscale, rotary_embedding_mscale_)                                                    \
    DEF_PROPERTY(rotary_embedding_offset, rotary_embedding_offset_)                                                    \
    DEF_PROPERTY(rotary_embedding_extrapolation_factor, rotary_embedding_extrapolation_factor_)                        \
    DEF_PROPERTY(use_mla, use_mla_)                                                                                    \
    DEF_PROPERTY(mla_ops_type, mla_ops_type_)                                                                          \
    DEF_PROPERTY(q_lora_rank, q_lora_rank_)                                                                            \
    DEF_PROPERTY(kv_lora_rank, kv_lora_rank_)                                                                          \
    DEF_PROPERTY(nope_head_dim, nope_head_dim_)                                                                        \
    DEF_PROPERTY(rope_head_dim, rope_head_dim_)                                                                        \
    DEF_PROPERTY(v_head_dim, v_head_dim_)                                                                              \
    DEF_PROPERTY(moe_n_group, moe_n_group_)                                                                            \
    DEF_PROPERTY(moe_topk_group, moe_topk_group_)                                                                      \
    DEF_PROPERTY(routed_scaling_factor, routed_scaling_factor_)                                                        \
    DEF_PROPERTY(softmax_extra_scale, softmax_extra_scale_)                                                            \
    DEF_PROPERTY(vit_separation, vit_separation_)                                                                      \
    DEF_PROPERTY(enable_speculative_decoding, enable_speculative_decoding_)                                            \
    DEF_PROPERTY(model_name, model_name_)                                                                              \
    DEF_PROPERTY(deepseek_rope_mscale, deepseek_rope_mscale_)                                                          \
    DEF_PROPERTY(deepseek_mscale_all_dim, deepseek_mscale_all_dim_)                                                    \
    DEF_PROPERTY(reverse_e_h_norm, reverse_e_h_norm_)                                                                  \
    DEF_PROPERTY(enable_eplb, enable_eplb_)                                                                            \
    DEF_PROPERTY(phy_exp_num, phy_exp_num_)                                                                            \
    DEF_PROPERTY(eplb_update_time, eplb_update_time_)                                                                  \
    DEF_PROPERTY(eplb_mode, eplb_mode_)                                                                                \
    DEF_PROPERTY(py_eplb, py_eplb_)

    pybind11::class_<GptInitParameter>(m, "GptInitParameter")
        .def(pybind11::init<int64_t,  // head_num
                            int64_t,  // size_per_head
                            int64_t,  // num_layers
                            int64_t,  // max_seq_len
                            int64_t,  // vocab_size
                            int64_t   // hidden_size
                            >(),
             py::arg("head_num"),
             py::arg("size_per_head"),
             py::arg("num_layers"),
             py::arg("max_seq_len"),
             py::arg("vocab_size"),
             py::arg("hidden_size"))
        .def("insertMultiTaskPromptTokens",
             &GptInitParameter::insertMultiTaskPromptTokens,
             py::arg("task_id"),
             py::arg("tokens_id"))
        .def("setLayerNormType", &GptInitParameter::setLayerNormType)
        .def("setNormType", &GptInitParameter::setNormType)
        .def("setActivationType", &GptInitParameter::setActivationType)
        .def("setTaskType", &GptInitParameter::setTaskType, py::arg("task"))
        .def("setDataType", &GptInitParameter::setDataType)
        .def("setKvCacheDataType", &GptInitParameter::setKvCacheDataType)
        .def("showDebugInfo", &GptInitParameter::showDebugInfo)
        .def("isGatedActivation", &GptInitParameter::isGatedActivation)
        .def("isKvCacheQuant", &GptInitParameter::isKvCacheQuant)
        // 新增配置结构体成员暴露
        .def_readwrite("parallelism_distributed_config", &GptInitParameter::parallelism_distributed_config)
        .def_readwrite("concurrency_config", &GptInitParameter::concurrency_config)
        .def_readwrite("fmha_config", &GptInitParameter::fmha_config)
        .def_readwrite("kv_cache_config", &GptInitParameter::kv_cache_config)
        .def_readwrite("profiling_debug_logging_config", &GptInitParameter::profiling_debug_logging_config)
        .def_readwrite("hw_kernel_config", &GptInitParameter::hw_kernel_config)
        .def_readwrite("device_resource_config", &GptInitParameter::device_resource_config)
        .def_readwrite("sampler_config", &GptInitParameter::sampler_config)
        .def_readwrite("moe_config", &GptInitParameter::moe_config)
        .def_readwrite("model_specific_config", &GptInitParameter::model_specific_config)
        .def_readwrite("sp_config", &GptInitParameter::sp_config)
        .def_readwrite("service_discovery_config", &GptInitParameter::service_discovery_config)
        .def_readwrite("cache_store_config", &GptInitParameter::cache_store_config)
        .def_readwrite("scheduler_config", &GptInitParameter::scheduler_config)
        .def_readwrite("batch_decode_scheduler_config", &GptInitParameter::batch_decode_scheduler_config)
        .def_readwrite("fifo_scheduler_config", &GptInitParameter::fifo_scheduler_config)
        .def_readwrite("misc_config", &GptInitParameter::misc_config)
        .def_readwrite("arpc_config", &GptInitParameter::arpc_config)
        .def_readwrite("grpc_config", &GptInitParameter::grpc_config)
        .def_readwrite("ffn_disaggregate_config", &GptInitParameter::ffn_disaggregate_config) REGISTER_PROPERTYS;
}

PYBIND11_MODULE(libth_transformer_config, m) {
    register_parallelism_distributed_config(m);
    register_concurrency_config(m);
    register_fmha_config(m);
    register_kvcache_config(m);
    register_profiling_debug_logging_config(m);
    register_hwkernel_config(m);
    register_device_resource_config(m);
    register_sampler_config(m);
    register_moe_config(m);
    register_model_specific_config(m);
    register_speculative_execution_config(m);
    register_service_discovery_config(m);
    register_cache_store_config(m);
    register_scheduler_config(m);
    register_batch_decode_scheduler_config(m);
    register_fifo_scheduler_config(m);
    register_misc_config(m);
    register_arpc_config(m);
    register_grpc_config(m);
    registerFMHAType(m);
    register_ffn_disaggregate_config(m);
    registerGptInitParameter(m);

    registerCommon(m);
}

}  // namespace rtp_llm
