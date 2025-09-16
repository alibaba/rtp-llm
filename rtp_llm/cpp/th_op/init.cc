#include <torch/library.h>
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"
#include "rtp_llm/cpp/dataclass/WorkerStatusInfo.h"
#include "rtp_llm/cpp/th_op/GptInitParameterRegister.h"
#include "rtp_llm/cpp/th_op/common/blockUtil.h"

#ifndef USE_FRONTEND
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#endif

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

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
        .def(pybind11::init<int, int, int, int, int, int, int, int>(),
             pybind11::arg("tp_size")          = 1,
             pybind11::arg("ep_size")          = 1,
             pybind11::arg("dp_size")          = 1,
             pybind11::arg("pp_size")          = 1,
             pybind11::arg("world_size")       = 1,
             pybind11::arg("world_rank")       = 0,
             pybind11::arg("local_world_size") = 1,
             pybind11::arg("ffn_sp_size")      = 1)
        .def("to_string", &ParallelismDistributedConfig::to_string)
        .def("update_from_env", &ParallelismDistributedConfig::update_from_env_for_test)
        .def_readwrite("tp_size", &ParallelismDistributedConfig::tp_size)
        .def_readwrite("ep_size", &ParallelismDistributedConfig::ep_size)
        .def_readwrite("dp_size", &ParallelismDistributedConfig::dp_size)
        .def_readwrite("pp_size", &ParallelismDistributedConfig::pp_size)
        .def_readwrite("world_size", &ParallelismDistributedConfig::world_size)
        .def_readwrite("world_rank", &ParallelismDistributedConfig::world_rank)
        .def_readwrite("local_world_size", &ParallelismDistributedConfig::local_world_size)
        .def_readwrite("ffn_sp_size", &ParallelismDistributedConfig::ffn_sp_size);
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
        .def(pybind11::init<bool, std::string, std::string, bool, int, int, int, int, int, int, int64_t, int64_t>(),
             pybind11::arg("reuse_cache")              = false,
             pybind11::arg("multi_task_prompt")        = "",
             pybind11::arg("multi_task_prompt_str")    = "",
             pybind11::arg("enable_3fs")               = false,
             pybind11::arg("match_timeout_ms")         = 1000,
             pybind11::arg("rpc_get_cache_timeout_ms") = 2000,
             pybind11::arg("rpc_put_cache_timeout_ms") = 2000,
             pybind11::arg("threefs_read_timeout_ms")  = 1000,
             pybind11::arg("threefs_write_timeout_ms") = 2000,
             pybind11::arg("max_block_size_per_item")  = 16,
             pybind11::arg("threefs_read_iov_size")    = 1LL << 32,
             pybind11::arg("threefs_write_iov_size")   = 1LL << 32)
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
        .def_readwrite("threefs_write_iov_size", &KVCacheConfig::threefs_write_iov_size);
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
             pybind11::arg("enable_detail_log")         = false)
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
        .def_readwrite("enable_detail_log", &ProfilingDebugLoggingConfig::enable_detail_log);
}

void register_hwkernel_config(pybind11::module& m) {
    pybind11::class_<HWKernelConfig>(m, "HWKernelConfig")
        .def(pybind11::init<int, bool, bool, bool, bool, std::string, bool, bool, bool, bool, int>(),
             pybind11::arg("deep_gemm_num_sm")             = -1,
             pybind11::arg("arm_gemm_use_kai")             = false,
             pybind11::arg("enable_stable_scatter_add")    = false,
             pybind11::arg("enable_multi_block_mode")      = true,
             pybind11::arg("ft_disable_custom_ar")         = true,
             pybind11::arg("rocm_hipblaslt_config")        = "gemm_config.csv",
             pybind11::arg("enable_cuda_graph")            = false,
             pybind11::arg("enable_cuda_graph_debug_mode") = false,
             pybind11::arg("use_aiter_pa")                 = true,
             pybind11::arg("enable_native_cuda_graph")     = false,
             pybind11::arg("num_native_cuda_graph")        = 200)
        .def("to_string", &HWKernelConfig::to_string)
        .def("update_from_env", &HWKernelConfig::update_from_env_for_test)
        .def_readwrite("deep_gemm_num_sm", &HWKernelConfig::deep_gemm_num_sm)
        .def_readwrite("arm_gemm_use_kai", &HWKernelConfig::arm_gemm_use_kai)
        .def_readwrite("enable_stable_scatter_add", &HWKernelConfig::enable_stable_scatter_add)
        .def_readwrite("enable_multi_block_mode", &HWKernelConfig::enable_multi_block_mode)
        .def_readwrite("ft_disable_custom_ar", &HWKernelConfig::ft_disable_custom_ar)
        .def_readwrite("rocm_hipblaslt_config", &HWKernelConfig::rocm_hipblaslt_config)
        .def_readwrite("enable_cuda_graph", &HWKernelConfig::enable_cuda_graph)
        .def_readwrite("enable_cuda_graph_debug_mode", &HWKernelConfig::enable_cuda_graph_debug_mode)
        .def_readwrite("use_aiter_pa", &HWKernelConfig::use_aiter_pa)
        .def_readwrite("enable_native_cuda_graph", &HWKernelConfig::enable_native_cuda_graph)
        .def_readwrite("num_native_cuda_graph", &HWKernelConfig::num_native_cuda_graph);
}

// DeviceResourceConfig
void register_device_resource_config(pybind11::module& m) {
    pybind11::class_<DeviceResourceConfig>(m, "DeviceResourceConfig")
        .def(pybind11::init<int64_t, int64_t, int, int, int, bool, int, bool>(),
             pybind11::arg("device_reserve_memory_bytes") = 0,
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
        .def(pybind11::init<bool>(), pybind11::arg("use_batch_decode_scheduler") = false)
        .def("to_string", &SchedulerConfig::to_string)
        .def("update_from_env", &SchedulerConfig::update_from_env_for_test)
        .def_readwrite("use_batch_decode_scheduler", &SchedulerConfig::use_batch_decode_scheduler);
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
        .def(pybind11::init<int, int, int, bool>(),
             pybind11::arg("load_balance")            = 0,
             pybind11::arg("step_records_time_range") = 60 * 1000 * 1000,
             pybind11::arg("step_records_max_size")   = 1000,
             pybind11::arg("disable_pdl")             = true)
        .def("to_string", &MiscellaneousConfig::to_string)
        .def("update_from_env", &MiscellaneousConfig::update_from_env_for_test)
        .def_readwrite("load_balance", &MiscellaneousConfig::load_balance)
        .def_readwrite("step_records_time_range", &MiscellaneousConfig::step_records_time_range)
        .def_readwrite("step_records_max_size", &MiscellaneousConfig::step_records_max_size)
        .def_readwrite("disable_pdl", &MiscellaneousConfig::disable_pdl);
}

// TODO(wangyin.yx): organize these regsiter function into classified registration functions

PYBIND11_MODULE(libth_transformer, m) {

    registerKvCacheInfo(m);
    registerWorkerStatusInfo(m);
    registerEngineScheduleInfo(m);
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
    registerFMHAType(m);
    register_ffn_disaggregate_config(m);
    registerGptInitParameter(m);

#ifndef USE_FRONTEND
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingHandler(m);
#endif

    registerDeviceOps(m);
    registerCommon(m);
    registerPyOpDefs(m);

    py::module rtp_ops_m = m.def_submodule("rtp_llm_ops", "rtp llm custom ops");
    registerPyModuleOps(rtp_ops_m);
}

}  // namespace rtp_llm
