#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "rtp_llm/cpp/pybind/common/blockUtil.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/config/SpecialTokens.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/model_utils/QuantInfo.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/layernorm_types.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace rtp_llm;

PYBIND11_MODULE(libth_transformer_config, m) {
    // Register get_block_cache_keys function
    registerCommon(m);

    // Register enums
    py::enum_<RoleType>(m, "RoleType")
        .value("PDFUSION", RoleType::PDFUSION)
        .value("PREFILL", RoleType::PREFILL)
        .value("DECODE", RoleType::DECODE)
        .value("VIT", RoleType::VIT)
        .value("FRONTEND", RoleType::FRONTEND);

    py::enum_<VitSeparation>(m, "VitSeparation")
        .value("VIT_SEPARATION_LOCAL", VitSeparation::VIT_SEPARATION_LOCAL)
        .value("VIT_SEPARATION_ROLE", VitSeparation::VIT_SEPARATION_ROLE)
        .value("VIT_SEPARATION_REMOTE", VitSeparation::VIT_SEPARATION_REMOTE);

    py::enum_<EplbMode>(m, "EplbMode")
        .value("NONE", EplbMode::NONE)
        .value("STATS", EplbMode::STATS)
        .value("EPLB", EplbMode::EPLB)
        .value("ALL", EplbMode::ALL);

    py::enum_<FMHAType>(m, "FMHAType")
        .value("FLASH_INFER", FMHAType::FLASH_INFER)
        .value("NONE", FMHAType::NONE)
        .value("OPEN_SOURCE", FMHAType::OPEN_SOURCE)
        .value("PAGED_OPEN_SOURCE", FMHAType::PAGED_OPEN_SOURCE)
        .value("PAGED_TRT_V2", FMHAType::PAGED_TRT_V2)
        .value("TRT_V1", FMHAType::TRT_V1)
        .value("TRT_V2", FMHAType::TRT_V2)
        .value("XQA", FMHAType::XQA)
        .value("AITER_PREFILL", FMHAType::AITER_PREFILL)
        .value("AITER_ASM_PREFILL", FMHAType::AITER_ASM_PREFILL)
        .value("AITER_DECODE", FMHAType::AITER_DECODE)
        .value("AITER_ASM_DECODE", FMHAType::AITER_ASM_DECODE)
        .value("PY_FLASHINFER_PREFILL", FMHAType::PY_FLASHINFER_PREFILL)
        .value("PY_FLASHINFER_DECODE", FMHAType::PY_FLASHINFER_DECODE);

    py::enum_<MlaOpsType>(m, "MlaOpsType")
        .value("AUTO", MlaOpsType::AUTO)
        .value("MHA", MlaOpsType::MHA)
        .value("FLASH_INFER", MlaOpsType::FLASH_INFER)
        .value("FLASH_MLA", MlaOpsType::FLASH_MLA);

    // Register LayerNormType enum
    py::enum_<LayerNormType>(m, "LayerNormType")
        .value("pre_layernorm", LayerNormType::pre_layernorm)
        .value("post_layernorm", LayerNormType::post_layernorm)
        .value("invalid_type", LayerNormType::invalid_type);

    // Register NormType enum
    py::enum_<NormType>(m, "NormType")
        .value("layernorm", NormType::layernorm)
        .value("rmsnorm", NormType::rmsnorm)
        .value("alphanorm", NormType::alphanorm)
        .value("add_bias", NormType::add_bias)
        .value("invalid_type", NormType::invalid_type);

    // Register ActivationType enum
    py::enum_<ActivationType>(m, "ActivationType")
        .value("Gelu", ActivationType::Gelu)
        .value("Relu", ActivationType::Relu)
        .value("Silu", ActivationType::Silu)
        .value("Swiglu", ActivationType::Swiglu)
        .value("Geglu", ActivationType::Geglu)
        .value("Identity", ActivationType::Identity)
        .value("GeluNoneApproximate", ActivationType::GeluNoneApproximate)
        .value("GeGluNoneApproximate", ActivationType::GeGluNoneApproximate)
        .value("Sigmoid", ActivationType::Sigmoid)
        .value("InvalidType", ActivationType::InvalidType);

    // Register TaskType enum
    py::enum_<TaskType>(m, "TaskType")
        .value("DENSE_EMBEDDING", TaskType::DENSE_EMBEDDING)
        .value("ALL_EMBEDDING", TaskType::ALL_EMBEDDING)
        .value("SPARSE_EMBEDDING", TaskType::SPARSE_EMBEDDING)
        .value("COLBERT_EMBEDDING", TaskType::COLBERT_EMBEDDING)
        .value("LANGUAGE_MODEL", TaskType::LANGUAGE_MODEL)
        .value("SEQ_CLASSIFICATION", TaskType::SEQ_CLASSIFICATION)
        .value("RERANKER", TaskType::RERANKER)
        .value("LINEAR_SOFTMAX", TaskType::LINEAR_SOFTMAX)
        .value("BGE_M3", TaskType::BGE_M3);

    // Register ArpcConfig
    py::class_<ArpcConfig>(m, "ArpcConfig")
        .def(py::init<>())
        .def_readwrite("threadNum", &ArpcConfig::threadNum)
        .def_readwrite("queueNum", &ArpcConfig::queueNum)
        .def_readwrite("ioThreadNum", &ArpcConfig::ioThreadNum)
        .def("to_string", &ArpcConfig::to_string)
        .def(py::pickle(
            [](const ArpcConfig& self) { return py::make_tuple(self.threadNum, self.queueNum, self.ioThreadNum); },
            [](py::tuple t) {
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");
                ArpcConfig c;
                try {
                    c.threadNum   = t[0].cast<int>();
                    c.queueNum    = t[1].cast<int>();
                    c.ioThreadNum = t[2].cast<int>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("ArpcConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    pybind11::class_<GrpcConfig>(m, "GrpcConfig")
        .def(pybind11::init<>())  // Default constructor
        .def(pybind11::init<const std::string&>(),
             pybind11::arg("json_str"))  // JSON string constructor
        .def("to_string", &GrpcConfig::to_string)
        .def("from_json", &GrpcConfig::from_json, "Initialize from JSON string")
        .def("get_client_config", &GrpcConfig::get_client_config)
        .def("get_server_config", &GrpcConfig::get_server_config)
        .def(py::pickle(
            [](const GrpcConfig& self) {
                // Convert maps to Python dicts for serialization
                py::dict client_dict;
                py::dict server_dict;
                auto     client_config = self.get_client_config();
                auto     server_config = self.get_server_config();
                for (const auto& pair : client_config) {
                    client_dict[py::str(pair.first)] = pair.second;
                }
                for (const auto& pair : server_config) {
                    server_dict[py::str(pair.first)] = pair.second;
                }
                return py::make_tuple(client_dict, server_dict);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                GrpcConfig c;
                try {
                    py::dict client_dict = t[0].cast<py::dict>();
                    py::dict server_dict = t[1].cast<py::dict>();

                    // Convert Python dicts to JSON string
                    std::ostringstream oss;
                    oss << "{\"client_config\": {";
                    bool first = true;
                    for (auto item : client_dict) {
                        if (!first)
                            oss << ", ";
                        first = false;
                        oss << "\"" << py::str(item.first).cast<std::string>() << "\": " << py::cast<int>(item.second);
                    }
                    oss << "}, \"server_config\": {";
                    first = true;
                    for (auto item : server_dict) {
                        if (!first)
                            oss << ", ";
                        first = false;
                        oss << "\"" << py::str(item.first).cast<std::string>() << "\": " << py::cast<int>(item.second);
                    }
                    oss << "}}";
                    c.from_json(oss.str());
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("GrpcConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register ConcurrencyConfig
    py::class_<ConcurrencyConfig>(m, "ConcurrencyConfig")
        .def(py::init<>())
        .def_readwrite("concurrency_with_block", &ConcurrencyConfig::concurrency_with_block)
        .def_readwrite("concurrency_limit", &ConcurrencyConfig::concurrency_limit)
        .def("to_string", &ConcurrencyConfig::to_string)
        .def(py::pickle(
            [](const ConcurrencyConfig& self) {
                return py::make_tuple(self.concurrency_with_block, self.concurrency_limit);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                ConcurrencyConfig c;
                try {
                    c.concurrency_with_block = t[0].cast<int>();
                    c.concurrency_limit      = t[1].cast<int>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("ConcurrencyConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register FMHAConfig
    py::class_<FMHAConfig>(m, "FMHAConfig")
        .def(py::init<>())
        .def_readwrite("enable_fmha", &FMHAConfig::enable_fmha)
        .def_readwrite("enable_trt_fmha", &FMHAConfig::enable_trt_fmha)
        .def_readwrite("enable_paged_trt_fmha", &FMHAConfig::enable_paged_trt_fmha)
        .def_readwrite("enable_open_source_fmha", &FMHAConfig::enable_open_source_fmha)
        .def_readwrite("enable_paged_open_source_fmha", &FMHAConfig::enable_paged_open_source_fmha)
        .def_readwrite("enable_trtv1_fmha", &FMHAConfig::enable_trtv1_fmha)
        .def_readwrite("disable_flash_infer", &FMHAConfig::disable_flash_infer)
        .def_readwrite("enable_xqa", &FMHAConfig::enable_xqa)
        .def_readwrite("use_aiter_pa", &FMHAConfig::use_aiter_pa)
        .def_readwrite("use_asm_pa", &FMHAConfig::use_asm_pa)
        .def_readwrite("absorb_opt_len", &FMHAConfig::absorb_opt_len)
        .def("to_string", &FMHAConfig::to_string)
        .def(py::pickle(
            [](const FMHAConfig& self) {
                return py::make_tuple(self.enable_fmha,
                                      self.enable_trt_fmha,
                                      self.enable_paged_trt_fmha,
                                      self.enable_open_source_fmha,
                                      self.enable_paged_open_source_fmha,
                                      self.enable_trtv1_fmha,
                                      self.disable_flash_infer,
                                      self.enable_xqa,
                                      self.use_aiter_pa,
                                      self.use_asm_pa,
                                      self.absorb_opt_len);
            },
            [](py::tuple t) {
                if (t.size() != 11)
                    throw std::runtime_error("Invalid state!");
                FMHAConfig c;
                try {
                    c.enable_fmha                   = t[0].cast<bool>();
                    c.enable_trt_fmha               = t[1].cast<bool>();
                    c.enable_paged_trt_fmha         = t[2].cast<bool>();
                    c.enable_open_source_fmha       = t[3].cast<bool>();
                    c.enable_paged_open_source_fmha = t[4].cast<bool>();
                    c.enable_trtv1_fmha             = t[5].cast<bool>();
                    c.disable_flash_infer           = t[6].cast<bool>();
                    c.enable_xqa                    = t[7].cast<bool>();
                    c.use_aiter_pa                  = t[8].cast<bool>();
                    c.use_asm_pa                    = t[9].cast<bool>();
                    c.absorb_opt_len                = t[10].cast<int64_t>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("FMHAConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register KVCacheConfig
    py::class_<KVCacheConfig>(m, "KVCacheConfig")
        .def(py::init<>())
        .def_readwrite("reuse_cache", &KVCacheConfig::reuse_cache)
        .def_readwrite("multi_task_prompt", &KVCacheConfig::multi_task_prompt)
        .def_readwrite("multi_task_prompt_str", &KVCacheConfig::multi_task_prompt_str)
        .def_readwrite("multi_task_prompt_tokens", &KVCacheConfig::multi_task_prompt_tokens)
        .def_readwrite("reserve_block_ratio", &KVCacheConfig::reserve_block_ratio)
        .def_readwrite("enable_3fs", &KVCacheConfig::enable_3fs)
        .def_readwrite("match_timeout_ms", &KVCacheConfig::match_timeout_ms)
        .def_readwrite("rpc_get_cache_timeout_ms", &KVCacheConfig::rpc_get_cache_timeout_ms)
        .def_readwrite("rpc_put_cache_timeout_ms", &KVCacheConfig::rpc_put_cache_timeout_ms)
        .def_readwrite("threefs_read_timeout_ms", &KVCacheConfig::threefs_read_timeout_ms)
        .def_readwrite("threefs_write_timeout_ms", &KVCacheConfig::threefs_write_timeout_ms)
        .def_readwrite("max_block_size_per_item", &KVCacheConfig::max_block_size_per_item)
        .def_readwrite("threefs_read_iov_size", &KVCacheConfig::threefs_read_iov_size)
        .def_readwrite("threefs_write_iov_size", &KVCacheConfig::threefs_write_iov_size)
        .def_readwrite("memory_cache_size_mb", &KVCacheConfig::memory_cache_size_mb)
        .def_readwrite("memory_cache_sync_timeout_ms", &KVCacheConfig::memory_cache_sync_timeout_ms)
        .def_readwrite("int8_kv_cache", &KVCacheConfig::int8_kv_cache)
        .def_readwrite("fp8_kv_cache", &KVCacheConfig::fp8_kv_cache)
        .def_readwrite("kv_cache_mem_mb", &KVCacheConfig::kv_cache_mem_mb)
        .def_readwrite("seq_size_per_block", &KVCacheConfig::seq_size_per_block)
        .def_readwrite("test_block_num", &KVCacheConfig::test_block_num)
        .def_readwrite("use_block_cache", &KVCacheConfig::use_block_cache)
        .def_readwrite("enable_device_cache", &KVCacheConfig::enable_device_cache)
        .def_readwrite("enable_memory_cache", &KVCacheConfig::enable_memory_cache)
        .def("insertMultiTaskPromptTokens", &KVCacheConfig::insertMultiTaskPromptTokens)
        .def("to_string", &KVCacheConfig::to_string)
        .def(py::pickle(
            [](const KVCacheConfig& self) {
                return py::make_tuple(self.reuse_cache,
                                      self.multi_task_prompt,
                                      self.multi_task_prompt_str,
                                      self.multi_task_prompt_tokens,
                                      self.reserve_block_ratio,
                                      self.enable_3fs,
                                      self.match_timeout_ms,
                                      self.rpc_get_cache_timeout_ms,
                                      self.rpc_put_cache_timeout_ms,
                                      self.threefs_read_timeout_ms,
                                      self.threefs_write_timeout_ms,
                                      self.max_block_size_per_item,
                                      self.threefs_read_iov_size,
                                      self.threefs_write_iov_size,
                                      self.memory_cache_size_mb,
                                      self.memory_cache_sync_timeout_ms,
                                      self.int8_kv_cache,
                                      self.fp8_kv_cache,
                                      self.kv_cache_mem_mb,
                                      self.seq_size_per_block,
                                      self.test_block_num,
                                      self.use_block_cache,
                                      self.enable_device_cache,
                                      self.enable_memory_cache);
            },
            [](py::tuple t) {
                if (t.size() != 24)
                    throw std::runtime_error("Invalid state!");
                KVCacheConfig c;
                try {
                    c.reuse_cache                  = t[0].cast<bool>();
                    c.multi_task_prompt            = t[1].cast<std::string>();
                    c.multi_task_prompt_str        = t[2].cast<std::string>();
                    c.multi_task_prompt_tokens     = t[3].cast<std::map<std::string, std::vector<int>>>();
                    c.reserve_block_ratio          = t[4].cast<int64_t>();
                    c.enable_3fs                   = t[5].cast<bool>();
                    c.match_timeout_ms             = t[6].cast<int>();
                    c.rpc_get_cache_timeout_ms     = t[7].cast<int>();
                    c.rpc_put_cache_timeout_ms     = t[8].cast<int>();
                    c.threefs_read_timeout_ms      = t[9].cast<int>();
                    c.threefs_write_timeout_ms     = t[10].cast<int>();
                    c.max_block_size_per_item      = t[11].cast<int>();
                    c.threefs_read_iov_size        = t[12].cast<int64_t>();
                    c.threefs_write_iov_size       = t[13].cast<int64_t>();
                    c.memory_cache_size_mb         = t[14].cast<int64_t>();
                    c.memory_cache_sync_timeout_ms = t[15].cast<int64_t>();
                    c.int8_kv_cache                = t[16].cast<int>();
                    c.fp8_kv_cache                 = t[17].cast<int>();
                    c.kv_cache_mem_mb              = t[18].cast<int64_t>();
                    c.seq_size_per_block           = t[19].cast<int>();
                    c.test_block_num               = t[20].cast<int>();
                    c.use_block_cache              = t[21].cast<int>();
                    c.enable_device_cache          = t[22].cast<bool>();
                    c.enable_memory_cache          = t[23].cast<bool>();

                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("KVCacheConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register ProfilingDebugLoggingConfig
    py::class_<ProfilingDebugLoggingConfig>(m, "ProfilingDebugLoggingConfig")
        .def(py::init<>())
        .def_readwrite("trace_memory", &ProfilingDebugLoggingConfig::trace_memory)
        .def_readwrite("trace_malloc_stack", &ProfilingDebugLoggingConfig::trace_malloc_stack)
        .def_readwrite("enable_device_perf", &ProfilingDebugLoggingConfig::enable_device_perf)
        .def_readwrite("ft_core_dump_on_exception", &ProfilingDebugLoggingConfig::ft_core_dump_on_exception)
        .def_readwrite("ft_alog_conf_path", &ProfilingDebugLoggingConfig::ft_alog_conf_path)
        .def_readwrite("gen_timeline_sync", &ProfilingDebugLoggingConfig::gen_timeline_sync)
        .def_readwrite("torch_cuda_profiler_dir", &ProfilingDebugLoggingConfig::torch_cuda_profiler_dir)
        .def_readwrite("log_file_backup_count", &ProfilingDebugLoggingConfig::log_file_backup_count)
        .def_readwrite("debug_load_server", &ProfilingDebugLoggingConfig::debug_load_server)
        .def_readwrite("hack_layer_num", &ProfilingDebugLoggingConfig::hack_layer_num)
        .def_readwrite("debug_start_fake_process", &ProfilingDebugLoggingConfig::debug_start_fake_process)
        .def_readwrite("enable_detail_log", &ProfilingDebugLoggingConfig::enable_detail_log)
        .def_readwrite("check_nan", &ProfilingDebugLoggingConfig::check_nan)
        .def_readwrite("enable_torch_alloc_profile", &ProfilingDebugLoggingConfig::enable_torch_alloc_profile)
        .def("to_string", &ProfilingDebugLoggingConfig::to_string)
        .def(py::pickle(
            [](const ProfilingDebugLoggingConfig& self) {
                return py::make_tuple(self.trace_memory,
                                      self.trace_malloc_stack,
                                      self.enable_device_perf,
                                      self.ft_core_dump_on_exception,
                                      self.ft_alog_conf_path,
                                      self.gen_timeline_sync,
                                      self.torch_cuda_profiler_dir,
                                      self.log_file_backup_count,
                                      self.debug_load_server,
                                      self.hack_layer_num,
                                      self.debug_start_fake_process,
                                      self.enable_detail_log,
                                      self.check_nan,
                                      self.enable_torch_alloc_profile);
            },
            [](py::tuple t) {
                if (t.size() != 14)
                    throw std::runtime_error("Invalid state!");
                ProfilingDebugLoggingConfig c;
                try {
                    c.trace_memory               = t[0].cast<bool>();
                    c.trace_malloc_stack         = t[1].cast<bool>();
                    c.enable_device_perf         = t[2].cast<bool>();
                    c.ft_core_dump_on_exception  = t[3].cast<bool>();
                    c.ft_alog_conf_path          = t[4].cast<std::string>();
                    c.gen_timeline_sync          = t[5].cast<bool>();
                    c.torch_cuda_profiler_dir    = t[6].cast<std::string>();
                    c.log_file_backup_count      = t[7].cast<int>();
                    c.debug_load_server          = t[8].cast<bool>();
                    c.hack_layer_num             = t[9].cast<int>();
                    c.debug_start_fake_process   = t[10].cast<bool>();
                    c.enable_detail_log          = t[11].cast<bool>();
                    c.check_nan                  = t[12].cast<bool>();
                    c.enable_torch_alloc_profile = t[13].cast<bool>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("ProfilingDebugLoggingConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register HWKernelConfig
    py::class_<HWKernelConfig>(m, "HWKernelConfig")
        .def(py::init<>())
        .def_readwrite("deep_gemm_num_sm", &HWKernelConfig::deep_gemm_num_sm)
        .def_readwrite("arm_gemm_use_kai", &HWKernelConfig::arm_gemm_use_kai)
        .def_readwrite("enable_stable_scatter_add", &HWKernelConfig::enable_stable_scatter_add)
        .def_readwrite("enable_multi_block_mode", &HWKernelConfig::enable_multi_block_mode)
        .def_readwrite("ft_disable_custom_ar", &HWKernelConfig::ft_disable_custom_ar)
        .def_readwrite("rocm_hipblaslt_config", &HWKernelConfig::rocm_hipblaslt_config)
        .def_readwrite("use_swizzleA", &HWKernelConfig::use_swizzleA)
        .def_readwrite("enable_cuda_graph", &HWKernelConfig::enable_cuda_graph)
        .def_readwrite("enable_cuda_graph_debug_mode", &HWKernelConfig::enable_cuda_graph_debug_mode)
        .def_readwrite("enable_native_cuda_graph", &HWKernelConfig::enable_native_cuda_graph)
        .def_readwrite("num_native_cuda_graph", &HWKernelConfig::num_native_cuda_graph)
        .def_readwrite("prefill_capture_seq_lens", &HWKernelConfig::prefill_capture_seq_lens)
        .def_readwrite("decode_capture_batch_sizes", &HWKernelConfig::decode_capture_batch_sizes)
        .def_readwrite("disable_dpc_random", &HWKernelConfig::disable_dpc_random)
        .def_readwrite("rocm_disable_custom_ag", &HWKernelConfig::rocm_disable_custom_ag)
        .def("to_string", &HWKernelConfig::to_string)
        .def(py::pickle(
            [](const HWKernelConfig& self) {
                return py::make_tuple(self.deep_gemm_num_sm,
                                      self.arm_gemm_use_kai,
                                      self.enable_stable_scatter_add,
                                      self.enable_multi_block_mode,
                                      self.ft_disable_custom_ar,
                                      self.rocm_hipblaslt_config,
                                      self.use_swizzleA,
                                      self.enable_cuda_graph,
                                      self.enable_cuda_graph_debug_mode,
                                      self.enable_native_cuda_graph,
                                      self.num_native_cuda_graph,
                                      self.prefill_capture_seq_lens,
                                      self.decode_capture_batch_sizes,
                                      self.disable_dpc_random,
                                      self.rocm_disable_custom_ag);
            },
            [](py::tuple t) {
                if (t.size() != 15)
                    throw std::runtime_error("Invalid state!");
                HWKernelConfig c;
                try {
                    c.deep_gemm_num_sm             = t[0].cast<int>();
                    c.arm_gemm_use_kai             = t[1].cast<bool>();
                    c.enable_stable_scatter_add    = t[2].cast<bool>();
                    c.enable_multi_block_mode      = t[3].cast<bool>();
                    c.ft_disable_custom_ar         = t[4].cast<bool>();
                    c.rocm_hipblaslt_config        = t[5].cast<std::string>();
                    c.use_swizzleA                 = t[6].cast<bool>();
                    c.enable_cuda_graph            = t[7].cast<bool>();
                    c.enable_cuda_graph_debug_mode = t[8].cast<bool>();
                    c.enable_native_cuda_graph     = t[9].cast<bool>();
                    c.num_native_cuda_graph        = t[10].cast<int>();
                    c.prefill_capture_seq_lens     = t[11].cast<std::vector<int>>();
                    c.decode_capture_batch_sizes   = t[12].cast<std::vector<int>>();
                    c.disable_dpc_random           = t[13].cast<bool>();
                    c.rocm_disable_custom_ag       = t[14].cast<bool>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("HWKernelConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register DeviceResourceConfig
    py::class_<DeviceResourceConfig>(m, "DeviceResourceConfig")
        .def(py::init<>())
        .def_readwrite("device_reserve_memory_bytes", &DeviceResourceConfig::device_reserve_memory_bytes)
        .def_readwrite("host_reserve_memory_bytes", &DeviceResourceConfig::host_reserve_memory_bytes)
        .def_readwrite("overlap_math_sm_count", &DeviceResourceConfig::overlap_math_sm_count)
        .def_readwrite("overlap_comm_type", &DeviceResourceConfig::overlap_comm_type)
        .def_readwrite("m_split", &DeviceResourceConfig::m_split)
        .def_readwrite("enable_comm_overlap", &DeviceResourceConfig::enable_comm_overlap)
        .def_readwrite("enable_layer_micro_batch", &DeviceResourceConfig::enable_layer_micro_batch)
        .def("to_string", &DeviceResourceConfig::to_string)
        .def(py::pickle(
            [](const DeviceResourceConfig& self) {
                return py::make_tuple(self.device_reserve_memory_bytes,
                                      self.host_reserve_memory_bytes,
                                      self.overlap_math_sm_count,
                                      self.overlap_comm_type,
                                      self.m_split,
                                      self.enable_comm_overlap,
                                      self.enable_layer_micro_batch);
            },
            [](py::tuple t) {
                if (t.size() != 7)
                    throw std::runtime_error("Invalid state!");
                DeviceResourceConfig c;
                try {
                    c.device_reserve_memory_bytes = t[0].cast<int64_t>();
                    c.host_reserve_memory_bytes   = t[1].cast<int64_t>();
                    c.overlap_math_sm_count       = t[2].cast<int>();
                    c.overlap_comm_type           = t[3].cast<int>();
                    c.m_split                     = t[4].cast<int>();
                    c.enable_comm_overlap         = t[5].cast<bool>();
                    c.enable_layer_micro_batch    = t[6].cast<int>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("DeviceResourceConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register MoeConfig
    py::class_<MoeConfig>(m, "MoeConfig")
        .def(py::init<>())
        .def_readwrite("use_deepep_moe", &MoeConfig::use_deepep_moe)
        .def_readwrite("use_deepep_internode", &MoeConfig::use_deepep_internode)
        .def_readwrite("use_deepep_low_latency", &MoeConfig::use_deepep_low_latency)
        .def_readwrite("use_deepep_p2p_low_latency", &MoeConfig::use_deepep_p2p_low_latency)
        .def_readwrite("fake_balance_expert", &MoeConfig::fake_balance_expert)
        .def_readwrite("hack_moe_expert", &MoeConfig::hack_moe_expert)
        .def_readwrite("deep_ep_num_sm", &MoeConfig::deep_ep_num_sm)
        .def_readwrite("max_moe_normal_masked_token_num", &MoeConfig::max_moe_normal_masked_token_num)
        .def_readwrite("use_all_gather", &MoeConfig::use_all_gather)
        .def("to_string", &MoeConfig::to_string)
        .def(py::pickle(
            [](const MoeConfig& self) {
                return py::make_tuple(self.use_deepep_moe,
                                      self.use_deepep_internode,
                                      self.use_deepep_low_latency,
                                      self.use_deepep_p2p_low_latency,
                                      self.fake_balance_expert,
                                      self.hack_moe_expert,
                                      self.deep_ep_num_sm,
                                      self.max_moe_normal_masked_token_num,
                                      self.use_all_gather);
            },
            [](py::tuple t) {
                if (t.size() != 9)
                    throw std::runtime_error("Invalid state!");
                MoeConfig c;
                try {
                    c.use_deepep_moe                  = t[0].cast<bool>();
                    c.use_deepep_internode            = t[1].cast<bool>();
                    c.use_deepep_low_latency          = t[2].cast<bool>();
                    c.use_deepep_p2p_low_latency      = t[3].cast<bool>();
                    c.fake_balance_expert             = t[4].cast<bool>();
                    c.hack_moe_expert                 = t[5].cast<bool>();
                    c.deep_ep_num_sm                  = t[6].cast<int>();
                    c.max_moe_normal_masked_token_num = t[7].cast<int>();
                    c.use_all_gather                  = t[8].cast<bool>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("MoeConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register ModelSpecificConfig
    py::class_<ModelSpecificConfig>(m, "ModelSpecificConfig")
        .def(py::init<>())
        .def_readwrite("max_lora_model_size", &ModelSpecificConfig::max_lora_model_size)
        .def_readwrite("load_python_model", &ModelSpecificConfig::load_python_model)
        .def("to_string", &ModelSpecificConfig::to_string)
        .def(py::pickle(
            [](const ModelSpecificConfig& self) {
                return py::make_tuple(self.max_lora_model_size, self.load_python_model);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                ModelSpecificConfig c;
                try {
                    c.max_lora_model_size = t[0].cast<int64_t>();
                    c.load_python_model   = t[1].cast<bool>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("ModelSpecificConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // LinearAttentionConfig
    pybind11::class_<LinearAttentionConfig>(m, "LinearAttentionConfig")
        .def(pybind11::init<int, int, int, int, int>(),
             pybind11::arg("linear_conv_kernel_dim") = 0,
             pybind11::arg("linear_key_head_dim")    = 0,
             pybind11::arg("linear_num_key_heads")   = 0,
             pybind11::arg("linear_num_value_heads") = 0,
             pybind11::arg("linear_value_head_dim")  = 0)
        .def("to_string", &LinearAttentionConfig::to_string)
        .def_readwrite("linear_conv_kernel_dim", &LinearAttentionConfig::linear_conv_kernel_dim)
        .def_readwrite("linear_key_head_dim", &LinearAttentionConfig::linear_key_head_dim)
        .def_readwrite("linear_num_key_heads", &LinearAttentionConfig::linear_num_key_heads)
        .def_readwrite("linear_num_value_heads", &LinearAttentionConfig::linear_num_value_heads)
        .def_readwrite("linear_value_head_dim", &LinearAttentionConfig::linear_value_head_dim);

    // HybridAttentionConfig
    py::enum_<HybridAttentionType>(m, "HybridAttentionType")
        .value("NONE", HybridAttentionType::NONE)
        .value("LINEAR", HybridAttentionType::LINEAR)
        .value("SLIDING_WINDOW", HybridAttentionType::SLIDING_WINDOW);

    pybind11::class_<HybridAttentionConfig>(m, "HybridAttentionConfig")
        .def(pybind11::init<bool, std::vector<HybridAttentionType>>(),
             pybind11::arg("enable_hybrid_attention") = false,
             pybind11::arg("hybrid_attention_types")  = std::vector<HybridAttentionType>{})
        .def("to_string", &HybridAttentionConfig::to_string)
        .def_readwrite("enable_hybrid_attention", &HybridAttentionConfig::enable_hybrid_attention)
        .def_readwrite("hybrid_attention_types", &HybridAttentionConfig::hybrid_attention_types);

    // Register SpeculativeType enum
    py::enum_<SpeculativeType>(m, "SpeculativeType")
        .value("NONE", SP_TYPE_NONE)
        .value("VANILLA", SP_TYPE_VANILLA)
        .value("MTP", SP_TYPE_MTP)
        .value("EAGLE3", SP_TYPE_EAGLE3)
        .value("EAGLE", SP_TYPE_EAGLE)
        .value("DETERMINISTIC", SP_TYPE_DETERMINISTIC);

    // Register SpeculativeExecutionConfig
    py::class_<SpeculativeExecutionConfig>(m, "SpeculativeExecutionConfig")
        .def(py::init<>())
        .def_readwrite("model_type", &SpeculativeExecutionConfig::model_type)
        .def_property(
            "type",
            [](const SpeculativeExecutionConfig& self) { return self.type; },
            [](SpeculativeExecutionConfig& self, py::object value) {
                if (py::isinstance<py::str>(value)) {
                    self.type = SpeculativeExecutionConfig::from_string(value.cast<std::string>());
                } else if (py::isinstance<SpeculativeType>(value)) {
                    self.type = value.cast<SpeculativeType>();
                } else {
                    throw py::type_error("type must be a string or SpeculativeType enum");
                }
            })
        .def_readwrite("sp_min_token_match", &SpeculativeExecutionConfig::sp_min_token_match)
        .def_readwrite("sp_max_token_match", &SpeculativeExecutionConfig::sp_max_token_match)
        .def_readwrite("tree_decode_config", &SpeculativeExecutionConfig::tree_decode_config)
        .def_readwrite("gen_num_per_cycle", &SpeculativeExecutionConfig::gen_num_per_cycle)
        .def_readwrite("force_stream_sample", &SpeculativeExecutionConfig::force_stream_sample)
        .def_readwrite("force_score_context_attention", &SpeculativeExecutionConfig::force_score_context_attention)
        .def_readwrite("quantization", &SpeculativeExecutionConfig::quantization)
        .def_readwrite("checkpoint_path", &SpeculativeExecutionConfig::checkpoint_path)
        .def_readwrite("use_new_sp_engine", &SpeculativeExecutionConfig::use_new_sp_engine)
        .def("to_string", [](const SpeculativeExecutionConfig& self) { return self.to_string(); })
        .def(py::pickle(
            [](const SpeculativeExecutionConfig& self) {
                return py::make_tuple(self.model_type,
                                      SpeculativeExecutionConfig::to_string(self.type),
                                      self.sp_min_token_match,
                                      self.sp_max_token_match,
                                      self.tree_decode_config,
                                      self.gen_num_per_cycle,
                                      self.force_stream_sample,
                                      self.force_score_context_attention,
                                      self.quantization,
                                      self.checkpoint_path,
                                      self.use_new_sp_engine);
            },
            [](py::tuple t) {
                if (t.size() != 11)
                    throw std::runtime_error("Invalid state!");
                SpeculativeExecutionConfig c;
                try {
                    c.model_type                    = t[0].cast<std::string>();
                    c.type                          = SpeculativeExecutionConfig::from_string(t[1].cast<std::string>());
                    c.sp_min_token_match            = t[2].cast<int64_t>();
                    c.sp_max_token_match            = t[3].cast<int64_t>();
                    c.tree_decode_config            = t[4].cast<std::string>();
                    c.gen_num_per_cycle             = t[5].cast<int64_t>();
                    c.force_stream_sample           = t[6].cast<bool>();
                    c.force_score_context_attention = t[7].cast<bool>();
                    c.quantization                  = t[8].cast<std::string>();
                    c.checkpoint_path               = t[9].cast<std::string>();
                    c.use_new_sp_engine             = t[10].cast<bool>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("SpeculativeExecutionConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register CacheStoreConfig
    py::class_<CacheStoreConfig>(m, "CacheStoreConfig")
        .def(py::init<>())
        .def_readwrite("cache_store_rdma_mode", &CacheStoreConfig::cache_store_rdma_mode)
        .def_readwrite("wrr_available_ratio", &CacheStoreConfig::wrr_available_ratio)
        .def_readwrite("rank_factor", &CacheStoreConfig::rank_factor)
        .def_readwrite("thread_count", &CacheStoreConfig::thread_count)
        .def_readwrite("rdma_connect_timeout_ms", &CacheStoreConfig::rdma_connect_timeout_ms)
        .def_readwrite("rdma_qp_count_per_connection", &CacheStoreConfig::rdma_qp_count_per_connection)
        .def_readwrite("rdma_io_thread_count", &CacheStoreConfig::rdma_io_thread_count)
        .def_readwrite("rdma_worker_thread_count", &CacheStoreConfig::rdma_worker_thread_count)
        .def_readwrite("messager_io_thread_count", &CacheStoreConfig::messager_io_thread_count)
        .def_readwrite("messager_worker_thread_count", &CacheStoreConfig::messager_worker_thread_count)
        .def("to_string", &CacheStoreConfig::to_string)
        .def(py::pickle(
            [](const CacheStoreConfig& self) {
                return py::make_tuple(self.cache_store_rdma_mode,
                                      self.wrr_available_ratio,
                                      self.rank_factor,
                                      self.thread_count,
                                      self.rdma_connect_timeout_ms,
                                      self.rdma_qp_count_per_connection,
                                      self.rdma_io_thread_count,
                                      self.rdma_worker_thread_count,
                                      self.messager_io_thread_count,
                                      self.messager_worker_thread_count);
            },
            [](py::tuple t) {
                if (t.size() != 10)
                    throw std::runtime_error("Invalid state!");
                CacheStoreConfig c;
                try {
                    c.cache_store_rdma_mode        = t[0].cast<bool>();
                    c.wrr_available_ratio          = t[1].cast<int>();
                    c.rank_factor                  = t[2].cast<int>();
                    c.thread_count                 = t[3].cast<int>();
                    c.rdma_connect_timeout_ms      = t[4].cast<int>();
                    c.rdma_qp_count_per_connection = t[5].cast<int>();
                    c.rdma_io_thread_count         = t[6].cast<int>();
                    c.rdma_worker_thread_count     = t[7].cast<int>();
                    c.messager_io_thread_count     = t[8].cast<int>();
                    c.messager_worker_thread_count = t[9].cast<int>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("CacheStoreConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register MiscellaneousConfig
    py::class_<MiscellaneousConfig>(m, "MiscellaneousConfig")
        .def(py::init<>())
        .def_readwrite("disable_pdl", &MiscellaneousConfig::disable_pdl)
        .def_property(
            "aux_string",
            [](const MiscellaneousConfig& self) { return self.aux_string; },
            [](MiscellaneousConfig& self, const std::string& value) { self.aux_string = value; })
        .def("to_string", &MiscellaneousConfig::to_string)
        .def(py::pickle(
            [](const MiscellaneousConfig& self) { return py::make_tuple(self.disable_pdl, self.aux_string); },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                MiscellaneousConfig c;
                try {
                    c.disable_pdl = t[0].cast<bool>();
                    c.aux_string  = t[1].cast<std::string>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("MiscellaneousConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register FfnDisAggregateConfig
    py::class_<FfnDisAggregateConfig>(m, "FfnDisAggregateConfig")
        .def(py::init<>())
        .def_readwrite("enable_ffn_disaggregate", &FfnDisAggregateConfig::enable_ffn_disaggregate)
        .def_readwrite("attention_tp_size", &FfnDisAggregateConfig::attention_tp_size)
        .def_readwrite("attention_dp_size", &FfnDisAggregateConfig::attention_dp_size)
        .def_readwrite("ffn_tp_size", &FfnDisAggregateConfig::ffn_tp_size)
        .def_readwrite("ffn_dp_size", &FfnDisAggregateConfig::ffn_dp_size)
        .def_readwrite("is_ffn_rank", &FfnDisAggregateConfig::is_ffn_rank)
        .def("to_string", &FfnDisAggregateConfig::to_string)
        .def("is_ffn_service", &FfnDisAggregateConfig::is_ffn_service)
        .def(py::pickle(
            [](const FfnDisAggregateConfig& self) {
                return py::make_tuple(self.enable_ffn_disaggregate,
                                      self.attention_tp_size,
                                      self.attention_dp_size,
                                      self.ffn_tp_size,
                                      self.ffn_dp_size,
                                      self.is_ffn_rank);
            },
            [](py::tuple t) {
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");
                FfnDisAggregateConfig c;
                try {
                    c.enable_ffn_disaggregate = t[0].cast<bool>();
                    c.attention_tp_size       = t[1].cast<int>();
                    c.attention_dp_size       = t[2].cast<int>();
                    c.ffn_tp_size             = t[3].cast<int>();
                    c.ffn_dp_size             = t[4].cast<int>();
                    c.is_ffn_rank             = t[5].cast<bool>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("FfnDisAggregateConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register SpecialTokens
    py::class_<RoleSpecialTokens>(m, "RoleSpecialTokens")
        .def(py::init<>())
        .def_readwrite("token_ids", &RoleSpecialTokens::token_ids)
        .def_readwrite("eos_token_ids", &RoleSpecialTokens::eos_token_ids);

    py::class_<SpecialTokens>(m, "SpecialTokens")
        .def(py::init<>())
        .def_readwrite("bos_token_id", &SpecialTokens::bos_token_id)
        .def_readwrite("eos_token_id", &SpecialTokens::eos_token_id)
        .def_readwrite("pad_token_id", &SpecialTokens::pad_token_id)
        .def_readwrite("decoder_start_token_id", &SpecialTokens::decoder_start_token_id)
        .def_readwrite("user", &SpecialTokens::user)
        .def_readwrite("assistant", &SpecialTokens::assistant)
        .def_readwrite("system", &SpecialTokens::system)
        .def_readwrite("stop_words_id_list", &SpecialTokens::stop_words_id_list)
        .def_readwrite("stop_words_str_list", &SpecialTokens::stop_words_str_list);

    // Register QuantMethod enum
    py::enum_<QuantMethod>(m, "QuantMethod")
        .value("None", QuantMethod::None)
        .value("WeightOnlyPerCol", QuantMethod::WeightOnlyPerCol)
        .value("GptQ", QuantMethod::GptQ)
        .value("Awq", QuantMethod::Awq)
        .value("SmoothQuant", QuantMethod::SmoothQuant)
        .value("OmniQuant", QuantMethod::OmniQuant)
        .value("PerTensorQuant", QuantMethod::PerTensorQuant)
        .value("FP8Quant", QuantMethod::FP8Quant)
        .value("FP8PTPC", QuantMethod::FP8PTPC);

    // Register QuantAlgo
    py::class_<QuantAlgo>(m, "QuantAlgo")
        .def(py::init<>())
        .def(py::init<QuantMethod, int, int>(), py::arg("method"), py::arg("bits"), py::arg("group_size"))
        .def("isWeightOnlyPerCol", &QuantAlgo::isWeightOnlyPerCol)
        .def("isPerTensorQuant", &QuantAlgo::isPerTensorQuant)
        .def("isGptq", &QuantAlgo::isGptq)
        .def("isAwq", &QuantAlgo::isAwq)
        .def("isSmoothQuant", &QuantAlgo::isSmoothQuant)
        .def("isOmniQuant", &QuantAlgo::isOmniQuant)
        .def("isFp8", &QuantAlgo::isFp8)
        .def("isFp8PTPC", &QuantAlgo::isFp8PTPC)
        .def("isQuant", &QuantAlgo::isQuant)
        .def("isGroupwise", &QuantAlgo::isGroupwise)
        .def("getQuantMethod", &QuantAlgo::getQuantMethod)
        .def("getGroupSize", &QuantAlgo::getGroupSize)
        .def("getWeightBits", &QuantAlgo::getWeightBits)
        .def("getActivationBits", &QuantAlgo::getActivationBits)
        .def("setQuantAlgo", &QuantAlgo::setQuantAlgo);

    // Register ParallelismConfig
    py::class_<ParallelismConfig>(m, "ParallelismConfig")
        .def(py::init<>())
        .def_readwrite("tp_size", &ParallelismConfig::tp_size)
        .def_readwrite("ep_size", &ParallelismConfig::ep_size)
        .def_readwrite("dp_size", &ParallelismConfig::dp_size)
        .def_readwrite("pp_size", &ParallelismConfig::pp_size)
        .def_readwrite("world_size", &ParallelismConfig::world_size)
        .def_readwrite("world_rank", &ParallelismConfig::world_rank)
        .def_readwrite("local_world_size", &ParallelismConfig::local_world_size)
        .def_readwrite("local_rank", &ParallelismConfig::local_rank)
        .def_readwrite("ffn_sp_size", &ParallelismConfig::ffn_sp_size)
        .def_readwrite("tp_rank", &ParallelismConfig::tp_rank)
        .def_readwrite("ep_rank", &ParallelismConfig::ep_rank)
        .def_readwrite("dp_rank", &ParallelismConfig::dp_rank)
        .def_readwrite("ffn_tp_size", &ParallelismConfig::ffn_tp_size)
        .def_readwrite("ffn_tp_rank", &ParallelismConfig::ffn_tp_rank)
        .def_readwrite("enable_sp", &ParallelismConfig::enable_sp)
        .def_readwrite("nccl_ip", &ParallelismConfig::nccl_ip)
        .def_readwrite("tp_nccl_port", &ParallelismConfig::tp_nccl_port)
        .def_readwrite("dp_tp_nccl_port", &ParallelismConfig::dp_tp_nccl_port)
        .def_readwrite("ffn_tp_nccl_port", &ParallelismConfig::ffn_tp_nccl_port)
        .def_readwrite("th_nccl_port", &ParallelismConfig::th_nccl_port)
        .def_readwrite("http_port", &ParallelismConfig::http_port)
        .def_readwrite("model_rpc_port", &ParallelismConfig::model_rpc_port)
        .def_readwrite("embedding_rpc_server_port", &ParallelismConfig::embedding_rpc_server_port)
        .def_readwrite("ffn_disaggregate_config", &ParallelismConfig::ffn_disaggregate_config)
        .def("to_string", &ParallelismConfig::to_string)
        .def(py::pickle(
            [](const ParallelismConfig& self) {
                return py::make_tuple(self.tp_size,
                                      self.ep_size,
                                      self.dp_size,
                                      self.pp_size,
                                      self.world_size,
                                      self.world_rank,
                                      self.local_world_size,
                                      self.ffn_sp_size,
                                      self.tp_rank,
                                      self.ep_rank,
                                      self.dp_rank,
                                      self.ffn_tp_size,
                                      self.ffn_tp_rank,
                                      self.enable_sp,
                                      self.nccl_ip,
                                      self.tp_nccl_port,
                                      self.dp_tp_nccl_port,
                                      self.ffn_tp_nccl_port,
                                      self.th_nccl_port,
                                      self.http_port,
                                      self.model_rpc_port,
                                      self.embedding_rpc_server_port,
                                      self.ffn_disaggregate_config);
            },
            [](py::tuple t) {
                if (t.size() != 23)
                    throw std::runtime_error("Invalid state!");
                ParallelismConfig c;
                try {
                    c.tp_size                   = t[0].cast<int64_t>();
                    c.ep_size                   = t[1].cast<int64_t>();
                    c.dp_size                   = t[2].cast<int64_t>();
                    c.pp_size                   = t[3].cast<int64_t>();
                    c.world_size                = t[4].cast<int64_t>();
                    c.world_rank                = t[5].cast<int64_t>();
                    c.local_world_size          = t[6].cast<int64_t>();
                    c.ffn_sp_size               = t[7].cast<int64_t>();
                    c.tp_rank                   = t[8].cast<int64_t>();
                    c.ep_rank                   = t[9].cast<int64_t>();
                    c.dp_rank                   = t[10].cast<int64_t>();
                    c.ffn_tp_size               = t[11].cast<int64_t>();
                    c.ffn_tp_rank               = t[12].cast<int64_t>();
                    c.enable_sp                 = t[13].cast<bool>();
                    c.nccl_ip                   = t[14].cast<std::string>();
                    c.tp_nccl_port              = t[15].cast<int64_t>();
                    c.dp_tp_nccl_port           = t[16].cast<int64_t>();
                    c.ffn_tp_nccl_port          = t[17].cast<int64_t>();
                    c.th_nccl_port              = t[18].cast<int64_t>();
                    c.http_port                 = t[19].cast<int64_t>();
                    c.model_rpc_port            = t[20].cast<int64_t>();
                    c.embedding_rpc_server_port = t[21].cast<int64_t>();
                    c.ffn_disaggregate_config   = t[22].cast<FfnDisAggregateConfig>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("ParallelismConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register BatchDecodeSchedulerConfig
    py::class_<BatchDecodeSchedulerConfig>(m, "BatchDecodeSchedulerConfig")
        .def(py::init<>())
        .def_readwrite("batch_decode_scheduler_batch_size",
                       &BatchDecodeSchedulerConfig::batch_decode_scheduler_batch_size)
        .def_readwrite("batch_decode_scheduler_warmup_type",
                       &BatchDecodeSchedulerConfig::batch_decode_scheduler_warmup_type)
        .def("to_string", &BatchDecodeSchedulerConfig::to_string)
        .def(py::pickle(
            [](const BatchDecodeSchedulerConfig& self) {
                return py::make_tuple(self.batch_decode_scheduler_batch_size, self.batch_decode_scheduler_warmup_type);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                BatchDecodeSchedulerConfig c;
                try {
                    c.batch_decode_scheduler_batch_size  = t[0].cast<int64_t>();
                    c.batch_decode_scheduler_warmup_type = t[1].cast<int64_t>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("BatchDecodeSchedulerConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register FIFOSchedulerConfig
    py::class_<FIFOSchedulerConfig>(m, "FIFOSchedulerConfig")
        .def(py::init<>())
        .def_readwrite("max_context_batch_size", &FIFOSchedulerConfig::max_context_batch_size)
        .def_readwrite("max_batch_tokens_size", &FIFOSchedulerConfig::max_batch_tokens_size)
        .def("to_string", &FIFOSchedulerConfig::to_string)
        .def(py::pickle(
            [](const FIFOSchedulerConfig& self) {
                return py::make_tuple(self.max_context_batch_size, self.max_batch_tokens_size);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                FIFOSchedulerConfig c;
                try {
                    c.max_context_batch_size = t[0].cast<int64_t>();
                    c.max_batch_tokens_size  = t[1].cast<int64_t>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("FIFOSchedulerConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register RuntimeConfig - only expose its own members, not sub-config members
    py::class_<RuntimeConfig> runtime_config(m, "RuntimeConfig");
    runtime_config.def(py::init<>())
        .def_readwrite("max_generate_batch_size", &RuntimeConfig::max_generate_batch_size)
        .def_readwrite("pre_allocate_op_mem", &RuntimeConfig::pre_allocate_op_mem)
        .def_readwrite("max_block_size_per_item", &RuntimeConfig::max_block_size_per_item)
        .def_readwrite("reserve_runtime_mem_mb", &RuntimeConfig::reserve_runtime_mem_mb)
        .def_readwrite("warm_up", &RuntimeConfig::warm_up)
        .def_readwrite("warm_up_with_loss", &RuntimeConfig::warm_up_with_loss)
        .def_readwrite("use_batch_decode_scheduler", &RuntimeConfig::use_batch_decode_scheduler)
        .def_readwrite("use_gather_batch_scheduler", &RuntimeConfig::use_gather_batch_scheduler)
        .def_readwrite("model_name", &RuntimeConfig::model_name)
        .def_readwrite("worker_grpc_addrs", &RuntimeConfig::worker_grpc_addrs)
        .def_readwrite("worker_addrs", &RuntimeConfig::worker_addrs)
        // Fields merged from PyDeviceResourceConfig
        .def_readwrite("specify_gpu_arch", &RuntimeConfig::specify_gpu_arch)
        .def_readwrite("acext_gemm_config_dir", &RuntimeConfig::acext_gemm_config_dir)
        // Add sub-configs as properties that return references
        .def_property_readonly(
            "batch_decode_scheduler_config",
            [](RuntimeConfig& self) -> BatchDecodeSchedulerConfig& { return self.batch_decode_scheduler_config; },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "fifo_scheduler_config",
            [](RuntimeConfig& self) -> FIFOSchedulerConfig& { return self.fifo_scheduler_config; },
            py::return_value_policy::reference_internal)
        .def("to_string", &RuntimeConfig::to_string)
        .def(py::pickle(
            [](const RuntimeConfig& self) {
                return py::make_tuple(self.max_generate_batch_size,
                                      self.pre_allocate_op_mem,
                                      self.max_block_size_per_item,
                                      self.reserve_runtime_mem_mb,
                                      self.warm_up,
                                      self.warm_up_with_loss,
                                      self.use_batch_decode_scheduler,
                                      self.use_gather_batch_scheduler,
                                      self.batch_decode_scheduler_config,
                                      self.fifo_scheduler_config,
                                      self.model_name,
                                      self.worker_grpc_addrs,
                                      self.worker_addrs,
                                      self.specify_gpu_arch,
                                      self.acext_gemm_config_dir);
            },
            [](py::tuple t) {
                if (t.size() != 15)
                    throw std::runtime_error("Invalid state!");
                RuntimeConfig c;
                try {
                    c.max_generate_batch_size       = t[0].cast<int64_t>();
                    c.pre_allocate_op_mem           = t[1].cast<bool>();
                    c.max_block_size_per_item       = t[2].cast<int64_t>();
                    c.reserve_runtime_mem_mb        = t[3].cast<int64_t>();
                    c.warm_up                       = t[4].cast<bool>();
                    c.warm_up_with_loss             = t[5].cast<bool>();
                    c.use_batch_decode_scheduler    = t[6].cast<bool>();
                    c.use_gather_batch_scheduler    = t[7].cast<bool>();
                    c.batch_decode_scheduler_config = t[8].cast<BatchDecodeSchedulerConfig>();
                    c.fifo_scheduler_config         = t[9].cast<FIFOSchedulerConfig>();
                    c.model_name                    = t[10].cast<std::string>();
                    c.worker_grpc_addrs             = t[11].cast<std::vector<std::string>>();
                    c.worker_addrs                  = t[12].cast<std::vector<std::string>>();
                    c.specify_gpu_arch              = t[13].cast<std::string>();
                    c.acext_gemm_config_dir         = t[14].cast<std::string>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("RuntimeConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("TYPE_INVALID", DataType::TYPE_INVALID)
        .value("TYPE_BOOL", DataType::TYPE_BOOL)
        .value("TYPE_UINT8", DataType::TYPE_UINT8)
        .value("TYPE_UINT16", DataType::TYPE_UINT16)
        .value("TYPE_UINT32", DataType::TYPE_UINT32)
        .value("TYPE_UINT64", DataType::TYPE_UINT64)
        .value("TYPE_INT8", DataType::TYPE_INT8)
        .value("TYPE_INT16", DataType::TYPE_INT16)
        .value("TYPE_INT32", DataType::TYPE_INT32)
        .value("TYPE_INT64", DataType::TYPE_INT64)
        .value("TYPE_FP16", DataType::TYPE_FP16)
        .value("TYPE_FP32", DataType::TYPE_FP32)
        .value("TYPE_FP64", DataType::TYPE_FP64)
        .value("TYPE_BYTES", DataType::TYPE_BYTES)
        .value("TYPE_BF16", DataType::TYPE_BF16)
        .value("TYPE_FP8_E4M3", DataType::TYPE_FP8_E4M3)
        .value("TYPE_STR", DataType::TYPE_STR)
        .value("TYPE_VOID", DataType::TYPE_VOID)
        .value("TYPE_QINT8", DataType::TYPE_QINT8)
        .value("TYPE_INT4X2", DataType::TYPE_INT4X2)
        .value("TYPE_QINT4X2", DataType::TYPE_QINT4X2)
        .value("TYPE_QFP8_E4M3", DataType::TYPE_QFP8_E4M3);

    // Register KvCacheDataType enum
    py::enum_<KvCacheDataType>(m, "KvCacheDataType")
        .value("BASE", KvCacheDataType::BASE)
        .value("INT8", KvCacheDataType::INT8)
        .value("FP8", KvCacheDataType::FP8);

    // Register RopeStyle enum
    py::enum_<RopeStyle>(m, "RopeStyle")
        .value("No", RopeStyle::No)
        .value("Base", RopeStyle::Base)
        .value("Glm2", RopeStyle::Glm2)
        .value("DynamicNTK", RopeStyle::DynamicNTK)
        .value("QwenDynamicNTK", RopeStyle::QwenDynamicNTK)
        .value("Yarn", RopeStyle::Yarn)
        .value("Llama3", RopeStyle::Llama3)
        .value("Mrope", RopeStyle::Mrope);

    // Register RopeConfig
    py::class_<RopeConfig>(m, "RopeConfig")
        .def(py::init<>())
        .def_property(
            "style",
            [](const RopeConfig& self) { return self.style; },
            [](RopeConfig& self, py::object value) {
                if (py::isinstance<RopeStyle>(value)) {
                    self.style = value.cast<RopeStyle>();
                } else if (py::isinstance<py::int_>(value)) {
                    self.style = static_cast<RopeStyle>(value.cast<int>());
                } else {
                    throw std::runtime_error("style must be RopeStyle enum or int");
                }
            },
            py::return_value_policy::reference_internal)
        .def_readwrite("dim", &RopeConfig::dim)
        .def_property(
            "base",
            [](const RopeConfig& self) { return self.base; },
            [](RopeConfig& self, py::object value) {
                if (py::isinstance<py::int_>(value)) {
                    self.base = value.cast<int>();
                } else if (py::isinstance<py::float_>(value)) {
                    self.base = static_cast<int>(value.cast<double>());
                } else {
                    throw std::runtime_error("base must be int or float");
                }
            })
        .def_readwrite("scale", &RopeConfig::scale)
        .def_readwrite("factor1", &RopeConfig::factor1)
        .def_readwrite("factor2", &RopeConfig::factor2)
        .def_readwrite("max_pos", &RopeConfig::max_pos)
        .def_readwrite("extrapolation_factor", &RopeConfig::extrapolation_factor)
        .def_readwrite("mscale", &RopeConfig::mscale)
        .def_readwrite("offset", &RopeConfig::offset)
        .def_readwrite("index_factor", &RopeConfig::index_factor)
        .def_readwrite("mrope_dim1", &RopeConfig::mrope_dim1)
        .def_readwrite("mrope_dim2", &RopeConfig::mrope_dim2)
        .def_readwrite("mrope_dim3", &RopeConfig::mrope_dim3);

    // Register AttentionConfigs
    py::class_<AttentionConfigs>(m, "AttentionConfigs")
        .def(py::init<>())
        .def_readwrite("head_num", &AttentionConfigs::head_num)
        .def_readwrite("kv_head_num", &AttentionConfigs::kv_head_num)
        .def_readwrite("size_per_head", &AttentionConfigs::size_per_head)
        .def_readwrite("rope_config", &AttentionConfigs::rope_config)
        .def_readwrite("tokens_per_block", &AttentionConfigs::tokens_per_block)
        .def_readwrite("q_scaling", &AttentionConfigs::q_scaling)
        .def_readwrite("fuse_qkv_add_bias", &AttentionConfigs::fuse_qkv_add_bias)
        .def_readwrite("use_logn_attn", &AttentionConfigs::use_logn_attn)
        .def_readwrite("is_causal", &AttentionConfigs::is_causal)
        .def_readwrite("use_mla", &AttentionConfigs::use_mla)
        .def_readwrite("q_lora_rank", &AttentionConfigs::q_lora_rank)
        .def_readwrite("kv_lora_rank", &AttentionConfigs::kv_lora_rank)
        .def_readwrite("nope_head_dim", &AttentionConfigs::nope_head_dim)
        .def_readwrite("rope_head_dim", &AttentionConfigs::rope_head_dim)
        .def_readwrite("v_head_dim", &AttentionConfigs::v_head_dim)
        .def_readwrite("softmax_extra_scale", &AttentionConfigs::softmax_extra_scale)
        .def_readwrite("kv_cache_dtype", &AttentionConfigs::kv_cache_dtype)
        .def_readwrite("skip_append_kv_cache", &AttentionConfigs::skip_append_kv_cache)
        .def_readwrite("dtype", &AttentionConfigs::dtype);

    py::class_<EPLBConfig>(m, "EPLBConfig")
        .def(py::init<>())
        .def_readwrite("eplb_update_time", &EPLBConfig::eplb_update_time)
        .def_property(
            "eplb_mode",
            [](const EPLBConfig& self) { return self.eplb_mode; },
            [](EPLBConfig& self, py::object value) {
                if (py::isinstance<py::str>(value)) {
                    self.eplb_mode = EPLBConfig::from_string(value.cast<std::string>());
                } else if (py::isinstance<EplbMode>(value)) {
                    self.eplb_mode = value.cast<EplbMode>();
                } else {
                    throw py::type_error("eplb_mode must be a string or EplbMode enum");
                }
            })
        .def_readwrite("redundant_expert", &EPLBConfig::redundant_expert)
        .def_readwrite("balance_method", &EPLBConfig::balance_method)
        .def_readwrite("eplb_force_repack", &EPLBConfig::eplb_force_repack)
        .def_readwrite("eplb_stats_window_size", &EPLBConfig::eplb_stats_window_size)
        .def_readwrite("eplb_control_step", &EPLBConfig::eplb_control_step)
        .def_readwrite("eplb_test_mode", &EPLBConfig::eplb_test_mode)
        .def_readwrite("eplb_balance_layer_per_step", &EPLBConfig::eplb_balance_layer_per_step)
        .def("enable_eplb", &EPLBConfig::enable_eplb, "Get enable_eplb status")
        .def("phy_exp_num", &EPLBConfig::phy_exp_num, py::arg("expert_num"), "Get physical expert number")
        .def(py::pickle(
            [](const EPLBConfig& self) {
                return py::make_tuple(self.eplb_update_time,
                                      EPLBConfig::to_string(self.eplb_mode),
                                      self.redundant_expert,
                                      self.balance_method,
                                      self.eplb_force_repack,
                                      self.eplb_stats_window_size,
                                      self.eplb_control_step,
                                      self.eplb_test_mode,
                                      self.eplb_balance_layer_per_step);
            },
            [](py::tuple t) {
                if (t.size() != 9)
                    throw std::runtime_error("Invalid state!");
                EPLBConfig c;
                try {
                    c.eplb_update_time = t[0].cast<int64_t>();
                    if (py::isinstance<py::str>(t[1])) {
                        c.eplb_mode = EPLBConfig::from_string(t[1].cast<std::string>());
                    } else {
                        c.eplb_mode = t[1].cast<EplbMode>();
                    }
                    c.redundant_expert            = t[2].cast<int64_t>();
                    c.balance_method              = t[3].cast<std::string>();
                    c.eplb_force_repack           = t[4].cast<int64_t>();
                    c.eplb_stats_window_size      = t[5].cast<int64_t>();
                    c.eplb_control_step           = t[6].cast<int>();
                    c.eplb_test_mode              = t[7].cast<bool>();
                    c.eplb_balance_layer_per_step = t[8].cast<int>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("EPLBConfig unpickle error: ") + e.what());
                }
                return c;
            }));

    // Register MMModelConfig
    py::class_<MMModelConfig>(m, "MMModelConfig")
        .def(py::init<>())
        .def_readwrite("is_multimodal", &MMModelConfig::is_multimodal)
        .def_readwrite("mm_sep_tokens", &MMModelConfig::mm_sep_tokens)
        .def_readwrite("include_sep_tokens", &MMModelConfig::include_sep_tokens)
        .def_readwrite("mm_position_ids_style", &MMModelConfig::mm_position_ids_style);

    // Register ModelConfig
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("num_layers", &ModelConfig::num_layers)
        .def_readwrite("max_seq_len", &ModelConfig::max_seq_len)
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("hidden_size", &ModelConfig::hidden_size)
        .def_readwrite("attn_config", &ModelConfig::attn_config)
        .def_readwrite("linear_attention_config", &ModelConfig::linear_attention_config)
        .def_readwrite("hybrid_attention_config", &ModelConfig::hybrid_attention_config)
        .def_readwrite("special_tokens", &ModelConfig::special_tokens)
        .def_readwrite("quant_algo", &ModelConfig::quant_algo)
        .def_readwrite("eplb_config", &ModelConfig::eplb_config)
        // task_type is defined as property below
        .def_readwrite("ckpt_path", &ModelConfig::ckpt_path)
        .def_readwrite("tokenizer_path", &ModelConfig::tokenizer_path)
        .def_readwrite("lora_infos", &ModelConfig::lora_infos)
        .def_readwrite("position_ids_style", &ModelConfig::position_ids_style)
        .def_readwrite("pre_seq_len", &ModelConfig::pre_seq_len)
        .def_readwrite("use_kvcache", &ModelConfig::use_kvcache)
        .def_readwrite("logit_scale", &ModelConfig::logit_scale)
        .def_readwrite("qk_norm", &ModelConfig::qk_norm)
        .def_readwrite("expert_num", &ModelConfig::expert_num)
        .def_readwrite("moe_n_group", &ModelConfig::moe_n_group)
        .def_readwrite("moe_k", &ModelConfig::moe_k)
        .def_readwrite("moe_style", &ModelConfig::moe_style)
        .def_readwrite("moe_layer_index", &ModelConfig::moe_layer_index)
        // Additional C++ fields
        .def_readwrite("deepseek_rope_mscale", &ModelConfig::deepseek_rope_mscale)
        .def_readwrite("deepseek_mscale_all_dim", &ModelConfig::deepseek_mscale_all_dim)
        .def_readwrite("moe_topk_group", &ModelConfig::moe_topk_group)
        .def_readwrite("routed_scaling_factor", &ModelConfig::routed_scaling_factor)
        .def_readwrite("layernorm_eps", &ModelConfig::layernorm_eps)
        .def_readwrite("partial_rotary_factor", &ModelConfig::partial_rotary_factor)
        .def_readwrite("input_embedding_scalar", &ModelConfig::input_embedding_scalar)
        .def_readwrite("residual_scalar", &ModelConfig::residual_scalar)
        .def_readwrite("use_norm_input_residual", &ModelConfig::use_norm_input_residual)
        .def_readwrite("use_norm_attn_out_residual", &ModelConfig::use_norm_attn_out_residual)
        .def_readwrite("input_vocab_size", &ModelConfig::input_vocab_size)
        .def_readwrite("type_vocab_size", &ModelConfig::type_vocab_size)
        .def_readwrite("embedding_size", &ModelConfig::embedding_size)
        .def_readwrite("moe_normalize_expert_scale", &ModelConfig::moe_normalize_expert_scale)
        .def_readwrite("scoring_func", &ModelConfig::scoring_func)
        .def_readwrite("has_positional_encoding", &ModelConfig::has_positional_encoding)
        .def_readwrite("has_pre_decoder_layernorm", &ModelConfig::has_pre_decoder_layernorm)
        .def_readwrite("has_post_decoder_layernorm", &ModelConfig::has_post_decoder_layernorm)
        .def_readwrite("has_lm_head", &ModelConfig::has_lm_head)
        .def_readwrite("use_attention_linear_bias", &ModelConfig::use_attention_linear_bias)
        .def_readwrite("use_fp32_to_compute_logit", &ModelConfig::use_fp32_to_compute_logit)
        .def_readwrite("add_bias_linear", &ModelConfig::add_bias_linear)
        .def_readwrite("has_moe_norm", &ModelConfig::has_moe_norm)
        .def_readwrite("prefix_projection", &ModelConfig::prefix_projection)
        .def_readwrite("reverse_e_h_norm", &ModelConfig::reverse_e_h_norm)
        // Properties with enum getter and string setter for type conversion
        .def_property(
            "data_type",
            [](const ModelConfig& self) { return self.data_type; },
            [](ModelConfig& self, const std::string& value) { self.set_data_type(value); })
        .def_property(
            "activation_type",
            [](const ModelConfig& self) { return self.activation_type; },
            [](ModelConfig& self, py::object value) {
                if (py::isinstance<ActivationType>(value)) {
                    self.activation_type = value.cast<ActivationType>();
                } else if (py::isinstance<py::str>(value)) {
                    self.set_activation_type(value.cast<std::string>());
                } else {
                    throw std::runtime_error("activation_type must be ActivationType enum or string");
                }
            })
        .def_property(
            "norm_type",
            [](const ModelConfig& self) { return self.norm_type; },
            [](ModelConfig& self, const std::string& value) { self.set_norm_type(value); })
        .def_property(
            "layernorm_type",
            [](const ModelConfig& self) { return self.layernorm_type; },
            [](ModelConfig& self, const std::string& value) { self.set_layer_norm_type(value); })
        .def_property(
            "task_type",
            [](const ModelConfig& self) { return self.task_type; },
            [](ModelConfig& self, py::object value) {
                if (py::isinstance<TaskType>(value)) {
                    self.task_type = value.cast<TaskType>();
                } else if (py::isinstance<py::str>(value)) {
                    self.set_task_type(value.cast<std::string>());
                } else {
                    throw std::runtime_error("task_type must be TaskType enum or string");
                }
            })
        .def_property(
            "mla_ops_type",
            [](const ModelConfig& self) { return self.mla_ops_type; },
            [](ModelConfig& self, const std::string& value) { self.set_mla_ops_type(value); })
        // Fields merged from PyModelConfig
        .def_readwrite("extra_data_path", &ModelConfig::extra_data_path)
        .def_readwrite("local_extra_data_path", &ModelConfig::local_extra_data_path)
        .def_readwrite("model_type", &ModelConfig::model_type)
        .def_readwrite("ptuning_path", &ModelConfig::ptuning_path)
        .def_readwrite("mm_model_config", &ModelConfig::mm_model_config)
        .def("getAttentionConfigs", &ModelConfig::getAttentionConfigs)
        .def("isGatedActivation", &ModelConfig::isGatedActivation)
        .def("isKvCacheQuant", &ModelConfig::isKvCacheQuant)
        .def("to_string", &ModelConfig::to_string);

    // Register VitConfig
    py::class_<VitConfig>(m, "VitConfig")
        .def(py::init<>())
        .def_readwrite("vit_separation", &VitConfig::vit_separation)
        .def("to_string", &VitConfig::to_string)
        .def(py::pickle([](const VitConfig& self) { return py::make_tuple(self.vit_separation); },
                        [](py::tuple t) {
                            if (t.size() != 1)
                                throw std::runtime_error("Invalid state!");
                            VitConfig c;
                            try {
                                c.vit_separation = t[0].cast<VitSeparation>();
                            } catch (const std::exception& e) {
                                throw std::runtime_error(std::string("VitConfig unpickle error: ") + e.what());
                            }
                            return c;
                        }));

    // Register PDSepConfig
    py::class_<PDSepConfig>(m, "PDSepConfig")
        .def(py::init<>())
        .def_readwrite("role_type", &PDSepConfig::role_type)
        .def_readwrite("cache_store_rdma_mode", &PDSepConfig::cache_store_rdma_mode)
        .def_readwrite("cache_store_listen_port", &PDSepConfig::cache_store_listen_port)
        .def_readwrite("cache_store_connect_port", &PDSepConfig::cache_store_connect_port)
        .def_readwrite("cache_store_rdma_listen_port", &PDSepConfig::cache_store_rdma_listen_port)
        .def_readwrite("cache_store_rdma_connect_port", &PDSepConfig::cache_store_rdma_connect_port)
        .def_readwrite("remote_rpc_server_port", &PDSepConfig::remote_rpc_server_port)
        .def_readwrite("prefill_retry_times", &PDSepConfig::prefill_retry_times)
        .def_readwrite("prefill_retry_timeout_ms", &PDSepConfig::prefill_retry_timeout_ms)
        .def_readwrite("prefill_max_wait_timeout_ms", &PDSepConfig::prefill_max_wait_timeout_ms)
        .def_readwrite("decode_retry_times", &PDSepConfig::decode_retry_times)
        .def_readwrite("decode_retry_timeout_ms", &PDSepConfig::decode_retry_timeout_ms)
        .def_readwrite("decode_retry_interval_ms", &PDSepConfig::decode_retry_interval_ms)
        .def_readwrite("decode_polling_kv_cache_step_ms", &PDSepConfig::decode_polling_kv_cache_step_ms)
        .def_readwrite("decode_polling_call_prefill_ms", &PDSepConfig::decode_polling_call_prefill_ms)
        .def_readwrite("rdma_connect_retry_times", &PDSepConfig::rdma_connect_retry_times)
        .def_readwrite("load_cache_timeout_ms", &PDSepConfig::load_cache_timeout_ms)
        .def_readwrite("max_rpc_timeout_ms", &PDSepConfig::max_rpc_timeout_ms)
        .def_readwrite("worker_port_offset", &PDSepConfig::worker_port_offset)
        .def_readwrite("decode_entrance", &PDSepConfig::decode_entrance)
        .def("to_string", &PDSepConfig::to_string)
        .def(py::pickle(
            [](const PDSepConfig& self) {
                return py::make_tuple(self.role_type,
                                      self.cache_store_rdma_mode,
                                      self.cache_store_listen_port,
                                      self.cache_store_connect_port,
                                      self.cache_store_rdma_listen_port,
                                      self.cache_store_rdma_connect_port,
                                      self.remote_rpc_server_port,
                                      self.prefill_retry_times,
                                      self.prefill_retry_timeout_ms,
                                      self.prefill_max_wait_timeout_ms,
                                      self.decode_retry_times,
                                      self.decode_retry_timeout_ms,
                                      self.decode_retry_interval_ms,
                                      self.decode_polling_kv_cache_step_ms,
                                      self.decode_polling_call_prefill_ms,
                                      self.rdma_connect_retry_times,
                                      self.load_cache_timeout_ms,
                                      self.max_rpc_timeout_ms,
                                      self.worker_port_offset,
                                      self.decode_entrance);
            },
            [](py::tuple t) {
                if (t.size() != 20)
                    throw std::runtime_error("Invalid state!");
                PDSepConfig c;
                try {
                    c.role_type                       = t[0].cast<RoleType>();
                    c.cache_store_rdma_mode           = t[1].cast<bool>();
                    c.cache_store_listen_port         = t[2].cast<int64_t>();
                    c.cache_store_connect_port        = t[3].cast<int64_t>();
                    c.cache_store_rdma_listen_port    = t[4].cast<int64_t>();
                    c.cache_store_rdma_connect_port   = t[5].cast<int64_t>();
                    c.remote_rpc_server_port          = t[6].cast<int64_t>();
                    c.prefill_retry_times             = t[7].cast<int64_t>();
                    c.prefill_retry_timeout_ms        = t[8].cast<int64_t>();
                    c.prefill_max_wait_timeout_ms     = t[9].cast<int64_t>();
                    c.decode_retry_times              = t[10].cast<int64_t>();
                    c.decode_retry_timeout_ms         = t[11].cast<int64_t>();
                    c.decode_retry_interval_ms        = t[12].cast<int64_t>();
                    c.decode_polling_kv_cache_step_ms = t[13].cast<int64_t>();
                    c.decode_polling_call_prefill_ms  = t[14].cast<int64_t>();
                    c.rdma_connect_retry_times        = t[15].cast<int64_t>();
                    c.load_cache_timeout_ms           = t[16].cast<int64_t>();
                    c.max_rpc_timeout_ms              = t[17].cast<int64_t>();
                    c.worker_port_offset              = t[18].cast<int64_t>();
                    c.decode_entrance                 = t[19].cast<bool>();
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("PDSepConfig unpickle error: ") + e.what());
                }
                return c;
            }));

}  // namespace rtp_llm
