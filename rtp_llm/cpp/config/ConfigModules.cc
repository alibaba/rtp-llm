#include "rtp_llm/cpp/config/ConfigModules.h"
#include "autil/EnvUtil.h"
#include <map>
#include <sstream>
#include <algorithm>
#include <string>
#include <cctype>
#include <regex>

namespace rtp_llm {


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
        << "local_rank: " << local_rank << "\n"
        << "ffn_sp_size: " << ffn_sp_size << "\n"
        << "tp_rank: " << tp_rank << "\n"
        << "ep_rank: " << ep_rank << "\n"
        << "dp_rank: " << dp_rank << "\n"
        << "ffn_tp_size: " << ffn_tp_size << "\n"
        << "ffn_tp_rank: " << ffn_tp_rank << "\n"
        << "enable_sp: " << enable_sp << "\n"
        << "nccl_ip: " << nccl_ip << "\n"
        << "tp_nccl_port: " << tp_nccl_port << "\n"
        << "dp_tp_nccl_port: " << dp_tp_nccl_port << "\n"
        << "ffn_tp_nccl_port: " << ffn_tp_nccl_port << "\n"
        << "th_nccl_port: " << th_nccl_port << "\n"
        << "http_port: " << http_port << "\n"
        << "model_rpc_port: " << model_rpc_port << "\n"
        << "embedding_rpc_server_port: " << embedding_rpc_server_port << "\n"
        << "ffn_disaggregate_config: {\n" << ffn_disaggregate_config.to_string() << "\n}";
    return oss.str();
}

// ConcurrencyConfig
std::string ConcurrencyConfig::to_string() const {
    std::ostringstream oss;
    oss << "concurrency_with_block: " << concurrency_with_block << "\n"
        << "concurrency_limit: " << concurrency_limit;
    return oss.str();
}

// FMHAConfig
std::string FMHAConfig::to_string() const {
    std::ostringstream oss;
    oss << "enable_fmha: " << enable_fmha << "\n"
        << "enable_trt_fmha: " << enable_trt_fmha << "\n"
        << "enable_paged_trt_fmha: " << enable_paged_trt_fmha << "\n"
        << "enable_open_source_fmha: " << enable_open_source_fmha << "\n"
        << "enable_paged_open_source_fmha: " << enable_paged_open_source_fmha << "\n"
        << "enable_trtv1_fmha: " << enable_trtv1_fmha << "\n"
        << "disable_flash_infer: " << disable_flash_infer << "\n"
        << "enable_xqa: " << enable_xqa << "\n"
        << "use_aiter_pa: " << use_aiter_pa << "\n"
        << "use_asm_pa: " << use_asm_pa << "\n"
        << "absorb_opt_len: " << absorb_opt_len << "\n";
    return oss.str();
}

// KVCacheConfig
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
        << "memory_block_cache_sync_timeout_ms: " << memory_block_cache_sync_timeout_ms << "\n"
        << "int8_kv_cache: " << int8_kv_cache << "\n"
        << "fp8_kv_cache: " << fp8_kv_cache << "\n"
        << "kv_cache_mem_mb: " << kv_cache_mem_mb << "\n"
        << "seq_size_per_block: " << seq_size_per_block << "\n"
        << "test_block_num: " << test_block_num << "\n"
        << "use_block_cache: " << use_block_cache << "\n";
    return oss.str();
}

// ProfilingDebugLoggingConfig
std::string ProfilingDebugLoggingConfig::to_string() const {
    std::ostringstream oss;
    oss << "trace_memory: " << trace_memory << "\n"
        << "trace_malloc_stack: " << trace_malloc_stack << "\n"
        << "enable_device_perf: " << enable_device_perf << "\n"
        << "ft_core_dump_on_exception: " << ft_core_dump_on_exception << "\n"
        << "ft_alog_conf_path: " << ft_alog_conf_path << "\n"
        << "gen_timeline_sync: " << gen_timeline_sync << "\n"
        << "torch_cuda_profiler_dir: " << torch_cuda_profiler_dir << "\n"
        << "log_file_backup_count: " << log_file_backup_count << "\n"
        << "debug_load_server: " << debug_load_server << "\n"
        << "hack_layer_num: " << hack_layer_num << "\n"
        << "debug_start_fake_process: " << debug_start_fake_process << "\n"
        << "enable_detail_log: " << enable_detail_log << "\n"
        << "check_nan: " << check_nan << "\n";
    return oss.str();
}

// HWKernelConfig
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
        << "enable_native_cuda_graph: " << enable_native_cuda_graph << "\n"
        << "num_native_cuda_graph: " << num_native_cuda_graph << "\n"
        << "prefill_capture_seq_lens size: " << prefill_capture_seq_lens.size() << "\n"
        << "decode_capture_batch_sizes size: " << decode_capture_batch_sizes.size() << "\n"
        << "disable_dpc_random: " << disable_dpc_random;
    return oss.str();
}

// DeviceResourceConfig
std::string DeviceResourceConfig::to_string() const {
    std::ostringstream oss;
    oss << "device_reserve_memory_bytes: " << device_reserve_memory_bytes << "\n"
        << "host_reserve_memory_bytes: " << host_reserve_memory_bytes << "\n"
        << "overlap_math_sm_count: " << overlap_math_sm_count << "\n"
        << "overlap_comm_type: " << overlap_comm_type << "\n"
        << "m_split: " << m_split << "\n"
        << "enable_comm_overlap: " << enable_comm_overlap << "\n"
        << "enable_layer_micro_batch: " << enable_layer_micro_batch << "\n"
        << "not_use_default_stream: " << not_use_default_stream;
    return oss.str();
}

std::string SamplerConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_batch_size: " << max_batch_size << "\n"
        << "enable_flashinfer_sample_kernel: " << enable_flashinfer_sample_kernel;
    return oss.str();
}

// MoeConfig
std::string MoeConfig::to_string() const {
    std::ostringstream oss;
    oss << "use_deepep_moe: " << use_deepep_moe << "\n"
        << "use_deepep_internode: " << use_deepep_internode << "\n"
        << "use_deepep_low_latency: " << use_deepep_low_latency << "\n"
        << "use_deepep_p2p_low_latency: " << use_deepep_p2p_low_latency << "\n"
        << "fake_balance_expert: " << fake_balance_expert << "\n"
        << "hack_moe_expert: " << hack_moe_expert << "\n"
        << "deep_ep_num_sm: " << deep_ep_num_sm << "\n"
        << "max_moe_normal_masked_token_num: " << max_moe_normal_masked_token_num << "\n"
        << "use_all_gather: " << use_all_gather;
    return oss.str();
}

// ModelSpecificConfig
std::string ModelSpecificConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_lora_model_size: " << max_lora_model_size << "\n";
    oss << "load_python_model:" << load_python_model << "\n";
    return oss.str();
}

// SpeculativeExecutionConfig
SpeculativeType SpeculativeExecutionConfig::from_string(const std::string& str) {
    if (str.empty() || str == "none") {
        return SP_TYPE_NONE;
    } else if (str == "vanilla") {
        return SP_TYPE_VANILLA;
    } else if (str == "mtp") {
        return SP_TYPE_MTP;
    } else if (str == "eagle3") {
        return SP_TYPE_EAGLE3;
    } else if (str == "eagle") {
        return SP_TYPE_EAGLE;
    } else if (str == "deterministic") {
        return SP_TYPE_DETERMINISTIC;
    } else {
        return SP_TYPE_NONE;  // Default to NONE for unknown values
    }
}

std::string SpeculativeExecutionConfig::to_string(SpeculativeType type) {
    switch (type) {
        case SP_TYPE_NONE:
            return "none";
        case SP_TYPE_VANILLA:
            return "vanilla";
        case SP_TYPE_MTP:
            return "mtp";
        case SP_TYPE_EAGLE3:
            return "eagle3";
        case SP_TYPE_EAGLE:
            return "eagle";
        case SP_TYPE_DETERMINISTIC:
            return "deterministic";
        default:
            return "none";
    }
}

std::string SpeculativeExecutionConfig::to_string() const {
    std::ostringstream oss;
    oss << "model_type: " << model_type << "\n"
        << "type: " << SpeculativeExecutionConfig::to_string(type) << "\n"
        << "sp_min_token_match: " << sp_min_token_match << "\n"
        << "sp_max_token_match: " << sp_max_token_match << "\n"
        << "tree_decode_config: " << tree_decode_config << "\n"
        << "gen_num_per_cycle: " << gen_num_per_cycle << "\n"
        << "force_stream_sample: " << force_stream_sample << "\n"
        << "force_score_context_attention: " << force_score_context_attention << "\n"
        << "quantization: " << quantization << "\n"
        << "checkpoint_path: " << checkpoint_path;
    return oss.str();
}

// VitConfig
std::string VitConfig::to_string() const {
    std::ostringstream oss;
    std::string vit_sep_str;
    switch (vit_separation) {
        case VitSeparation::VIT_SEPARATION_LOCAL:
            vit_sep_str = "LOCAL";
            break;
        case VitSeparation::VIT_SEPARATION_ROLE:
            vit_sep_str = "ROLE";
            break;
        case VitSeparation::VIT_SEPARATION_REMOTE:
            vit_sep_str = "REMOTE";
            break;
        default:
            vit_sep_str = "UNKNOWN(" + std::to_string(static_cast<int>(vit_separation)) + ")";
            break;
    }
    oss << "vit_separation: " << vit_sep_str;
    return oss.str();
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
std::string MiscellaneousConfig::to_string() const {
    std::ostringstream oss;
    oss << "disable_pdl: " << disable_pdl << "\n"
        << "aux_string: " << aux_string << "\n";
    return oss.str();
}

// SchedulerConfig
std::string SchedulerConfig::to_string() const {
    std::ostringstream oss;
    oss << "use_batch_decode_scheduler: " << use_batch_decode_scheduler << "\n"
        << "use_gather_batch_scheduler: " << use_gather_batch_scheduler;
    return oss.str();
}

// BatchDecodeSchedulerConfig
std::string BatchDecodeSchedulerConfig::to_string() const {
    std::ostringstream oss;
    oss << "batch_decode_scheduler_batch_size: " << batch_decode_scheduler_batch_size << "\n"
        << "batch_decode_scheduler_warmup_type: " << batch_decode_scheduler_warmup_type;
    return oss.str();
}

// FIFOSchedulerConfig
std::string FIFOSchedulerConfig::to_string() const {
    std::ostringstream oss;
    oss << "enable_fast_gen: " << enable_fast_gen << "\n"
        << "enable_partial_fallback: " << enable_partial_fallback << "\n"
        << "fast_gen_context_budget: " << fast_gen_context_budget << "\n"
        << "max_context_batch_size: " << max_context_batch_size << "\n"
        << "scheduler_reserve_resource_ratio: " << scheduler_reserve_resource_ratio << "\n"
        << "fast_gen_max_context_len: " << fast_gen_max_context_len << "\n"
        << "max_batch_tokens_size: " << max_batch_tokens_size;
    return oss.str();
}

// RuntimeConfig
std::string RuntimeConfig::to_string() const {
    std::ostringstream oss;
    oss << "max_generate_batch_size: " << max_generate_batch_size << "\n"
        << "pre_allocate_op_mem: " << pre_allocate_op_mem << "\n"
        << "max_block_size_per_item: " << max_block_size_per_item << "\n"
        << "reserve_runtime_mem_mb: " << reserve_runtime_mem_mb << "\n"
        << "warm_up: " << warm_up << "\n"
        << "warm_up_with_loss: " << warm_up_with_loss << "\n"
        << "use_batch_decode_scheduler: " << use_batch_decode_scheduler << "\n"
        << "use_gather_batch_scheduler: " << use_gather_batch_scheduler << "\n"
        << "batch_decode_scheduler_config: {\n" << batch_decode_scheduler_config.to_string() << "\n}\n"
        << "fifo_scheduler_config: {\n" << fifo_scheduler_config.to_string() << "\n}\n"
        << "model_name: " << model_name << "\n"
        << "worker_grpc_addrs: [";
    for (size_t i = 0; i < worker_grpc_addrs.size(); ++i) {
        oss << worker_grpc_addrs[i];
        if (i < worker_grpc_addrs.size() - 1) oss << ", ";
    }
    oss << "]\n"
        << "worker_addrs: [";
    for (size_t i = 0; i < worker_addrs.size(); ++i) {
        oss << worker_addrs[i];
        if (i < worker_addrs.size() - 1) oss << ", ";
    }
    oss << "]\n"
        << "specify_gpu_arch: " << specify_gpu_arch << "\n"
        << "acext_gemm_config_dir: " << acext_gemm_config_dir;
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

GrpcConfig::GrpcConfig(const std::string& json_str) {
    from_json(json_str);
}

std::string GrpcConfig::to_string() const {
    std::ostringstream oss;

    // Output client config
    oss << "Client Config:\n";
    for (auto it = client_config.begin(); it != client_config.end(); ++it) {
        oss << "  " << it->first << ": " << it->second << "\n";
    }

    // Output server config
    oss << "Server Config:\n";
    for (auto it = server_config.begin(); it != server_config.end(); ++it) {
        oss << "  " << it->first << ": " << it->second << "\n";
    }

    return oss.str();
}


void GrpcConfig::from_json(const std::string& json_str) {
    if (json_str.empty()) {
        return;
    }

    // Clear existing configs
    client_config.clear();
    server_config.clear();

    // Parse 2-level JSON structure
    // Expected format: {"client_config": {"key1": value1, ...}, "server_config": {"key2": value2, ...}}

    // Find client_config section
    std::regex  client_section_pattern("\"client_config\"\\s*:\\s*\\{([^}]+)\\}");
    std::smatch client_match;
    if (std::regex_search(json_str, client_match, client_section_pattern)) {
        std::string          client_section = client_match[1].str();
        std::regex           key_value_pattern("\"([^\"]+)\"\\s*:\\s*(\\d+)");
        std::sregex_iterator iter(client_section.begin(), client_section.end(), key_value_pattern);
        std::sregex_iterator end;

        while (iter != end) {
            std::string key    = (*iter)[1].str();
            int         value  = std::stoi((*iter)[2].str());
            client_config[key] = value;
            ++iter;
        }
    }

    // Find server_config section
    std::regex  server_section_pattern("\"server_config\"\\s*:\\s*\\{([^}]+)\\}");
    std::smatch server_match;
    if (std::regex_search(json_str, server_match, server_section_pattern)) {
        std::string          server_section = server_match[1].str();
        std::regex           key_value_pattern("\"([^\"]+)\"\\s*:\\s*(\\d+)");
        std::sregex_iterator iter(server_section.begin(), server_section.end(), key_value_pattern);
        std::sregex_iterator end;

        while (iter != end) {
            std::string key    = (*iter)[1].str();
            int         value  = std::stoi((*iter)[2].str());
            server_config[key] = value;
            ++iter;
        }
    }
}

// FfnDisAggregateConfig
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

// Helper function to convert RoleType enum to string
static std::string roleTypeToString(RoleType role_type) {
    switch (role_type) {
        case RoleType::PDFUSION:
            return "PDFUSION";
        case RoleType::PREFILL:
            return "PREFILL";
        case RoleType::DECODE:
            return "DECODE";
        case RoleType::VIT:
            return "VIT";
        case RoleType::FRONTEND:
            return "FRONTEND";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(role_type)) + ")";
    }
}

// PDSepConfig
std::string PDSepConfig::to_string() const {
    std::ostringstream oss;
    oss << "role_type: " << roleTypeToString(role_type) << "\n"
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
        << "decode_retry_interval_ms: " << decode_retry_interval_ms << "\n"
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
