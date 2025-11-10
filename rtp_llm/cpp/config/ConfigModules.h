#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <map>
namespace rtp_llm {

struct ParallelismDistributedConfig {
    int         tp_size          = 1;
    int         ep_size          = 1;
    int         dp_size          = 1;
    int         pp_size          = 1;
    int         world_size       = 1;
    int         world_rank       = 0;
    int         local_world_size = 1;
    int         ffn_sp_size      = 1;
    bool        use_all_gather   = true;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct ConcurrencyConfig {
    bool        concurrency_with_block = false;
    int         concurrency_limit      = 32;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct FMHAConfig {
    bool        enable_fmha                   = true;
    bool        enable_trt_fmha               = true;
    bool        enable_paged_trt_fmha         = true;
    bool        enable_open_source_fmha       = true;
    bool        enable_paged_open_source_fmha = true;
    bool        enable_trtv1_fmha             = true;
    bool        fmha_perf_instrument          = false;
    bool        fmha_show_params              = false;
    bool        disable_flash_infer           = false;
    bool        enable_xqa                    = true;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct KVCacheConfig {
    bool        reuse_cache                        = false;
    std::string multi_task_prompt                  = "";
    std::string multi_task_prompt_str              = "";
    bool        enable_3fs                         = false;
    int         match_timeout_ms                   = 1000;
    int         rpc_get_cache_timeout_ms           = 2000;
    int         rpc_put_cache_timeout_ms           = 2000;
    int         threefs_read_timeout_ms            = 1000;
    int         threefs_write_timeout_ms           = 2000;
    int         max_block_size_per_item            = 16;
    int64_t     threefs_read_iov_size              = 1LL << 32;  // 4GB
    int64_t     threefs_write_iov_size             = 1LL << 32;  // 4GB
    int64_t     memory_block_cache_size_mb         = 0;
    int64_t     memory_block_cache_sync_timeout_ms = 10000;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct ProfilingDebugLoggingConfig {
    bool        trace_memory              = false;
    bool        trace_malloc_stack        = false;
    bool        enable_device_perf        = false;
    bool        ft_core_dump_on_exception = false;
    std::string ft_alog_conf_path         = "";
    std::string log_level                 = "INFO";
    bool        gen_timeline_sync         = false;
    std::string torch_cuda_profiler_dir   = "";
    std::string log_path                  = "logs";
    int         log_file_backup_count     = 16;
    std::string nccl_debug_file           = "";
    bool        debug_load_server         = false;
    int         hack_layer_num            = 0;
    bool        debug_start_fake_process  = false;
    bool        dg_print_reg_reuse        = false;
    bool        qwen_agent_debug          = false;
    bool        disable_dpc_random        = false;
    bool        enable_detail_log         = false;
    bool        check_nan                 = false;

    std::string to_string() const;
    void        update_from_env_for_test();
};

struct HWKernelConfig {
    int         deep_gemm_num_sm             = -1;
    bool        arm_gemm_use_kai             = false;
    bool        enable_stable_scatter_add    = false;
    bool        enable_multi_block_mode      = true;
    bool        ft_disable_custom_ar         = true;
    std::string rocm_hipblaslt_config        = "gemm_config.csv";
    bool        use_swizzleA                 = false;
    bool        enable_cuda_graph            = false;
    bool        enable_cuda_graph_debug_mode = false;
    bool        use_aiter_pa                 = true;
    bool        use_asm_pa                   = true;
    bool        enable_native_cuda_graph     = false;
    int         num_native_cuda_graph        = 200;
    // Prefill CUDA Graph capture configuration
    // Can be set via: prefill_capture_file_path, prefill_capture_seq_lens, or prefill_capture_max_seq_len + step
    std::vector<int> prefill_capture_seq_lens;
    // Decode CUDA Graph capture configuration
    // Comma-separated list of batch sizes, e.g., "1,2,4,8,16,32"
    std::vector<int> decode_capture_batch_sizes;
    std::string      to_string() const;
    void             update_from_env_for_test();
};

struct DeviceResourceConfig {
    int64_t     device_reserve_memory_bytes = -1073741824;
    int64_t     host_reserve_memory_bytes   = 4LL * 1024 * 1024 * 1024;
    int         overlap_math_sm_count       = 0;
    int         overlap_comm_type           = 0;
    int         m_split                     = 0;
    bool        enable_comm_overlap         = true;
    int         enable_layer_micro_batch    = 0;
    bool        not_use_default_stream      = false;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct SamplerConfig {
    int64_t     max_batch_size                  = 0;
    bool        enable_flashinfer_sample_kernel = true;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct MoeConfig {
    bool        use_deepep_moe                  = false;
    bool        use_deepep_internode            = false;
    bool        use_deepep_low_latency          = true;
    bool        use_deepep_p2p_low_latency      = false;
    bool        fake_balance_expert             = false;
    int         eplb_control_step               = 100;
    bool        eplb_test_mode                  = false;
    bool        hack_moe_expert                 = false;
    int         eplb_balance_layer_per_step     = 1;
    int         deep_ep_num_sm                  = 0;
    int         max_moe_normal_masked_token_num = 1024;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct ModelSpecificConfig {
    int64_t     max_lora_model_size = -1;
    bool        load_python_model   = false;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct SpeculativeExecutionConfig {
    std::string sp_model_type                 = "";
    std::string sp_type                       = "";
    int64_t     sp_min_token_match            = 2;
    int64_t     sp_max_token_match            = 2;
    std::string tree_decode_config            = "";
    int64_t     gen_num_per_cycle             = 1;
    bool        force_stream_sample           = false;
    bool        force_score_context_attention = true;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct ServiceDiscoveryConfig {
    bool        use_local = false;
    std::string remote_rpc_server_ip;
    std::string decode_cm2_config;
    std::string remote_vit_server_ip;
    std::string multimodal_part_cm2_config;
    std::string remote_backend_ip;
    std::string backend_cm2_config;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct CacheStoreConfig {
    bool        cache_store_rdma_mode        = false;
    int         wrr_available_ratio          = 80;
    int         rank_factor                  = 0;
    int         thread_count                 = 16;
    int         rdma_connect_timeout_ms      = 250;
    int         rdma_qp_count_per_connection = 2;
    int         messager_io_thread_count     = 2;
    int         messager_worker_thread_count = 16;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct SchedulerConfig {
    bool        use_batch_decode_scheduler = false;
    bool        use_gather_batch_scheduler = false;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct BatchDecodeSchedulerConfig {
    int64_t batch_decode_scheduler_batch_size = 1;
    // 0: use decode warmup, others: use prefill warmup
    int64_t     batch_decode_scheduler_warmup_type = 0;
    std::string to_string() const;
    void        update_from_env_for_test();
};

struct FIFOSchedulerConfig {
    int64_t max_context_batch_size           = 1;
    int     scheduler_reserve_resource_ratio = 5;
    bool    enable_fast_gen                  = false;
    bool    enable_partial_fallback          = false;
    int64_t fast_gen_context_budget          = -1;

    std::string to_string() const;
    void        update_from_env_for_test();
};

struct MiscellaneousConfig {
    bool        disable_pdl        = true;
    bool        disable_access_log = false;
    std::string aux_string         = "";
    std::string to_string() const;
    void        update_from_env_for_test();
};

class ParallelInfo final {
public:
    ParallelInfo(int tp_size          = 1,
                 int pp_size          = 1,
                 int ep_size          = 1,
                 int dp_size          = 1,
                 int world_size       = 1,
                 int world_rank       = 0,
                 int local_world_size = 1):
        tp_size_(tp_size),
        pp_size_(pp_size),
        ep_size_(ep_size),
        dp_size_(dp_size),
        world_size_(world_size),
        world_rank_(world_rank),
        local_world_size_(local_world_size) {}

public:
    static ParallelInfo& globalParallelInfo() {
        static ParallelInfo parallel_info;
        return parallel_info;
    }
    void setTpSize(int tp_size) {
        tp_size_ = tp_size;
    }
    int getTpSize() const {
        return tp_size_;
    }
    void setPpSize(int pp_size) {
        pp_size_ = pp_size;
    }
    int getPpSize() const {
        return pp_size_;
    }
    void setDpSize(int dp_size) {
        dp_size_ = dp_size;
    }
    int getDpSize() const {
        return dp_size_;
    }
    void setEpSize(int ep_size) {
        ep_size_ = ep_size;
    }
    int getEpSize() const {
        return ep_size_;
    }
    int getTpRank() const {
        return world_rank_ % tp_size_;
    }
    void setWorldRank(int world_rank) {
        world_rank_ = world_rank;
    }
    int getWorldRank() const {
        return world_rank_;
    }
    int getLocalRank() const {
        return world_rank_ % local_world_size_;
    }
    void setWorldSize(int world_size) {
        world_size_ = world_size;
    }
    int getWorldSize() const {
        return world_size_;
    }
    void setLocalWorldSize(int local_world_size) {
        local_world_size_ = local_world_size;
    }
    int getLocalWorldSize() const {
        return local_world_size_;
    }
    bool isMaster() const {
        return world_rank_ == 0;
    }
    bool isWorker() const {
        return !isMaster();
    }
    std::string toString() const {
        std::ostringstream oss;
        oss << "ParallelInfo:[ "
            << "tp_size=" << tp_size_ << " pp_size=" << pp_size_ << " world_size=" << world_size_
            << " world_rank=" << world_rank_ << " local_world_size=" << local_world_size_ << " ]";
        return oss.str();
    }
    // only for test
    void reload() {
        ParallelismDistributedConfig parallelism_distributed_config;
        parallelism_distributed_config.update_from_env_for_test();
        tp_size_ = parallelism_distributed_config.tp_size;
        // in fact pipeline parallelism is not supported yet
        pp_size_          = parallelism_distributed_config.pp_size;
        ep_size_          = parallelism_distributed_config.ep_size;
        dp_size_          = parallelism_distributed_config.dp_size;
        world_size_       = parallelism_distributed_config.world_size;
        world_rank_       = parallelism_distributed_config.world_rank;
        local_world_size_ = parallelism_distributed_config.local_world_size;
    }

private:
    int tp_size_;
    int pp_size_;
    int ep_size_;
    int dp_size_;
    int world_size_;
    int world_rank_;
    int local_world_size_;
};

struct FfnDisAggregateConfig {
    bool        enable_ffn_disaggregate = false;
    int         attention_tp_size       = 1;
    int         attention_dp_size       = 1;
    int         ffn_tp_size             = 1;
    int         ffn_dp_size             = 1;
    bool        is_ffn_rank             = false;
    std::string to_string() const;
    void        update_from_env_for_test();
    bool        is_ffn_service() const {
        return enable_ffn_disaggregate && is_ffn_rank;
    }
};

struct ArpcConfig {
    int         threadNum   = 10;
    int         queueNum    = 50;
    int         ioThreadNum = 2;
    std::string to_string() const;
};

struct GrpcConfig {
    std::map<std::string, int> client_config;
    std::map<std::string, int> server_config;
    GrpcConfig() {};
    GrpcConfig(const std::string& json_str);
    std::string                to_string() const;
    void                       update_from_env_for_test();
    void                       from_json(const std::string& json_str);
    std::map<std::string, int> get_client_config() const {
        return client_config;
    }
    std::map<std::string, int> get_server_config() const {
        return server_config;
    }
};

std::string to_lower(const std::string& s);
bool        bool_from_env_for_test(std::string env_name, bool default_value);

}  // namespace rtp_llm
