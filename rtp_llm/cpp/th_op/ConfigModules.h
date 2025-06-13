#pragma once
#include <cstring>
#include <string>
#include <optional>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>

namespace rtp_llm {
struct ParallelismDistributedConfig {
    int tp_size = 1;
    int ep_size = 1;
    int dp_size = 1;
    int pp_size = 1;
    int world_size = 1;
    int world_rank = 0;
    int local_world_size = 1;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct ConcurrencyConfig {
    bool concurrency_with_block = false;
    int concurrency_limit = 32;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct FMHAConfig {
    bool enable_fmha = true;
    bool enable_trt_fmha = true;
    bool enable_paged_trt_fmha = true;
    bool enable_open_source_fmha = true;
    bool enable_paged_open_source_fmha = true;
    bool enable_trtv1_fmha = true;
    bool fmha_perf_instrument = false;
    bool fmha_show_params = false;
    bool disable_flash_infer = false;
    bool enable_xqa = true;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct KVCacheConfig {
    bool reuse_cache = false;
    std::string multi_task_prompt = "";
    std::string multi_task_prompt_str = "";
    std::string to_string() const;
    void update_from_env_for_test();
};

struct ProfilingDebugLoggingConfig {
    bool ft_nvtx = false;
    bool py_inference_log_response = false;
    bool rtp_llm_trace_memory = false;
    bool rtp_llm_trace_malloc_stack = false;
    bool enable_device_perf = false;
    bool ft_core_dump_on_exception = false;
    std::string ft_alog_conf_path = "";
    std::string log_level = "INFO";
    std::string to_string() const;
    void update_from_env_for_test();
};

struct HWKernelConfig {
    int deep_gemm_num_sm = -1;
    bool arm_gemm_use_kai = false;
    bool enable_stable_scatter_add = false;
    bool enable_multi_block_mode = true;
    bool ft_disable_custom_ar = true;
    std::string rocm_hipblaslt_config = "gemm_config.csv";
    std::string to_string() const;
    void update_from_env_for_test();
};

struct DeviceResourceConfig {
    int64_t device_reserve_memory_bytes = 0;
    int64_t host_reserve_memory_bytes = 4LL * 1024 * 1024 * 1024;
    int overlap_math_sm_count = 0;
    int overlap_comm_type = 0;
    int m_split = 0;
    bool enable_comm_overlap = true;
    int enable_layer_micro_batch = 0;
    bool not_use_default_stream = false;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct SamplerConfig {
    int64_t max_batch_size = 0;
    bool enable_flashinfer_sample_kernel = true;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct MoeConfig {
    bool use_deepep_moe = false;
    bool use_deepep_internode = false;
    bool use_deepep_low_latency = true;
    bool use_deepep_p2p_low_latency = false;
    bool fake_balance_expert = false;
    int eplb_control_step = 100;
    bool eplb_test_mode = false;
    bool hack_moe_expert = false;
    int eplb_balance_layer_per_step = 1;
    int deep_ep_num_sm = 0;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct ModelSpecificConfig {
    int64_t max_lora_model_size = -1;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct SpeculativeExecutionConfig {
    std::string sp_model_type = "";
    std::string sp_type = "";
    int64_t sp_min_token_match = 2;
    int64_t sp_max_token_match = 2;
    std::string tree_decode_config = "";
    int64_t gen_num_per_cycle = 1;
    bool force_stream_sample = false;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct ServiceDiscoveryConfig {
    bool use_local = false;
    std::string remote_rpc_server_ip;
    std::string rtp_llm_decode_cm2_config;
    std::string remote_vit_server_ip;
    std::string rtp_llm_multimodal_part_cm2_config;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct CacheStoreConfig {
    bool cache_store_rdma_mode = false;
    int wrr_available_ratio = 80;
    int rank_factor = 0;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct SchedulerConfig {
    bool use_batch_decode_scheduler = false;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct BatchDecodeSchedulerConfig {
    int64_t batch_decode_scheduler_batch_size = 1;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct FIFOSchedulerConfig {
    int64_t max_context_batch_size = 1;
    int scheduler_reserve_resource_ratio = 5;
    bool enable_fast_gen = false;
    bool enable_partial_fallback = false;
    int64_t fast_gen_context_budget = -1;
    std::string to_string() const;
    void update_from_env_for_test();
};

struct MiscellaneousConfig {
    int load_balance = 0;
    int64_t step_records_time_range = 60 * 1000 * 1000;
    int64_t step_records_max_size = 1000;
    std::string to_string() const;
    void update_from_env_for_test();
};

std::string to_lower(const std::string& s);
bool bool_from_env_for_test(std::string env_name, bool default_value);
void register_parallelism_distributed_config(pybind11::module& m);
void register_concurrency_config(pybind11::module& m);
void register_fmha_config(pybind11::module& m);
void register_kvcache_config(pybind11::module& m);
void register_profiling_debug_logging_config(pybind11::module& m);
void register_hwkernel_config(pybind11::module& m);
void register_device_resource_config(pybind11::module& m);
void register_sampler_config(pybind11::module& m);
void register_moe_config(pybind11::module& m);
void register_model_specific_config(pybind11::module& m);
void register_speculative_execution_config(pybind11::module& m);
void register_service_discovery_config(pybind11::module& m);
void register_cache_store_config(pybind11::module& m);
void register_scheduler_config(pybind11::module& m);
void register_batch_decode_scheduler_config(pybind11::module& m);
void register_fifo_scheduler_config(pybind11::module& m);
void register_misc_config(pybind11::module& m);

}
