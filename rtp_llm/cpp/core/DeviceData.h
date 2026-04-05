#pragma once
#include <cstdint>
#include <stddef.h>
#include <string>
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

enum class DeviceType {
    Cpu    = 0,
    Cuda   = 1,
    Yitian = 2,
    ArmCpu = 3,
    ROCm   = 4,
    Ppu    = 5,
};

inline DeviceType buildDeviceType() {
#if defined(USE_PPU)
    return DeviceType::Ppu;
#elif USING_CUDA
    return DeviceType::Cuda;
#elif USING_ROCM
    return DeviceType::ROCm;
#else
    return DeviceType::Cpu;
#endif
}

enum class MicroBatchType {
    NONE       = 0,
    DS_PREFILL = 1,
    DS_DECODE  = 2,
};

struct ExecInitParams {
    DeviceType device_type;
    size_t     device_id               = 0;
    size_t     max_generate_batch_size = 128;

    size_t tp_rank        = 0;
    size_t tp_size        = 1;
    size_t ep_rank        = 0;
    size_t ep_size        = 1;
    size_t dp_rank        = 0;
    size_t dp_size        = 1;
    size_t ffn_tp_rank    = 0;
    size_t ffn_tp_size    = 1;
    bool   use_all_gather = false;

    size_t tokens_per_block        = 0;
    size_t kernel_tokens_per_block = 0;

    // Used by CUDA graph capture/replay path to select per-layer kv cache block tables.
    std::vector<int32_t> kv_cache_layer_to_group;
    int32_t              kv_cache_group_num = 0;

    MlaOpsType mla_ops_type = MlaOpsType::AUTO;

    bool           enable_comm_overlap      = true;
    MicroBatchType enable_layer_micro_batch = MicroBatchType::NONE;

    bool   enable_sp = false;
    size_t m_split   = 0;

    bool enable_prefill_cp = false;

    // to init deepep
    int64_t max_seq_len    = 0;
    int64_t hidden_size    = 0;
    int64_t num_experts    = 0;
    int64_t extra_experts  = 0;
    bool    ffn_as_service = false;

    bool                       use_deepep_moe         = false;
    int                        user_deep_gemm_num_sm  = -1;
    bool                       use_deepep_internode   = false;
    bool                       use_deepep_low_latency = false;
    bool                       is_mtp                 = false;
    bool                       is_eagle3              = false;
    FMHAConfig                 fmha_config;
    HWKernelConfig             hw_kernel_config;
    DeviceResourceConfig       device_resource_config;
    MoeConfig                  moe_config;
    SpeculativeExecutionConfig sp_config;

    // FIFOSchedulerConfig fields are now in RuntimeConfig
    RuntimeConfig               runtime_config;
    MiscellaneousConfig         misc_config;
    ParallelismConfig           parallelism_config;
    ProfilingDebugLoggingConfig profile_debug_logging_config;
    ModelSpecificConfig         model_specific_config;
    ConcurrencyConfig           concurrency_config;
};

// immutable device properties. Can not change since device is initialized.
struct ExecProperties {
    DeviceType type;
    size_t     id = 0;

    /* -- distributed properties -- */
    size_t tp_rank = 0;
    size_t tp_size = 1;

    size_t dp_rank     = 0;
    size_t dp_size     = 1;
    size_t ffn_tp_rank = 0;
    size_t ffn_tp_size = 1;

    bool   enable_sp             = false;
    size_t overlap_math_sm_count = 0;
    size_t overlap_comm_type     = 0;
    size_t m_split               = 0;

    /* -- device implementation detail -- */
    // These two options are prepared for intel cpu device.
    // xfastertransformer fuses adding residual in their layer implementation.
    bool attn_fuse_add_residual  = false;
    bool ffn_fuse_add_residual   = false;
    bool sq_fuse_bias_activation = false;

    bool           enable_comm_overlap      = true;
    MicroBatchType enable_layer_micro_batch = MicroBatchType::NONE;

    bool          use_deepep_moe         = false;
    bool          use_deepep_internode   = false;
    bool          use_deepep_low_latency = false;
    bool          is_mtp                 = false;
    bool          use_all_gather         = false;
    bool          is_eagle3              = false;
    std::set<int> eagle3_selected_layer{1, 46, 90};
    // std::set<int> eagle3_selected_layer{0,1,2};
    bool ffn_as_service    = false;
    bool enable_prefill_cp = false;
};

struct MemoryStatus {
    size_t used_bytes         = 0;
    size_t free_bytes         = 0;
    size_t available_bytes    = 0;  // free GPU memory available for allocation
    size_t allocated_bytes    = 0;  // memory allocated via current device
    size_t max_consumed_bytes = 0;  // only applicable if RTP_LLM_TRACE_MEMORY is enabled.
};

// runtime device status, such as available memory.
struct ExecStatus {
    MemoryStatus device_memory_status;
    MemoryStatus host_memory_status;
};

inline ExecProperties buildExecProperties(const ExecInitParams& p) {
    ExecProperties props;
    props.type                     = buildDeviceType();
    props.id                       = p.device_id;
    props.use_all_gather           = p.use_all_gather;
    props.tp_rank                  = p.tp_rank;
    props.tp_size                  = p.tp_size;
    props.dp_rank                  = p.dp_rank;
    props.dp_size                  = p.dp_size;
    props.enable_comm_overlap      = p.enable_comm_overlap;
    props.enable_layer_micro_batch = p.enable_layer_micro_batch;
    props.enable_sp                = p.enable_sp;
    props.overlap_math_sm_count    = p.device_resource_config.overlap_math_sm_count;
    props.overlap_comm_type        = p.device_resource_config.overlap_comm_type;
    props.ffn_tp_size              = p.ffn_tp_size;
    props.ffn_tp_rank              = p.ffn_tp_rank;
    props.m_split                  = p.m_split;
    props.use_deepep_moe           = p.use_deepep_moe;
    props.use_deepep_internode     = p.use_deepep_internode;
    props.use_deepep_low_latency   = p.use_deepep_low_latency;
    props.is_mtp                   = p.is_mtp;
    props.is_eagle3                = p.is_eagle3;
    props.ffn_as_service           = p.ffn_as_service;
    props.enable_prefill_cp        = p.enable_prefill_cp;
    return props;
}

}  // namespace rtp_llm
