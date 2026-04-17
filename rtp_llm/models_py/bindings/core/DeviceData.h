#pragma once
#include <cstdint>
#include <stddef.h>
#include <string>
#include "rtp_llm/models_py/bindings/core/Types.h"
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

inline ExecProperties buildExecProperties(size_t                      device_id,
                                          const ParallelismConfig&    parallelism_config,
                                          const DeviceResourceConfig& device_resource_config,
                                          const MoeConfig&            moe_config,
                                          bool                        is_mtp,
                                          bool                        is_eagle3) {
    ExecProperties props;
    props.type = buildDeviceType();
    props.id   = device_id;
    // From parallelism_config
    props.tp_rank           = parallelism_config.tp_rank;
    props.tp_size           = parallelism_config.tp_size;
    props.dp_rank           = parallelism_config.dp_rank;
    props.dp_size           = parallelism_config.dp_size;
    props.ffn_tp_size       = parallelism_config.ffn_tp_size;
    props.ffn_tp_rank       = parallelism_config.ffn_tp_rank;
    props.enable_sp         = parallelism_config.enable_sp;
    props.enable_prefill_cp = parallelism_config.prefill_cp_config.is_enabled();
    props.ffn_as_service    = parallelism_config.ffn_disaggregate_config.is_ffn_service();
    // From device_resource_config
    props.enable_comm_overlap      = device_resource_config.enable_comm_overlap;
    props.enable_layer_micro_batch = static_cast<MicroBatchType>(device_resource_config.enable_layer_micro_batch);
    props.overlap_math_sm_count    = device_resource_config.overlap_math_sm_count;
    props.overlap_comm_type        = device_resource_config.overlap_comm_type;
    props.m_split                  = device_resource_config.m_split;
    // From moe_config
    props.use_deepep_moe         = moe_config.use_deepep_moe;
    props.use_deepep_internode   = moe_config.use_deepep_internode;
    props.use_deepep_low_latency = moe_config.use_deepep_low_latency;
    props.use_all_gather         = moe_config.use_all_gather && !moe_config.use_deepep_low_latency;
    // Computed during init
    props.is_mtp    = is_mtp;
    props.is_eagle3 = is_eagle3;
    return props;
}

}  // namespace rtp_llm
