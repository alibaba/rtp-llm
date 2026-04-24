#pragma once
#include <cstdint>
#include <stddef.h>
#include <string>
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

enum class MicroBatchType {
    NONE       = 0,
    DS_PREFILL = 1,
    DS_DECODE  = 2,
};

struct ExecProperties {
    size_t tp_rank = 0;
    size_t tp_size = 1;

    bool   enable_sp         = false;
    size_t overlap_comm_type = 0;
    size_t m_split           = 0;

    MicroBatchType enable_layer_micro_batch = MicroBatchType::NONE;

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

inline ExecProperties buildExecProperties(const ParallelismConfig&    parallelism_config,
                                          const DeviceResourceConfig& device_resource_config) {
    ExecProperties props;
    props.tp_rank                  = parallelism_config.tp_rank;
    props.tp_size                  = parallelism_config.tp_size;
    props.enable_sp                = parallelism_config.enable_sp;
    props.enable_prefill_cp        = parallelism_config.prefill_cp_config.is_enabled();
    props.ffn_as_service           = parallelism_config.ffn_disaggregate_config.is_ffn_service();
    props.enable_layer_micro_batch = static_cast<MicroBatchType>(device_resource_config.enable_layer_micro_batch);
    props.overlap_comm_type        = device_resource_config.overlap_comm_type;
    props.m_split                  = device_resource_config.m_split;
    return props;
}

}  // namespace rtp_llm
