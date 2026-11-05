#pragma once
#include <cstdint>
#include <stddef.h>
#include <string>
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

struct ExecProperties {
    size_t tp_rank = 0;
    size_t tp_size = 1;

    bool ffn_as_service              = false;
    bool enable_prefill_cp           = false;
    bool prefill_cp_kv_cache_sharded = false;
};

struct MemoryStatus {
    size_t used_bytes         = 0;
    size_t free_bytes         = 0;
    size_t available_bytes    = 0;  // free GPU memory available for allocation
    size_t allocated_bytes    = 0;  // memory allocated via current device
    size_t max_consumed_bytes = 0;  // peak device memory consumed; currently always 0.
};

// runtime device status, such as available memory.
struct ExecStatus {
    MemoryStatus device_memory_status;
    MemoryStatus host_memory_status;
};

inline ExecProperties buildExecProperties(const ParallelismConfig& parallelism_config) {
    ExecProperties props;
    props.tp_rank                     = parallelism_config.tp_rank;
    props.tp_size                     = parallelism_config.tp_size;
    props.enable_prefill_cp           = parallelism_config.prefill_cp_config.is_enabled();
    props.prefill_cp_kv_cache_sharded = parallelism_config.prefill_cp_config.is_enabled()
                                        && parallelism_config.prefill_cp_config.kv_cache_sharded
                                        && parallelism_config.tp_size > 1;
    props.ffn_as_service           = parallelism_config.ffn_disaggregate_config.is_ffn_service();
    return props;
}

}  // namespace rtp_llm
