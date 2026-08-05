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

    bool ffn_as_service              = false;
    bool enable_prefill_cp           = false;
    bool prefill_cp_kv_cache_sharded = false;
};

struct MemoryStatus {
    size_t used_bytes = 0;
    size_t free_bytes = 0;
    // Device total as reported by cudaMemGetInfo/hipMemGetInfo. Carried explicitly so
    // callers that need the total (the safety-ratio base in MemoryEvaluationHelper) do
    // not have to reconstruct it as used + free, which only works because used is
    // computed as total - free right here.
    size_t total_bytes     = 0;
    size_t available_bytes = 0;  // free GPU memory available for allocation
    size_t allocated_bytes = 0;  // memory allocated via current device
    // Torch allocator peak growth over the traced window; only set while tracing. Always part of
    // the KV-cache budget: the allocator high-water mark is released again when the traced
    // executor is destroyed, so serving has to re-earn it.
    size_t max_consumed_bytes = 0;
    // Non-torch (driver-side) resident delta at the moment of sampling. Only the share of it that
    // does NOT survive the warmup teardown belongs in the KV budget; sample it twice and use
    // transientNonTorchBytes() below to get that share.
    size_t non_torch_increase_bytes = 0;
};

struct MemoryGrowthBreakdown {
    size_t torch_peak_increase_bytes = 0;
    size_t non_torch_increase_bytes  = 0;
};

inline MemoryGrowthBreakdown calculateMemoryGrowth(size_t reserved_baseline_bytes,
                                                   size_t reserved_peak_bytes,
                                                   size_t reserved_current_bytes,
                                                   size_t cuda_used_baseline_bytes,
                                                   size_t cuda_used_current_bytes) {
    const size_t torch_peak_increase =
        reserved_peak_bytes > reserved_baseline_bytes ? reserved_peak_bytes - reserved_baseline_bytes : 0;
    const size_t non_torch_current =
        cuda_used_current_bytes > reserved_current_bytes ? cuda_used_current_bytes - reserved_current_bytes : 0;
    const size_t non_torch_baseline =
        cuda_used_baseline_bytes > reserved_baseline_bytes ? cuda_used_baseline_bytes - reserved_baseline_bytes : 0;
    const size_t non_torch_increase =
        non_torch_current > non_torch_baseline ? non_torch_current - non_torch_baseline : 0;
    return {torch_peak_increase, non_torch_increase};
}

// Share of the warmup's non-torch growth that serving has to allocate again.
//
// in_forward_bytes is sampled at the end of the traced forward, post_teardown_bytes after the
// traced executor has been released (and emptyCache() run) but still inside the trace window.
// Whatever is still there post teardown is process-global (lazily loaded kernel modules,
// cuBLAS/cuDNN handle state, comm buffers) and is therefore already missing from the
// available_bytes the KV budget is derived from -- reserving it would subtract it twice. The
// difference was released with the executor (driver-side CUDA graph bookkeeping, NCCL
// graph-capture registrations -- both scale with the number of captured graphs, so this term is
// far from negligible on the decode path), is back in available_bytes, and must be reserved.
inline size_t transientNonTorchBytes(size_t in_forward_bytes, size_t post_teardown_bytes) {
    return in_forward_bytes > post_teardown_bytes ? in_forward_bytes - post_teardown_bytes : 0;
}

// runtime device status, such as available memory.
struct ExecStatus {
    MemoryStatus device_memory_status;
    MemoryStatus host_memory_status;
};

inline ExecProperties buildExecProperties(const ParallelismConfig&    parallelism_config,
                                          const DeviceResourceConfig& device_resource_config) {
    ExecProperties props;
    props.tp_rank                     = parallelism_config.tp_rank;
    props.tp_size                     = parallelism_config.tp_size;
    props.enable_sp                   = parallelism_config.enable_sp;
    props.enable_prefill_cp           = parallelism_config.prefill_cp_config.is_enabled();
    props.prefill_cp_kv_cache_sharded = parallelism_config.prefill_cp_config.is_enabled()
                                        && parallelism_config.prefill_cp_config.kv_cache_sharded
                                        && parallelism_config.tp_size > 1;
    props.ffn_as_service           = parallelism_config.ffn_disaggregate_config.is_ffn_service();
    props.enable_layer_micro_batch = static_cast<MicroBatchType>(device_resource_config.enable_layer_micro_batch);
    props.overlap_comm_type        = device_resource_config.overlap_comm_type;
    props.m_split                  = device_resource_config.m_split;
    return props;
}

}  // namespace rtp_llm
