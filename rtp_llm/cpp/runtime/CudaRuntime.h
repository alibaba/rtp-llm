#pragma once

#include <cstdint>
#include <memory>
#include <torch/torch.h>

#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {

// ===================================================================
// Transitional initRuntime wrapper.
//
// Calls process::bootstrap, then resolves MlaOpsType::AUTO via current GPU
// arch and sets the legacy globals (g_device_id, g_enable_comm_overlap,
// g_runtime_initialized) consumed by the accessors below. To be deleted
// once all callers migrate to process::bootstrap + resolveMlaOpsType.
// ===================================================================

MlaOpsType initRuntime(std::size_t device_id, bool trace_memory, bool enable_comm_overlap, MlaOpsType mla_ops_type);

bool    isRuntimeInitialized();
int64_t getDeviceId();
bool    getEnableCommOverlap();

// Resolves AUTO to FLASH_MLA / FLASH_INFER based on the current GPU arch.
// Other values pass through unchanged.
MlaOpsType resolveMlaOpsType(MlaOpsType requested);

// ===================================================================
// Sync / error-check
// ===================================================================

void runtimeSyncAndCheck();
void cudaSyncAndCheck();
void cudaCheckLastError();
void cudaPreRun(int device_id);

// ===================================================================
// Profiling
// ===================================================================

void cudaProfilerBegin();
void cudaProfilerEnd();

// ===================================================================
// Events
// ===================================================================

std::shared_ptr<torch::Event> runtimeCreateEvent();

// ===================================================================
// Status queries
// ===================================================================

ExecStatus getGpuExecStatus();

}  // namespace rtp_llm
