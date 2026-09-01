#pragma once

#include <cstdint>
#include <memory>
#include <torch/torch.h>

#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/runtime/DeviceStatus.h"

namespace rtp_llm {

MlaOpsType initRuntime(std::size_t device_id, bool trace_memory, bool enable_comm_overlap, MlaOpsType mla_ops_type);

// Process runtime state used outside the initialization path:
// - cache-store allocation must reject requests before initialization;
// - engine worker threads must select the initialized device;
// - copy operations must honor the configured overlap policy.
bool    isRuntimeInitialized();
int64_t getDeviceId();
bool    getEnableCommOverlap();

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
