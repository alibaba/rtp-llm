#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_util.h"

#include <memory>
#include <atomic>
#include <mutex>

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#endif

namespace rtp_llm {

class CacheStore;

// ===================================================================
// Runtime lifecycle
// ===================================================================

// Perform one-time runtime init: cudaSetDevice, global flags, etc.
// Returns the resolved MlaOpsType (AUTO → FLASH_MLA/FLASH_INFER based on GPU arch).
MlaOpsType initRuntime(size_t device_id, bool trace_memory, bool enable_comm_overlap, MlaOpsType mla_ops_type);

bool isRuntimeInitialized();

int64_t getDeviceId();

// ===================================================================
// Sync / error-check
// ===================================================================

void runtimeSyncAndCheck();
void cudaSyncAndCheck();
void cudaCheckLastError();
void cudaPreRun(int device_id);

// ===================================================================
// Config accessors (set once during initRuntime)
// ===================================================================

bool getEnableCommOverlap();

// ===================================================================
// Profiling
// ===================================================================

void cudaProfilerBegin();
void cudaProfilerEnd();

// ===================================================================
// Status queries
// ===================================================================

ExecStatus    getGpuExecStatus();
torch::Device getTorchCudaDevice();
void          setTraceMemory(bool trace_memory);

// ===================================================================
// Copy ops
// ===================================================================

void runtimeCopy(const CopyParams& params);
void runtimeBatchCopy(const BatchCopyParams& params);
void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask);

void execNoBlockCopy(const CopyParams& params);
void execBatchCopy(const BatchCopyParams& params);
void execMultiMergeCopy(const MultiMergeCopyParams& params);

void fusedCopy(const FusedD2DCopyParams& params);
void fusedStridedCopy(const FusedStridedCopyParams& params);

// ===================================================================
// Sample ops
// ===================================================================

GreedyOutput     execSampleGreedy(const GreedyParams& params);
BeamSearchOutput execSampleBeamSearch(const BeamSearchParams& params);
void             execChainSpeculativeSampling(const SpeculativeSamplingParams& params);

// ===================================================================
// Communication ops (backed by c10d ProcessGroup)
// ===================================================================

void            execBroadcast(const BroadcastParams& params);
AllReduceOutput execAllReduce(const AllReduceParams& params);
void            execAllGather(const AllGatherParams& params);
void            execSyncCommunication(bool timeout = true);
void            execSyncCommunication(ParallelMode mode, bool timeout = true);

// ===================================================================
// MOE / EPLB
// ===================================================================

OverallExpertStats execCreateMoeExpertStates(const ExpertStatsParams& params);

// ===================================================================
// Events
// ===================================================================

std::shared_ptr<torch::Event> runtimeCreateEvent();

// ===================================================================
// CacheStore (cache_store passed explicitly; see KVCacheManager::getCacheStore)
// ===================================================================

void runtimeWriteCacheStore(const CacheStoreInputs&     inputs,
                            const KvCacheInfo&          kv_cache,
                            bool                        mla_kvcache,
                            std::shared_ptr<CacheStore> cache_store);
void execWriteCacheStore(const CacheStoreInputs&     inputs,
                         const KvCacheInfo&          kv_cache,
                         bool                        mla_kvcache,
                         std::shared_ptr<CacheStore> cache_store);

// ===================================================================
// Static ops (weight preprocessing)
// ===================================================================

torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai);
torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale);

}  // namespace rtp_llm
