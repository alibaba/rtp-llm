#pragma once

#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/core/DeviceData.h"
#include "rtp_llm/cpp/core/Event.h"
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

namespace torch_ext {
class ExecCtxExporter;
}

namespace rtp_llm {

class CacheStore;

// ===================================================================
// Runtime lifecycle
// ===================================================================

ExecInitParams initExecCtx(const ParallelismConfig&           parallelism_config,
                           const ModelConfig&                 model_config,
                           const EPLBConfig&                  eplb_config,
                           const FMHAConfig&                  fmha_config,
                           const DeviceResourceConfig&        device_resource_config,
                           const MoeConfig&                   moe_config,
                           const SpeculativeExecutionConfig&  sp_config,
                           const MiscellaneousConfig&         misc_config,
                           const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                           const HWKernelConfig&              hw_kernel_config,
                           const ConcurrencyConfig&           concurrency_config,
                           const FfnDisAggregateConfig&       ffn_disaggregate_config,
                           const RuntimeConfig&               runtime_config,
                           const ModelSpecificConfig&         model_specific_config,
                           const NcclCommConfig&              nccl_comm_config = NcclCommConfig{});

bool isRuntimeInitialized();

std::shared_ptr<torch_ext::ExecCtxExporter> getExecCtxExporter();

// ===================================================================
// Sync / error-check
// ===================================================================

void runtimeSyncAndCheck();
void cudaSyncAndCheck();
void cudaCheckLastError();
void cudaPreRun(int device_id);

// ===================================================================
// Config accessors (set once during initExecCtx)
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
MemoryStatus  getGpuMemoryStatus();
torch::Device getTorchCudaDevice();
void          setTraceMemory(bool trace_memory);
bool          getTraceMemory();

// ===================================================================
// Copy ops
// ===================================================================

void runtimeCopy(const CopyParams& params);
void runtimeBatchCopy(const BatchCopyParams& params);
void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask);

void execCopy(const CopyParams& params);
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

AsyncEventPtr runtimeCreateEvent();

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

// ===================================================================
// Misc
// ===================================================================

void execMaskLogits(torch::Tensor& logits, const torch::Tensor& mask);

}  // namespace rtp_llm
