#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "autil/StackTracer.h"
#include "autil/EnvUtil.h"
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <atomic>
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#elif USING_ROCM
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#endif
#include <pybind11/functional.h>

#if USING_CUDA
using DeviceGuard = c10::cuda::CUDAGuard;
#elif USING_ROCM
using DeviceGuard = c10::hip::HIPGuardMasqueradingAsCUDA;
#endif

namespace rtp_llm {
GreedyOutput     sampleGreedy(const GreedyParams& params);
BeamSearchOutput sampleBeamSearch(BeamSearchParams params);
void             chainSpeculativeSampling(const SpeculativeSamplingParams& params);
void             rejectionSampling(const RejectionSamplingParams& params);
void             multiMergeCopy(const MultiMergeCopyParams& params);
}  // namespace rtp_llm

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

using namespace std;

namespace py = pybind11;

namespace rtp_llm {

// ============================================================
// Module-level init guards (minimal state - no cache_store here)
// ============================================================

namespace {
static std::atomic<bool> g_runtime_initialized{false};
static std::once_flag    g_init_flag;

static bool g_enable_comm_overlap = true;

static int64_t g_device_id = 0;
}  // anonymous namespace

// ============================================================
// Runtime state query
// ============================================================

bool isRuntimeInitialized() {
    return g_runtime_initialized.load(std::memory_order_acquire);
}

// ============================================================
// Config accessors
// ============================================================

bool getEnableCommOverlap() {
    return g_enable_comm_overlap;
}

int64_t getDeviceId() {
    return g_device_id;
}

// ============================================================
// Sync / check
// ============================================================

#if USING_CUDA

void runtimeSyncAndCheck() {
    check_cuda_value(cudaDeviceSynchronize());
    check_cuda_error();
}

#else  // ROCm

void runtimeSyncAndCheck() {
    ROCM_CHECK(hipDeviceSynchronize());
    ROCM_CHECK_ERROR();
}

#endif  // USING_CUDA

// ============================================================
// Events
// ============================================================

#if USING_CUDA

std::shared_ptr<torch::Event> runtimeCreateEvent() {
    auto event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(at::cuda::getCurrentCUDAStream());
    return event;
}

#else  // ROCm

std::shared_ptr<torch::Event> runtimeCreateEvent() {
    auto event = std::make_shared<torch::Event>(torch::kHIP);
    event->record(at::hip::getCurrentHIPStream(at::hip::current_device()));
    return event;
}

#endif  // USING_CUDA

// ============================================================
// CacheStore (cache_store passed explicitly from KVCacheManager)
// ============================================================

void runtimeWriteCacheStore(const CacheStoreInputs&     cache_store_inputs,
                            const KvCacheInfo&          kv_cache,
                            bool                        mla_kvcache,
                            std::shared_ptr<CacheStore> cache_store) {
    if (cache_store_inputs.warmup) {
        RTP_LLM_LOG_DEBUG("is warmup, so ignore writeCacheStore");
        return;
    }
    if (!cache_store_inputs.pd_separation || cache_store_inputs.context_batch_size == 0) {
        RTP_LLM_LOG_DEBUG("pd_separation = %d, context_batch_size = %d, so ignore writeCacheStore",
                          cache_store_inputs.pd_separation,
                          cache_store_inputs.context_batch_size);
        return;
    }
    if (!cache_store) {
        RTP_LLM_LOG_DEBUG("cache_store is null, skip writeCacheStore");
        return;
    }

    // Wait for the CUDA event before reading pinned-host metadata.
    // The event was recorded on the main stream AFTER both the async D2H
    // copies (metadata) and KV cache writes were enqueued, so blocking
    // here guarantees all pinned buffers are populated.
    if (cache_store_inputs.pre_created_event) {
        cache_store_inputs.pre_created_event->synchronize();
    }

    const auto& param = cache_store_inputs;

    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.defined(), "failed to get host_kv_cache_offset");
    const int32_t* offset_addr          = nullptr;
    size_t         max_blocks_per_batch = 0;

    RTP_LLM_CHECK_WITH_INFO(!param.tag.empty(), "cache-store write requires a cache tag for layer=%d", param.layer_id);
    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.dim() == 2,
                            "cache-store block table for tag=%s must be group-local [batch, blocks], got dim=%ld",
                            param.tag.c_str(),
                            param.host_kv_cache_offset.dim());
    max_blocks_per_batch = param.host_kv_cache_offset.size(1);
    offset_addr          = param.host_kv_cache_offset.data_ptr<int32_t>();

    const auto policy_it = param.kv_cache_group_policies.find(param.tag);
    RTP_LLM_CHECK_WITH_INFO(policy_it != param.kv_cache_group_policies.end(),
                            "cache-store metadata has no group policy for tag=%s",
                            param.tag.c_str());
    const CacheGroupPolicy group_policy                    = policy_it->second;
    const bool             use_group_cache_transfer_policy = param.kv_cache_group_policies.size() > 1;

    const auto seq_it = param.tokens_per_block_by_tag.find(param.tag);
    const auto seq_size_per_block =
        seq_it != param.tokens_per_block_by_tag.end() ? seq_it->second : param.tokens_per_block;
    const auto kv_stride_it = param.kv_block_stride_bytes_by_tag.find(param.tag);
    const auto kv_block_stride_bytes =
        kv_stride_it != param.kv_block_stride_bytes_by_tag.end() ? kv_stride_it->second : param.kv_block_stride_bytes;
    const auto scale_stride_it       = param.kv_scale_stride_bytes_by_tag.find(param.tag);
    const auto kv_scale_stride_bytes = scale_stride_it != param.kv_scale_stride_bytes_by_tag.end() ?
                                           scale_stride_it->second :
                                           param.kv_scale_stride_bytes;
    const auto kv_transfer_it        = param.kv_block_transfer_bytes_by_tag.find(param.tag);
    const auto kv_block_transfer_bytes =
        kv_transfer_it != param.kv_block_transfer_bytes_by_tag.end() ? kv_transfer_it->second : kv_block_stride_bytes;
    const auto scale_transfer_it       = param.kv_scale_transfer_bytes_by_tag.find(param.tag);
    const auto kv_scale_transfer_bytes = scale_transfer_it != param.kv_scale_transfer_bytes_by_tag.end() ?
                                             scale_transfer_it->second :
                                             kv_scale_stride_bytes;
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "cache-store tag=%s has zero tokens_per_block", param.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        kv_block_stride_bytes > 0, "cache-store tag=%s has zero kv block stride", param.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        kv_block_transfer_bytes > 0, "cache-store tag=%s has zero kv transfer bytes", param.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(kv_block_transfer_bytes <= kv_block_stride_bytes,
                            "cache-store tag=%s transfer bytes=%zu exceed physical stride=%zu",
                            param.tag.c_str(),
                            kv_block_transfer_bytes,
                            kv_block_stride_bytes);
    RTP_LLM_CHECK_WITH_INFO(kv_scale_transfer_bytes <= kv_scale_stride_bytes,
                            "cache-store tag=%s scale transfer bytes=%zu exceed physical stride=%zu",
                            param.tag.c_str(),
                            kv_scale_transfer_bytes,
                            kv_scale_stride_bytes);
    auto       kv_cache_data  = (uint64_t*)kv_cache.kv_cache_buffer.data_ptr();
    auto       kv_cache_owner = std::make_shared<torch::Tensor>(kv_cache.kv_cache_buffer);
    const bool kv_gpu_mem     = kv_cache.kv_cache_buffer.is_cuda();
    const bool has_kv_scale   = kv_cache.kv_scale_buffer.defined() && kv_cache.kv_scale_buffer.numel() > 0
                              && kv_scale_stride_bytes > 0 && kv_scale_transfer_bytes > 0;
    uint64_t*                      kv_scale_data = nullptr;
    std::shared_ptr<torch::Tensor> kv_scale_owner;
    if (has_kv_scale) {
        kv_scale_data  = (uint64_t*)kv_cache.kv_scale_buffer.data_ptr();
        kv_scale_owner = std::make_shared<torch::Tensor>(kv_cache.kv_scale_buffer);
    }
    const bool kv_scale_gpu_mem = has_kv_scale && kv_cache.kv_scale_buffer.is_cuda();

    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_pd_separation.numel()),
                            "size not same");
    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_id.numel()),
                            "context batch size and request id size is not same");

    RTP_LLM_LOG_DEBUG("write cache store, context_batch_size is %ld", param.context_batch_size);

    // cache_keys is laid out [batch, global_max_blocks]; this stride is INDEPENDENT
    // of `max_blocks_per_batch` (which is per-group offset stride and may be smaller
    // for CP-sharded FULL groups whose offset is rank-local-compact).
    const size_t cache_keys_per_batch =
        param.context_batch_size > 0 ? (param.cache_keys.size() / param.context_batch_size) : 0;

    for (size_t batch_id = 0; batch_id < param.context_batch_size; batch_id++) {
        if (*(param.request_pd_separation.data_ptr<bool>() + batch_id) == false) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.defined() && param.input_lengths_host.defined(),
                                "failed to get prefix_length_host and input_length_host for cache store");
        const bool uses_cp_canonical_keys = param.cp_size > 1 && group_policy.cp_mapping != CpBlockMappingMode::NONE
                                            && seq_size_per_block % param.cp_size == 0;
        const size_t canonical_seq_size_per_block =
            uses_cp_canonical_keys ? seq_size_per_block / static_cast<size_t>(param.cp_size) : seq_size_per_block;
        const int prefix_length = param.prefix_lengths_host.data_ptr<int>()[batch_id];
        RTP_LLM_CHECK_WITH_INFO(prefix_length % static_cast<int>(canonical_seq_size_per_block) == 0,
                                "cache-store tag=%s prefix_length=%d is not aligned to canonical tokens_per_block=%zu "
                                "(physical tokens_per_block=%zu, cp_size=%d)",
                                param.tag.c_str(),
                                prefix_length,
                                canonical_seq_size_per_block,
                                seq_size_per_block,
                                param.cp_size);
        int reuse_block_num = prefix_length / seq_size_per_block;
        int block_num =
            (param.input_lengths_host.data_ptr<int>()[param.decoder_batch_size + batch_id] + seq_size_per_block - 1)
            / seq_size_per_block;
        int canonical_reuse_block_num = prefix_length / canonical_seq_size_per_block;
        int canonical_block_num       = (param.input_lengths_host.data_ptr<int>()[param.decoder_batch_size + batch_id]
                                   + canonical_seq_size_per_block - 1)
                                  / canonical_seq_size_per_block;
        auto request_id     = *(param.request_id.data_ptr<int64_t>() + batch_id);
        auto event          = param.pre_created_event ? param.pre_created_event : runtimeCreateEvent();
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), event);
        RTP_LLM_LOG_DEBUG(
            "write cache store, request id is %ld, blocks num is %ld", request_id, block_num + reuse_block_num);

        const int canonical_total_blocks = canonical_block_num + canonical_reuse_block_num;
        const int total_blocks = uses_cp_canonical_keys ? (canonical_total_blocks + param.cp_size - 1) / param.cp_size :
                                                          block_num + reuse_block_num;
        if (total_blocks <= 0) {
            continue;
        }

        auto addBlock = [&](int key_index, int offset_index) {
            RTP_LLM_CHECK_WITH_INFO(offset_index >= 0 && offset_index < static_cast<int>(max_blocks_per_batch),
                                    "invalid block offset_index=%d (max_blocks_per_batch=%zu)",
                                    offset_index,
                                    max_blocks_per_batch);
            RTP_LLM_CHECK_WITH_INFO(key_index >= 0 && key_index < static_cast<int>(cache_keys_per_batch),
                                    "invalid block key_index=%d (cache_keys_per_batch=%zu)",
                                    key_index,
                                    cache_keys_per_batch);
            std::string cache_key = makeCacheKey(param.model_id,
                                                 param.cache_keys[batch_id * cache_keys_per_batch + key_index],
                                                 param.layer_id,
                                                 param.tag);
            auto        block_id =
                *(offset_addr + (param.decoder_batch_size + batch_id) * max_blocks_per_batch + offset_index);
            // Host block-offset tables use -1 as the null block sentinel.
            if (block_id == -1) {
                RTP_LLM_LOG_DEBUG(
                    "PD_CACHE_KEY_WRITE_SKIP_NULL key=kv_%s request_id=%ld tag=%s layer=%d cp_rank=%d cp_size=%d "
                    "key_index=%d offset_index=%d block_id=%d",
                    cache_key.c_str(),
                    request_id,
                    param.tag.c_str(),
                    param.layer_id,
                    param.cp_rank,
                    param.cp_size,
                    key_index,
                    offset_index,
                    block_id);
                return;
            }
            const bool has_policy_cp_slice = param.cp_size > 1 && group_policy.cp_slice != CpBlockSliceMode::NONE;
            if (has_policy_cp_slice) {
                RTP_LLM_CHECK_WITH_INFO(param.cp_rank >= 0 && param.cp_rank < param.cp_size,
                                        "cache-store tag=%s invalid cp_rank=%d cp_size=%d",
                                        param.tag.c_str(),
                                        param.cp_rank,
                                        param.cp_size);
                // The prefill topology already materializes each rank's local
                // STATE/SWA row. Send that complete local row from offset zero;
                // decode applies the peer-rank offset in the corresponding
                // full row. Dividing here would slice an already-sliced row.
            }

            const bool use_opaque_key_prefix =
                param.use_opaque_kv_cache_store || use_group_cache_transfer_policy || mla_kvcache;
            void*                 kv_addr = (void*)((int8_t*)kv_cache_data + block_id * kv_block_stride_bytes);
            std::shared_ptr<void> kv_block_addr(kv_cache_owner, kv_addr);
            RTP_LLM_LOG_DEBUG("PD_CACHE_KEY_WRITE_BLOCK key=kv_%s request_id=%ld tag=%s layer=%d cp_rank=%d "
                              "cp_size=%d cp_slice=%d key_index=%d offset_index=%d block_id=%d addr=%p "
                              "physical_stride=%zu len=%zu",
                              cache_key.c_str(),
                              request_id,
                              param.tag.c_str(),
                              param.layer_id,
                              param.cp_rank,
                              param.cp_size,
                              static_cast<int>(group_policy.cp_slice),
                              key_index,
                              offset_index,
                              block_id,
                              kv_addr,
                              kv_block_stride_bytes,
                              kv_block_transfer_bytes);
            if (use_opaque_key_prefix) {
                request_blocks->addBlock("kv_" + cache_key, kv_block_addr, kv_block_transfer_bytes, kv_gpu_mem, true);
            } else {
                RTP_LLM_CHECK_WITH_INFO(kv_block_transfer_bytes % 2 == 0,
                                        "KV transfer bytes must split evenly into K/V");
                const uint32_t        kv_half = static_cast<uint32_t>(kv_block_transfer_bytes / 2);
                void*                 k_addr  = kv_addr;
                void*                 v_addr  = (void*)((int8_t*)kv_addr + kv_half);
                std::shared_ptr<void> k_block_addr(kv_cache_owner, k_addr);
                std::shared_ptr<void> v_block_addr(kv_cache_owner, v_addr);
                request_blocks->addBlock("k_" + cache_key, k_block_addr, kv_half, kv_gpu_mem, true);
                request_blocks->addBlock("v_" + cache_key, v_block_addr, kv_half, kv_gpu_mem, true);
            }

            if (kv_scale_data) {
                void* kv_scale_addr = (void*)((int8_t*)kv_scale_data + block_id * kv_scale_stride_bytes);

                std::shared_ptr<void> kv_scale_block_addr(kv_scale_owner, kv_scale_addr);
                if (use_opaque_key_prefix) {
                    request_blocks->addBlock(
                        "kv_scale_" + cache_key, kv_scale_block_addr, kv_scale_transfer_bytes, kv_scale_gpu_mem, true);
                } else {
                    RTP_LLM_CHECK_WITH_INFO(kv_scale_transfer_bytes % 2 == 0,
                                            "scale transfer bytes must split evenly into K/V");
                    const uint32_t        sc_half = static_cast<uint32_t>(kv_scale_transfer_bytes / 2);
                    void*                 k_sc    = kv_scale_addr;
                    void*                 v_sc    = (void*)((int8_t*)kv_scale_addr + sc_half);
                    std::shared_ptr<void> k_scale_block_addr(kv_scale_owner, k_sc);
                    std::shared_ptr<void> v_scale_block_addr(kv_scale_owner, v_sc);
                    request_blocks->addBlock(
                        "k_scale_" + cache_key, k_scale_block_addr, sc_half, kv_scale_gpu_mem, true);
                    request_blocks->addBlock(
                        "v_scale_" + cache_key, v_scale_block_addr, sc_half, kv_scale_gpu_mem, true);
                }
            }
        };

        // Under CP sharding, kv_cache_offset can be rank-local-compact while
        // cache_keys stays in the full logical namespace. The common cache
        // policy owns the key/offset projection for both legacy and sharded cases.
        // Clamp by cache_keys_per_batch (global stride) -- NOT max_blocks_per_batch,
        // which under CP shard is the local-compact stride for FULL groups.
        const auto block_plan = buildCacheStorePlan(
            group_policy,
            static_cast<size_t>(std::min<int>(canonical_total_blocks, static_cast<int>(cache_keys_per_batch))),
            /*reuse_block_size=*/0,
            use_group_cache_transfer_policy,
            param.cp_rank,
            param.cp_size);
        for (const auto& pair : block_plan) {
            addBlock(pair.key_index, pair.offset_index);
        }

        auto storeCallback = [layer_id = param.layer_id,
                              model_id = param.model_id,
                              tag      = param.tag,
                              request_id,
                              request_blocks](bool success, CacheStoreErrorCode ec) {
            if (!success) {
                RTP_LLM_LOG_WARNING("PD_CACHE_KEY_WRITE_FAILED request_id=%ld model_id=%zu local_layer_id=%d tag=%s "
                                    "error_code=%d error=%s buffer={%s}",
                                    static_cast<long>(request_id),
                                    model_id,
                                    layer_id,
                                    tag.c_str(),
                                    static_cast<int>(ec),
                                    ErrorCodeToString(transCacheStoreErrorCode(ec)).c_str(),
                                    request_blocks->debugInfo().c_str());
            }
        };
        if (request_blocks->getBlocksCount() > 0) {
            cache_store->store(request_blocks, storeCallback);
        } else {
            RTP_LLM_LOG_DEBUG("skip cache store because all selected blocks are null, request id [%ld], layer id [%d]",
                              request_id,
                              param.layer_id);
        }
    }
}

// ============================================================
// Static ops (weight preprocessing)
// ============================================================

#if USING_CUDA
torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai) {
    return weight;
}

torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) {
    return weight;
}
#elif USING_ROCM
torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool user_arm_gemm_use_kai) {
    return weight;
}

torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) {
    return weight;
}
#endif

// ============================================================
// Sync / error check wrappers
// ============================================================

void cudaSyncAndCheck() {
    runtimeSyncAndCheck();
}

void cudaCheckLastError() {
#if USING_CUDA
    check_cuda_error();
#elif USING_ROCM
    auto err = hipGetLastError();
    if (err != hipSuccess) {
        RTP_LLM_LOG_ERROR("ROCm error: %s", hipGetErrorString(err));
    }
#endif
}

void cudaPreRun(int device_id) {
    setCurrentThreadDevice(device_id);
}

// ============================================================
// Profiling
// ============================================================

void cudaProfilerBegin() {
#if USING_CUDA
    check_cuda_value(cudaProfilerStart());
#endif
}

void cudaProfilerEnd() {
#if USING_CUDA
    check_cuda_value(cudaProfilerStop());
#endif
}

// ============================================================
// Status queries
// ============================================================

namespace {
// Gates forward peak-memory tracking; only enabled during warmup (setTraceMemory(true)).
static bool g_trace_memory = false;
#if USING_CUDA
// Baselines snapshotted right after emptyCache()+resetPeakStats() in setTraceMemory(true).
// Used to turn absolute readings into the forward's transient deltas (see getGpuExecStatus).
static size_t g_reserved_baseline_bytes  = 0;  // torch reserved at baseline
static size_t g_cuda_used_baseline_bytes = 0;  // device used (total-free) at baseline
#endif
}  // namespace

ExecStatus getGpuExecStatus() {
    MemoryStatus mem;
    size_t       total_bytes = 0;
#if USING_CUDA
    auto error = cudaMemGetInfo(&mem.free_bytes, &total_bytes);
    RTP_LLM_CHECK(error == cudaSuccess);
#elif USING_ROCM
    hipMemGetInfo(&mem.free_bytes, &total_bytes);
#endif
    mem.used_bytes      = total_bytes - mem.free_bytes;
    mem.available_bytes = mem.free_bytes;
#if USING_CUDA
    if (g_trace_memory) {
        // max_consumed_bytes = forward transient growth = torch_peak_increase + non_torch_increase
        // (vLLM-style decomposition). available_bytes downstream already excludes the steady-state
        // (weights/context), so we report only the growth on top of the baseline -- counting the
        // baseline here too would double-subtract it from the KV cache budget.
        const auto&  stats      = c10::cuda::CUDACachingAllocator::getDeviceStats(at::cuda::current_device());
        const size_t torch_peak = static_cast<size_t>(stats.reserved_bytes[0].peak);  // [0] = AGGREGATE
        const size_t torch_cur  = static_cast<size_t>(stats.reserved_bytes[0].current);
        mem.allocated_bytes     = static_cast<size_t>(stats.allocated_bytes[0].current);

        // cudaMemGetInfo is device-global, so this decomposition assumes the warmup rank has
        // exclusive use of its GPU during the measurement window. External allocations would be
        // attributed to non-torch growth.
        const auto growth = calculateMemoryGrowth(
            g_reserved_baseline_bytes, torch_peak, torch_cur, g_cuda_used_baseline_bytes, mem.used_bytes);
        mem.torch_peak_increase_bytes = growth.torch_peak_increase_bytes;
        mem.non_torch_increase_bytes  = growth.non_torch_increase_bytes;
        mem.max_consumed_bytes        = growth.max_consumed_bytes;
    }
#endif
    ExecStatus status;
    status.device_memory_status = mem;
    return status;
}

torch::Device getTorchCudaDevice() {
    return torch::Device(torch::kCUDA);
}

bool isTraceMemory() {
    return g_trace_memory;
}

void setTraceMemory(bool trace_memory) {
    g_trace_memory = trace_memory;
#if USING_CUDA
    if (trace_memory) {
        // Release loader-cached free blocks so the baseline is pure steady-state (weights), then
        // zero the peak high-water mark and snapshot the baselines. Without emptyCache, the forward
        // could reuse cached free blocks without growing reserved, making the measured delta too
        // small -> KV cache over-allocated -> runtime OOM.
        c10::cuda::CUDACachingAllocator::emptyCache();
        const auto device = at::cuda::current_device();
        c10::cuda::CUDACachingAllocator::resetPeakStats(device);
        g_reserved_baseline_bytes =
            static_cast<size_t>(c10::cuda::CUDACachingAllocator::getDeviceStats(device).reserved_bytes[0].current);
        size_t free_bytes = 0, total_bytes = 0;
        check_cuda_value(cudaMemGetInfo(&free_bytes, &total_bytes));
        g_cuda_used_baseline_bytes = total_bytes - free_bytes;  // for non_torch_increase
    } else {
        g_reserved_baseline_bytes  = 0;
        g_cuda_used_baseline_bytes = 0;
    }
#endif
}

// === Copy ops ===

void execNoBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
#if USING_CUDA
    const auto  copy_device = getCopyDevice(dst, src);
    DeviceGuard device_guard(copy_device);
    auto        stream = getNoBlockCopyStream(copy_device).stream();
    check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), src.nbytes(), cudaMemcpyDefault, stream));
    check_cuda_value(cudaStreamSynchronize(stream));
    check_cuda_error();
#elif USING_ROCM
    dst.copy_(src);
#else
    dst.copy_(src);
#endif
}

void execBatchCopy(const BatchCopyParams& params) {
    runtimeBatchCopy(params);
}

void execMultiMergeCopy(const MultiMergeCopyParams& params) {
    multiMergeCopy(params);
}

// === Sample ops ===

GreedyOutput execSampleGreedy(const GreedyParams& params) {
    return sampleGreedy(params);
}

BeamSearchOutput execSampleBeamSearch(BeamSearchParams params) {
    return sampleBeamSearch(std::move(params));
}

void execChainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    chainSpeculativeSampling(params);
}

void execRejectionSampling(const RejectionSamplingParams& params) {
    rejectionSampling(params);
}

// === Communication ops (Python callbacks via pybind11) ===

namespace {
std::mutex   g_comm_mutex;
py::function g_broadcast_fn;  // (tensors: list[Tensor], root: int, mode: int) -> None
py::function g_allreduce_fn;  // (tensor: Tensor, op: int, mode: int, dest: Optional[Tensor]) -> Tensor
py::function
    g_allgather_fn;  // (recv_buffers: list[Tensor], mode: int, send_buffers: list[Tensor], inplace: bool) -> None
}  // anonymous namespace

void execBroadcast(const BroadcastParams& params) {
    py::function fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        fn = g_broadcast_fn;
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execBroadcast called but broadcast callback not registered via register_comm_ops");
    py::gil_scoped_acquire gil;
    py::list               tensors;
    for (auto& t : params.buffers)
        tensors.append(t);
    fn(tensors, params.root, static_cast<int>(params.mode));
}

AllReduceOutput execAllReduce(const AllReduceParams& params) {
    py::function fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        fn = g_allreduce_fn;
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllReduce called but allreduce callback not registered via register_comm_ops");
    py::gil_scoped_acquire gil;
    auto                   result = fn(params.buffer,
                     static_cast<int>(params.op),
                     static_cast<int>(params.mode),
                     params.dest.defined() ? py::cast(params.dest) : py::none());
    return AllReduceOutput{result.cast<torch::Tensor>()};
}

void execAllGather(const AllGatherParams& params) {
    py::function fn;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        fn = g_allgather_fn;
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllGather called but allgather callback not registered via register_comm_ops");
    py::gil_scoped_acquire gil;
    py::list               recv_list, send_list;
    for (auto& t : params.recv_buffers)
        recv_list.append(t);
    for (auto& t : params.send_buffers)
        send_list.append(t);
    fn(recv_list, static_cast<int>(params.mode), send_list, params.inplace);
}

void execSyncCommunication(bool timeout) {
    (void)timeout;  // Python ops are synchronous
}

void execSyncCommunication(ParallelMode mode, bool timeout) {
    (void)mode;
    (void)timeout;  // Python ops are synchronous
}

// === CacheStore wrapper ===

void execWriteCacheStore(const CacheStoreInputs&     inputs,
                         const KvCacheInfo&          kv_cache,
                         bool                        mla_kvcache,
                         std::shared_ptr<CacheStore> cache_store) {
    runtimeWriteCacheStore(inputs, kv_cache, mla_kvcache, std::move(cache_store));
}

// ============================================================
// initRuntime — one-time runtime init (side effects only)
// ============================================================

MlaOpsType initRuntime(size_t device_id, bool trace_memory, bool enable_comm_overlap, MlaOpsType mla_ops_type) {
    MlaOpsType resolved_mla_ops_type = mla_ops_type;

    // Guard against double-init
    if (g_runtime_initialized.load(std::memory_order_acquire)) {
        RTP_LLM_LOG_WARNING("Runtime is already initialized! will do nothing.");
        return resolved_mla_ops_type;
    }

    std::call_once(g_init_flag, [&]() {
        setlinebuf(stdout);

        if (trace_memory) {
            autil::EnvUtil::setEnv("STACK_TRACER_LOG", "true");
            DECLARE_STACK_TRACER_FILE("rtp_llm_stack.log");
        }

#if USING_CUDA
        RTP_LLM_LOG_INFO("Initialize runtime. device_id=%zu", device_id);
        check_cuda_value(cudaSetDevice(device_id));
        at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());

        if (resolved_mla_ops_type == MlaOpsType::AUTO) {
            auto* prop            = at::cuda::getCurrentDeviceProperties();
            resolved_mla_ops_type = prop->major >= 9 ? MlaOpsType::FLASH_MLA : MlaOpsType::FLASH_INFER;
        }
#elif USING_ROCM
        RTP_LLM_LOG_INFO("Initialize runtime (ROCm). device_id=%zu", device_id);
        ROCM_CHECK(hipSetDevice(device_id));
#endif

        g_enable_comm_overlap = enable_comm_overlap;
        g_device_id           = device_id;

        g_runtime_initialized.store(true, std::memory_order_release);
        RTP_LLM_LOG_INFO("Runtime init done (communication via c10d ProcessGroup)");
    });

    RTP_LLM_LOG_INFO("init devices done");
    return resolved_mla_ops_type;
}

// === MOE / EPLB ===

OverallExpertStats execCreateMoeExpertStates(const ExpertStatsParams& params) {
    OverallExpertStats states;
    states.layer_num               = params.layer_num;
    states.ep_size                 = params.ep_size;
    states.log_exp_num             = params.log_exp_num;
    states.phy_exp_num             = params.phy_exp_num;
    states.stats_buf.log_stats_buf = torch::zeros({(int64_t)params.layer_num, (int64_t)params.log_exp_num},
                                                  torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    states.stats_buf.gpu_loads_buf = torch::zeros({(int64_t)params.layer_num, (int64_t)params.ep_size},
                                                  torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    return states;
}

// ============================================================
// Pybind registration
// ============================================================

void registerExecCtxOps(pybind11::module& m) {
    m.def("get_device_id", &getDeviceId);
    m.def("is_trace_memory", &isTraceMemory, "True while a warmup forward is being memory-traced.");
    m.def("preprocess_gemm_weight_by_key",
          &preprocessGemmWeightByKey,
          py::arg("key"),
          py::arg("weight"),
          py::arg("user_arm_gemm_use_kai"));
    m.def("preprocess_weight_scale", &preprocessWeightScale, py::arg("weight"), py::arg("scale"));

    m.def(
        "init_exec_ctx",
        [](size_t device_id, bool trace_memory, bool enable_comm_overlap, int mla_ops_type) {
            (void)initRuntime(device_id, trace_memory, enable_comm_overlap, static_cast<MlaOpsType>(mla_ops_type));
        },
        py::arg("device_id"),
        py::arg("trace_memory"),
        py::arg("enable_comm_overlap"),
        py::arg("mla_ops_type"));

    m.def(
        "register_comm_ops",
        [](py::function broadcast_fn, py::function allreduce_fn, py::function allgather_fn) {
            std::lock_guard<std::mutex> lock(g_comm_mutex);
            g_broadcast_fn = std::move(broadcast_fn);
            g_allreduce_fn = std::move(allreduce_fn);
            g_allgather_fn = std::move(allgather_fn);
        },
        py::arg("broadcast_fn"),
        py::arg("allreduce_fn"),
        py::arg("allgather_fn"),
        "Register Python callbacks for C++ communication ops.");

    m.def(
        "clear_comm_ops",
        []() {
            std::lock_guard<std::mutex> lock(g_comm_mutex);
            g_broadcast_fn = py::function();
            g_allreduce_fn = py::function();
            g_allgather_fn = py::function();
        },
        "Clear registered Python communication callbacks.");
}

}  // namespace rtp_llm
