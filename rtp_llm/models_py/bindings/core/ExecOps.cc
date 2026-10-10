#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/distribute/context_parallel/CpTransferProjection.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CachePhysicalRegion.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "autil/StackTracer.h"
#include "autil/EnvUtil.h"
#include <algorithm>
#include <cstdint>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>
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
torch::Tensor    sampleFromProbs(const torch::Tensor& probabilities);
BeamSearchOutput sampleBeamSearch(BeamSearchParams params);
void             rejectionSampling(const RejectionSamplingParams& params);
void             mappingDraft2Target(const MappingDraft2TargetParams& params);
void             multiMergeCopy(const MultiMergeCopyParams& params);
}  // namespace rtp_llm

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
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
// CacheStore (cache_store passed explicitly from CacheStoreAsyncWriter)
// ============================================================

void runtimeWriteCacheStore(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                            const torch_ext::LayerKVCache&       layer_kv,
                            const CacheConfig&                   cache_config,
                            std::shared_ptr<CacheStore>          cache_store,
                            size_t                               cache_model_id,
                            int                                  cp_rank,
                            int                                  cp_size,
                            std::shared_ptr<torch::Event>        pre_created_event,
                            CacheStoreCompletionRegistrar        register_store_completion) {
    const auto& param = cache_store_inputs;
    const auto  requireHostTensor =
        [](const torch::Tensor& tensor, const char* name, int64_t expected_dim, c10::ScalarType expected_type) {
            RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "cache-store %s must be defined", name);
            RTP_LLM_CHECK_WITH_INFO(tensor.dim() == expected_dim,
                                    "cache-store %s must be %ld-D, got dim=%ld",
                                    name,
                                    expected_dim,
                                    tensor.dim());
            RTP_LLM_CHECK_WITH_INFO(tensor.device().is_cpu(), "cache-store %s must be a CPU tensor", name);
            RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == expected_type,
                                    "cache-store %s must use %s, got %s",
                                    name,
                                    c10::toString(expected_type),
                                    c10::toString(tensor.scalar_type()));
        };

    requireHostTensor(param.request_id, "request_id", 1, torch::kInt64);
    const size_t context_batch_size = static_cast<size_t>(param.request_id.numel());
    if (context_batch_size == 0) {
        return;
    }
    requireHostTensor(param.input_lengths_host, "input_lengths_host", 1, torch::kInt32);
    requireHostTensor(param.prefix_lengths_host, "prefix_lengths_host", 1, torch::kInt32);
    requireHostTensor(param.publish_start_tokens, "publish_start_tokens", 1, torch::kInt32);
    RTP_LLM_CHECK_WITH_INFO(param.publish_start_tokens.numel() == static_cast<int64_t>(context_batch_size),
                            "cache-store publish_start_tokens size must match context batch");
    requireHostTensor(param.host_kv_cache_offset, "host_kv_cache_offset", 2, torch::kInt32);
    requireHostTensor(param.request_pd_separation, "request_pd_separation", 1, torch::kBool);
    requireHostTensor(param.cache_keys, "cache_keys", 2, torch::kInt64);

    if (!cache_store) {
        RTP_LLM_CHECK_WITH_INFO(!register_store_completion,
                                "writeCacheStore has tracked publication but cache_store is null; "
                                "refusing to silently drop the KV transfer");
        RTP_LLM_LOG_DEBUG("cache_store is null, skip writeCacheStore");
        return;
    }

    // Wait for the CUDA event before reading pinned-host metadata.
    // The event was recorded on the main stream AFTER both the async D2H
    // copies (metadata) and KV cache writes were enqueued, so blocking
    // here guarantees all pinned buffers are populated.
    if (pre_created_event) {
        pre_created_event->synchronize();
    }

    RTP_LLM_CHECK_WITH_INFO(
        !layer_kv.tag.empty(), "cache-store write requires a cache tag for layer=%d", layer_kv.layer_id);

    const size_t max_blocks_per_batch = static_cast<size_t>(param.host_kv_cache_offset.size(1));

    const auto& group = cache_config.groupForLayer(layer_kv.layer_id, layer_kv.tag);
    RTP_LLM_CHECK_WITH_INFO(
        group.spec != nullptr, "cache-store tag=%s has no KVCacheSpec attached", layer_kv.tag.c_str());

    // LayerKVCache may expose kernel-page views; CacheStore keys and block IDs use physical pages.
    const size_t seq_size_per_block              = group.seqSizePerBlock();
    const bool   use_group_cache_transfer_policy = cache_config.topology().groups().size() > 1;

    RTP_LLM_CHECK_WITH_INFO(
        seq_size_per_block > 0, "cache-store tag=%s has zero tokens_per_block", layer_kv.tag.c_str());
    auto       kv_cache_owner = std::make_shared<torch::Tensor>(layer_kv.kv_cache_base);
    const bool has_kv_scale =
        layer_kv.kv_scale_base.defined() && layer_kv.kv_scale_base.numel() > 0 && group.kvScaleStrideBytes() > 0;
    std::shared_ptr<torch::Tensor> kv_scale_owner;
    if (has_kv_scale) {
        kv_scale_owner = std::make_shared<torch::Tensor>(layer_kv.kv_scale_base);
    }

    const size_t total_batch_size = static_cast<size_t>(param.input_lengths_host.numel());
    RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.numel() == static_cast<int64_t>(context_batch_size),
                            "cache-store tag=%s prefix_lengths numel=%ld != context batch=%zu",
                            layer_kv.tag.c_str(),
                            param.prefix_lengths_host.numel(),
                            context_batch_size);
    RTP_LLM_CHECK_WITH_INFO(param.request_pd_separation.numel() == static_cast<int64_t>(context_batch_size),
                            "cache-store tag=%s request_pd_separation numel=%ld != context batch=%zu",
                            layer_kv.tag.c_str(),
                            param.request_pd_separation.numel(),
                            context_batch_size);
    RTP_LLM_CHECK_WITH_INFO(total_batch_size >= context_batch_size,
                            "cache-store tag=%s input_lengths numel=%zu < context batch=%zu",
                            layer_kv.tag.c_str(),
                            total_batch_size,
                            context_batch_size);
    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.size(0) == static_cast<int64_t>(total_batch_size),
                            "cache-store tag=%s block table rows=%ld != total batch=%zu",
                            layer_kv.tag.c_str(),
                            param.host_kv_cache_offset.size(0),
                            total_batch_size);
    RTP_LLM_CHECK_WITH_INFO(param.cache_keys.size(0) == static_cast<int64_t>(context_batch_size),
                            "cache-store tag=%s cache_keys rows=%ld != context batch=%zu",
                            layer_kv.tag.c_str(),
                            param.cache_keys.size(0),
                            context_batch_size);

    const size_t decoder_batch_size = total_batch_size - context_batch_size;
    // cache_keys is laid out [batch, global_max_blocks]; this logical width is INDEPENDENT
    // of `max_blocks_per_batch` (which is per-group offset width and may be smaller
    // for CP-sharded FULL groups whose offset is rank-local-compact).
    const size_t cache_keys_per_batch  = static_cast<size_t>(param.cache_keys.size(1));
    const auto   host_kv_cache_offset  = param.host_kv_cache_offset.accessor<int32_t, 2>();
    const auto   input_lengths_host    = param.input_lengths_host.accessor<int32_t, 1>();
    const auto   prefix_lengths_host   = param.prefix_lengths_host.accessor<int32_t, 1>();
    const auto   publish_start_tokens = param.publish_start_tokens.accessor<int32_t, 1>();
    const auto   request_ids           = param.request_id.accessor<int64_t, 1>();
    const auto   request_pd_separation = param.request_pd_separation.accessor<bool, 1>();
    const auto   cache_keys            = param.cache_keys.accessor<int64_t, 2>();

    RTP_LLM_LOG_DEBUG("write cache store, context_batch_size is %zu", context_batch_size);
    for (size_t batch_id = 0; batch_id < context_batch_size; ++batch_id) {
        const auto context_index = static_cast<int64_t>(batch_id);
        if (!request_pd_separation[context_index]) {
            continue;
        }

        const int prefix_length = prefix_lengths_host[context_index];
        const auto   input_index  = static_cast<int64_t>(decoder_batch_size + batch_id);
        const int    input_length = input_lengths_host[input_index];
        RTP_LLM_CHECK_WITH_INFO(cp_size > 0 && cp_rank >= 0 && cp_rank < cp_size,
                                "producer cache-store requires a valid rank and CP size");
        RTP_LLM_CHECK_WITH_INFO(cache_config.seq_size_per_block > 0 && input_length >= 0 && prefix_length >= 0
                                    && static_cast<size_t>(prefix_length) % cache_config.seq_size_per_block == 0,
                                "cache-store requires nonnegative input/prefix and a block-aligned prefix");
        const size_t key_count    = std::min(cache_keys_per_batch,
                                          (static_cast<size_t>(input_length) + cache_config.seq_size_per_block - 1)
                                                  / cache_config.seq_size_per_block
                                              + static_cast<size_t>(prefix_length) / cache_config.seq_size_per_block);
        const bool   compact_tail = cache_config.groupUsesCompactTransferTail(layer_kv.tag);
        RTP_LLM_CHECK_WITH_INFO(!compact_tail
                                    || cache_config.groupKeyScale(layer_kv.tag) == static_cast<size_t>(cp_size),
                                "producer compact block scale does not match CP size");
        const auto pages = selectProducerTransferPages(group.policy,
                                                       cache_config.logicalBlockCountForGroup(layer_kv.tag, key_count),
                                                       use_group_cache_transfer_policy,
                                                       compact_tail);
        if (pages.empty()) {
            continue;
        }

        const int64_t request_id       = request_ids[context_index];
        const int publish_start_token = publish_start_tokens[context_index];
        RTP_LLM_CHECK_WITH_INFO(publish_start_token >= 0 && publish_start_token <= prefix_length,
                                "cache-store tag=%s invalid publish_start_token=%d for prefix_length=%d",
                                layer_kv.tag.c_str(), publish_start_token, prefix_length);
        const size_t publish_start_block =
            group.policy.group_type == CacheGroupType::FULL && group.policy.active_tail_blocks == 0 ?
                static_cast<size_t>(publish_start_token) / seq_size_per_block : 0;
        auto event          = pre_created_event ? pre_created_event : runtimeCreateEvent();
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), event);
        std::vector<int64_t> publication_lease_keys;
        std::vector<int32_t> publication_lease_blocks;
        RTP_LLM_LOG_DEBUG(
            "write cache store, request id is %ld, blocks num is %zu", static_cast<long>(request_id), pages.size());

        // Tensor owners stay with the asynchronous request; the cache query
        // resolves only this rank's selected physical parts.
        auto tensorBase = [](const torch::Tensor& tensor) {
            BlockInfo base;
            base.addr         = tensor.data_ptr();
            base.size_bytes   = static_cast<size_t>(tensor.numel()) * tensor.element_size();
            base.is_cuda      = tensor.is_cuda();
            base.device_index = tensor.is_cuda() ? tensor.get_device() : 0;
            base.scalar_type  = static_cast<int32_t>(tensor.scalar_type());
            return base;
        };
        std::vector<BlockInfo> part_bases{tensorBase(layer_kv.kv_cache_base)};
        if (has_kv_scale) {
            part_bases.push_back(tensorBase(layer_kv.kv_scale_base));
        }
        std::vector<int32_t> block_ids(max_blocks_per_batch);
        for (size_t slot = 0; slot < max_blocks_per_batch; ++slot) {
            block_ids[slot] = host_kv_cache_offset[input_index][static_cast<int64_t>(slot)];
        }
        const bool use_opaque_key_prefix = cache_config.use_opaque_kv_cache_store || use_group_cache_transfer_policy
                                           || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention;
        for (size_t logical_page : pages) {
            if (logical_page < publish_start_block) {
                continue;
            }
            const auto regions = queryProducerCacheBlockRegions(cache_config,
                                                                layer_kv.layer_id,
                                                                layer_kv.tag,
                                                                key_count,
                                                                logical_page,
                                                                cp_rank,
                                                                cp_size,
                                                                use_opaque_key_prefix,
                                                                block_ids.data(),
                                                                block_ids.size(),
                                                                part_bases);
            if (regions.empty()) {
                // Another producer owns the page, or the source block is NULL.
                continue;
            }
            const auto& first = regions.front();
            RTP_LLM_CHECK_WITH_INFO(first.key_index < cache_keys_per_batch,
                                    "producer cache key index=%zu exceeds table width=%zu",
                                    first.key_index,
                                    cache_keys_per_batch);
            RTP_LLM_CHECK_WITH_INFO(first.local_slot < block_ids.size(),
                                    "producer slot=%zu exceeds block table width=%zu",
                                    first.local_slot,
                                    block_ids.size());
            cacheTransferPartPrefix(use_opaque_key_prefix, 0, regions.size());
            const int32_t block_id = block_ids[first.local_slot];
            const int64_t cache_key_id =
                cache_keys[static_cast<int64_t>(batch_id)][static_cast<int64_t>(first.key_index)];
            const std::string cache_key =
                makeCacheKey(cache_model_id, std::to_string(cache_key_id), layer_kv.layer_id, layer_kv.tag);
            publication_lease_keys.push_back(cache_key_id);
            publication_lease_blocks.push_back(block_id);
            for (const auto& region : regions) {
                const std::string wire_key =
                    std::string(cacheTransferPartPrefix(use_opaque_key_prefix, region.part_index, regions.size()))
                    + cache_key;
                const bool  scale_part = region.part_index >= (use_opaque_key_prefix ? 1u : 2u);
                const auto& owner      = scale_part ? kv_scale_owner : kv_cache_owner;
                RTP_LLM_CHECK_WITH_INFO(owner != nullptr && region.buffer.addr != nullptr
                                            && region.buffer.size_bytes > 0,
                                        "invalid producer cache part key=%s",
                                        wire_key.c_str());
                std::shared_ptr<void> addr(owner, region.buffer.addr);
                RTP_LLM_LOG_DEBUG("PD_CACHE_KEY_WRITE_BLOCK key=%s request_id=%ld tag=%s layer=%d cp_rank=%d "
                                  "cp_size=%d key_index=%zu offset_index=%zu block_id=%d addr=%p len=%zu",
                                  wire_key.c_str(),
                                  static_cast<long>(request_id),
                                  layer_kv.tag.c_str(),
                                  layer_kv.layer_id,
                                  cp_rank,
                                  cp_size,
                                  region.key_index,
                                  region.local_slot,
                                  block_id,
                                  region.buffer.addr,
                                  region.buffer.size_bytes);
                request_blocks->addBlock(
                    wire_key, addr, static_cast<uint32_t>(region.buffer.size_bytes), region.buffer.is_cuda, true);
            }
        }

        if (request_blocks->getBlocksCount() > 0) {
            CacheStoreCompletionCallback store_completion;
            if (register_store_completion) {
                store_completion =
                    register_store_completion(publication_lease_keys, publication_lease_blocks, layer_kv.tag);
            }
            auto store_callback = [layer_id = layer_kv.layer_id,
                                   cache_model_id,
                                   tag = layer_kv.tag,
                                   request_id,
                                   request_blocks,
                                   store_completion](bool success, CacheStoreErrorCode ec) {
                if (!success) {
                    RTP_LLM_LOG_WARNING("PD_CACHE_KEY_WRITE_FAILED request_id=%ld model_id=%zu local_layer_id=%d "
                                        "tag=%s error_code=%d error=%s buffer={%s}",
                                        static_cast<long>(request_id),
                                        cache_model_id,
                                        layer_id,
                                        tag.c_str(),
                                        static_cast<int>(ec),
                                        ErrorCodeToString(transCacheStoreErrorCode(ec)).c_str(),
                                        request_blocks->debugInfo().c_str());
                }
                if (store_completion) {
                    if (success) {
                        store_completion(nullptr);
                    } else {
                        store_completion(std::make_exception_ptr(std::runtime_error(
                            "cache-store publication failed for request " + std::to_string(request_id) + ", model "
                            + std::to_string(cache_model_id) + ", layer " + std::to_string(layer_id) + ", tag " + tag
                            + ", error code " + std::to_string(static_cast<int>(ec)))));
                    }
                }
            };
            try {
                cache_store->store(request_blocks, std::move(store_callback));
            } catch (...) {
                if (store_completion) {
                    store_completion(std::current_exception());
                }
                throw;
            }
        } else {
            RTP_LLM_LOG_DEBUG("skip cache store because no blocks require publication, request id [%ld], layer id [%d]",
                              static_cast<long>(request_id),
                              layer_kv.layer_id);
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
    ExecStatus status;
    status.device_memory_status = mem;
    return status;
}

torch::Device getTorchCudaDevice() {
    return torch::Device(torch::kCUDA);
}

namespace {
static bool g_trace_memory = false;
}

void setTraceMemory(bool trace_memory) {
    g_trace_memory = trace_memory;
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

torch::Tensor execSampleFromProbs(const torch::Tensor& probabilities) {
    return sampleFromProbs(probabilities);
}

BeamSearchOutput execSampleBeamSearch(BeamSearchParams params) {
    return sampleBeamSearch(std::move(params));
}

void execRejectionSampling(const RejectionSamplingParams& params) {
    rejectionSampling(params);
}

void execMappingDraft2Target(const MappingDraft2TargetParams& params) {
    mappingDraft2Target(params);
}

// === Communication ops (Python callbacks via pybind11) ===

namespace {
std::mutex g_comm_mutex;

// Avoid destroying static Python objects after interpreter finalization.
py::function* g_broadcast_fn = nullptr;
py::function* g_allreduce_fn = nullptr;
py::function* g_allgather_fn = nullptr;

void clearCommOpsUnlocked() {
    py::function broadcast_fn;
    py::function allreduce_fn;
    py::function allgather_fn;
    if (g_broadcast_fn != nullptr) {
        broadcast_fn = std::move(*g_broadcast_fn);
        delete g_broadcast_fn;
        g_broadcast_fn = nullptr;
    }
    if (g_allreduce_fn != nullptr) {
        allreduce_fn = std::move(*g_allreduce_fn);
        delete g_allreduce_fn;
        g_allreduce_fn = nullptr;
    }
    if (g_allgather_fn != nullptr) {
        allgather_fn = std::move(*g_allgather_fn);
        delete g_allgather_fn;
        g_allgather_fn = nullptr;
    }
}
}  // anonymous namespace

void execBroadcast(const BroadcastParams& params) {
    py::function           fn;
    py::gil_scoped_acquire gil;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        if (g_broadcast_fn != nullptr) {
            fn = *g_broadcast_fn;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execBroadcast called but broadcast callback not registered via register_comm_ops");
    py::list tensors;
    for (auto& t : params.buffers)
        tensors.append(t);
    fn(tensors, params.root, static_cast<int>(params.mode));
}

void execBroadcastCpu(const BroadcastParams& params) {
    RTP_LLM_CHECK_WITH_INFO(
        params.root == 0, "execBroadcastCpu supports only root=0; got %ld", static_cast<long>(params.root));
    RTP_LLM_CHECK_WITH_INFO(params.mode == ParallelMode::TP,
                            "execBroadcastCpu supports only ParallelMode::TP; got %d",
                            static_cast<int>(params.mode));

    auto& broadcaster = CpuTpBroadcaster::instance();
    if (broadcaster.isInitialized()) {
        for (auto& tensor : params.buffers) {
            RTP_LLM_CHECK_WITH_INFO(tensor.is_cpu(),
                                    "execBroadcastCpu requires CPU tensors (got device=%s)",
                                    tensor.device().str().c_str());
            auto contiguous = tensor.contiguous();
            broadcaster.broadcast(contiguous.data_ptr(), contiguous.nbytes(), params.root);
            if (!contiguous.is_same(tensor)) {
                tensor.copy_(contiguous);
            }
        }
        return;
    }
    execBroadcast(params);
    execSyncCommunication(false);
    cudaSyncAndCheck();
}

bool isCpuTpBroadcasterInitialized() {
    return CpuTpBroadcaster::instance().isInitialized();
}

AllReduceOutput execAllReduce(const AllReduceParams& params) {
    py::function           fn;
    py::gil_scoped_acquire gil;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        if (g_allreduce_fn != nullptr) {
            fn = *g_allreduce_fn;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllReduce called but allreduce callback not registered via register_comm_ops");
    auto result = fn(params.buffer,
                     static_cast<int>(params.op),
                     static_cast<int>(params.mode),
                     params.dest.defined() ? py::cast(params.dest) : py::none());
    return AllReduceOutput{result.cast<torch::Tensor>()};
}

void execAllGather(const AllGatherParams& params) {
    py::function           fn;
    py::gil_scoped_acquire gil;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        if (g_allgather_fn != nullptr) {
            fn = *g_allgather_fn;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<bool>(fn),
                            "execAllGather called but allgather callback not registered via register_comm_ops");
    py::list recv_list, send_list;
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
            clearCommOpsUnlocked();
            g_broadcast_fn = new py::function(std::move(broadcast_fn));
            g_allreduce_fn = new py::function(std::move(allreduce_fn));
            g_allgather_fn = new py::function(std::move(allgather_fn));
        },
        py::arg("broadcast_fn"),
        py::arg("allreduce_fn"),
        py::arg("allgather_fn"),
        "Register Python callbacks for C++ communication ops.");

    m.def(
        "clear_comm_ops",
        []() {
            std::lock_guard<std::mutex> lock(g_comm_mutex);
            clearCommOpsUnlocked();
        },
        "Clear registered Python communication callbacks.");

    m.def(
        "init_cpu_tp_broadcaster",
        [](int tp_rank, int tp_size, const std::string& base_path) {
            py::gil_scoped_release release;
            CpuTpBroadcaster::instance().initialize(tp_rank, tp_size, base_path);
        },
        py::arg("tp_rank"),
        py::arg("tp_size"),
        py::arg("base_path"));

    m.def("destroy_cpu_tp_broadcaster", []() {
        py::gil_scoped_release release;
        CpuTpBroadcaster::instance().reset();
    });
}

}  // namespace rtp_llm
