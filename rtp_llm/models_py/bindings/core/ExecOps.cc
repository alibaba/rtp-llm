#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
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
#include <algorithm>
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#elif USING_ROCM
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#endif
#include <pybind11/functional.h>

#if USING_CUDA
using DeviceGuard = at::cuda::CUDAGuard;
#elif USING_ROCM
using DeviceGuard = c10::hip::HIPGuardMasqueradingAsCUDA;
#endif

namespace rtp_llm {
GreedyOutput     sampleGreedy(const GreedyParams& params);
BeamSearchOutput sampleBeamSearch(const BeamSearchParams& params);
void             chainSpeculativeSampling(const SpeculativeSamplingParams& params);
void             rejectionSampling(const RejectionSamplingParams& params);
void             mappingDraft2Target(const MappingDraft2TargetParams& params);
void             multiMergeCopy(const MultiMergeCopyParams& params);
}  // namespace rtp_llm

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/cuda/ops/CudaFlashInfer.h"
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

    // Legacy host path uses data_ptr pointer arithmetic; sync D2H when callers
    // pass device tensors. TODO(async): consume device tensors directly.
    auto to_cpu_sync = [](const torch::Tensor& t) -> torch::Tensor {
        if (!t.defined() || t.device().is_cpu()) {
            return t;
        }
        return t.cpu();
    };

    CacheStoreInputs local             = cache_store_inputs;
    local.host_kv_cache_offset         = to_cpu_sync(local.host_kv_cache_offset);
    local.prefix_lengths_host          = to_cpu_sync(local.prefix_lengths_host);
    local.input_lengths_host           = to_cpu_sync(local.input_lengths_host);
    local.kv_cache_layer_to_group_host = to_cpu_sync(local.kv_cache_layer_to_group_host);
    local.kv_cache_group_types_host    = to_cpu_sync(local.kv_cache_group_types_host);
    local.request_id                   = to_cpu_sync(local.request_id);
    local.request_pd_separation        = to_cpu_sync(local.request_pd_separation);

    auto& param = local;

    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.defined(), "failed to get host_kv_cache_offset");
    const int32_t* offset_addr          = nullptr;
    size_t         max_blocks_per_batch = 0;

    const size_t group_num = param.kv_cache_group_types_host.defined() ? param.kv_cache_group_types_host.size(0) : 1;
    const bool   use_group_cache_transfer_policy = group_num > 1;

    int  gid             = 0;
    auto mapped_group_id = [&param, group_num]() -> int {
        if (param.kv_cache_layer_region_to_group_host.defined() && param.kv_cache_layer_region_to_group_host.dim() == 2
            && param.layer_id >= 0
            && static_cast<int64_t>(param.layer_id) < param.kv_cache_layer_region_to_group_host.size(0)) {
            const auto region = static_cast<int64_t>(param.region_name);
            if (region >= 0 && region < param.kv_cache_layer_region_to_group_host.size(1)) {
                const int candidate = param.kv_cache_layer_region_to_group_host.data_ptr<
                    int32_t>()[param.layer_id * param.kv_cache_layer_region_to_group_host.size(1) + region];
                if (candidate >= 0) {
                    return candidate;
                }
            }
        }
        if (param.kv_cache_layer_to_group_host.defined() && param.layer_id >= 0
            && static_cast<size_t>(param.layer_id) < static_cast<size_t>(param.kv_cache_layer_to_group_host.numel())) {
            return param.kv_cache_layer_to_group_host.data_ptr<int32_t>()[param.layer_id];
        }
        return group_num > 0 ? 0 : -1;
    };
    if (param.host_kv_cache_offset.dim() == 3) {
        gid = mapped_group_id();
        RTP_LLM_CHECK_WITH_INFO(
            gid >= 0 && gid < static_cast<int32_t>(group_num), "invalid kv cache group id [%d]", gid);
        const auto group_offset_view = param.host_kv_cache_offset[static_cast<int64_t>(gid)];
        max_blocks_per_batch         = group_offset_view.size(1);
        offset_addr                  = group_offset_view.data_ptr<int32_t>();
    } else {
        gid = mapped_group_id();
        RTP_LLM_CHECK_WITH_INFO(
            gid >= 0 && gid < static_cast<int32_t>(group_num), "invalid kv cache group id [%d]", gid);
        max_blocks_per_batch = param.host_kv_cache_offset.size(1);
        offset_addr          = param.host_kv_cache_offset.data_ptr<int32_t>();
    }

    const auto seq_size_per_block = param.tokens_per_block;
    auto       kv_cache_data      = (uint64_t*)kv_cache.kv_cache_buffer.data_ptr();
    auto kv_scale_data = kv_cache.kv_scale_buffer.defined() ? (uint64_t*)kv_cache.kv_scale_buffer.data_ptr() : nullptr;

    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_pd_separation.numel()),
                            "size not same");
    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_id.numel()),
                            "context batch size and request id size is not same");

    RTP_LLM_LOG_DEBUG("write cache store, context_batch_size is %ld", param.context_batch_size);

    for (size_t batch_id = 0; batch_id < param.context_batch_size; batch_id++) {
        if (*(param.request_pd_separation.data_ptr<bool>() + batch_id) == false) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.defined() && param.input_lengths_host.defined(),
                                "failed to get prefix_length_host and input_length_host for cache store");
        RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.data_ptr<int>()[batch_id] % seq_size_per_block == 0,
                                "prefix_length %% seq_size_per_block != 0");
        int reuse_block_num = param.prefix_lengths_host.data_ptr<int>()[batch_id] / seq_size_per_block;
        int block_num =
            (param.input_lengths_host.data_ptr<int>()[param.decoder_batch_size + batch_id] + seq_size_per_block - 1)
            / seq_size_per_block;
        auto request_id     = *(param.request_id.data_ptr<int64_t>() + batch_id);
        auto event          = param.pre_created_event ? param.pre_created_event : runtimeCreateEvent();
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), event);
        RTP_LLM_LOG_DEBUG(
            "write cache store, request id is %ld, blocks num is %ld", request_id, block_num + reuse_block_num);

        CacheGroupType group_type = CacheGroupType::FULL;
        if (param.kv_cache_group_types_host.defined()) {
            group_type = static_cast<CacheGroupType>(param.kv_cache_group_types_host.data_ptr<int32_t>()[gid]);
        }

        const int total_blocks = block_num + reuse_block_num;
        if (total_blocks <= 0) {
            continue;
        }

        auto addBlock = [&](int index) {
            RTP_LLM_CHECK_WITH_INFO(index >= 0 && index < static_cast<int>(max_blocks_per_batch),
                                    "invalid block index=%d (max_blocks_per_batch=%zu)",
                                    index,
                                    max_blocks_per_batch);
            auto block_id = *(offset_addr + (param.decoder_batch_size + batch_id) * max_blocks_per_batch + index);
            if (isNullBlockIdx(block_id)) {
                RTP_LLM_LOG_DEBUG("skip null kv cache block, request id [%ld], layer id [%d], region [%d], index [%d]",
                                  request_id,
                                  param.layer_id,
                                  static_cast<int>(param.region_name),
                                  index);
                return;
            }
            std::string cache_key = makeCacheKey(param.model_id,
                                                 param.cache_keys[batch_id * max_blocks_per_batch + index],
                                                 param.layer_id,
                                                 param.region_name);

            void*                 kv_addr = (void*)((int8_t*)kv_cache_data + block_id * param.kv_block_stride_bytes);
            std::shared_ptr<void> kv_block_addr(kv_addr, [](void* p) {});

            // Some layouts treat the block as a single opaque KV chunk. Only
            // the legacy MHA path splits k/v.
            if (param.use_opaque_kv_cache_store || mla_kvcache) {
                request_blocks->addBlock("kv_" + cache_key, kv_block_addr, param.kv_block_stride_bytes, true, true);
            } else {
                const uint32_t        kv_half = static_cast<uint32_t>(param.kv_block_stride_bytes / 2);
                void*                 k_addr  = kv_addr;
                void*                 v_addr  = (void*)((int8_t*)kv_addr + kv_half);
                std::shared_ptr<void> k_block_addr(k_addr, [](void* p) {});
                std::shared_ptr<void> v_block_addr(v_addr, [](void* p) {});
                request_blocks->addBlock("k_" + cache_key, k_block_addr, kv_half, true, true);
                request_blocks->addBlock("v_" + cache_key, v_block_addr, kv_half, true, true);
            }

            if (kv_scale_data) {
                void* kv_scale_addr = (void*)((int8_t*)kv_scale_data + block_id * param.kv_scale_stride_bytes);
                std::shared_ptr<void> kv_scale_block_addr(kv_scale_addr, [](void* p) {});
                if (param.use_opaque_kv_cache_store || mla_kvcache) {
                    request_blocks->addBlock(
                        "kv_scale_" + cache_key, kv_scale_block_addr, param.kv_scale_stride_bytes, true, true);
                } else {
                    const uint32_t        sc_half = static_cast<uint32_t>(param.kv_scale_stride_bytes / 2);
                    void*                 k_sc    = kv_scale_addr;
                    void*                 v_sc    = (void*)((int8_t*)kv_scale_addr + sc_half);
                    std::shared_ptr<void> k_scale_block_addr(k_sc, [](void* p) {});
                    std::shared_ptr<void> v_scale_block_addr(v_sc, [](void* p) {});
                    request_blocks->addBlock("k_scale_" + cache_key, k_scale_block_addr, sc_half, true, true);
                    request_blocks->addBlock("v_scale_" + cache_key, v_scale_block_addr, sc_half, true, true);
                }
            }
        };

        const auto block_positions = blockPositionsForCacheTransfer(
            static_cast<size_t>(std::min<int>(total_blocks, static_cast<int>(max_blocks_per_batch))),
            /*reuse_block_size=*/0,
            use_group_cache_transfer_policy,
            group_type,
            /*hybrid_full_from_begin=*/true);
        for (const auto block_pos : block_positions) {
            addBlock(static_cast<int>(block_pos));
        }

        auto storeCallback = [layer_id = param.layer_id, request_id](bool success, CacheStoreErrorCode ec) {
            if (!success) {
                RTP_LLM_LOG_WARNING(
                    "query [%ld], layer id [%d], call store kv cache failed, ec is %d, error msg is [%s]",
                    request_id,
                    layer_id,
                    ec,
                    ErrorCodeToString(transCacheStoreErrorCode(ec)).c_str());
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
#if USING_CUDA
    check_cuda_value(cudaSetDevice(device_id));
    at::cuda::set_device(device_id);
    at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream(device_id));
#elif USING_ROCM
    hipSetDevice(device_id);
#endif
}

// === Profiling ===

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

// === Status queries ===

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

namespace {
#if USING_CUDA
at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}
#endif
}  // anonymous namespace

void execNoBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
#if USING_CUDA
    auto stream = getNoBlockCopyStream().stream();
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

BeamSearchOutput execSampleBeamSearch(const BeamSearchParams& params) {
    return sampleBeamSearch(params);
}

void execChainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    chainSpeculativeSampling(params);
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

// These callbacks are Python objects. Do not store them as static py::function
// values: their C++ static destructors may run after Python has started
// finalizing, which aborts in pybind11::function::~function() without a GIL.
// Keep raw pointers instead; normal shutdown deletes them under the GIL via
// clearCommOpsUnlocked(), and abnormal process exit intentionally leaks them.
py::function* g_broadcast_fn = nullptr;  // (tensors: list[Tensor], root: int, mode: int) -> None
py::function* g_allreduce_fn = nullptr;  // (tensor: Tensor, op: int, mode: int, dest: Optional[Tensor]) -> Tensor
py::function* g_allgather_fn =
    nullptr;  // (recv_buffers: list[Tensor], mode: int, send_buffers: list[Tensor], inplace: bool) -> None

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

    auto& bcast = CpuTpBroadcaster::instance();
    if (bcast.isInitialized()) {
        // Pure CPU path via UDS (no GPU stream, no Python, no cudaSync).
        // Caller must guarantee CPU tensors with identical (count, nbytes)
        // on every rank — see execBroadcastCpu doc in ExecOps.h.
        for (auto& t : params.buffers) {
            RTP_LLM_CHECK_WITH_INFO(
                t.is_cpu(), "execBroadcastCpu requires CPU tensors (got device=%s)", t.device().str().c_str());
            // Pinned tensors from torch::empty(...).pin_memory() are already
            // contiguous; .contiguous() is a no-op fast path.
            auto contig = t.contiguous();
            bcast.broadcast(contig.data_ptr(), contig.nbytes(), params.root);
            if (!contig.is_same(t)) {
                t.copy_(contig);
            }
        }
        return;
    }
    // Fallback to NCCL via Python callback, typically for cross-node TP.
    // Preserve immediate-read semantics with the original sync sequence.
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
            // Release GIL while peers block in accept/connect retry.
            // initialize() touches no Python state.
            py::gil_scoped_release release;
            CpuTpBroadcaster::instance().initialize(tp_rank, tp_size, base_path);
        },
        py::arg("tp_rank"),
        py::arg("tp_size"),
        py::arg("base_path"),
        "Bootstrap the UDS-backed intra-node TP broadcaster used by tpSyncModelInputs. "
        "Must be called by every TP rank with the same base_path; rank 0 binds, others connect.");

    m.def(
        "destroy_cpu_tp_broadcaster",
        []() {
            py::gil_scoped_release release;
            CpuTpBroadcaster::instance().reset();
        },
        "Tear down the UDS-backed intra-node TP broadcaster and clear its singleton state.");
}

}  // namespace rtp_llm
