#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/ExecCtxExport.h"
#include "rtp_llm/cpp/core/DistributedComm.h"
#include "rtp_llm/cpp/core/CommonDefines.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
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
#include <cstddef>
#include <mutex>
#include <atomic>
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#elif USING_ROCM
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#endif
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#if USING_CUDA
using DeviceGuard = at::cuda::CUDAGuard;
#elif USING_ROCM
using DeviceGuard = c10::hip::HIPGuardMasqueradingAsCUDA;
#endif

namespace rtp_llm {
GreedyOutput     sampleGreedy(const GreedyParams& params);
BeamSearchOutput sampleBeamSearch(const BeamSearchParams& params);
void             chainSpeculativeSampling(const SpeculativeSamplingParams& params);
void             multiMergeCopy(const MultiMergeCopyParams& params);
}  // namespace rtp_llm

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/ops/CudaFlashInfer.h"
#include "rtp_llm/cpp/core/torch_utils/TorchEvent.h"
#include "rtp_llm/cpp/kernels/sm_utils/sm_copy_kernel.h"
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
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

static std::shared_ptr<torch_ext::ExecCtxExporter> g_exporter;
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

AsyncEventPtr runtimeCreateEvent() {
    return std::make_unique<TorchEvent>(at::cuda::getCurrentCUDAStream());
}

#else  // ROCm

AsyncEventPtr runtimeCreateEvent() {
    auto torch_stream = at::hip::getCurrentHIPStream(at::hip::current_device());
    auto event_ptr    = std::make_shared<torch::Event>(torch::kHIP);
    event_ptr->record(torch_stream);
    struct HipEvent: public AsyncEvent {
        explicit HipEvent(std::shared_ptr<torch::Event> e): event(std::move(e)) {}
        void synchronize() const override {
            event->synchronize();
        }
        bool checkReadiness() const override {
            return event->query();
        }
        std::shared_ptr<torch::Event> event;
    };
    return std::make_unique<HipEvent>(std::move(event_ptr));
}

#endif  // USING_CUDA

// ============================================================
// CacheStore (cache_store passed explicitly from KVCacheManager)
// ============================================================

void runtimeWriteCacheStore(const CacheStoreInputs&     cache_store_inputs,
                            const KvCacheInfo&          kv_cache,
                            bool                        mla_kvcache,
                            std::shared_ptr<CacheStore> cache_store) {
    auto& param = cache_store_inputs;
    if (param.warmup) {
        RTP_LLM_LOG_DEBUG("is warmup, so ignore writeCacheStore");
        return;
    }
    if (!param.pd_separation || param.context_batch_size == 0) {
        RTP_LLM_LOG_DEBUG("pd_separation = %d, context_batch_size = %d, so ignore writeCacheStore",
                          param.pd_separation,
                          param.context_batch_size);
        return;
    }
    if (!cache_store) {
        RTP_LLM_LOG_DEBUG("cache_store is null, skip writeCacheStore");
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.defined(), "failed to get host_kv_cache_offset");
    const int32_t* offset_addr          = nullptr;
    size_t         max_blocks_per_batch = 0;

    bool is_hybrid = false;
    if (param.kv_cache_group_types_host.defined() && param.kv_cache_group_types_host.size(0) > 1) {
        is_hybrid =
            !torch::all(param.kv_cache_group_types_host.index({param.kv_cache_layer_to_group_host}) == 1).item<bool>();
    }

    const size_t group_num = is_hybrid ? param.kv_cache_group_types_host.size(0) : 1;

    int gid = 0;
    if (param.host_kv_cache_offset.dim() == 3) {
        gid = -1;
        if (param.kv_cache_layer_to_group_host.defined() && param.layer_id >= 0
            && static_cast<size_t>(param.layer_id) < static_cast<size_t>(param.kv_cache_layer_to_group_host.numel())) {
            gid = param.kv_cache_layer_to_group_host.data_ptr<int32_t>()[param.layer_id];
        }
        RTP_LLM_CHECK_WITH_INFO(
            gid >= 0 && gid < static_cast<int32_t>(group_num), "invalid kv cache group id [%d]", gid);
        const auto group_offset_view = param.host_kv_cache_offset[static_cast<int64_t>(gid)];
        max_blocks_per_batch         = group_offset_view.size(1);
        offset_addr                  = group_offset_view.data_ptr<int32_t>();
    } else {
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
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), runtimeCreateEvent());
        RTP_LLM_LOG_DEBUG(
            "write cache store, request id is %ld, blocks num is %ld", request_id, block_num + reuse_block_num);

        CacheGroupType group_type = CacheGroupType::FULL;
        group_type = static_cast<CacheGroupType>(param.kv_cache_group_types_host.data_ptr<int32_t>()[gid]);

        const int total_blocks = block_num + reuse_block_num;
        if (total_blocks <= 0) {
            continue;
        }

        auto addBlock = [&](int index, CacheGroupType group_type) {
            RTP_LLM_CHECK_WITH_INFO(index >= 0 && index < static_cast<int>(max_blocks_per_batch),
                                    "invalid block index=%d (max_blocks_per_batch=%zu)",
                                    index,
                                    max_blocks_per_batch);
            auto block_id = *(offset_addr + (param.decoder_batch_size + batch_id) * max_blocks_per_batch + index);
            std::string cache_key;
            cache_key =
                makeCacheKey(param.model_id, param.cache_keys[batch_id * max_blocks_per_batch + index], param.layer_id);

            void*                 kv_addr = (void*)((int8_t*)kv_cache_data + block_id * param.kv_block_stride_bytes);
            std::shared_ptr<void> kv_block_addr(kv_addr, [](void* p) {});

            if (is_hybrid || mla_kvcache) {
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
                if (is_hybrid || mla_kvcache) {
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

        if (group_type == CacheGroupType::LINEAR) {
            addBlock(total_blocks - 1, group_type);
        } else {
            for (int index = 0; index < total_blocks; ++index) {
                addBlock(index, group_type);
            }
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
        cache_store->store(request_blocks, storeCallback);
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

MemoryStatus getGpuMemoryStatus() {
    return getGpuExecStatus().device_memory_status;
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

bool getTraceMemory() {
    return g_trace_memory;
}

// === Copy ops ===

namespace {
#if USING_CUDA
at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}

static bool splitKvTensorsByteContiguous(const std::vector<torch::Tensor>& t, size_t off, size_t layer_num) {
    const size_t n = 2u * layer_num;
    if (off + n > t.size()) {
        return false;
    }
    for (size_t i = 0; i + 1 < n; ++i) {
        const auto& a = t[off + i];
        const auto& b = t[off + i + 1];
        if (!a.defined() || !b.defined() || !a.is_contiguous() || !b.is_contiguous()) {
            return false;
        }
        auto* pa = static_cast<const char*>(a.data_ptr());
        if (pa + static_cast<ptrdiff_t>(a.nbytes()) != static_cast<const char*>(b.data_ptr())) {
            return false;
        }
    }
    return true;
}

// Mirrors legacy SplitKvCacheCopyCuda: reuse device buffers, cudaMalloc only when capacity grows,
// cudaFreeAsync on the same no-block-copy stream. Release via releaseSplitKvTensorCopyCudaState() on the worker thread.
class SplitKvTensorCopyCudaState {
public:
    SplitKvTensorCopyCudaState()  = default;
    ~SplitKvTensorCopyCudaState() = default;

    SplitKvTensorCopyCudaState(const SplitKvTensorCopyCudaState&)            = delete;
    SplitKvTensorCopyCudaState& operator=(const SplitKvTensorCopyCudaState&) = delete;

    void releaseDeviceResources() {
        releaseAllDeviceResources();
    }

    bool run(const MultiCopyParams& params) {
        const int L = params.split_kv_layer_num;
        if (L <= 0) {
            return false;
        }
        const size_t tpi = static_cast<size_t>(2 * L);
        const size_t n   = params.multi_src.size();
        if (n != params.multi_dst.size() || n % tpi != 0) {
            return false;
        }
        const size_t kv    = params.split_kv_cache_stride_bytes;
        const size_t scale = params.split_kv_scale_stride_bytes;
        if (kv + scale == 0) {
            return false;
        }

        const auto& s0  = params.multi_src[0];
        const bool  h2d = s0.is_cpu();
        const bool  d2h = s0.is_cuda();
        if (!h2d && !d2h) {
            return false;
        }

        cudaStream_t stream = getNoBlockCopyStream().stream();
        copy_stream_        = stream;

        int ptr_device = -1;
        if (h2d) {
            for (size_t i = 0; i < tpi; ++i) {
                const auto& d = params.multi_dst[i];
                if (!d.is_cuda()) {
                    return false;
                }
                int di = static_cast<int>(d.get_device());
                if (ptr_device < 0) {
                    ptr_device = di;
                } else if (di != ptr_device) {
                    return false;
                }
            }
        } else {
            cudaPointerAttributes attr{};
            check_cuda_value(cudaPointerGetAttributes(&attr, params.multi_src[0].data_ptr()));
            if (attr.type != cudaMemoryTypeDevice) {
                return false;
            }
            ptr_device = attr.device;
            for (size_t i = 0; i < tpi; ++i) {
                if (!params.multi_src[i].is_cuda()) {
                    return false;
                }
            }
            for (size_t i = 0; i < tpi; ++i) {
                if (!params.multi_dst[i].is_cpu()) {
                    return false;
                }
            }
        }

        at::cuda::CUDAGuard device_guard(ptr_device);
        check_cuda_value(cudaSetDevice(ptr_device));

        if (buffer_device_ >= 0 && ptr_device != buffer_device_) {
            releaseAllDeviceResources();
        }
        buffer_device_ = ptr_device;

        const size_t block_size      = kv * static_cast<size_t>(L) + scale * static_cast<size_t>(L);
        const size_t ptr_table_bytes = static_cast<size_t>(L) * sizeof(void*);
        if (ptr_table_bytes == 0) {
            return false;
        }

        const size_t       block_nums = n / tpi;
        std::vector<void*> h_kv(static_cast<size_t>(L));
        std::vector<void*> h_scale(static_cast<size_t>(L));

        for (size_t b = 0; b < block_nums; ++b) {
            const size_t off = b * tpi;
            for (int i = 0; i < L; ++i) {
                if (params.multi_src[off + static_cast<size_t>(2 * i)].nbytes() != kv
                    || params.multi_src[off + static_cast<size_t>(2 * i + 1)].nbytes() != scale) {
                    return false;
                }
            }
            if (h2d) {
                if (!splitKvTensorsByteContiguous(params.multi_src, off, static_cast<size_t>(L))) {
                    return false;
                }
            } else if (!splitKvTensorsByteContiguous(params.multi_dst, off, static_cast<size_t>(L))) {
                return false;
            }
        }

        ensureBuffers(block_size, ptr_table_bytes, stream);

        void* const d_staging     = staging_;
        void* const d_kv_table    = ptr0_;
        void* const d_scale_table = ptr1_;

        for (size_t b = 0; b < block_nums; ++b) {
            const size_t off = b * tpi;
            if (h2d) {
                for (int i = 0; i < L; ++i) {
                    h_kv[static_cast<size_t>(i)]    = params.multi_dst[off + static_cast<size_t>(2 * i)].data_ptr();
                    h_scale[static_cast<size_t>(i)] = params.multi_dst[off + static_cast<size_t>(2 * i + 1)].data_ptr();
                }
                check_cuda_value(cudaMemcpyAsync(
                    d_staging, params.multi_src[off].data_ptr(), block_size, cudaMemcpyHostToDevice, stream));
                check_cuda_value(
                    cudaMemcpyAsync(d_kv_table, h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                check_cuda_value(
                    cudaMemcpyAsync(d_scale_table, h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                sDevMPS::launch_scatter_copy_split(d_staging,
                                                   reinterpret_cast<void**>(d_kv_table),
                                                   reinterpret_cast<void**>(d_scale_table),
                                                   kv,
                                                   scale,
                                                   L,
                                                   /*block_num=*/0,
                                                   stream);
            } else {
                for (int i = 0; i < L; ++i) {
                    h_kv[static_cast<size_t>(i)]    = params.multi_src[off + static_cast<size_t>(2 * i)].data_ptr();
                    h_scale[static_cast<size_t>(i)] = params.multi_src[off + static_cast<size_t>(2 * i + 1)].data_ptr();
                }
                check_cuda_value(
                    cudaMemcpyAsync(d_kv_table, h_kv.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                check_cuda_value(
                    cudaMemcpyAsync(d_scale_table, h_scale.data(), ptr_table_bytes, cudaMemcpyHostToDevice, stream));
                sDevMPS::launch_gather_copy_split(reinterpret_cast<const void**>(d_kv_table),
                                                  reinterpret_cast<const void**>(d_scale_table),
                                                  kv,
                                                  scale,
                                                  d_staging,
                                                  L,
                                                  /*block_num=*/0,
                                                  stream);
                check_cuda_value(cudaMemcpyAsync(
                    params.multi_dst[off].data_ptr(), d_staging, block_size, cudaMemcpyDeviceToHost, stream));
            }
        }

        check_cuda_value(cudaStreamSynchronize(stream));
        check_cuda_error();
        return true;
    }

private:
    void releaseAllDeviceResources() {
        if (buffer_device_ < 0) {
            staging_cap_   = 0;
            ptr_table_cap_ = 0;
            return;
        }
        check_cuda_value(cudaSetDevice(buffer_device_));
        if (copy_stream_) {
            check_cuda_value(cudaStreamSynchronize(copy_stream_));
            if (staging_) {
                check_cuda_value(cudaFreeAsync(staging_, copy_stream_));
                staging_ = nullptr;
            }
            if (ptr0_) {
                check_cuda_value(cudaFreeAsync(ptr0_, copy_stream_));
                ptr0_ = nullptr;
            }
            if (ptr1_) {
                check_cuda_value(cudaFreeAsync(ptr1_, copy_stream_));
                ptr1_ = nullptr;
            }
            check_cuda_value(cudaStreamSynchronize(copy_stream_));
        } else {
            if (staging_) {
                check_cuda_value(cudaFree(staging_));
                staging_ = nullptr;
            }
            if (ptr0_) {
                check_cuda_value(cudaFree(ptr0_));
                ptr0_ = nullptr;
            }
            if (ptr1_) {
                check_cuda_value(cudaFree(ptr1_));
                ptr1_ = nullptr;
            }
        }
        staging_cap_   = 0;
        ptr_table_cap_ = 0;
        buffer_device_ = -1;
        copy_stream_   = nullptr;
    }

    void ensureBuffers(size_t staging_bytes, size_t ptr_table_bytes, cudaStream_t stream) {
        RUNTIME_ASSERT_OP_ARG(buffer_device_ >= 0, "split KV copy: buffer device not set before ensureBuffers");
        check_cuda_value(cudaSetDevice(buffer_device_));
        if (staging_bytes > staging_cap_) {
            if (staging_) {
                check_cuda_value(cudaStreamSynchronize(stream));
                check_cuda_value(cudaFreeAsync(staging_, stream));
                check_cuda_value(cudaStreamSynchronize(stream));
                staging_     = nullptr;
                staging_cap_ = 0;
            }
            check_cuda_value(cudaMalloc(&staging_, staging_bytes));
            staging_cap_ = staging_bytes;
        }
        if (ptr_table_bytes > ptr_table_cap_) {
            if (ptr0_ || ptr1_) {
                check_cuda_value(cudaStreamSynchronize(stream));
                if (ptr0_) {
                    check_cuda_value(cudaFreeAsync(ptr0_, stream));
                    ptr0_ = nullptr;
                }
                if (ptr1_) {
                    check_cuda_value(cudaFreeAsync(ptr1_, stream));
                    ptr1_ = nullptr;
                }
                check_cuda_value(cudaStreamSynchronize(stream));
                ptr_table_cap_ = 0;
            }
            check_cuda_value(cudaMalloc(&ptr0_, ptr_table_bytes));
            check_cuda_value(cudaMalloc(&ptr1_, ptr_table_bytes));
            ptr_table_cap_ = ptr_table_bytes;
        }
    }

    void*        staging_{nullptr};
    void*        ptr0_{nullptr};
    void*        ptr1_{nullptr};
    size_t       staging_cap_{0};
    size_t       ptr_table_cap_{0};
    int          buffer_device_{-1};
    cudaStream_t copy_stream_{nullptr};
};

static SplitKvTensorCopyCudaState& splitKvTensorCopyState() {
    thread_local SplitKvTensorCopyCudaState state;
    return state;
}

static bool tryExecNoBlockCopySplitKv(const MultiCopyParams& params) {
    return splitKvTensorCopyState().run(params);
}

void splitKvReleaseTensorCopyStateCurrentThread() {
    splitKvTensorCopyState().releaseDeviceResources();
}
#endif
}  // anonymous namespace

#if USING_CUDA
void releaseSplitKvTensorCopyCudaState() {
    splitKvReleaseTensorCopyStateCurrentThread();
}
#endif

void execCopy(const CopyParams& params) {
    runtimeCopy(params);
}

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

void execNoBlockCopy(const MultiCopyParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.multi_src.size() == params.multi_dst.size(),
                          "multi_src and multi_dst must have the same size");
#if USING_CUDA
    auto stream = getNoBlockCopyStream().stream();
    if (params.split_kv_layer_num > 0 && tryExecNoBlockCopySplitKv(params)) {
        check_cuda_error();
        return;
    }
    for (size_t i = 0; i < params.multi_src.size(); i++) {
        cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                        params.multi_src[i].data_ptr(),
                        params.multi_src[i].nbytes(),
                        cudaMemcpyDefault,
                        stream);
    }
    cudaStreamSynchronize(stream);
    check_cuda_error();
#elif USING_ROCM
    for (size_t i = 0; i < params.multi_src.size(); i++) {
        params.multi_dst[i].copy_(params.multi_src[i]);
    }
#else
    for (size_t i = 0; i < params.multi_src.size(); i++) {
        params.multi_dst[i].copy_(params.multi_src[i]);
    }
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

// === Communication ops (backed by c10d ProcessGroup) ===

void execBroadcast(const BroadcastParams& params) {
    c10dBroadcast(params);
}

AllReduceOutput execAllReduce(const AllReduceParams& params) {
    return c10dAllReduce(params);
}

void execAllGather(const AllGatherParams& params) {
    c10dAllGather(params);
}

void execSyncCommunication(bool timeout) {
    c10dSyncCommunication(timeout);
}

void execSyncCommunication(ParallelMode mode, bool timeout) {
    c10dSyncCommunication(mode, timeout);
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

// === Misc ===

void execMaskLogits(torch::Tensor& logits, const torch::Tensor& mask) {
    runtimeMaskLogits(logits, mask);
}

// ============================================================
// initExecCtx — high-level init from config objects
// ============================================================

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
                           const NcclCommConfig&              nccl_comm_config) {
    ExecInitParams params;

    // Guard against double-init (same side effects as old initGlobalRuntime)
    if (g_runtime_initialized.load(std::memory_order_acquire)) {
        RTP_LLM_LOG_WARNING("Runtime is already initialized! will do nothing.");
        return params;
    }

    std::call_once(g_init_flag, [&]() {
#if USING_CUDA
#if defined(USE_PPU)
        params.device_type = DeviceType::Ppu;
#else
        params.device_type = DeviceType::Cuda;
#endif
#elif USING_ROCM
        params.device_type = DeviceType::ROCm;
#endif
        params.tp_size           = parallelism_config.tp_size;
        params.dp_size           = parallelism_config.dp_size;
        params.ep_size           = parallelism_config.ep_size;
        params.ep_rank           = parallelism_config.ep_rank;
        params.tp_rank           = parallelism_config.tp_rank;
        params.dp_rank           = parallelism_config.dp_rank;
        params.ffn_tp_size       = parallelism_config.ffn_tp_size;
        params.ffn_tp_rank       = parallelism_config.ffn_tp_rank;
        params.enable_sp         = parallelism_config.enable_sp;
        params.enable_prefill_cp = parallelism_config.prefill_cp_config.is_enabled();

        params.use_all_gather             = moe_config.use_all_gather && !moe_config.use_deepep_low_latency;
        params.device_id                  = parallelism_config.world_rank % parallelism_config.local_world_size;
        params.tokens_per_block           = model_config.attn_config.tokens_per_block;
        params.kernel_tokens_per_block    = model_config.attn_config.kernel_tokens_per_block;
        params.mla_ops_type               = model_config.mla_ops_type;
        params.max_seq_len                = model_config.max_seq_len;
        params.hidden_size                = model_config.hidden_size;
        params.num_experts                = model_config.expert_num;
        params.extra_experts              = eplb_config.phy_exp_num(model_config.expert_num) - model_config.expert_num;
        params.fmha_config                = fmha_config;
        params.device_resource_config     = device_resource_config;
        params.moe_config                 = moe_config;
        params.sp_config                  = sp_config;
        params.runtime_config             = runtime_config;
        params.misc_config                = misc_config;
        params.parallelism_config.tp_size = parallelism_config.tp_size;
        params.parallelism_config.ep_size = parallelism_config.ep_size;
        params.parallelism_config.dp_size = parallelism_config.dp_size;
        params.parallelism_config.pp_size = parallelism_config.pp_size;
        params.parallelism_config.world_size       = parallelism_config.world_size;
        params.parallelism_config.world_rank       = parallelism_config.world_rank;
        params.parallelism_config.local_world_size = parallelism_config.local_world_size;
        params.parallelism_config.ffn_sp_size      = parallelism_config.ffn_sp_size;
        params.profile_debug_logging_config        = profiling_debug_logging_config;
        params.hw_kernel_config                    = hw_kernel_config;
        params.concurrency_config                  = concurrency_config;
        params.ffn_as_service                      = ffn_disaggregate_config.is_ffn_service();
        RTP_LLM_LOG_INFO("set overlap type to be %d", params.device_resource_config.overlap_comm_type);
        params.m_split                 = device_resource_config.m_split;
        params.max_generate_batch_size = runtime_config.max_generate_batch_size;

        params.enable_comm_overlap      = device_resource_config.enable_comm_overlap;
        params.enable_layer_micro_batch = static_cast<MicroBatchType>(device_resource_config.enable_layer_micro_batch);
        RTP_LLM_LOG_INFO("enable comm overlap: %d, enable layer micro batch: %d",
                         params.enable_comm_overlap,
                         params.enable_layer_micro_batch);
        params.user_deep_gemm_num_sm  = hw_kernel_config.deep_gemm_num_sm;
        params.use_aiter_pa           = fmha_config.use_aiter_pa;
        params.use_asm_pa             = fmha_config.use_asm_pa;
        params.use_deepep_moe         = moe_config.use_deepep_moe;
        params.use_deepep_internode   = moe_config.use_deepep_internode;
        params.use_deepep_low_latency = moe_config.use_deepep_low_latency;
        auto sp_type_str              = SpeculativeExecutionConfig::to_string(sp_config.type);
        auto sp_model_type            = sp_config.model_type;
        RTP_LLM_LOG_INFO("device_params sp_type is %s", sp_type_str.c_str());
        RTP_LLM_LOG_INFO("device_params sp_model_type is %s", sp_model_type.c_str());
        if (((sp_config.type == SP_TYPE_VANILLA) && (sp_model_type == "mixtbstars-mtp"))
            || ((sp_config.type == SP_TYPE_VANILLA) && (sp_model_type == "deepseek-v3-mtp"))
            || (sp_config.type == SP_TYPE_MTP) || (sp_config.type == SP_TYPE_EAGLE)) {
            params.is_mtp = true;
            RTP_LLM_LOG_INFO("device_params.is_mtp true");
        }

        if (((sp_config.type == SP_TYPE_VANILLA) && (sp_model_type == "qwen_3_moe_eagle"))
            || (sp_config.type == SP_TYPE_EAGLE3)) {
            params.is_eagle3 = true;
            RTP_LLM_LOG_INFO("device_params.eagle3 true");
        }

        RTP_LLM_LOG_INFO(
            "use deepep moe: %d, use deepep low latency: %d", params.use_deepep_moe, params.use_deepep_low_latency);

        params.model_specific_config = model_specific_config;

        // --- perform the runtime init that was in initGlobalRuntime ---
        setlinebuf(stdout);

        if (params.profile_debug_logging_config.trace_memory) {
            autil::EnvUtil::setEnv("STACK_TRACER_LOG", "true");
            DECLARE_STACK_TRACER_FILE("rtp_llm_stack.log");
        }

        int device_id = params.device_id;
#if USING_CUDA
        RTP_LLM_LOG_INFO("Initialize runtime. device_id=%d", device_id);
        check_cuda_value(cudaSetDevice(device_id));
        at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());

        if (!sDevMPS::warmup_sm_copy_split_kernels(at::cuda::getCurrentCUDAStream().stream())) {
            RTP_LLM_LOG_WARNING("warmup_sm_copy_split_kernels failed (split KV copy may JIT on first use)");
        }

        if (params.mla_ops_type == MlaOpsType::AUTO) {
            auto* prop          = at::cuda::getCurrentDeviceProperties();
            params.mla_ops_type = prop->major >= 9 ? MlaOpsType::FLASH_MLA : MlaOpsType::FLASH_INFER;
        }
#elif USING_ROCM
        RTP_LLM_LOG_INFO("Initialize runtime (ROCm). device_id=%d", device_id);
        ROCM_CHECK(hipSetDevice(device_id));
#endif

        // Set module-level config for CudaOps copy overlap
        g_enable_comm_overlap = params.enable_comm_overlap;

        // Create the ExecCtxExporter eagerly
        struct RuntimeExporter: public torch_ext::ExecCtxExporter {
            explicit RuntimeExporter(const ExecInitParams& p): ExecCtxExporter(p) {}
            torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool b) override {
                return rtp_llm::preprocessGemmWeightByKey(key, weight, b);
            }
            torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) override {
                return rtp_llm::preprocessWeightScale(weight, scale);
            }
        };
        g_exporter = std::make_shared<RuntimeExporter>(params);

        g_runtime_initialized.store(true, std::memory_order_release);
        RTP_LLM_LOG_INFO("Runtime init done (communication via c10d ProcessGroup)");
    });

    RTP_LLM_LOG_INFO("init devices done");
    return params;
}

// ============================================================
// ExecCtxExporter (pybind helper)
// ============================================================

std::shared_ptr<torch_ext::ExecCtxExporter> getExecCtxExporter() {
    if (g_exporter) {
        return g_exporter;
    }
    // Fallback: runtime not yet initialized — return a default exporter
    static ExecInitParams default_params;
#if USING_CUDA
#if defined(USE_PPU)
    default_params.device_type = DeviceType::Ppu;
#else
    default_params.device_type = DeviceType::Cuda;
#endif
#elif USING_ROCM
    default_params.device_type = DeviceType::ROCm;
#endif
    struct DefaultExporter: public torch_ext::ExecCtxExporter {
        explicit DefaultExporter(const ExecInitParams& p): ExecCtxExporter(p) {}
        torch::Tensor preprocessGemmWeightByKey(const std::string& key, torch::Tensor weight, bool b) override {
            return rtp_llm::preprocessGemmWeightByKey(key, weight, b);
        }
        torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor scale) override {
            return rtp_llm::preprocessWeightScale(weight, scale);
        }
    };
    static auto default_exporter = std::make_shared<DefaultExporter>(default_params);
    return default_exporter;
}

// ============================================================
// Pybind registration
// ============================================================

void registerExecCtxOps(pybind11::module& m) {
    pybind11::enum_<DeviceType>(m, "DeviceType")
        .value("Cpu", DeviceType::Cpu)
        .value("Cuda", DeviceType::Cuda)
        .value("Yitian", DeviceType::Yitian)
        .value("ArmCpu", DeviceType::ArmCpu)
        .value("ROCm", DeviceType::ROCm)
        .value("Ppu", DeviceType::Ppu);

    auto exec_ctx_exporter_class =
        pybind11::class_<torch_ext::ExecCtxExporter, std::shared_ptr<torch_ext::ExecCtxExporter>>(m, "ExecCtxExporter")
            .def("get_device_type", &torch_ext::ExecCtxExporter::getDeviceType)
            .def("get_device_id", &torch_ext::ExecCtxExporter::getDeviceId)
            .def("preprocess_gemm_weight_by_key",
                 &torch_ext::ExecCtxExporter::preprocessGemmWeightByKey,
                 py::arg("key"),
                 py::arg("weight"),
                 py::arg("user_arm_gemm_use_kai"))
            .def("preprocess_weight_scale",
                 &torch_ext::ExecCtxExporter::preprocessWeightScale,
                 py::arg("weight"),
                 py::arg("scale"));
    m.attr("DeviceExporter") = exec_ctx_exporter_class;

    m.def("get_exec_ctx", &getExecCtxExporter);
    m.def("get_device", &getExecCtxExporter);

    m.def(
        "init_exec_ctx",
        [](const ParallelismConfig&           parallelism_config,
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
           const NcclCommConfig&              nccl_comm_config) {
            (void)initExecCtx(parallelism_config,
                              model_config,
                              eplb_config,
                              fmha_config,
                              device_resource_config,
                              moe_config,
                              sp_config,
                              misc_config,
                              profiling_debug_logging_config,
                              hw_kernel_config,
                              concurrency_config,
                              ffn_disaggregate_config,
                              runtime_config,
                              model_specific_config,
                              nccl_comm_config);
        },
        py::arg("parallelism_config"),
        py::arg("model_config"),
        py::arg("eplb_config"),
        py::arg("fmha_config"),
        py::arg("device_resource_config"),
        py::arg("moe_config"),
        py::arg("sp_config"),
        py::arg("misc_config"),
        py::arg("profiling_debug_logging_config"),
        py::arg("hw_kernel_config"),
        py::arg("concurrency_config"),
        py::arg("ffn_disaggregate_config"),
        py::arg("runtime_config"),
        py::arg("model_specific_config"),
        py::arg("nccl_comm_config") = NcclCommConfig{});

    m.def(
        "init_device",
        [](const ParallelismConfig&           parallelism_config,
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
           const NcclCommConfig&              nccl_comm_config) {
            (void)initExecCtx(parallelism_config,
                              model_config,
                              eplb_config,
                              fmha_config,
                              device_resource_config,
                              moe_config,
                              sp_config,
                              misc_config,
                              profiling_debug_logging_config,
                              hw_kernel_config,
                              concurrency_config,
                              ffn_disaggregate_config,
                              runtime_config,
                              model_specific_config,
                              nccl_comm_config);
        },
        py::arg("parallelism_config"),
        py::arg("model_config"),
        py::arg("eplb_config"),
        py::arg("fmha_config"),
        py::arg("device_resource_config"),
        py::arg("moe_config"),
        py::arg("sp_config"),
        py::arg("misc_config"),
        py::arg("profiling_debug_logging_config"),
        py::arg("hw_kernel_config"),
        py::arg("concurrency_config"),
        py::arg("ffn_disaggregate_config"),
        py::arg("runtime_config"),
        py::arg("model_specific_config"),
        py::arg("nccl_comm_config") = NcclCommConfig{});

    m.def(
        "register_process_group_from_store",
        [](int mode, const std::string& host, int port, int rank, int world_size, int device_id) {
            DeviceGuard           guard(device_id);
            c10d::TCPStoreOptions opts;
            opts.port       = static_cast<uint16_t>(port);
            opts.isServer   = (rank == 0);
            opts.numWorkers = world_size;
            auto store      = c10::make_intrusive<c10d::TCPStore>(host, opts);
            auto backend    = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, world_size);
            auto pg         = c10::make_intrusive<c10d::ProcessGroup>(store, rank, world_size);
            pg->setBackend(c10::DeviceType::CUDA, c10d::ProcessGroup::BackendType::NCCL, backend);
            registerProcessGroup(static_cast<ParallelMode>(mode), std::move(pg), device_id);
        },
        py::arg("mode"),
        py::arg("host"),
        py::arg("port"),
        py::arg("rank"),
        py::arg("world_size"),
        py::arg("device_id"),
        "Create and register a ProcessGroup in C++ using a TCPStore.");

    m.def("clear_process_groups", &clearProcessGroups, "Clear all registered ProcessGroups.");
}

}  // namespace rtp_llm
