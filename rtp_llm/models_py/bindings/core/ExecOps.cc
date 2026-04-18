#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
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
#include <mutex>
#include <atomic>
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
        auto event          = param.pre_created_event ? param.pre_created_event : runtimeCreateEvent();
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), event);
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
