#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/OffloadDeviceBuffer.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace rtp_llm {

#if USING_CUDA
namespace {
void checkDriver(CUresult result, const char* operation) {
    const char* message = nullptr;
    if (result != CUDA_SUCCESS) {
        cuGetErrorString(result, &message);
    }
    RTP_LLM_CHECK_WITH_INFO(result == CUDA_SUCCESS, "%s: %s", operation, message ? message : "CUDA driver error");
}
}  // namespace

struct OffloadDeviceBuffer::Impl {
    CUdeviceptr address = 0;
    size_t bytes = 0;
    size_t reserved = 0;
    size_t granularity = 0;
    int device = -1;
    uint32_t maximum = 0;
    uint32_t page_rows = 0;
    uint32_t rows = 0;
    int64_t generation = 0;
    CUmemAllocationProp prop{};
    std::vector<OffloadDeviceSegment> segments;
    std::vector<CUmemGenericAllocationHandle> handles;
    torch::Tensor active;
    torch::Tensor epoch;
    mutable std::mutex mutex;

    ~Impl() {
        if (!address) {
            return;
        }
        c10::cuda::CUDAGuard guard(device);
        // Tensor views/graphs can outlive the cache manager. Their shared owner
        // releases the VA only after the final view has gone away.
        cudaDeviceSynchronize();
        for (size_t i = 0; i < handles.size(); ++i) {
            if (handles[i]) {
                cuMemUnmap(address + i * granularity, granularity);
                cuMemRelease(handles[i]);
            }
        }
        cuMemAddressFree(address, reserved);
    }
};

OffloadDeviceBuffer::OffloadDeviceBuffer(size_t bytes,
                                         uint32_t max_rows,
                                         uint32_t rows_per_page,
                                         std::vector<OffloadDeviceSegment> segments): impl_(std::make_unique<Impl>()) {
    auto& state = *impl_;
    RTP_LLM_CHECK(bytes > 0 && rows_per_page > 0 && max_rows > 0 && max_rows % rows_per_page == 0);
    RTP_LLM_CHECK(cudaGetDevice(&state.device) == cudaSuccess);
    checkDriver(cuInit(0), "initialize sparse KV CUDA driver");
    int supported = 0;
    checkDriver(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, state.device),
                "query sparse KV virtual-memory support");
    RTP_LLM_CHECK_WITH_INFO(supported, "dynamic sparse KV hot cache requires CUDA virtual memory management");
    state.bytes = bytes;
    state.maximum = max_rows;
    state.page_rows = rows_per_page;
    state.segments = std::move(segments);
    state.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    state.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    state.prop.location.id = state.device;
    int rdma_supported = 0;
    checkDriver(cuDeviceGetAttribute(&rdma_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                    state.device), "query sparse KV VMM RDMA support");
    state.prop.allocFlags.gpuDirectRDMACapable = rdma_supported != 0;
    checkDriver(cuMemGetAllocationGranularity(&state.granularity, &state.prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                "query sparse KV allocation granularity");
    RTP_LLM_CHECK_WITH_INFO(kOffloadAllocationChunkBytes % state.granularity == 0,
                            "CUDA minimum mapping granularity=%zu must divide sparse KV chunk bytes=%zu",
                            state.granularity, kOffloadAllocationChunkBytes);
    state.granularity = kOffloadAllocationChunkBytes;
    RTP_LLM_CHECK(bytes <= std::numeric_limits<size_t>::max() - state.granularity + 1);
    state.reserved = (bytes + state.granularity - 1) / state.granularity * state.granularity;
    state.handles.resize(state.reserved / state.granularity, 0);
    for (const auto& segment : state.segments) {
        RTP_LLM_CHECK(segment.offset <= bytes && segment.fixed_bytes <= bytes - segment.offset);
        RTP_LLM_CHECK(segment.page_bytes > 0
                      && max_rows / rows_per_page <= (bytes - segment.offset - segment.fixed_bytes) / segment.page_bytes);
    }
    auto options = torch::TensorOptions().device(torch::Device(torch::kCUDA, state.device));
    state.active = torch::zeros({}, options.dtype(torch::kInt32));
    state.epoch = torch::zeros({max_rows}, options.dtype(torch::kInt64));
    checkDriver(cuMemAddressReserve(&state.address, state.reserved, 0, 0, 0), "reserve sparse KV virtual addresses");
    resize(0);
}

OffloadDeviceBuffer::~OffloadDeviceBuffer() = default;

torch::Tensor OffloadDeviceBuffer::tensor() {
    auto owner = shared_from_this();
    return torch::from_blob(reinterpret_cast<void*>(impl_->address),
                             {static_cast<int64_t>(impl_->bytes)},
                             [owner = std::move(owner)](void*) {},
                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, impl_->device)));
}

void OffloadDeviceBuffer::resize(uint32_t rows, bool grow_only) {
    auto& state = *impl_;
    std::lock_guard<std::mutex> lock(state.mutex);
    RTP_LLM_CHECK_WITH_INFO(rows <= state.maximum, "hot cache rows=%u exceed maximum=%u", rows, state.maximum);
    rows = (static_cast<uint64_t>(rows) + state.page_rows - 1) / state.page_rows * state.page_rows;
    if (grow_only) {
        rows = std::max(rows, state.rows);
    }
    c10::cuda::CUDAGuard guard(state.device);
    cudaStreamCaptureStatus capture;
    RTP_LLM_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &capture) == cudaSuccess);
    RTP_LLM_CHECK_WITH_INFO(capture == cudaStreamCaptureStatusNone, "hot cache resize must run outside graph capture");
    if (rows == state.rows && state.handles.front()) {
        return;
    }
    std::vector<uint8_t> needed(state.handles.size(), 0);
    for (const auto& segment : state.segments) {
        const size_t end = segment.offset + segment.fixed_bytes + rows / state.page_rows * segment.page_bytes;
        for (size_t i = segment.offset / state.granularity; i < (end + state.granularity - 1) / state.granularity; ++i) {
            needed[i] = 1;
        }
    }
    // Resize only at an engine forward boundary. That stream joins all offload
    // transfer streams before a forward returns, including exceptional exits.
    // Constant concurrency takes the early return and never synchronizes here.
    RTP_LLM_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()) == cudaSuccess);
    std::vector<size_t> added;
    try {
        for (size_t i = 0; i < needed.size(); ++i) {
            if (!needed[i] || state.handles[i]) {
                continue;
            }
            CUmemGenericAllocationHandle handle = 0;
            checkDriver(cuMemCreate(&handle, state.granularity, &state.prop, 0), "allocate sparse KV physical pages");
            const auto mapped = cuMemMap(state.address + i * state.granularity, state.granularity, 0, handle, 0);
            if (mapped != CUDA_SUCCESS) {
                cuMemRelease(handle);
                checkDriver(mapped, "map sparse KV physical pages");
            }
            state.handles[i] = handle;
            added.push_back(i);
            CUmemAccessDesc access{};
            access.location = state.prop.location;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            checkDriver(cuMemSetAccess(state.address + i * state.granularity, state.granularity, &access, 1),
                        "set sparse KV page access");
        }
    } catch (...) {
        for (size_t i : added) {
            cuMemUnmap(state.address + i * state.granularity, state.granularity);
            cuMemRelease(state.handles[i]);
            state.handles[i] = 0;
        }
        throw;
    }
    for (size_t i = 0; i < needed.size(); ++i) {
        if (!needed[i] && state.handles[i]) {
            checkDriver(cuMemUnmap(state.address + i * state.granularity, state.granularity), "unmap sparse KV hot pages");
            checkDriver(cuMemRelease(state.handles[i]), "release sparse KV hot pages");
            state.handles[i] = 0;
        }
    }
    if (rows < state.rows) {
        // A later grow can reuse the same addresses with different physical
        // pages. Runtime tags from before the shrink must never be cache hits.
        state.epoch.slice(0, rows, state.rows).fill_(++state.generation);
    }
    state.rows = rows;
    state.active.fill_(rows);
    const auto pages = std::count_if(state.handles.begin(), state.handles.end(), [](auto handle) { return handle != 0; });
    RTP_LLM_LOG_INFO("sparse KV hot cache active_rows=%u max_rows=%u committed_bytes=%zu granularity=%zu",
                     rows, state.maximum, pages * state.granularity, state.granularity);
}

size_t OffloadDeviceBuffer::committedBytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return std::count_if(impl_->handles.begin(), impl_->handles.end(), [](auto handle) { return handle != 0; })
           * impl_->granularity;
}

const torch::Tensor& OffloadDeviceBuffer::activeRows() const { return impl_->active; }
const torch::Tensor& OffloadDeviceBuffer::epoch() const { return impl_->epoch; }

#else
struct OffloadDeviceBuffer::Impl {};
OffloadDeviceBuffer::OffloadDeviceBuffer(size_t, uint32_t, uint32_t, std::vector<OffloadDeviceSegment>) {
    RTP_LLM_CHECK_WITH_INFO(false, "dynamic sparse KV hot cache requires CUDA");
}
OffloadDeviceBuffer::~OffloadDeviceBuffer() = default;
torch::Tensor OffloadDeviceBuffer::tensor() { return {}; }
void OffloadDeviceBuffer::resize(uint32_t, bool) { RTP_LLM_CHECK(false); }
size_t OffloadDeviceBuffer::committedBytes() const { return 0; }
const torch::Tensor& OffloadDeviceBuffer::activeRows() const { static const torch::Tensor empty; return empty; }
const torch::Tensor& OffloadDeviceBuffer::epoch() const { static const torch::Tensor empty; return empty; }
#endif

}  // namespace rtp_llm
