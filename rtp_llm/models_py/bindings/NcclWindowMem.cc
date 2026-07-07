#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <torch/extension.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define RTP_CHECK_CUDA(cmd)                                                                            \
    do {                                                                                               \
        cudaError_t e = (cmd);                                                                         \
        if (e != cudaSuccess) {                                                                        \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));             \
        }                                                                                              \
    } while (0)

#define RTP_CHECK_NCCL(cmd)                                                                            \
    do {                                                                                               \
        ncclResult_t r = (cmd);                                                                        \
        if (r != ncclSuccess) {                                                                        \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(r));             \
        }                                                                                              \
    } while (0)

namespace torch_ext {
namespace {

constexpr size_t kMaxWindowTotalBytes = 64ul * 1024ul * 1024ul;

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22700
constexpr bool kNcclWindowSupported = true;
#else
constexpr bool kNcclWindowSupported = false;
#endif

ncclDataType_t ncclDtypeFromTensor(const torch::Tensor& tensor) {
    if (tensor.scalar_type() == at::ScalarType::BFloat16) {
        return ncclBfloat16;
    }
    if (tensor.scalar_type() == at::ScalarType::Float) {
        return ncclFloat32;
    }
    throw std::runtime_error("NCCL window all_gather only supports bf16/fp32");
}

struct WindowBuffer {
    void*        send = nullptr;
    void*        recv = nullptr;
    size_t       send_bytes = 0;
    size_t       recv_bytes = 0;
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22700
    ncclWindow_t send_win = nullptr;
    ncclWindow_t recv_win = nullptr;
#endif
};

}  // namespace

class NcclWindowAllGather {
public:
    NcclWindowAllGather(torch::Tensor unique_id, int rank, int world_size, int device, bool zero_cta):
        rank_(rank), world_size_(world_size), device_(device) {
        if (!unique_id.device().is_cpu() || unique_id.scalar_type() != at::ScalarType::Byte
            || !unique_id.is_contiguous()) {
            throw std::runtime_error("unique_id must be a contiguous CPU uint8 tensor");
        }
        if (unique_id.numel() != static_cast<int64_t>(sizeof(ncclUniqueId))) {
            throw std::runtime_error("unique_id has unexpected size");
        }
        if (world_size_ <= 0) {
            throw std::runtime_error("world_size must be positive");
        }

        ncclUniqueId id;
        std::memcpy(&id, unique_id.data_ptr(), sizeof(ncclUniqueId));
        if (!kNcclWindowSupported) {
            throw std::runtime_error("NCCL registered-window API is not available");
        }

        RTP_CHECK_CUDA(cudaSetDevice(device_));
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22700
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        if (zero_cta) {
            config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
        }
        RTP_CHECK_NCCL(ncclCommInitRankConfig(&comm_, world_size_, id, rank_, &config));
#else
        (void)zero_cta;
        (void)id;
#endif
    }

    ~NcclWindowAllGather() {
        for (auto& kv : buffers_) {
            releaseBuffer(kv.second.get());
        }
        if (comm_ != nullptr) {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }
    }

    torch::Tensor allGather(torch::Tensor shard, const std::string& key) {
        validateShard(shard);
        const int64_t shard_numel = shard.numel();
        const size_t  send_bytes  = static_cast<size_t>(shard_numel) * shard.element_size();
        const size_t  recv_bytes  = send_bytes * static_cast<size_t>(world_size_);
        WindowBuffer* buf         = ensureBuffer(key, send_bytes, recv_bytes);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(shard.device().index()).stream();
        RTP_CHECK_CUDA(cudaMemcpyAsync(buf->send, shard.data_ptr(), send_bytes, cudaMemcpyDeviceToDevice, stream));
        RTP_CHECK_NCCL(ncclAllGather(buf->send, buf->recv, shard_numel, ncclDtypeFromTensor(shard), comm_, stream));

        std::vector<int64_t> sizes = {shard.size(0) * world_size_, shard.size(1)};
        return torch::from_blob(buf->recv, sizes, [](void*) {}, shard.options());
    }

    void prepareAllGather(torch::Tensor shard, const std::string& key) {
        validateShard(shard);
        const size_t send_bytes = static_cast<size_t>(shard.numel()) * shard.element_size();
        const size_t recv_bytes = send_bytes * static_cast<size_t>(world_size_);
        ensureBuffer(key, send_bytes, recv_bytes);
    }

private:
    void validateShard(const torch::Tensor& shard) const {
        if (!shard.is_cuda() || !shard.is_contiguous() || shard.dim() != 2) {
            throw std::runtime_error("shard must be a contiguous CUDA 2D tensor");
        }
        if (shard.device().index() != device_) {
            throw std::runtime_error("shard device does not match NCCL window communicator device");
        }
        const int64_t shard_numel = shard.numel();
        if (shard_numel <= 0) {
            throw std::runtime_error("empty shard is not supported");
        }
    }

    WindowBuffer* ensureBuffer(const std::string& key, size_t send_bytes, size_t recv_bytes) {
        const size_t max_send_bytes = kMaxWindowTotalBytes / static_cast<size_t>(world_size_);
        if (recv_bytes > kMaxWindowTotalBytes || send_bytes > max_send_bytes) {
            throw std::runtime_error("NCCL window all_gather payload exceeds registered buffer limit");
        }
        const size_t send_alloc_bytes = max_send_bytes;
        const size_t recv_alloc_bytes = kMaxWindowTotalBytes;
        auto it = buffers_.find(key);
        if (it != buffers_.end() && it->second->send_bytes >= send_bytes && it->second->recv_bytes >= recv_bytes) {
            return it->second.get();
        }
        if (it != buffers_.end()) {
            releaseBuffer(it->second.get());
            buffers_.erase(it);
        }

        auto buf = std::make_unique<WindowBuffer>();
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22700
        try {
            RTP_CHECK_NCCL(ncclMemAlloc(&buf->send, send_alloc_bytes));
            RTP_CHECK_NCCL(ncclMemAlloc(&buf->recv, recv_alloc_bytes));
            RTP_CHECK_NCCL(
                ncclCommWindowRegister(comm_, buf->send, send_alloc_bytes, &buf->send_win, NCCL_WIN_COLL_SYMMETRIC));
            RTP_CHECK_NCCL(
                ncclCommWindowRegister(comm_, buf->recv, recv_alloc_bytes, &buf->recv_win, NCCL_WIN_COLL_SYMMETRIC));
            buf->send_bytes = send_alloc_bytes;
            buf->recv_bytes = recv_alloc_bytes;
        } catch (...) {
            releaseBuffer(buf.get());
            throw;
        }
#else
        throw std::runtime_error("NCCL registered-window API is not available");
#endif
        WindowBuffer* raw = buf.get();
        buffers_[key] = std::move(buf);
        return raw;
    }

    void releaseBuffer(WindowBuffer* buf) {
        if (buf == nullptr) {
            return;
        }
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22700
        if (buf->send_win != nullptr) {
            ncclCommWindowDeregister(comm_, buf->send_win);
            buf->send_win = nullptr;
        }
        if (buf->recv_win != nullptr) {
            ncclCommWindowDeregister(comm_, buf->recv_win);
            buf->recv_win = nullptr;
        }
        if (buf->send != nullptr) {
            ncclMemFree(buf->send);
            buf->send = nullptr;
        }
        if (buf->recv != nullptr) {
            ncclMemFree(buf->recv);
            buf->recv = nullptr;
        }
#endif
    }

    int rank_ = 0;
    int world_size_ = 1;
    int device_ = 0;
    ncclComm_t comm_ = nullptr;
    std::unordered_map<std::string, std::unique_ptr<WindowBuffer>> buffers_;
};

torch::Tensor getNcclWindowUniqueId() {
    ncclUniqueId id;
    RTP_CHECK_NCCL(ncclGetUniqueId(&id));
    auto out = torch::empty(
        {static_cast<int64_t>(sizeof(ncclUniqueId))},
        torch::dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(out.data_ptr(), &id, sizeof(ncclUniqueId));
    return out;
}

void registerPyNcclWindowMem(pybind11::module& m) {
    m.def("nccl_window_supported", []() { return kNcclWindowSupported; });
    m.def("get_nccl_window_unique_id", &getNcclWindowUniqueId, "Create a NCCL unique id");
    pybind11::class_<NcclWindowAllGather>(m, "NcclWindowAllGather")
        .def(pybind11::init<torch::Tensor, int, int, int, bool>())
        .def("prepare_all_gather", &NcclWindowAllGather::prepareAllGather)
        .def("all_gather", &NcclWindowAllGather::allGather);
}

}  // namespace torch_ext
