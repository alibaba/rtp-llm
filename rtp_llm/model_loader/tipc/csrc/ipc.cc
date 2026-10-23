#include "ipc.h"
#include <cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>
#include <cstring>
#include <cstdint>
#include <limits>

namespace py = pybind11;

namespace tipc {

#pragma pack(push, 1)
struct IpcCudaTensorMeta {
    cudaIpcMemHandle_t handle;  // 64 B
    int32_t            dtype;   // at::ScalarType
    int32_t            ndim;    // dims
    int32_t            device;  // CUDA ordinal
    int64_t            numel;   // total elements
    int64_t            sizes[8];
    int64_t            strides[8];
};
#pragma pack(pop)

py::bytes export_tensor_ipc(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "tensor must be on CUDA");
    TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(t.dim() <= 8, "dim>8 not supported in demo");

    IpcCudaTensorMeta meta{};
    meta.dtype  = static_cast<int32_t>(t.scalar_type());
    meta.ndim   = t.dim();
    meta.device = t.device().index();
    meta.numel  = t.numel();

    for (int i = 0; i < meta.ndim; ++i) {
        meta.sizes[i]   = t.size(i);
        meta.strides[i] = t.stride(i);
    }

    cudaIpcGetMemHandle(&meta.handle, t.data_ptr());
    return py::bytes(reinterpret_cast<char*>(&meta), sizeof(meta));
}

torch::Tensor import_tensor_ipc(py::bytes b) {
    std::string buf = b;
    TORCH_CHECK(buf.size() == sizeof(IpcCudaTensorMeta), "bytes length mismatch");

    IpcCudaTensorMeta meta;
    std::memcpy(&meta, buf.data(), sizeof(meta));

    /* ---------- 2. 把当前线程绑到该 GPU ---------- */
    cudaSetDevice(meta.device);

    /* ---------- 1. 让 PyTorch 把目标 GPU 的 primary context 建立起来 ---------- */
    // 构造一个 0 元素的临时张量即可激活 context
    // auto dummy = torch::empty({0}, torch::device(torch::kCUDA));

    /* ---------- 3. 打开句柄 ---------- */
    void*       dev_ptr = nullptr;
    cudaError_t st      = cudaIpcOpenMemHandle(&dev_ptr, meta.handle, cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(st == cudaSuccess, "cudaIpcOpenMemHandle failed: ", cudaGetErrorString(st));

    /* ---------- 4. 构造张量 ---------- */
    std::vector<int64_t> sizes(meta.sizes, meta.sizes + meta.ndim);
    std::vector<int64_t> strides(meta.strides, meta.strides + meta.ndim);
    auto opts = torch::TensorOptions().device(torch::kCUDA, meta.device).dtype(static_cast<at::ScalarType>(meta.dtype));

    return torch::from_blob(dev_ptr, sizes, strides, [dev_ptr](void*) { cudaIpcCloseMemHandle(dev_ptr); }, opts);
}

namespace {
constexpr uint32_t kIpcV2Magic   = 0x52545032;  // RTP2
constexpr uint32_t kIpcV2Version = 2;

#pragma pack(push, 1)
struct IpcCudaTensorMetaV2 {
    uint32_t           magic;
    uint32_t           version;
    cudaIpcMemHandle_t handle;
    cudaUUID_t         device_uuid;
    int32_t            dtype;
    int32_t            ndim;
    uint64_t           allocation_bytes;
    uint64_t           offset_bytes;
    uint64_t           numel;
    int64_t            sizes[8];
};
#pragma pack(pop)

void checkCuda(cudaError_t status, const char* operation) {
    TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

int findVisibleDevice(const cudaUUID_t& uuid) {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cudaDeviceProp properties;
        checkCuda(cudaGetDeviceProperties(&properties, ordinal), "cudaGetDeviceProperties");
        if (std::memcmp(properties.uuid.bytes, uuid.bytes, sizeof(uuid.bytes)) == 0) {
            return ordinal;
        }
    }
    TORCH_CHECK(false, "IPC producer GPU UUID is not visible to this process");
}
}  // namespace

py::bytes export_tensor_ipc_v2(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "IPC v2 requires a contiguous CUDA tensor");
    TORCH_CHECK(tensor.dim() <= 8 && tensor.numel() > 0, "IPC v2 requires a nonempty tensor with at most 8 dimensions");
    c10::cuda::CUDAGuard guard(tensor.device());
    IpcCudaTensorMetaV2  meta{};
    meta.magic   = kIpcV2Magic;
    meta.version = kIpcV2Version;
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16 || tensor.scalar_type() == at::kFloat
                    || tensor.scalar_type() == at::kHalf,
                "IPC v2 supports BF16, FP16, or FP32 weights");
    meta.dtype = tensor.scalar_type() == at::kBFloat16 ? 1 : (tensor.scalar_type() == at::kFloat ? 2 : 3);
    meta.ndim  = tensor.dim();
    meta.numel = tensor.numel();
    for (int i = 0; i < meta.ndim; ++i) {
        meta.sizes[i] = tensor.size(i);
    }
    cudaDeviceProp properties;
    checkCuda(cudaGetDeviceProperties(&properties, tensor.device().index()), "cudaGetDeviceProperties");
    meta.device_uuid             = properties.uuid;
    CUdeviceptr allocation_base  = 0;
    size_t      allocation_bytes = 0;
    const auto  data             = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
    const auto  status           = cuMemGetAddressRange(&allocation_base, &allocation_bytes, data);
    TORCH_CHECK(status == CUDA_SUCCESS,
                "cuMemGetAddressRange failed (",
                static_cast<int>(status),
                "); IPC v2 requires a CUDA allocation exportable by cudaIpcGetMemHandle");
    meta.allocation_bytes = allocation_bytes;
    meta.offset_bytes     = data - allocation_base;
    TORCH_CHECK(meta.offset_bytes <= allocation_bytes && tensor.nbytes() <= allocation_bytes - meta.offset_bytes,
                "Tensor exceeds its CUDA allocation");
    checkCuda(cudaIpcGetMemHandle(&meta.handle, reinterpret_cast<void*>(allocation_base)), "cudaIpcGetMemHandle");
    return py::bytes(reinterpret_cast<const char*>(&meta), sizeof(meta));
}

torch::Tensor import_tensor_ipc_v2(py::bytes bytes) {
    const std::string buffer = bytes;
    TORCH_CHECK(buffer.size() == sizeof(IpcCudaTensorMetaV2), "IPC v2 metadata size mismatch");
    IpcCudaTensorMetaV2 meta;
    std::memcpy(&meta, buffer.data(), sizeof(meta));
    TORCH_CHECK(meta.magic == kIpcV2Magic && meta.version == kIpcV2Version, "IPC v2 protocol version mismatch");
    TORCH_CHECK(meta.ndim >= 0 && meta.ndim <= 8, "IPC v2 invalid dimension count");
    // Restrict RL weights to the exact dtypes supported by the BF16 integration.
    TORCH_CHECK(meta.dtype >= 1 && meta.dtype <= 3, "IPC v2 supports BF16, FP16, or FP32 weights");
    const auto dtype = meta.dtype == 1 ? at::kBFloat16 : (meta.dtype == 2 ? at::kFloat : at::kHalf);
    uint64_t   count = 1;
    for (int i = 0; i < meta.ndim; ++i) {
        TORCH_CHECK(meta.sizes[i] > 0 && count <= std::numeric_limits<uint64_t>::max() / meta.sizes[i],
                    "IPC v2 invalid tensor shape");
        count *= meta.sizes[i];
    }
    const auto element_bytes = c10::elementSize(dtype);
    TORCH_CHECK(count == meta.numel && meta.offset_bytes <= meta.allocation_bytes
                    && meta.offset_bytes % element_bytes == 0
                    && count <= (meta.allocation_bytes - meta.offset_bytes) / element_bytes,
                "IPC v2 tensor range exceeds its allocation");
    const int            device = findVisibleDevice(meta.device_uuid);
    c10::cuda::CUDAGuard guard(device);
    void*                base = nullptr;
    checkCuda(cudaIpcOpenMemHandle(&base, meta.handle, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
    try {
        CUdeviceptr actual_base  = 0;
        size_t      actual_bytes = 0;
        TORCH_CHECK(cuMemGetAddressRange(&actual_base, &actual_bytes, reinterpret_cast<CUdeviceptr>(base))
                        == CUDA_SUCCESS,
                    "Cannot validate imported CUDA allocation");
        TORCH_CHECK(actual_base == reinterpret_cast<CUdeviceptr>(base) && actual_bytes == meta.allocation_bytes,
                    "IPC v2 allocation size mismatch");
        std::vector<int64_t> sizes(meta.sizes, meta.sizes + meta.ndim);
        auto                 options = torch::TensorOptions().device(torch::kCUDA, device).dtype(dtype);
        return torch::from_blob(
            static_cast<char*>(base) + meta.offset_bytes,
            sizes,
            [base, device](void*) {
                c10::cuda::CUDAGuard guard(device);
                cudaIpcCloseMemHandle(base);
            },
            options);
    } catch (...) {
        cudaIpcCloseMemHandle(base);
        throw;
    }
}

}  // namespace tipc
