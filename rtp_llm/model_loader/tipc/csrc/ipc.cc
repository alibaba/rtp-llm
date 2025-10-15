# include "ipc.h"

namespace py = pybind11;

namespace tipc {

#pragma pack(push,1)
struct IpcCudaTensorMeta {
    cudaIpcMemHandle_t handle;  // 64 B
    int32_t  dtype;             // at::ScalarType
    int32_t  ndim;              // dims
    int32_t  device;            // CUDA ordinal
    int64_t  numel;             // total elements
    int64_t  sizes[8];
    int64_t  strides[8];
};
#pragma pack(pop)

py::bytes export_tensor_ipc(const torch::Tensor& t)
{
    TORCH_CHECK(t.is_cuda(), "tensor must be on CUDA");
    TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(t.dim() <= 8, "dim>8 not supported in demo");

    IpcCudaTensorMeta meta{};
    meta.dtype  = static_cast<int32_t>(t.scalar_type());
    meta.ndim   = t.dim();
    meta.device = t.device().index();
    meta.numel  = t.numel();

    for (int i = 0;i < meta.ndim; ++i) {
        meta.sizes[i]   = t.size(i);
        meta.strides[i] = t.stride(i);
    }

    cudaIpcGetMemHandle(&meta.handle, t.data_ptr());
    return py::bytes(reinterpret_cast<char*>(&meta), sizeof(meta));
}

torch::Tensor import_tensor_ipc(py::bytes b)
{
    std::string buf = b;
    TORCH_CHECK(buf.size() == sizeof(IpcCudaTensorMeta),
                "bytes length mismatch");

    IpcCudaTensorMeta meta;
    std::memcpy(&meta, buf.data(), sizeof(meta));

    /* ---------- 2. 把当前线程绑到该 GPU ---------- */
    cudaSetDevice(meta.device);

    /* ---------- 1. 让 PyTorch 把目标 GPU 的 primary context 建立起来 ---------- */
    // 构造一个 0 元素的临时张量即可激活 context
    // auto dummy = torch::empty({0}, torch::device(torch::kCUDA));

    /* ---------- 3. 打开句柄 ---------- */
    void* dev_ptr = nullptr;
    cudaError_t st = cudaIpcOpenMemHandle(&dev_ptr,
                                          meta.handle,
                                          cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(st == cudaSuccess,
                "cudaIpcOpenMemHandle failed: ",
                cudaGetErrorString(st));

    /* ---------- 4. 构造张量 ---------- */
    std::vector<int64_t> sizes(meta.sizes, meta.sizes + meta.ndim);
    std::vector<int64_t> strides(meta.strides, meta.strides + meta.ndim);
    auto opts = torch::TensorOptions()
                    .device(torch::kCUDA, meta.device)
                    .dtype(static_cast<at::ScalarType>(meta.dtype));

    return torch::from_blob(
        dev_ptr, sizes, strides,
        [dev_ptr](void*) { cudaIpcCloseMemHandle(dev_ptr); },
        opts);
}

} // namespace tipc
