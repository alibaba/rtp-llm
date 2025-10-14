/**
 * tensor_ipc.cpp
 *
 * A minimal, self-contained C++/CUDA extension for PyTorch that
 * enables zero-copy sharing of contiguous CUDA tensors between
 * independent Python processes via CUDA IPC (cudaIpcMemHandle_t).
 *
 *  - export_tensor_ipc : pack a tensor into an IPC descriptor (bytes)
 *  - import_tensor_ipc : reconstruct a tensor from an IPC descriptor
 *
 *  Requirements:
 *      - C++17
 *      - CUDA Toolkit 11.0+
 *      - PyTorch >= 1.8
 */
#include "rtp_llm/tools/tipc/cpp/IpcOp.h"
#include <cstring>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
namespace torch_ext {

#if USING_CUDA

/* ================================================================== */
/* 2.  Export                                                          */
/* ================================================================== */

/**
 * export_tensor_ipc
 *
 * Pack a contiguous CUDA tensor into an IPC descriptor that can be
 * transmitted to another process (e.g. via socket, shared memory, …).
 *
 * @param tensor  A contiguous CUDA tensor with <= 8 dimensions.
 * @return        A `bytes` object containing the raw IPC descriptor.
 *
 * @throws c10::Error if tensor is not CUDA, not contiguous, or too large.
 */
py::bytes export_tensor_ipc(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "export_tensor_ipc: tensor must be on CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "export_tensor_ipc: tensor must be contiguous");
    TORCH_CHECK(tensor.dim() <= 8, "export_tensor_ipc: tensor dim > 8 is not supported");

    IpcCudaTensorMeta meta{};
    meta.dtype  = static_cast<int32_t>(tensor.scalar_type());
    meta.ndim   = tensor.dim();
    meta.device = tensor.device().index();
    meta.numel  = tensor.numel();

    for (int64_t i = 0; i < meta.ndim; ++i) {
        meta.sizes[i]   = tensor.size(i);
        meta.strides[i] = tensor.stride(i);
    }

    // Ensure any pending work on the tensor is complete.
    cudaStreamSynchronize(nullptr);

    cudaError_t err = cudaIpcGetMemHandle(&meta.handle, tensor.data_ptr());
    TORCH_CHECK(err == cudaSuccess, "export_tensor_ipc: cudaIpcGetMemHandle failed: ", cudaGetErrorString(err));

    return py::bytes(reinterpret_cast<const char*>(&meta), sizeof(meta));
}

/* ================================================================== */
/* 3.  Import                                                          */
/* ================================================================== */

/**
 * import_tensor_ipc
 *
 * Reconstruct a CUDA tensor from an IPC descriptor previously created by
 * `export_tensor_ipc`.
 *
 * @param packed  A `bytes` object exactly sizeof(IpcCudaTensorMeta) bytes.
 * @return        A new torch::Tensor that shares the underlying GPU memory
 *                with the exporting process.  The memory is automatically
 *                released via `cudaIpcCloseMemHandle` when the last Python
 *                reference to the tensor is dropped.
 *
 * @throws c10::Error on size mismatch or CUDA runtime failure.
 */
torch::Tensor import_tensor_ipc(py::bytes packed) {
    std::string buf = packed;
    TORCH_CHECK(buf.size() == sizeof(IpcCudaTensorMeta),
                "import_tensor_ipc: bytes length mismatch (expected ",
                sizeof(IpcCudaTensorMeta),
                ", got ",
                buf.size(),
                ")");

    IpcCudaTensorMeta meta{};
    std::memcpy(&meta, buf.data(), sizeof(meta));

    // Switch to the correct GPU.
    cudaError_t err = cudaSetDevice(meta.device);
    TORCH_CHECK(err == cudaSuccess, "import_tensor_ipc: cudaSetDevice failed: ", cudaGetErrorString(err));

    // Map the IPC handle into this process.
    void* dev_ptr = nullptr;
    err           = cudaIpcOpenMemHandle(&dev_ptr, meta.handle, cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(err == cudaSuccess, "import_tensor_ipc: cudaIpcOpenMemHandle failed: ", cudaGetErrorString(err));

    // Build sizes / strides vectors.
    std::vector<int64_t> sizes(meta.sizes, meta.sizes + meta.ndim);
    std::vector<int64_t> strides(meta.strides, meta.strides + meta.ndim);

    auto options =
        torch::TensorOptions().device(torch::kCUDA, meta.device).dtype(static_cast<at::ScalarType>(meta.dtype));

    // Deleter ensures cudaIpcCloseMemHandle is called on destruction.
    auto deleter = [dev_ptr](void*) { cudaIpcCloseMemHandle(dev_ptr); };

    return torch::from_blob(dev_ptr, sizes, strides, deleter, options);
}

#else

py::bytes export_tensor_ipc(const torch::Tensor& tensor) {
    throw std::runtime_error("RtpLLM is compiled without CUDA; tensor IPC function is not available.");
}

torch::Tensor import_tensor_ipc(py::bytes packed) {
    throw std::runtime_error("RtpLLM is compiled without CUDA; tensor IPC function is not available.");
}

#endif



}  // namespace torch_ext