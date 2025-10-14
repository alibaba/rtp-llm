// tensor_ipc.h
#pragma once

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#include <torch/extension.h>

namespace torch_ext {

#if USING_CUDA

/* ------------------------------------------------------------------ */
/* 1.  IPC descriptor (64-byte aligned, trivially copyable)           */
/* ------------------------------------------------------------------ */
struct IpcCudaTensorMeta {
    cudaIpcMemHandle_t handle{};      //!< CUDA IPC handle (64 B)
    int32_t            dtype{};       //!< at::ScalarType enum value
    int32_t            ndim{};        //!< Number of dimensions
    int32_t            device{};      //!< CUDA device ordinal
    int64_t            numel{};       //!< Total number of elements
    int64_t            sizes[8]{};    //!< Shape (max 8-D)
    int64_t            strides[8]{};  //!< Stride (max 8-D)
};

#endif

/**
 * export_tensor_ipc
 *
 * 将一块连续 CUDA 张量打包成 IPC 描述符（bytes），
 * 可在进程间通过 socket / shm 等方式零拷贝共享。
 *
 * @param tensor  连续 CUDA 张量
 * @return        长度为 sizeof(IpcCudaTensorMeta) 的 bytes 对象
 */
pybind11::bytes export_tensor_ipc(const torch::Tensor& tensor);

/**
 * import_tensor_ipc
 *
 * 根据 IPC 描述符重建共享张量。
 *
 * @param packed  export_tensor_ipc 返回的 bytes
 * @return        与原张量共享显存的新张量
 */
torch::Tensor import_tensor_ipc(pybind11::bytes packed);

void registerIpcOp(py::module& m);

}  // namespace torch_ext