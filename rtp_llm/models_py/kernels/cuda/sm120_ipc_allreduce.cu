// Reuse FlashInfer's vLLM / SGLang kernel with a fixed, registered IPC input.
// Upstream: vllm v0.8.2 csrc/custom_all_reduce.cuh (Apache-2.0).
#include <cstdint>
#include "flashinfer/comm/vllm_custom_all_reduce.cuh"
#include "tvm_ffi_utils.h"

void registered_allreduce(TensorView input,
                          TensorView output,
                          TensorView pointers,
                          int64_t    buffer,
                          int64_t    buffer_bytes,
                          int64_t    signal0,
                          int64_t    signal1,
                          int64_t    rank,
                          int64_t    ctas) {
    ffi::CUDADeviceGuard guard(input.device().device_id);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code);
    TVM_FFI_ICHECK(input.IsContiguous() && output.IsContiguous());
    TVM_FFI_ICHECK_EQ(input.numel(), output.numel());
    TVM_FFI_ICHECK_GT(input.numel(), 0);
    TVM_FFI_ICHECK_EQ(input.numel() % 8, 0);
    TVM_FFI_ICHECK_LE(input.numel() * 2, buffer_bytes);
    TVM_FFI_ICHECK_GT(ctas, 0);
    TVM_FFI_ICHECK_LE(ctas, vllm::kMaxBlocks);
    TVM_FFI_ICHECK(rank == 0 || rank == 1);
    TVM_FFI_ICHECK_EQ(pointers.numel(), 8);
    TVM_FFI_ICHECK(pointers.IsContiguous());
    TVM_FFI_ICHECK_EQ(pointers.dtype().code, kDLInt);
    TVM_FFI_ICHECK_EQ(pointers.dtype().bits, 64);
    TVM_FFI_ICHECK_EQ(output.device().device_id, input.device().device_id);
    TVM_FFI_ICHECK_EQ(pointers.device().device_id, input.device().device_id);
    TVM_FFI_ICHECK(buffer != 0 && signal0 != 0 && signal1 != 0);
    auto* staging = reinterpret_cast<void*>(buffer);
    auto  stream  = get_stream(input.device());
    CHECK_CUDA_SUCCESS(cudaMemcpyAsync(staging, input.data_ptr(), input.numel() * 2, cudaMemcpyDeviceToDevice, stream));
    const int         size   = input.numel() / 8;
    const int         blocks = std::min<int>(ctas, (size + 511) / 512);
    vllm::RankSignals signals{};
    signals.signals[0] = reinterpret_cast<vllm::Signal*>(signal0);
    signals.signals[1] = reinterpret_cast<vllm::Signal*>(signal1);
    // Peer pointers are registered before capture. Every graph references the
    // same staging storage, so there is no post-capture registration callback.
    vllm::cross_device_reduce_1stage<nv_bfloat16, 2>
        <<<blocks, 512, 0, stream>>>(reinterpret_cast<vllm::RankData*>(pointers.data_ptr()),
                                     signals,
                                     signals.signals[rank],
                                     reinterpret_cast<nv_bfloat16*>(output.data_ptr()),
                                     rank,
                                     size);
    CHECK_CUDA_SUCCESS(cudaGetLastError());
}

int64_t signal_size() {
    return sizeof(vllm::Signal);
}
void zero_signal(int64_t pointer) {
    CHECK_CUDA_SUCCESS(cudaMemset(reinterpret_cast<void*>(pointer), 0, sizeof(vllm::Signal)));
}
void close_ipc_pointer(int64_t pointer) {
    CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(reinterpret_cast<void*>(pointer)));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(signal_size, signal_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(zero_signal, zero_signal);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(close_ipc_pointer, close_ipc_pointer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(registered_allreduce, registered_allreduce);
