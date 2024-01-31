#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/core/allocator_cuda.h"

namespace fastertransformer {

CudaDevice::CudaDevice() : device_id_(getDevice()) {
    allocator_.reset(new Allocator<AllocatorType::CUDA>(device_id_));

    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
    cublasSetStream(cublas_handle_, stream_);
    cudaGetDeviceProperties(&device_prop_, device_id_);

    cublas_algo_map_.reset(new cublasAlgoMap(GEMM_CONFIG));
    cublas_mm_wrapper_.reset(new cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(),
        &cublas_wrapper_mutex_, allocator_.get()));
}

CudaDevice::~CudaDevice() {
    cublas_mm_wrapper_.reset();
    cudaStreamDestroy(stream_);
    cublasDestroy(cublas_handle_);
    cublasLtDestroy(cublaslt_handle_);
}

}; // namespace fastertransformer

