#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"

namespace fastertransformer {

CudaDevice::CudaDevice() : DeviceBase(), device_id_(getDevice()) {
    cudaSetDevice(device_id_);
    cudaStreamCreate(&stream_);
    
    auto allocator_ptr = new Allocator<AllocatorType::CUDA>(device_id_);
    allocator_ptr->setStream(stream_);
    allocator_.reset(allocator_ptr);
    host_allocator_.reset(new Allocator<AllocatorType::CUDA_HOST>(device_id_));

    check_cuda_error(cublasCreate(&cublas_handle_));
    cublasLtCreate(&cublaslt_handle_);
    check_cuda_error(cublasSetStream(cublas_handle_, stream_));
    cudaGetDeviceProperties(&device_prop_, device_id_);

    cublas_algo_map_.reset(new cublasAlgoMap(GEMM_CONFIG));
    cublas_mm_wrapper_.reset(new cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(),
        &cublas_wrapper_mutex_, allocator_.get()));

    cublas_mm_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
}

CudaDevice::~CudaDevice() {
    cublas_mm_wrapper_.reset();
    cudaStreamDestroy(stream_);
    cublasDestroy(cublas_handle_);
    cublasLtDestroy(cublaslt_handle_);
}

void CudaDevice::syncAndCheck() {
    cudaDeviceSynchronize();
    sync_check_cuda_error();
}

RTP_LLM_REGISTER_DEVICE(Cuda);

}; // namespace fastertransformer

