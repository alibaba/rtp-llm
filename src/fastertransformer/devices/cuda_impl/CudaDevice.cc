#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"

#include <cuda_runtime.h>
#include <unistd.h>

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

    auto ret = nvmlInit();
    FT_CHECK(ret == NVML_SUCCESS);
    ret = nvmlDeviceGetHandleByIndex(device_id_, &nvml_device_);
    FT_CHECK(ret == NVML_SUCCESS);
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

DeviceProperties CudaDevice::getDeviceProperties() {
    static DeviceProperties* prop = nullptr;
    if (prop == nullptr) {
        prop = new DeviceProperties();
        prop->type = DeviceType::Cuda;
    }
    return *prop;
}

// TODO(wangyin.yx): fill all memory status.
DeviceStatus CudaDevice::getDeviceStatus() {
    DeviceStatus status;

    size_t total_bytes;
    auto error = cudaMemGetInfo(&status.device_memory_status.free_bytes, &total_bytes);
    FT_CHECK(error == cudaSuccess);
    status.device_memory_status.used_bytes = total_bytes - status.device_memory_status.free_bytes;

    const auto buffer_status = queryBufferStatus();
    status.device_memory_status.allocated_bytes = buffer_status.device_allocated_bytes;
    status.device_memory_status.preserved_bytes = buffer_status.device_preserved_bytes;
    status.host_memory_status.allocated_bytes = buffer_status.host_allocated_bytes;

    nvmlUtilization_t utilization;
    auto ret = nvmlDeviceGetUtilizationRates(nvml_device_, &utilization);
    FT_CHECK(ret == NVML_SUCCESS);
    status.device_utilization = (float)utilization.gpu;

    return status;
}

RTP_LLM_REGISTER_DEVICE(Cuda);

}; // namespace fastertransformer

