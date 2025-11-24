#pragma once
#include "rtp_llm/cpp/devices/GraphUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

class CudaGraphStreamLife {
public:
    // CudaDevice's `stream_` is torch default stream
    CudaGraphStreamLife(at::cuda::CUDAStream capture_stream, rtp_llm::DeviceBase* device):
        origin_stream_(at::cuda::getCurrentCUDAStream(at::cuda::current_device())) {
        cuda_device_ = dynamic_cast<rtp_llm::CudaDevice*>(device);
        // Set `capture_stream` for capture. All kernels should use this stream while capturing.
        origin_cuda_device_stream_ = cuda_device_->getStream();
        cuda_device_->setStream(capture_stream.stream());
        RTP_LLM_LOG_INFO("Set Cuda Stream: capture_stream -> %d, set_stream -> %d, origin_cuda_device_stream_-> %d",
                         capture_stream.stream(),
                         reinterpret_cast<int64_t>(cuda_device_->getStream()),
                         origin_cuda_device_stream_);
        at::cuda::setCurrentCUDAStream(capture_stream);
    }

    ~CudaGraphStreamLife() {
        at::cuda::setCurrentCUDAStream(origin_stream_);
        cuda_device_->setStream(origin_cuda_device_stream_);
    }

private:
    at::cuda::CUDAStream origin_stream_;
    cudaStream_t         origin_cuda_device_stream_;
    rtp_llm::CudaDevice* cuda_device_;
};

namespace CudaGraphUtils {

// 设备同步
inline void deviceSynchronize() {
    cudaDeviceSynchronize();
}

// 流管理
inline at::cuda::CUDAStream getStreamFromPool(bool high_priority = false) {
    return at::cuda::getStreamFromPool(high_priority);
}

inline at::cuda::CUDAStream getCurrentStream() {
    return at::cuda::getCurrentCUDAStream();
}

template<typename DeviceType>
inline auto createStreamLife(at::cuda::CUDAStream capture_stream, DeviceType* device) {
    return CudaGraphStreamLife(capture_stream, device);
}

}  // namespace CudaGraphUtils

}  // namespace rtp_llm
