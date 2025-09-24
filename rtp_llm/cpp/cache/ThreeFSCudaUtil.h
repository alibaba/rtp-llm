#pragma once

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#define CHECK_CUDA_ERROR(cuda_call)                                                                                    \
    do {                                                                                                               \
        cudaError_t err = (cuda_call);                                                                                 \
        if (err != cudaSuccess) {                                                                                      \
            RTP_LLM_LOG_WARNING("%s failed, err: %s ", #cuda_call, cudaGetErrorString(err));                           \
        }                                                                                                              \
    } while (0)

namespace rtp_llm::threefs {

class ThreeFSCudaUtil final {
public:
    ThreeFSCudaUtil() = default;
    ~ThreeFSCudaUtil() {
        release();
    }

public:
    bool init() {
        auto err = cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            RTP_LLM_LOG_WARNING("init failed, create cuda stream failed, err: %s", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    void release() {
        if (cuda_stream_) {
            CHECK_CUDA_ERROR(cudaStreamDestroy(cuda_stream_));
            cuda_stream_ = nullptr;
        }
    }

    void copyAsyncHostToDevice(void* dst, const void* src, size_t size) {
        check_cuda_value(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cuda_stream_));
    }

    void copyAsyncDeviceToHost(void* dst, const void* src, size_t size) {
        check_cuda_value(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cuda_stream_));
    }

    void sync() {
        check_cuda_value(cudaStreamSynchronize(cuda_stream_));
    }

    bool registerHost(void* ptr, size_t size) const {
        if (ptr == nullptr || size == 0) {
            return false;
        }
        auto err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            RTP_LLM_LOG_WARNING(
                "cuda host register failed, err: %s, ptr: %p, size: %zu", cudaGetErrorString(err), ptr, size);
            return false;
        }
        return true;
    }

    void unregisterHost(void* ptr) const {
        if (ptr != nullptr) {
            CHECK_CUDA_ERROR(cudaHostUnregister(ptr));
        }
    }

private:
    cudaStream_t cuda_stream_{nullptr};
};

}  // namespace rtp_llm::threefs