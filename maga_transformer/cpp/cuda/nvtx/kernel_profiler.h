#pragma once
#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#endif
#include <fstream>
#include <iostream>
#include <string>
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
class KernelProfiler {
public:
    KernelProfiler(cudaStream_t stream, std::string kernel_name, kmonitor::MetricsReporterPtr metrics_reporter);
    ~KernelProfiler();
    void start();
    void stop();
    void reportMetrics(float time);

protected:
    cudaStream_t                 stream_;
    std::string                  kernel_name_;
    cudaEvent_t                  start_, stop_;
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
};

}  // namespace rtp_llm