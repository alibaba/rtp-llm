#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"

namespace fastertransformer {
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

}  // namespace fastertransformer