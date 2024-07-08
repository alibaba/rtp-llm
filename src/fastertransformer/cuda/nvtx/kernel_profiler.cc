#include "kernel_profiler.h"
#if USING_CUDA
#include "src/fastertransformer/cuda/cuda_utils.h"
#endif
#if USING_ROCM
#include "src/fastertransformer/rocm/hip_utils.h"
#endif
namespace fastertransformer {

void KernelProfiler::start() {
    cudaDeviceSynchronize();
    cudaEventSynchronize(start_);
    cudaEventRecord(start_, stream_);
}

void KernelProfiler::stop() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start_, stop_);
    reportMetrics(kernel_time);
}

void KernelProfiler::reportMetrics(float time) {
    rtp_llm::RtpLLMKernelMetricsCollector collector;
    collector.kernel_exec_time = time;
    kmonitor::MetricsTags tags("kernel_name", kernel_name_);
    metrics_reporter_->report<rtp_llm::RtpLLMKernelMetrics, rtp_llm::RtpLLMKernelMetricsCollector>(&tags, &collector);
}

KernelProfiler::KernelProfiler(cudaStream_t                 stream,
                               std::string                  kernel_name,
                               kmonitor::MetricsReporterPtr metrics_reporter):
    stream_(stream), kernel_name_(kernel_name), metrics_reporter_(metrics_reporter) {
    FT_CHECK_WITH_INFO(metrics_reporter_ != nullptr, "metric reporter should not be nullptr");
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

KernelProfiler::~KernelProfiler() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

}  // namespace fastertransformer