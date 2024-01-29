#include "kernel_profiler.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

namespace fastertransformer {

bool record_kernel_time() {
    static bool record = [] {
        bool         should_record = false;
        static char* record_kernel = std::getenv("RECORD_KERNEL");
        if (record_kernel != nullptr) {
            static std::string level = std::string(record_kernel);
            should_record            = level == "ON";
        }
        return should_record;
    }();
    return record;
}

void KernelProfiler::start() {
    if (record_kernel_time_) {
        cudaDeviceSynchronize();
        cudaEventSynchronize(start_);
        cudaEventRecord(start_, stream_);
    }
}

void KernelProfiler::stop() {
    if (record_kernel_time_) {
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float kernel_time = 0;
        cudaEventElapsedTime(&kernel_time, start_, stop_);
        FT_LOG_INFO("kernel: " + std::string(kernel_name_.c_str()) + " time: " + std::to_string(kernel_time));
    }
}

KernelProfiler::KernelProfiler(cudaStream_t stream, std::string kernel_name):
    stream_(stream), kernel_name_(kernel_name), record_kernel_time_(record_kernel_time()) {

    if (record_kernel_time_) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
}

KernelProfiler::~KernelProfiler() {
    if (record_kernel_time_) {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
}
}  // namespace fastertransformer