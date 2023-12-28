#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>

namespace fastertransformer {
class KernelProfiler {
public:
    KernelProfiler(cudaStream_t stream, std::string kernel_name);
    ~KernelProfiler();
    void start();
    void stop();

protected:
    cudaStream_t stream_;
    bool         record_kernel_time_;
    std::string  kernel_name_;
    cudaEvent_t  start_, stop_;
};

}  // namespace fastertransformer