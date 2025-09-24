#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include <string>
#include "rtp_llm/cpp/config/StaticConfig.h"
namespace rtp_llm {
namespace tap = torch::autograd::profiler;

size_t CudaProfiler::count = 0;

CudaProfiler::CudaProfiler(const std::string& prefix, std::string user_torch_cuda_profiler_dir): prefix_(prefix) {
    dest_dir_ = (user_torch_cuda_profiler_dir != "") ? user_torch_cuda_profiler_dir : ".";
    tap::prepareProfiler(config_, activities_);
}

CudaProfiler::~CudaProfiler() {
    if (!stoped_) {
        stoped_ = true;
        stop();
    }
}

void CudaProfiler::start() {
    count += 1;
    stoped_ = false;
    tap::enableProfiler(config_, activities_);
}

void CudaProfiler::stop() {
    std::unique_ptr<tap::ProfilerResult> res       = tap::disableProfiler();
    std::string                          file_name = dest_dir_ + "/" + prefix_ + std::to_string(count) + ".json";
    res->save(file_name);
    stoped_ = true;
}

}  // namespace rtp_llm
