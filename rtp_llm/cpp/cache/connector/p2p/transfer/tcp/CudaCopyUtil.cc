#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <torch/torch.h>

namespace rtp_llm {
namespace transfer {
namespace tcp {

static torch::Tensor wrapRawPtr(void* ptr, size_t size, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    return torch::from_blob(ptr, {static_cast<int64_t>(size)}, options);
}

bool CudaCopyUtil::batchCopyToHost(std::vector<CopyTask>& tasks) {
    if (tasks.empty()) {
        return true;
    }

    MultiCopyParams params;
    params.multi_src.reserve(tasks.size());
    params.multi_dst.reserve(tasks.size());

    for (auto& task : tasks) {
        if (!task.dst_ptr) {
            RTP_LLM_LOG_WARNING("dst_ptr is nullptr, caller must pre-allocate dst_ptr");
            return false;
        }
        params.multi_src.push_back(wrapRawPtr(task.src_ptr, task.size, torch::kCUDA));
        params.multi_dst.push_back(wrapRawPtr(task.dst_ptr, task.size, torch::kCPU));
    }

    execNoBlockCopy(params);
    return true;
}

bool CudaCopyUtil::batchCopyToDevice(std::vector<CopyTask>& tasks) {
    if (tasks.empty()) {
        return true;
    }

    MultiCopyParams params;
    params.multi_src.reserve(tasks.size());
    params.multi_dst.reserve(tasks.size());

    for (auto& task : tasks) {
        if (!task.dst_ptr) {
            RTP_LLM_LOG_WARNING("dst_ptr is nullptr, caller must pre-allocate dst_ptr");
            return false;
        }
        params.multi_src.push_back(wrapRawPtr(task.src_ptr, task.size, torch::kCPU));
        params.multi_dst.push_back(wrapRawPtr(task.dst_ptr, task.size, torch::kCUDA));
    }

    execNoBlockCopy(params);
    return true;
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
