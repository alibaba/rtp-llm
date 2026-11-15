#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <torch/torch.h>
#include <cstring>
#include <limits>

namespace rtp_llm {
namespace transfer {
namespace tcp {

static torch::Tensor wrapRawPtr(void* ptr, size_t size, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    return torch::from_blob(ptr, {static_cast<int64_t>(size)}, options);
}

static bool copyBlocks(const std::vector<CopyTask>& tasks, bool to_host) {
    MultiCopyParams params;
    params.multi_src.reserve(tasks.size());
    params.multi_dst.reserve(tasks.size());

    for (const auto& task : tasks) {
        if (!task.src_ptr || !task.dst_ptr) {
            RTP_LLM_LOG_WARNING("TCP copy source or destination is null");
            return false;
        }
        if (!task.is_cuda) {
            continue;
        }
        if (task.device_index < -1 || task.device_index > std::numeric_limits<c10::DeviceIndex>::max()) {
            RTP_LLM_LOG_WARNING("TCP copy device index is invalid: %d", task.device_index);
            return false;
        }
        const torch::Device gpu(torch::kCUDA, static_cast<c10::DeviceIndex>(task.device_index));
        const torch::Device cpu(torch::kCPU);
        params.multi_src.push_back(wrapRawPtr(task.src_ptr, task.size, to_host ? gpu : cpu));
        params.multi_dst.push_back(wrapRawPtr(task.dst_ptr, task.size, to_host ? cpu : gpu));
    }

    if (!params.multi_src.empty()) {
        execNoBlockCopy(params);
    }
    for (const auto& task : tasks) {
        if (!task.is_cuda) {
            std::memcpy(task.dst_ptr, task.src_ptr, task.size);
        }
    }
    return true;
}

bool CudaCopyUtil::batchCopyToHost(std::vector<CopyTask>& tasks) {
    return copyBlocks(tasks, /*to_host=*/true);
}

bool CudaCopyUtil::batchCopyToDevice(std::vector<CopyTask>& tasks) {
    return copyBlocks(tasks, /*to_host=*/false);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
