#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <torch/torch.h>
#include <cstring>
#include <limits>
#include <map>

namespace rtp_llm {
namespace transfer {
namespace tcp {

static torch::Tensor wrapRawPtr(void* ptr, size_t size, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    return torch::from_blob(ptr, {static_cast<int64_t>(size)}, options);
}

namespace {

int32_t deviceType(const CopyTask& task) {
    return task.device_type >= 0 ? task.device_type :
                                  static_cast<int32_t>(task.is_cuda ? torch::kCUDA : torch::kCPU);
}

bool copyTasks(const std::vector<CopyTask>& tasks, bool to_host) {
    // Validate the whole batch before touching any destination. In particular,
    // PrivateUse1/NPU memory must not fall through to a CPU memcpy.
    for (const auto& task : tasks) {
        if (task.size == 0) {
            continue;
        }
        const auto type = deviceType(task);
        if (!task.src_ptr || !task.dst_ptr || task.size > static_cast<size_t>(std::numeric_limits<int64_t>::max())
            || task.device_type < -1 || task.device_index < -1
            || task.device_index > std::numeric_limits<c10::DeviceIndex>::max()
            || (type != static_cast<int32_t>(torch::kCPU) && type != static_cast<int32_t>(torch::kCUDA))) {
            RTP_LLM_LOG_WARNING("invalid or unsupported TCP copy buffer: type=%d device=%d size=%zu",
                                type, task.device_index, task.size);
            return false;
        }
    }

    // execNoBlockCopy selects one device stream per invocation. Keep different
    // CUDA device indices in separate batches instead of using the current GPU.
    std::map<int32_t, MultiCopyParams> device_batches;
    try {
        for (const auto& task : tasks) {
            if (task.size == 0 || deviceType(task) == static_cast<int32_t>(torch::kCPU)) {
                continue;
            }
            const torch::Device device(torch::kCUDA, static_cast<c10::DeviceIndex>(task.device_index));
            auto& params = device_batches[task.device_index];
            params.multi_src.push_back(wrapRawPtr(task.src_ptr, task.size, to_host ? device : torch::Device(torch::kCPU)));
            params.multi_dst.push_back(wrapRawPtr(task.dst_ptr, task.size, to_host ? torch::Device(torch::kCPU) : device));
        }
        for (const auto& task : tasks) {
            if (task.size > 0 && deviceType(task) == static_cast<int32_t>(torch::kCPU)) {
                std::memmove(task.dst_ptr, task.src_ptr, task.size);
            }
        }
        for (const auto& [index, params] : device_batches) {
            (void)index;
            execNoBlockCopy(params);
        }
    } catch (const c10::Error& error) {
        RTP_LLM_LOG_WARNING("TCP buffer copy failed: %s", error.what());
        return false;
    }
    return true;
}

}  // namespace

bool CudaCopyUtil::batchCopyToHost(std::vector<CopyTask>& tasks) {
    return copyTasks(tasks, true);
}

bool CudaCopyUtil::batchCopyToDevice(std::vector<CopyTask>& tasks) {
    return copyTasks(tasks, false);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
