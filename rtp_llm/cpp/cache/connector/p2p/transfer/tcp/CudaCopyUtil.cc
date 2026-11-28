#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <torch/torch.h>
#include <cstring>
#include <exception>
#include <limits>

namespace rtp_llm {
namespace transfer {
namespace tcp {

static torch::Tensor wrapRawPtr(void* ptr, size_t size, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    return torch::from_blob(ptr, {static_cast<int64_t>(size)}, options);
}

static TransferErrorCode copyBlocks(const std::vector<CopyTask>& tasks,
                                    bool to_host,
                                    int64_t deadline_ms,
                                    const std::function<bool()>& is_cancelled = {}) try {
    MultiCopyParams params;
    params.multi_src.reserve(tasks.size());
    params.multi_dst.reserve(tasks.size());

    for (const auto& task : tasks) {
        if (!task.src_ptr || !task.dst_ptr) {
            RTP_LLM_LOG_WARNING("TCP copy source or destination is null");
            return TransferErrorCode::COPY_FAILED;
        }
        if (!task.is_cuda) {
            continue;
        }
        if (task.device_index < -1 || task.device_index > std::numeric_limits<c10::DeviceIndex>::max()) {
            RTP_LLM_LOG_WARNING("TCP copy device index is invalid: %d", task.device_index);
            return TransferErrorCode::COPY_FAILED;
        }
        const torch::Device gpu(torch::kCUDA, static_cast<c10::DeviceIndex>(task.device_index));
        const torch::Device cpu(torch::kCPU);
        auto device_view = wrapRawPtr(to_host ? task.src_ptr : task.dst_ptr, task.size, gpu);
        auto host_view = wrapRawPtr(to_host ? task.dst_ptr : task.src_ptr, task.size, cpu);
        params.multi_src.push_back(to_host ? device_view : host_view);
        params.multi_dst.push_back(to_host ? host_view : device_view);
    }

    if (is_cancelled && is_cancelled()) {
        return TransferErrorCode::CANCELLED;
    }
    if (currentTimeMs() >= deadline_ms) {
        return TransferErrorCode::TIMEOUT;
    }
#if USE_PPU
    // The RPC owns the host payload only until completion. PPU can reject
    // DMA on the pooled side stream even with pinned staging. Use ATen's
    // blocking copy on the KV device's current stream in both directions.
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        params.multi_dst[i].copy_(params.multi_src[i], /*non_blocking=*/false);
    }
#else
    if (!params.multi_src.empty()) {
        execNoBlockCopy(params);
    }
#endif
    for (const auto& task : tasks) {
        if (!task.is_cuda) {
            std::memcpy(task.dst_ptr, task.src_ptr, task.size);
        }
    }
    return TransferErrorCode::OK;
} catch (const std::exception& e) {
    // Complete the transport request through its error callback. Letting a
    // copy exception escape the Decode worker leaves its peer waiting.
    RTP_LLM_LOG_WARNING("TCP %s copy failed: %s", to_host ? "D2H" : "H2D", e.what());
    return TransferErrorCode::COPY_FAILED;
}

bool CudaCopyUtil::batchCopyToHost(std::vector<CopyTask>& tasks) {
    return copyBlocks(tasks, /*to_host=*/true, std::numeric_limits<int64_t>::max()) == TransferErrorCode::OK;
}

bool CudaCopyUtil::batchCopyToDevice(std::vector<CopyTask>& tasks) {
    return batchCopyToDevice(tasks, std::numeric_limits<int64_t>::max()) == TransferErrorCode::OK;
}

TransferErrorCode CudaCopyUtil::batchCopyToDevice(std::vector<CopyTask>& tasks,
                                                  int64_t deadline_ms,
                                                  const std::function<bool()>& is_cancelled) {
    return copyBlocks(tasks, /*to_host=*/false, deadline_ms, is_cancelled);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
