#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

std::shared_ptr<DeviceHostCopyStreams> acquireDeviceHostCopyStreams(int device_index) {
    return std::make_shared<DeviceHostCopyStreams>(device_index, 0, 0);
}

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        params.multi_dst[i].copy_(params.multi_src[i]);
    }
}

void execNoBlockCopy(const MultiCopyParams& params, const DeviceHostCopyExecutionContext&) {
    execNoBlockCopy(params);
}

BatchedMemoryCopyStatus execBatchedMemoryCopy(const BatchedMemoryCopyParams& params,
                                              const DeviceHostCopyExecutionContext&) {
    return params.tiles.empty() ? BatchedMemoryCopyStatus::SUCCESS : BatchedMemoryCopyStatus::NOT_SUPPORTED;
}

bool execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch*) {
    return params.tiles.empty();
}

void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch&) {}

void warmupNoBlockCopy() {}

}  // namespace rtp_llm
