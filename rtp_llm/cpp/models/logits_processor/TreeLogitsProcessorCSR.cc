#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessorCSR.h"

namespace rtp_llm {

// ---------------------------------------------------------------------------
// 构造函数
// ---------------------------------------------------------------------------

TreeLogitsProcessorCSR::TreeLogitsProcessorCSR(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device) {}

TreeLogitsProcessorCSR::TreeLogitsProcessorCSR(rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos):
    BaseLogitsProcessor(device), tree_infos_(std::move(tree_infos)) {}

// process / updateStatus / updateMultiSeqStatus / fromGenerateInput
// 均在 TreeLogitsProcessorCSRGpu.cc 中实现（GPU 版本）。

}  // namespace rtp_llm
