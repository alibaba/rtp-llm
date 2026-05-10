#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessorCSR.h"

namespace rtp_llm {

// ---------------------------------------------------------------------------
// 构造函数
// ---------------------------------------------------------------------------

TreeLogitsProcessorCSR::TreeLogitsProcessorCSR(std::vector<StreamTreeInfo> tree_infos):
    tree_infos_(std::move(tree_infos)) {}

// process / updateStatus / updateMultiSeqStatus / fromGenerateInput
// 均在 TreeLogitsProcessorCSRGpu.cc 中实现（GPU 版本）。

}  // namespace rtp_llm
