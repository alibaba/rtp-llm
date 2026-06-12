#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/SpecLogitsProcessorTypes.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

// MTP verify bitmask output produced by MtpExecutor before target-model sampling.
struct MtpSpecLogitsVerifyResult {
    torch::Tensor                      spec_vocab_mask_gpu;
    torch::Tensor                      spec_cap_gpu;
    bool                               has_active_processor = false;
    std::vector<SpecLogitsProcessorId> applied_processors;

    torch::Tensor spec_vocab_mask_cpu_owner;
    torch::Tensor spec_cap_cpu_owner;
};

}  // namespace rtp_llm
