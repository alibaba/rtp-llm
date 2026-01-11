#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class LogitsProcessorFactory {
public:
    static void init(const std::string& ckpt_path, const std::string& tree_decode_config);

    static std::vector<BaseLogitsProcessorPtr> createLogitsProcessors(rtp_llm::DeviceBase*           device,
                                                                      std::shared_ptr<GenerateInput> generate_input,
                                                                      int32_t                        init_batch_size,
                                                                      int32_t                        max_batch_size,
                                                                      int64_t                        eos_token_id);
};

}  // namespace rtp_llm