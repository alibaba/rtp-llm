#pragma once

#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

class FakeMultimodalProcessor: public MultimodalProcessor {
public:
    FakeMultimodalProcessor(py::object                               mm_process_engine,
                            const std::vector<std::vector<int64_t>>& sep_token_ids,
                            bool                                     include_sep_tokens,
                            int64_t                                  max_seq_len):
        MultimodalProcessor(mm_process_engine, MMModelConfig{true, sep_token_ids, include_sep_tokens}, max_seq_len) {}

    static FakeMultimodalProcessor createFakeMultimodalProcessor(const std::vector<std::vector<int64_t>>& sep_token_ids,
                                                                 bool    include_sep_tokens,
                                                                 int64_t max_seq_len) {
        return FakeMultimodalProcessor(py::none(), sep_token_ids, include_sep_tokens, max_seq_len);
    }

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "",
                                                      const std::string& rendered_prompt = "") override {
        MultimodalOutput output;
        if (!rendered_prompt.empty() && mm_inputs.size() == 1 && mm_inputs[0].mm_type == 2) {
            output.mm_features = {torch::zeros({1, 1}), torch::zeros({1, 1})};
            MultimodalExpansionMetadata frame0;
            frame0.is_video              = true;
            frame0.fps                   = 24.0;
            frame0.frame_indices         = {0};
            frame0.frame_number          = 0;
            frame0.frame_count           = 2;
            frame0.soft_tokens_per_frame = 1;
            auto frame1                  = frame0;
            frame1.frame_indices         = {12};
            frame1.frame_number          = 1;
            output.mm_expansion_metadata = std::vector<MultimodalExpansionMetadata>{frame0, frame1};
            output.expanded_token_ids    = torch::tensor({0, 10, 99, 11, 10, 98, 11, 3}, torch::kInt32);
            return output;
        }
        for (const auto& input : mm_inputs) {
            int embed_len = std::stoi(input.url);
            output.mm_features.push_back(torch::zeros({embed_len, 1}));
        }
        return output;
    }
};

}  // namespace rtp_llm
