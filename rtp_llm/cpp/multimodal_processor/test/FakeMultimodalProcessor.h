#pragma once

#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

class FakeMultimodalProcessor: public MultimodalProcessor {
public:
    FakeMultimodalProcessor(py::object                               mm_process_engine,
                            const std::vector<std::vector<int64_t>>& sep_token_ids,
                            bool                                     include_sep_tokens,
                            int64_t                                  max_seq_len,
                            int64_t                                  feature_width = 1):
        MultimodalProcessor(mm_process_engine, MMModelConfig{true, sep_token_ids, include_sep_tokens}, max_seq_len),
        feature_width_(feature_width) {}

    static FakeMultimodalProcessor createFakeMultimodalProcessor(const std::vector<std::vector<int64_t>>& sep_token_ids,
                                                                 bool    include_sep_tokens,
                                                                 int64_t max_seq_len,
                                                                 int64_t feature_width = 1) {
        return FakeMultimodalProcessor(py::none(), sep_token_ids, include_sep_tokens, max_seq_len, feature_width);
    }

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string ip_port = "") override {
        MultimodalOutput output;
        for (const auto& input : mm_inputs) {
            int embed_len = std::stoi(input.url);
            output.mm_features.push_back(torch::full({embed_len, feature_width_}, embed_len));
        }
        return output;
    }

    int64_t feature_width_;
};

}  // namespace rtp_llm
