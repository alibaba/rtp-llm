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
                                                      std::string ip_port = "") override {
        MultimodalOutput output;
        for (const auto& input : mm_inputs) {
            int embed_len = std::stoi(input.url);
            output.mm_features.push_back(torch::zeros({embed_len, 1}));
        }
        return output;
    }
};

}  // namespace rtp_llm
