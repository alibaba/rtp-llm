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

    float feature_value = 0.0f;

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<MultimodalInput>& mm_inputs,
                                                      const std::string&,
                                                      int64_t,
                                                      grpc::ServerContext*,
                                                      std::optional<bool>) override {
        MultimodalOutput output;
        for (const auto& input : mm_inputs) {
            int embed_len = std::stoi(input.url);
            output.mm_features.push_back(torch::full({embed_len, 1}, feature_value));
        }
        return output;
    }
};

}  // namespace rtp_llm
