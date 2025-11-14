#pragma once

#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace py = pybind11;

namespace rtp_llm {

struct ExpandedOutput {
    rtp_llm::BufferPtr expanded_ids;
    rtp_llm::BufferPtr token_type_ids;
    rtp_llm::BufferPtr text_tokens_mask;
    rtp_llm::BufferPtr locs;
    ExpandedOutput(rtp_llm::BufferPtr expanded_ids     = nullptr,
                   rtp_llm::BufferPtr token_type_ids   = nullptr,
                   rtp_llm::BufferPtr text_tokens_mask = nullptr,
                   rtp_llm::BufferPtr locs             = nullptr):
        expanded_ids(expanded_ids), token_type_ids(token_type_ids), text_tokens_mask(text_tokens_mask), locs(locs) {}
};

class MultimodalProcessor {
public:
    MultimodalProcessor(py::object mm_process_engine,
                       const MMModelConfig& mm_model_config,
                       int64_t max_seq_len):
        mm_process_engine_(mm_process_engine),
        sep_token_ids_(mm_model_config.mm_sep_tokens),
        include_sep_tokens_(mm_model_config.include_sep_tokens),
        max_seq_len_(max_seq_len) {}

protected:
    py::object                mm_process_engine_;

private:
    std::vector<std::vector<int64_t>> sep_token_ids_;
    bool                              include_sep_tokens_;
    int64_t                           max_seq_len_;

    ErrorInfo getStrHash(int32_t* token_ids, std::string& url, int mm_emb_len);

    virtual ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                              std::string ip_port = "") = 0;

    ErrorResult<ExpandedOutput> expandTokenIds(const std::vector<torch::Tensor>&           mm_embedding,
                                               rtp_llm::BufferPtr                          token_ids,
                                               const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                               rtp_llm::BufferPtr                          token_type_ids = nullptr);

    ErrorResult<std::vector<std::pair<int32_t, int32_t>>> getMultimodalTags(rtp_llm::BufferPtr token_ids);

    ErrorInfo checkExpandLength(const ExpandedOutput& expand_output);

public:
    ErrorInfo updateMultimodalFeatures(std::shared_ptr<rtp_llm::GenerateInput>& input);

    ErrorInfo updateMultimodalFeatures(std::shared_ptr<rtp_llm::EmbeddingInput>&    input,
                                       const std::vector<rtp_llm::MultimodalInput>& mm_inputs);

    ErrorResult<MultimodalFeature> getMultimodalFeatures(const rtp_llm::BufferPtr&                    input_ids,
                                                         const std::vector<rtp_llm::MultimodalInput>& mm_inputs);
};

}  // namespace rtp_llm
