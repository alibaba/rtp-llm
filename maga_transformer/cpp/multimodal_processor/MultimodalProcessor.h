#pragma once

#include <vector>
#include <torch/python.h>
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/utils/ErrorCode.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/utils/PyUtils.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace ft = fastertransformer;
namespace py = pybind11;

namespace rtp_llm {

struct ExpandedOutput {
    ft::BufferPtr expanded_ids;
    ft::BufferPtr text_tokens_mask;
    ft::BufferPtr locs;
    ExpandedOutput(ft::BufferPtr expanded_ids = nullptr, ft::BufferPtr text_tokens_mask = nullptr, ft::BufferPtr locs = nullptr):
        expanded_ids(expanded_ids), text_tokens_mask(text_tokens_mask), locs(locs) {}
};

class MultimodalProcessor {
public:
        MultimodalProcessor(py::object mm_process_engine, ft::GptInitParameter params): 
        mm_process_engine_(mm_process_engine), gpt_init_parameter_(params), sep_token_ids_(params.mm_sep_tokens_), include_sep_tokens_(params.include_sep_tokens_), max_seq_len_(params.max_seq_len_) {}

protected:
    py::object mm_process_engine_;
    ft::GptInitParameter gpt_init_parameter_;

private:
    std::vector<std::vector<int64_t>> sep_token_ids_;
    bool include_sep_tokens_;
    int64_t max_seq_len_;

    ErrorInfo getStrHash(int32_t* token_ids, std::string& url, int mm_emb_len);

    virtual ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs) = 0;

    ErrorResult<ExpandedOutput> expandTokenIds(const std::vector<torch::Tensor>& mm_embedding, ft::BufferPtr token_ids, const std::vector<rtp_llm::MultimodalInput> mm_inputs);

    ErrorResult<std::vector<std::pair<int32_t, int32_t>>> getMultimodalTags(ft::BufferPtr token_ids);

    ErrorInfo checkExpandLength(const ExpandedOutput& expand_output);

public:
    ErrorInfo updateMultimodalFeatures(std::shared_ptr<rtp_llm::GenerateInput>& input);

    ErrorResult<MultimodalFeature> getMultimodalFeatures(const ft::BufferPtr& input_ids, const std::vector<MultimodalInput> &mm_inputs);
};

}
