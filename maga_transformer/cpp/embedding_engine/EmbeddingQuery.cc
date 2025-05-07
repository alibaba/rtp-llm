#include "maga_transformer/cpp/embedding_engine/EmbeddingQuery.h"

using namespace std;

namespace rtp_llm {

EmbeddingInput::EmbeddingInput(const std::shared_ptr<rtp_llm::Buffer>& token_ids,
                               const std::shared_ptr<rtp_llm::Buffer>& token_type_ids,
                               const std::shared_ptr<rtp_llm::Buffer>& input_lengths,
                               const int64_t                      total_length,
                               int64_t request_id,
                               std::optional<MultimodalFeature> multimodal_features)
    : token_ids(token_ids)
    , token_type_ids(token_type_ids)
    , input_lengths(input_lengths)
    , total_length(total_length)
    , request_id(request_id)
    , multimodal_features(multimodal_features)
{
    checkVaild();
}

void EmbeddingInput::checkVaild() {
    if (token_ids->shape().size() != 1 || token_type_ids->shape().size() != 1)  {
        throw std::runtime_error("token id shape size or token type id shape size != 1");
    }
    if (token_ids->shape()[0] == 0) {
        throw std::runtime_error("input length can't be 0");
    }
    if (token_ids->shape()[0] != token_type_ids->shape()[0]) {
        throw std::runtime_error("token length should equal to token type length");
    }
    if (total_length != int64_t(token_ids->shape()[0])) {
        throw std::runtime_error("sum of token length don't equal to total_length");
    }
}

}  // namespace rtp_llm
