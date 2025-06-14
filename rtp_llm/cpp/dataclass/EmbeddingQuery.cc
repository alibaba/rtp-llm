#include "rtp_llm/cpp/dataclass/EmbeddingQuery.h"

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

EmbeddingInput::EmbeddingInput(
    const torch::Tensor& token_ids_,
    const torch::Tensor& token_type_ids_,
    const torch::Tensor& input_lengths_,
    int request_id_,
    std::optional<MultimodalFeature> multimodal_features_)
{
    token_ids =
        std::make_shared<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_CPU,
            rtp_llm::DataType::TYPE_INT32,
            std::vector<size_t>{(size_t)token_ids_.size(0)},
            token_ids_.data_ptr());

    token_type_ids =
        std::make_shared<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_CPU,
            rtp_llm::DataType::TYPE_INT32,
            std::vector<size_t>{(size_t)token_type_ids_.size(0)},
            token_type_ids_.data_ptr());

    total_length = std::accumulate((int32_t*)input_lengths_.data_ptr(), (int32_t*)input_lengths_.data_ptr() + input_lengths_.size(0), 0);

    input_lengths =
        std::make_shared<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_CPU,
            rtp_llm::DataType::TYPE_INT32,
            std::vector<size_t>{(size_t)input_lengths_.size(0)},
            input_lengths_.data_ptr());
    request_id = request_id_;
    multimodal_features = multimodal_features_;
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
