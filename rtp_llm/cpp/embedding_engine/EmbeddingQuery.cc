#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"

#include <numeric>

using namespace std;

namespace rtp_llm {

EmbeddingInput::EmbeddingInput(const std::vector<int32_t>&             token_ids_param,
                               const std::vector<int32_t>&             token_type_ids_param,
                               const std::vector<int32_t>&             input_lengths_param,
                               int64_t                                 request_id_param,
                               const std::optional<MultimodalFeature>& multimodal_features_param,
                               std::optional<torch::Tensor>            input_embeddings_param) {

    token_ids =
        torch::from_blob(const_cast<int32_t*>(token_ids_param.data()), {(int64_t)token_ids_param.size()}, torch::kInt32)
            .clone();

    token_type_ids = torch::from_blob(const_cast<int32_t*>(token_type_ids_param.data()),
                                      {(int64_t)token_type_ids_param.size()},
                                      torch::kInt32)
                         .clone();

    total_length = std::accumulate(input_lengths_param.begin(), input_lengths_param.end(), 0);

    input_lengths = torch::from_blob(const_cast<int32_t*>(input_lengths_param.data()),
                                     {(int64_t)input_lengths_param.size()},
                                     torch::kInt32)
                        .clone();

    request_id          = request_id_param;
    multimodal_features = multimodal_features_param;
    input_embeddings    = input_embeddings_param;
    checkVaild();
}

EmbeddingInput::EmbeddingInput(const torch::Tensor&                    token_ids_,
                               const torch::Tensor&                    token_type_ids_,
                               const torch::Tensor&                    input_lengths_,
                               int64_t                                 request_id_,
                               const std::optional<MultimodalFeature>& multimodal_features_,
                               std::optional<torch::Tensor>            input_embeddings_) {
    token_ids      = token_ids_;
    token_type_ids = token_type_ids_;

    total_length = std::accumulate(
        input_lengths_.data_ptr<int32_t>(), input_lengths_.data_ptr<int32_t>() + input_lengths_.size(0), 0);

    input_lengths       = input_lengths_;
    request_id          = request_id_;
    multimodal_features = multimodal_features_;
    input_embeddings    = input_embeddings_;
    checkVaild();
}

void EmbeddingInput::checkVaild() {
    if (token_ids.dim() != 1 || token_type_ids.dim() != 1) {
        throw std::runtime_error("token id shape size or token type id shape size != 1");
    }
    if (token_ids.size(0) == 0) {
        throw std::runtime_error("input length can't be 0");
    }
    if (token_ids.size(0) != token_type_ids.size(0)) {
        throw std::runtime_error("token length should equal to token type length");
    }
    if (total_length != token_ids.size(0)) {
        throw std::runtime_error("sum of token length don't equal to total_length");
    }
    if (input_embeddings.has_value() && input_embeddings.value().dim() != 2) {
        throw std::runtime_error("input_embeddings shape size != 2");
    }
    if (input_embeddings.has_value() && total_length != input_embeddings.value().size(0)) {
        throw std::runtime_error("sum of token length don't equal to total_length");
    }
}

}  // namespace rtp_llm
