#include "maga_transformer/cpp/embedding_engine/EmbeddingQuery.h"

using namespace std;

namespace rtp_llm {

EmbeddingInput::EmbeddingInput(const std::shared_ptr<ft::Buffer>& token_ids,
                               const std::shared_ptr<ft::Buffer>& token_type_ids,
                               const std::shared_ptr<ft::Buffer>& input_lengths,
                               const int64_t                      total_length,
                               int64_t request_id) : token_ids(token_ids), token_type_ids(token_type_ids), total_length(total_length), request_id(request_id), input_lengths(input_lengths) {
}

void EmbeddingOutput::setOutput(ft::BufferPtr& model_outputs, int64_t length) {
    output = std::move(model_outputs);
    input_length = length;
}

void EmbeddingOutput::setError(const std::string& error) {
    this->error_info.has_error = true;
    this->error_info.error_message = error;
}

}  // namespace rtp_llm
