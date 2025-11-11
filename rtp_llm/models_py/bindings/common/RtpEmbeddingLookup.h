#pragma once
namespace rtp_llm {

void embedding(at::Tensor& output, at::Tensor& input, at::Tensor& weight);
void embeddingBert(at::Tensor& output,
                   at::Tensor& input,
                   at::Tensor& weight,
                   at::Tensor  combo_position_ids,
                   at::Tensor  position_encoding,
                   at::Tensor  combo_tokens_type_ids,
                   at::Tensor  token_type_embedding,
                   float       input_embedding_scalar);
}  // namespace rtp_llm
