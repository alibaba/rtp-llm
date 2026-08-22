#pragma once
namespace rtp_llm {

// position_ids and token_type_ids are retained for binding compatibility but
// ignored because the generic op has no corresponding embedding tables.
void embedding(at::Tensor&               output,
               at::Tensor&               input,
               at::Tensor&               weight,
               std::optional<at::Tensor> position_ids,
               std::optional<at::Tensor> token_type_ids,
               std::optional<at::Tensor> text_tokens_mask);
// Callers must validate IDs before dispatch: position IDs are in
// [0, position_encoding.size(0)), token-type IDs in
// [0, token_type_embedding.size(0)), and unmasked word IDs in
// [0, weight.size(0)). text_tokens_mask == 0 only exempts the word ID.
void embeddingBert(at::Tensor&               output,
                   at::Tensor&               input,
                   at::Tensor&               weight,
                   at::Tensor                combo_position_ids,
                   at::Tensor                position_encoding,
                   at::Tensor                combo_tokens_type_ids,
                   at::Tensor                token_type_embedding,
                   float                     input_embedding_scalar,
                   std::optional<at::Tensor> text_tokens_mask);
}  // namespace rtp_llm
