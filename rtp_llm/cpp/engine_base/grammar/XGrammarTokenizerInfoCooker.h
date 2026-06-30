#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm::xgrammar_impl {

// Build the xgrammar-private TokenizerInfo blob from HF materials; throws on empty/invalid vocab.
// model_vocab_size lets a padded model vocab (model_config.vocab_size) widen the grammar vocab so
// the bitmask covers the full logits range, matching the dsv4 bootstrap behavior.
std::string cookTokenizerInfoOpaque(const std::unordered_map<std::string, int32_t>& vocab,
                                    const std::string&                              backend_tokenizer_str,
                                    const std::vector<int32_t>&                     stop_token_ids,
                                    int64_t                                         model_vocab_size = 0);

}  // namespace rtp_llm::xgrammar_impl
