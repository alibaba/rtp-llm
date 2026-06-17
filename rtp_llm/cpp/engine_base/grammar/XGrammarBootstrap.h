#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

struct SpecialTokens;

// Collect single-token EOS / stop ids from a model's SpecialTokens. Multi-token
// stop sequences are dropped — xgrammar's stop_token_ids list is per-token.
// Returned ids are deduplicated and sorted ascending.
std::vector<int32_t> collectStopTokenIds(const SpecialTokens& special_tokens);

// Build the xgrammar TokenizerInfo JSON from (vocab map, HF backend tokenizer
// JSON string, stop ids). Pure C++; throws on empty/invalid vocab.
std::string buildXGrammarTokenizerInfoJson(const std::unordered_map<std::string, int32_t>& vocab,
                                           const std::string&                              backend_tokenizer_str,
                                           const std::vector<int32_t>&                     stop_token_ids);

}  // namespace rtp_llm
