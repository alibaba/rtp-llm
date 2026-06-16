#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

// Build an xgrammar::TokenizerInfo serialized JSON from a HF tokenizer's
// vocab dict + backend tokenizer.to_str() + the union of stop token ids.
std::string buildXGrammarTokenizerInfoJson(const std::unordered_map<std::string, int32_t>& vocab,
                                           const std::string&                              backend_tokenizer_str,
                                           const std::vector<int32_t>&                     stop_token_ids);

// Called from RtpLLMOp::initModel only: read tokenizer + special_tokens from
// the loaded Python model and fill grammar_config.tokenizer_info_json.
void bootstrapGrammarConfigFromModel(pybind11::object model, GrammarConfig& grammar_config);

}  // namespace rtp_llm
