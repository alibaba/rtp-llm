#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

namespace rtp_llm {

// Build an xgrammar::TokenizerInfo serialized JSON from a HF tokenizer's
// vocab dict + backend tokenizer.to_str() + the union of stop token ids.
// C++ does the id→token reverse fill so Python doesn't have to allocate a
// per-id list. Wraps xgrammar's public API (DetectMetadataFromHF +
// FromVocabAndMetadata + SerializeJSON).
std::string buildXGrammarTokenizerInfoJson(
    const std::unordered_map<std::string, int32_t>& vocab,
    const std::string&                              backend_tokenizer_str,
    int64_t                                         vocab_size,
    const std::vector<int32_t>&                     stop_token_ids);

void registerXGrammarBootstrap(pybind11::module& m);

}  // namespace rtp_llm
