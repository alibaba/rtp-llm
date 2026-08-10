#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

namespace rtp_llm {

std::string buildXGrammarTokenizerInfoJson(const std::unordered_map<std::string, int32_t>& vocab,
                                           const std::string&                              backend_tokenizer_str,
                                           int64_t                                         vocab_size,
                                           const std::vector<int32_t>&                     stop_token_ids);

std::string buildXGrammarTokenizerInfoJsonFromVocab(const std::unordered_map<std::string, int32_t>& vocab,
                                                    int32_t                                         vocab_type,
                                                    int64_t                                         vocab_size,
                                                    const std::vector<int32_t>&                     stop_token_ids,
                                                    bool                                            add_prefix_space);

void registerXGrammarBootstrap(pybind11::module& m);

}  // namespace rtp_llm
