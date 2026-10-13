#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm::xgrammar_impl {

std::string serializeTokenizerInfo(const std::vector<std::string>& encoded_vocab,
                                   const std::string&              tokenizer_metadata_json);

}  // namespace rtp_llm::xgrammar_impl
