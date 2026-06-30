#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfoCooker.h"

#include <algorithm>
#include <stdexcept>

#include <xgrammar/tokenizer_info.h>

namespace rtp_llm::xgrammar_impl {

std::string cookTokenizerInfoOpaque(const std::unordered_map<std::string, int32_t>& vocab,
                                    const std::string&                              backend_tokenizer_str,
                                    const std::vector<int32_t>&                     stop_token_ids,
                                    int64_t                                         model_vocab_size) {
    static constexpr int64_t kMaxVocabSize = 1'000'000;
    if (vocab.empty()) {
        throw std::invalid_argument("cookTokenizerInfoOpaque: vocab is empty");
    }
    int64_t max_id = -1;
    for (const auto& [_, tid] : vocab) {
        if (tid < 0) {
            throw std::invalid_argument("cookTokenizerInfoOpaque: negative token id " + std::to_string(tid));
        }
        max_id = std::max(max_id, static_cast<int64_t>(tid));
    }
    // Widen to the (possibly padded) model vocab so the grammar bitmask spans the full logits range,
    // matching dsv4's max(model_config.vocab_size, max(vocab)+1).
    const int64_t vocab_size = std::max(model_vocab_size, max_id + 1);
    if (vocab_size > kMaxVocabSize) {
        throw std::invalid_argument("vocab_size must be in (0, " + std::to_string(kMaxVocabSize) + "], got "
                                    + std::to_string(vocab_size));
    }
    std::vector<std::string> encoded_vocab(static_cast<size_t>(vocab_size));
    for (const auto& [tok, tid] : vocab) {
        encoded_vocab[static_cast<size_t>(tid)] = tok;
    }

    std::string meta = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);

    std::string stops = "[";
    for (size_t i = 0; i < stop_token_ids.size(); ++i) {
        if (i)
            stops += ",";
        stops += std::to_string(stop_token_ids[i]);
    }
    stops += "]";

    const auto close_pos = meta.rfind('}');
    if (close_pos == std::string::npos) {
        throw std::runtime_error(
            "cookTokenizerInfoOpaque: DetectMetadataFromHF returned a non-object metadata string: '" + meta + "'");
    }
    const auto  first_content = meta.find_first_not_of(" \t\r\n", meta.find('{') + 1);
    const bool  has_members   = first_content != std::string::npos && first_content < close_pos;
    std::string injected      = std::string(has_members ? "," : "") + "\"vocab_size\":" + std::to_string(vocab_size)
                           + ",\"stop_token_ids\":" + stops;
    meta.insert(close_pos, injected);
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, meta).SerializeJSON();
}

}  // namespace rtp_llm::xgrammar_impl
