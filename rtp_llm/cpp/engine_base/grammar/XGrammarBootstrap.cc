#include "rtp_llm/cpp/engine_base/grammar/XGrammarBootstrap.h"

#include <algorithm>
#include <stdexcept>

#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/config/SpecialTokens.h"

namespace rtp_llm {

std::vector<int32_t> collectStopTokenIds(const SpecialTokens& st) {
    std::vector<int32_t> ids;
    auto                 add_id = [&](int64_t id) {
        if (id < 0) return;
        const auto v = static_cast<int32_t>(id);
        if (std::find(ids.begin(), ids.end(), v) == ids.end()) {
            ids.push_back(v);
        }
    };

    add_id(st.eos_token_id);
    for (const auto& seq : st.stop_words_id_list) {
        if (seq.size() == 1) {
            add_id(seq[0]);
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::string buildXGrammarTokenizerInfoJson(const std::unordered_map<std::string, int32_t>& vocab,
                                           const std::string&                              backend_tokenizer_str,
                                           const std::vector<int32_t>&                     stop_token_ids) {
    static constexpr int64_t kMaxVocabSize = 1'000'000;
    if (vocab.empty()) {
        throw std::invalid_argument("buildXGrammarTokenizerInfoJson: vocab is empty");
    }
    int64_t max_id = -1;
    for (const auto& [_, tid] : vocab) {
        if (tid < 0) {
            throw std::invalid_argument("buildXGrammarTokenizerInfoJson: negative token id " + std::to_string(tid));
        }
        max_id = std::max(max_id, static_cast<int64_t>(tid));
    }
    const int64_t vocab_size = max_id + 1;
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
        if (i) stops += ",";
        stops += std::to_string(stop_token_ids[i]);
    }
    stops += "]";

    const auto close_pos = meta.rfind('}');
    if (close_pos == std::string::npos) {
        throw std::runtime_error(
            "buildXGrammarTokenizerInfoJson: DetectMetadataFromHF returned a non-object metadata string: '" + meta
            + "'");
    }
    const auto  first_content = meta.find_first_not_of(" \t\r\n", meta.find('{') + 1);
    const bool  has_members   = first_content != std::string::npos && first_content < close_pos;
    std::string injected      = std::string(has_members ? "," : "")
                           + "\"vocab_size\":" + std::to_string(vocab_size) + ",\"stop_token_ids\":" + stops;
    meta.insert(close_pos, injected);
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, meta).SerializeJSON();
}

}  // namespace rtp_llm
