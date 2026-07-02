#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfoCooker.h"

#include <algorithm>
#include <optional>
#include <stdexcept>

#include <xgrammar/tokenizer_info.h>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"

namespace rtp_llm::xgrammar_impl {

namespace {

using JsonMap = autil::legacy::json::JsonMap;

int64_t jsonMapInt64(const JsonMap& map, const char* key) {
    const auto it = map.find(key);
    if (it == map.end()) {
        throw std::runtime_error(std::string("cookTokenizerInfoOpaque: metadata missing key '") + key + "'");
    }
    if (const auto* v = autil::legacy::AnyCast<int64_t>(&it->second)) {
        return *v;
    }
    if (const auto* v = autil::legacy::AnyCast<int>(&it->second)) {
        return *v;
    }
    if (const auto* v = autil::legacy::AnyCast<int32_t>(&it->second)) {
        return *v;
    }
    throw std::runtime_error(std::string("cookTokenizerInfoOpaque: metadata key '") + key + "' is not an integer");
}

bool jsonMapBool(const JsonMap& map, const char* key) {
    const auto it = map.find(key);
    if (it == map.end()) {
        throw std::runtime_error(std::string("cookTokenizerInfoOpaque: metadata missing key '") + key + "'");
    }
    if (const auto* v = autil::legacy::AnyCast<bool>(&it->second)) {
        return *v;
    }
    throw std::runtime_error(std::string("cookTokenizerInfoOpaque: metadata key '") + key + "' is not a bool");
}

struct HfTokenizerMetadata {
    xgrammar::VocabType vocab_type;
    bool                add_prefix_space;
};

HfTokenizerMetadata detectMetadataFromHf(const std::string& backend_tokenizer_str) {
    const std::string meta_json = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);
    autil::legacy::Any any;
    try {
        any = autil::legacy::json::ParseJson(meta_json);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("cookTokenizerInfoOpaque: invalid metadata JSON '") + meta_json
                                 + "': " + e.what());
    }
    const auto* map = autil::legacy::AnyCast<JsonMap>(&any);
    if (!map) {
        throw std::runtime_error("cookTokenizerInfoOpaque: metadata is not a JSON object: '" + meta_json + "'");
    }
    const int64_t vocab_type_val = jsonMapInt64(*map, "vocab_type");
    if (vocab_type_val < static_cast<int64_t>(xgrammar::VocabType::RAW)
        || vocab_type_val > static_cast<int64_t>(xgrammar::VocabType::BYTE_LEVEL)) {
        throw std::runtime_error("cookTokenizerInfoOpaque: unsupported vocab_type " + std::to_string(vocab_type_val));
    }
    return {static_cast<xgrammar::VocabType>(vocab_type_val), jsonMapBool(*map, "add_prefix_space")};
}

}  // namespace

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

    const auto                            meta = detectMetadataFromHf(backend_tokenizer_str);
    std::optional<std::vector<int32_t>> stops =
        stop_token_ids.empty() ? std::nullopt : std::optional<std::vector<int32_t>>(stop_token_ids);
    const xgrammar::TokenizerInfo info(encoded_vocab,
                                       meta.vocab_type,
                                       static_cast<int>(vocab_size),
                                       stops,
                                       meta.add_prefix_space);
    return info.SerializeJSON();
}

}  // namespace rtp_llm::xgrammar_impl
