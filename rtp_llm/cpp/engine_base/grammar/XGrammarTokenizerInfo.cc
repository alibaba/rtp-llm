#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfo.h"

#include <stdexcept>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"

#include <xgrammar/tokenizer_info.h>

namespace rtp_llm::xgrammar_impl {
namespace {

int64_t vocabTypeId(const std::string& vocab_type) {
    if (vocab_type == "RAW") {
        return static_cast<int64_t>(xgrammar::VocabType::RAW);
    }
    if (vocab_type == "BYTE_FALLBACK") {
        return static_cast<int64_t>(xgrammar::VocabType::BYTE_FALLBACK);
    }
    if (vocab_type == "BYTE_LEVEL") {
        return static_cast<int64_t>(xgrammar::VocabType::BYTE_LEVEL);
    }
    throw std::runtime_error("serializeTokenizerInfo: invalid xgrammar vocab_type '" + vocab_type + "'");
}

const autil::legacy::Any&
requireField(const autil::legacy::json::JsonMap& metadata, const std::string& field_name) {
    const auto it = metadata.find(field_name);
    if (it == metadata.end()) {
        throw std::runtime_error("serializeTokenizerInfo: tokenizer metadata missing " + field_name);
    }
    return it->second;
}

std::string xgrammarMetadataJson(const std::string& tokenizer_metadata_json) {
    using autil::legacy::AnyCast;
    using autil::legacy::json::ParseJson;
    using autil::legacy::json::JsonMap;
    using autil::legacy::json::JsonNumber;
    using autil::legacy::json::ToString;

    const auto metadata_any = ParseJson(tokenizer_metadata_json);
    const auto metadata     = AnyCast<JsonMap>(metadata_any);

    const auto hf_tokenizer_json_it = metadata.find("hf_tokenizer_json");
    if (hf_tokenizer_json_it != metadata.end()) {
        auto xgrammar_metadata =
            AnyCast<JsonMap>(ParseJson(xgrammar::TokenizerInfo::DetectMetadataFromHF(
                AnyCast<std::string>(hf_tokenizer_json_it->second))));
        xgrammar_metadata["vocab_size"]     = requireField(metadata, "vocab_size");
        xgrammar_metadata["stop_token_ids"] = requireField(metadata, "stop_token_ids");
        return ToString(autil::legacy::Any(xgrammar_metadata), true);
    }

    auto xgrammar_metadata = metadata;
    auto vocab_type_it     = xgrammar_metadata.find("vocab_type");
    if (vocab_type_it != xgrammar_metadata.end()
        && vocab_type_it->second.GetType() == typeid(std::string)) {
        vocab_type_it->second = JsonNumber(vocabTypeId(AnyCast<std::string>(vocab_type_it->second)));
    }
    return ToString(autil::legacy::Any(xgrammar_metadata), true);
}

}  // namespace

std::string serializeTokenizerInfo(const std::vector<std::string>& encoded_vocab,
                                   const std::string&              tokenizer_metadata_json) {
    if (encoded_vocab.empty()) {
        throw std::invalid_argument("serializeTokenizerInfo: encoded_vocab is empty");
    }
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, xgrammarMetadataJson(tokenizer_metadata_json))
        .SerializeJSON();
}

}  // namespace rtp_llm::xgrammar_impl
