#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarCompilerCache.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <variant>

namespace rtp_llm {

namespace {

constexpr const char* kBackendPayloadPrefix = "rtp-xgrammar-backend-payload-v1:";

struct ParsedBackendPayload {
    std::string tokenizer_info_json;
    std::string compiled_grammar_json;
};

ParsedBackendPayload parseBackendPayload(const std::string& payload) {
    const std::string prefix(kBackendPayloadPrefix);
    RTP_LLM_CHECK_WITH_INFO(payload.rfind(prefix, 0) == 0, "invalid xgrammar backend payload prefix");
    auto len_start = prefix.size();
    auto len_end   = payload.find(':', len_start);
    RTP_LLM_CHECK_WITH_INFO(len_end != std::string::npos, "invalid xgrammar backend payload length");
    auto tokenizer_json_len = static_cast<size_t>(std::stoull(payload.substr(len_start, len_end - len_start)));
    auto body_start         = len_end + 1;
    RTP_LLM_CHECK_WITH_INFO(payload.size() >= body_start + tokenizer_json_len,
                            "truncated xgrammar backend payload");
    ParsedBackendPayload parsed;
    parsed.tokenizer_info_json   = payload.substr(body_start, tokenizer_json_len);
    parsed.compiled_grammar_json = payload.substr(body_start + tokenizer_json_len);
    RTP_LLM_CHECK_WITH_INFO(!parsed.compiled_grammar_json.empty(), "empty xgrammar compiled grammar payload");
    return parsed;
}

#if RTP_LLM_ENABLE_XGRAMMAR_CPP
std::string serializationErrorMessage(const xgrammar::SerializationError& error) {
    return std::visit([](const auto& e) { return std::string(e.what()); }, error);
}

std::shared_ptr<xgrammar::CompiledGrammar> deserializeCompiledGrammar(const std::string& payload) {
    auto parsed                 = parseBackendPayload(payload);
    auto tokenizer_info_variant = xgrammar::TokenizerInfo::DeserializeJSON(parsed.tokenizer_info_json);
    RTP_LLM_CHECK_WITH_INFO(std::holds_alternative<xgrammar::TokenizerInfo>(tokenizer_info_variant),
                            "failed to deserialize xgrammar tokenizer info: %s",
                            serializationErrorMessage(std::get<xgrammar::SerializationError>(tokenizer_info_variant))
                                .c_str());
    auto tokenizer_info    = std::get<xgrammar::TokenizerInfo>(std::move(tokenizer_info_variant));
    auto compiled_variant = xgrammar::CompiledGrammar::DeserializeJSON(parsed.compiled_grammar_json, tokenizer_info);
    RTP_LLM_CHECK_WITH_INFO(std::holds_alternative<xgrammar::CompiledGrammar>(compiled_variant),
                            "failed to deserialize xgrammar compiled grammar: %s",
                            serializationErrorMessage(std::get<xgrammar::SerializationError>(compiled_variant))
                                .c_str());
    return std::make_shared<xgrammar::CompiledGrammar>(
        std::get<xgrammar::CompiledGrammar>(std::move(compiled_variant)));
}
#endif

}  // namespace

XGrammarCompilerCache& XGrammarCompilerCache::instance() {
    static XGrammarCompilerCache cache;
    return cache;
}

void XGrammarCompilerCache::init(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    while (entries_.size() > capacity_) {
        index_.erase(entries_.back().cache_key);
        entries_.pop_back();
        ++evictions_;
    }
}

size_t XGrammarCompilerCache::capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
}

size_t XGrammarCompilerCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

size_t XGrammarCompilerCache::evictionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return evictions_;
}

XGrammarDeviceTable XGrammarCompilerCache::getOrInsertDeviceTable(const std::string& cache_key,
                                                                  const std::string& lowering_blob) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = index_.find(cache_key);
    if (it != index_.end()) {
        entries_.splice(entries_.begin(), entries_, it->second);
        return *entries_.begin();
    }

    XGrammarDeviceTable table;
    table.cache_key     = cache_key;
    table.lowering_blob = lowering_blob;
    table.bytes         = lowering_blob.size();
#if RTP_LLM_ENABLE_XGRAMMAR_CPP
    table.compiled_grammar = deserializeCompiledGrammar(lowering_blob);
#endif
    if (capacity_ == 0) {
        return table;
    }

    entries_.push_front(table);
    index_[cache_key] = entries_.begin();
    while (entries_.size() > capacity_) {
        index_.erase(entries_.back().cache_key);
        entries_.pop_back();
        ++evictions_;
    }
    return table;
}

}  // namespace rtp_llm
