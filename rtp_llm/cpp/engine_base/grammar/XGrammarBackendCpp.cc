#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

#include <cassert>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <variant>

#include <xgrammar/exception.h>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using JsonMap   = autil::legacy::json::JsonMap;
using JsonArray = autil::legacy::json::JsonArray;

// Walk structural-format tree; inject empty `json_schema:{}` into any
// json_schema/qwen_xml_parameter node missing one (xgrammar requires the field).
void sanitizeStructuralFormat(autil::legacy::Any& any) {
    auto* map = autil::legacy::AnyCast<JsonMap>(&any);
    if (!map) {
        return;
    }

    std::string fmt_type;
    if (auto type_it = map->find("type"); type_it != map->end()) {
        if (auto* s = autil::legacy::AnyCast<std::string>(&type_it->second)) {
            fmt_type = *s;
        }
    }

    if (fmt_type == "json_schema" || fmt_type == "qwen_xml_parameter") {
        if (map->find("json_schema") == map->end()) {
            (*map)["json_schema"] = JsonMap{};
        }
    }

    if (fmt_type == "tag") {
        if (auto it = map->find("content"); it != map->end()) {
            sanitizeStructuralFormat(it->second);
        }
    } else if (fmt_type == "sequence" || fmt_type == "or") {
        if (auto it = map->find("elements"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& el : *arr) {
                    sanitizeStructuralFormat(el);
                }
            }
        }
    } else if (fmt_type == "triggered_tags" || fmt_type == "tags_with_separator") {
        if (auto it = map->find("tags"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& tag : *arr) {
                    sanitizeStructuralFormat(tag);
                }
            }
        }
    }
}

// Legacy structural_tag: default-fill `schema:{}` in `structures:[...]` items.
void sanitizeLegacyStructures(JsonMap& root) {
    auto it = root.find("structures");
    if (it == root.end()) {
        return;
    }
    auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second);
    if (!arr) {
        return;
    }
    for (auto& s : *arr) {
        if (auto* sm = autil::legacy::AnyCast<JsonMap>(&s)) {
            if (sm->find("schema") == sm->end()) {
                (*sm)["schema"] = JsonMap{};
            }
        }
    }
}

}  // namespace

std::string XGrammarBackendCpp::sanitizeStructuralTag(const std::string& tag_json) {
    // Parse → mutate → re-serialize. On any parse/serialize failure return
    // input unchanged so xgrammar's own InvalidGrammar path takes over.
    autil::legacy::Any any;
    try {
        autil::legacy::json::ParseJson(tag_json, any);
    } catch (...) {
        return tag_json;
    }
    auto* root = autil::legacy::AnyCast<JsonMap>(&any);
    if (!root) {
        return tag_json;
    }
    // Legacy carries `structures:[...]`; new carries `format:{...}`.
    if (root->find("structures") != root->end()) {
        sanitizeLegacyStructures(*root);
    } else if (auto fmt = root->find("format"); fmt != root->end()) {
        sanitizeStructuralFormat(fmt->second);
    }
    try {
        return autil::legacy::json::ToString(any, /*isCompact=*/true);
    } catch (...) {
        return tag_json;
    }
}

XGrammarBackendCpp::XGrammarBackendCpp(const std::string&            tokenizer_info_json,
                                       const XGrammarBackendOptions& options):
    options_(options),
    tokenizer_info_([&] {
        // Deserialize once at construction; xgrammar returns variant<info, err>.
        auto result = xgrammar::TokenizerInfo::DeserializeJSON(tokenizer_info_json);
        if (std::holds_alternative<xgrammar::TokenizerInfo>(result)) {
            return std::get<xgrammar::TokenizerInfo>(std::move(result));
        }
        std::string msg;
        std::visit([&msg](const auto& err) { msg = err.what(); }, std::get<1>(result));
        throw std::runtime_error(std::string("XGrammarBackendCpp: failed to deserialize TokenizerInfo: ") + msg);
    }()),
    compiler_(tokenizer_info_,
              std::max(1, options.max_compiler_threads),
              options.enable_compiler_cache,
              options.compiler_cache_bytes) {
    RTP_LLM_LOG_INFO(
        "XGrammarBackendCpp init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, "
        "compiler_threads=%d, compiler_cache=%d, compiler_cache_bytes=%lld, "
        "think_end_id=%s",
        tokenizer_info_.GetVocabSize(),
        static_cast<int>(options_.any_whitespace),
        static_cast<int>(options_.strict_mode),
        std::max(1, options_.max_compiler_threads),
        static_cast<int>(options_.enable_compiler_cache),
        static_cast<long long>(options_.compiler_cache_bytes),
        options_.think_end_id.has_value() ? std::to_string(*options_.think_end_id).c_str() : "<none>");
}

XGrammarBackendCpp::~XGrammarBackendCpp() = default;

std::shared_ptr<xgrammar::CompiledGrammar> XGrammarBackendCpp::getCached(const GrammarKeyCpp& key) const {
    const std::string kid = key.id();
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(kid);
        if (it != cache_.end()) {
            cache_hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
    }
    cache_misses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

std::string XGrammarBackendCpp::getCachedInvalid(const GrammarKeyCpp& key) const {
    const std::string kid = key.id();
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = invalid_cache_.find(kid);
    if (it == invalid_cache_.end()) {
        return {};
    }
    return it->second;
}

void XGrammarBackendCpp::setCache(const GrammarKeyCpp&                       key,
                                  std::shared_ptr<xgrammar::CompiledGrammar> compiled) {
    if (!compiled) {
        return;  // use setCacheInvalid for nullptrs.
    }
    const std::string kid             = key.id();
    const auto        new_size_bytes  = static_cast<int64_t>(compiled->MemorySizeBytes());
    int64_t           prev_size_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto                        it = cache_.find(kid);
        if (it != cache_.end()) {
            prev_size_bytes = static_cast<int64_t>(it->second->MemorySizeBytes());
        }
        cache_[kid] = std::move(compiled);
        invalid_cache_.erase(kid);  // success invalidates any prior failure marker.
    }
    bytes_in_cache_.fetch_add(new_size_bytes - prev_size_bytes, std::memory_order_relaxed);
}

void XGrammarBackendCpp::setCacheInvalid(const GrammarKeyCpp& key, const std::string& error_message) {
    const std::string kid = key.id();
    std::lock_guard<std::mutex> lock(cache_mutex_);
    invalid_cache_[kid] = error_message;
    auto it             = cache_.find(kid);
    if (it != cache_.end()) {
        bytes_in_cache_.fetch_sub(static_cast<int64_t>(it->second->MemorySizeBytes()),
                                  std::memory_order_relaxed);
        cache_.erase(it);
    }
}

CompileResult XGrammarBackendCpp::compileNow(const GrammarKeyCpp& key) {
    compile_calls_.fetch_add(1, std::memory_order_relaxed);
    const auto t_start = std::chrono::steady_clock::now();

    // Wrap an xgrammar Compile* call: schema-rejection exceptions become
    // is_invalid=true (cacheable); xgrammar's schema errors all inherit
    // runtime_error.
    auto compileWith = [](auto&& fn) -> CompileResult {
        CompileResult out;
        try {
            out.compiled = std::make_shared<xgrammar::CompiledGrammar>(fn());
        } catch (const std::runtime_error& e) {
            out.is_invalid    = true;
            out.error_message = e.what();
        }
        return out;
    };

    CompileResult result;
    const auto&   s = key.key_string;
    if (key.key_type == "json") {
        // "$$ANY$$" sentinel = "any JSON value" (response_format=json_object).
        result = compileWith([&] {
            return s == "$$ANY$$"
                       ? compiler_.CompileBuiltinJSONGrammar()
                       : compiler_.CompileJSONSchema(s, options_.any_whitespace, std::nullopt, std::nullopt,
                                                     options_.strict_mode);
        });
    } else if (key.key_type == "regex") {
        result = compileWith([&] { return compiler_.CompileRegex(s); });
    } else if (key.key_type == "ebnf") {
        result = compileWith([&] { return compiler_.CompileGrammar(s); });
    } else if (key.key_type == "structural_tag") {
        result = compileWith([&] { return compiler_.CompileStructuralTag(sanitizeStructuralTag(s)); });
    } else {
        result.is_invalid    = true;
        result.error_message = "Unknown grammar key_type: " + key.key_type;
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t_start)
                                .count();
    if (result.compiled) {
        RTP_LLM_LOG_DEBUG("XGrammarBackendCpp compile OK: type=%s, len=%zu, elapsed_ms=%lld, bytes=%zu",
                          key.key_type.c_str(),
                          key.key_string.size(),
                          static_cast<long long>(elapsed_ms),
                          result.compiled->MemorySizeBytes());
    } else {
        compile_failures_.fetch_add(1, std::memory_order_relaxed);
        RTP_LLM_LOG_WARNING(
            "XGrammarBackendCpp compile FAIL: type=%s, len=%zu, elapsed_ms=%lld, invalid=%d, err=%s",
            key.key_type.c_str(),
            key.key_string.size(),
            static_cast<long long>(elapsed_ms),
            static_cast<int>(result.is_invalid),
            result.error_message.c_str());
    }
    return result;
}

std::unique_ptr<RtpGrammarMatcher>
XGrammarBackendCpp::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled, bool require_reasoning) {
    assert(compiled && "createMatcher requires a non-null CompiledGrammar");
    return std::make_unique<RtpGrammarMatcher>(std::move(compiled),
                                               require_reasoning,
                                               options_.think_end_id,
                                               options_.override_stop_tokens,
                                               /*max_rollback_tokens=*/200);
}

void XGrammarBackendCpp::clear() {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
        invalid_cache_.clear();
    }
    bytes_in_cache_.store(0, std::memory_order_relaxed);
    compiler_.ClearCache();
    RTP_LLM_LOG_INFO("XGrammarBackendCpp clear: caches dropped");
}

XGrammarBackendCpp::Stats XGrammarBackendCpp::stats() const {
    Stats s;
    s.compile_calls    = compile_calls_.load(std::memory_order_relaxed);
    s.compile_failures = compile_failures_.load(std::memory_order_relaxed);
    s.cache_hits       = cache_hits_.load(std::memory_order_relaxed);
    s.cache_misses     = cache_misses_.load(std::memory_order_relaxed);
    s.bytes_in_cache   = bytes_in_cache_.load(std::memory_order_relaxed);
    return s;
}

}  // namespace rtp_llm
