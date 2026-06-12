#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBackend.h"

#include <cassert>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

#include <xgrammar/exception.h>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"


namespace rtp_llm {

namespace {

using JsonMap   = autil::legacy::json::JsonMap;
using JsonArray = autil::legacy::json::JsonArray;

// Cap on structural-format nesting depth. The legitimate xgrammar trees we
// see are at most ~5 deep (sequence/or wrapping triggered_tags wrapping
// json_schema), so 64 is generous; the bound exists purely to keep an
// adversarial / malformed payload from blowing the stack.
constexpr int kStructuralFormatMaxDepth = 64;

// Walk structural-format tree; inject empty `json_schema:{}` into any
// json_schema/qwen_xml_parameter node missing one (xgrammar requires the field).
void sanitizeStructuralFormat(autil::legacy::Any& any, int depth = 0) {
    if (depth > kStructuralFormatMaxDepth) {
        throw std::invalid_argument("structural_tag: format tree exceeds max depth "
                                    + std::to_string(kStructuralFormatMaxDepth));
    }
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
            sanitizeStructuralFormat(it->second, depth + 1);
        }
    } else if (fmt_type == "sequence" || fmt_type == "or") {
        if (auto it = map->find("elements"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& el : *arr) {
                    sanitizeStructuralFormat(el, depth + 1);
                }
            }
        }
    } else if (fmt_type == "triggered_tags" || fmt_type == "tags_with_separator") {
        if (auto it = map->find("tags"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& tag : *arr) {
                    sanitizeStructuralFormat(tag, depth + 1);
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

std::string XGrammarBackend::sanitizeStructuralTag(const std::string& tag_json) {
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

XGrammarBackend::XGrammarBackend(const std::string&            tokenizer_info_json,
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
        throw std::runtime_error(std::string("XGrammarBackend: failed to deserialize TokenizerInfo: ") + msg);
    }()),
    compiler_(tokenizer_info_,
              std::max(1, options.max_compiler_threads),
              options.enable_compiler_cache,
              options.compiler_cache_bytes) {
    RTP_LLM_LOG_INFO(
        "XGrammarBackend init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, "
        "compiler_threads=%d, compiler_cache=%d, compiler_cache_bytes=%lld",
        tokenizer_info_.GetVocabSize(),
        static_cast<int>(options_.any_whitespace),
        static_cast<int>(options_.strict_mode),
        std::max(1, options_.max_compiler_threads),
        static_cast<int>(options_.enable_compiler_cache),
        static_cast<long long>(options_.compiler_cache_bytes));

}


XGrammarBackend::~XGrammarBackend() = default;

std::shared_ptr<xgrammar::CompiledGrammar> XGrammarBackend::getCached(const GrammarKeyCpp& key) const {
    const std::string           kid = key.id();
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto                        it = cache_.find(kid);
    if (it == cache_.end()) {
        return nullptr;
    }
    // Touch: move to MRU. splice keeps it->second.lru_it valid.
    cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second.lru_it);
    return it->second.compiled;
}

std::string XGrammarBackend::getCachedInvalid(const GrammarKeyCpp& key) const {
    const std::string           kid = key.id();
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto                        it = invalid_cache_.find(kid);
    if (it == invalid_cache_.end()) {
        return {};
    }
    invalid_lru_.splice(invalid_lru_.begin(), invalid_lru_, it->second.lru_it);
    return it->second.error_message;
}

void XGrammarBackend::setCache(const GrammarKeyCpp&                       key,
                                  std::shared_ptr<xgrammar::CompiledGrammar> compiled) {
    if (!compiled) {
        return;  // use setCacheInvalid for nullptrs.
    }
    const std::string           kid = key.id();
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Drop any invalid entry for this key (it just compiled successfully).
    if (auto iit = invalid_cache_.find(kid); iit != invalid_cache_.end()) {
        invalid_lru_.erase(iit->second.lru_it);
        invalid_cache_.erase(iit);
    }

    if (auto it = cache_.find(kid); it != cache_.end()) {
        it->second.compiled = std::move(compiled);
        cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second.lru_it);
    } else {
        cache_lru_.push_front(kid);
        cache_[kid] = CompiledEntry{std::move(compiled), cache_lru_.begin()};
    }

    while (cache_.size() > kMaxCompiledCacheEntries) {
        const std::string victim = cache_lru_.back();  // LRU
        cache_lru_.pop_back();
        cache_.erase(victim);
    }
}

void XGrammarBackend::setCacheInvalid(const GrammarKeyCpp& key, const std::string& error_message) {
    const std::string           kid = key.id();
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Drop any compiled entry for this key (now known invalid).
    if (auto cit = cache_.find(kid); cit != cache_.end()) {
        cache_lru_.erase(cit->second.lru_it);
        cache_.erase(cit);
    }

    if (auto it = invalid_cache_.find(kid); it != invalid_cache_.end()) {
        it->second.error_message = error_message;
        invalid_lru_.splice(invalid_lru_.begin(), invalid_lru_, it->second.lru_it);
    } else {
        invalid_lru_.push_front(kid);
        invalid_cache_[kid] = InvalidEntry{error_message, invalid_lru_.begin()};
    }

    while (invalid_cache_.size() > kMaxInvalidCacheEntries) {
        const std::string victim = invalid_lru_.back();
        invalid_lru_.pop_back();
        invalid_cache_.erase(victim);
    }
}

// Thread-safe: xgrammar::GrammarCompiler uses ThreadSafeLRUCache (shared_mutex + shared_future) internally.
CompileResult XGrammarBackend::compileNow(const GrammarKeyCpp& key) {
    const auto t_start = std::chrono::steady_clock::now();

    // Wrap an xgrammar Compile* call: grammar/schema rejection exceptions
    // become is_invalid=true (permanently cacheable); system errors
    // (bad_alloc, etc.) leave is_invalid=false so the caller can retry.
    auto compileWith = [](auto&& fn) -> CompileResult {
        CompileResult out;
        try {
            out.compiled = std::make_shared<xgrammar::CompiledGrammar>(fn());
        } catch (const std::bad_alloc& e) {
            out.is_invalid    = false;
            out.error_message = std::string("system error (retryable): ") + e.what();
        } catch (const std::runtime_error& e) {
            out.is_invalid    = true;
            out.error_message = e.what();
        } catch (const std::exception& e) {
            out.is_invalid    = false;
            out.error_message = std::string("unexpected error (retryable): ") + e.what();
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
        RTP_LLM_LOG_INFO("XGrammarBackend compile OK: type=%s, len=%zu, elapsed_ms=%lld, bytes=%zu",
                          key.key_type.c_str(),
                          key.key_string.size(),
                          static_cast<long long>(elapsed_ms),
                          result.compiled->MemorySizeBytes());
    } else {
        RTP_LLM_LOG_WARNING(
            "XGrammarBackend compile FAIL: type=%s, len=%zu, elapsed_ms=%lld, invalid=%d, err=%s",
            key.key_type.c_str(),
            key.key_string.size(),
            static_cast<long long>(elapsed_ms),
            static_cast<int>(result.is_invalid),
            result.error_message.c_str());
    }
    return result;
}

std::unique_ptr<RtpGrammarMatcher>
XGrammarBackend::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                               bool                                       terminate_without_stop_token) {
    assert(compiled && "createMatcher requires a non-null CompiledGrammar");
    return std::make_unique<RtpGrammarMatcher>(std::move(compiled),
                                               options_.override_stop_tokens,
                                               terminate_without_stop_token,
                                               /*max_rollback_tokens=*/200);
}

void XGrammarBackend::clear() {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
        invalid_cache_.clear();
        cache_lru_.clear();
        invalid_lru_.clear();
    }
    compiler_.ClearCache();
    RTP_LLM_LOG_INFO("XGrammarBackend clear: caches dropped");
}

}  // namespace rtp_llm
