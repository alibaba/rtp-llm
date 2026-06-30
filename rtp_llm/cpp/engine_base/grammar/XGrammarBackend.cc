#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <algorithm>
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
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/TokenizerInfo.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

using JsonMap   = autil::legacy::json::JsonMap;
using JsonArray = autil::legacy::json::JsonArray;

// Stack guard for adversarial nesting; legitimate trees are ~5 deep.
constexpr int kStructuralFormatMaxDepth = 64;

// xgrammar requires json_schema:{} on json_schema/qwen_xml_parameter nodes.
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
    // On parse/serialize failure return input unchanged; xgrammar's InvalidGrammar takes over.
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

namespace {
XGrammarBackendOptions backendOptionsFromConfig(const GrammarConfig& cfg) {
    XGrammarBackendOptions opts;
    opts.any_whitespace        = !cfg.constrained_json_disable_any_whitespace;
    opts.strict_mode           = true;
    opts.max_compiler_threads  = std::max(1, cfg.num_workers);
    opts.enable_compiler_cache = true;
    // <=0 = unlimited; bounds adversarial unique-schema streams in xgrammar's compile cache.
    opts.compiler_cache_bytes = cfg.compiler_cache_bytes > 0 ? cfg.compiler_cache_bytes : -1;
    if (!cfg.override_stop_tokens.empty()) {
        opts.override_stop_tokens = cfg.override_stop_tokens;
    }
    return opts;
}
}  // namespace

std::shared_ptr<XGrammarBackend> XGrammarBackend::create(const TokenizerInfo& tokenizer_info,
                                                         const GrammarConfig& cfg) noexcept {
    try {
        if (tokenizer_info.empty()) {
            RTP_LLM_LOG_INFO("XGrammarBackend::create: structured output disabled (TokenizerInfo empty)");
            return nullptr;
        }
        const std::string&     opaque  = TokenizerInfo::BackendAccess::opaque(tokenizer_info);
        XGrammarBackendOptions opts    = backendOptionsFromConfig(cfg);
        auto                   backend = std::make_shared<XGrammarBackend>(opaque, opts);
        RTP_LLM_LOG_INFO("XGrammarBackend::create: ready (override_stop_tokens=%zu, threads=%d)",
                         cfg.override_stop_tokens.size(),
                         opts.max_compiler_threads);
        return backend;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("XGrammarBackend::create: build threw (%s); disabling grammar", e.what());
        return nullptr;
    }
}

XGrammarBackend::XGrammarBackend(const std::string& tokenizer_info_json, const XGrammarBackendOptions& options):
    options_(options),
    tokenizer_info_([&] {
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
    RTP_LLM_LOG_INFO("XGrammarBackend init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, "
                     "compiler_threads=%d, compiler_cache=%d, compiler_cache_bytes=%lld",
                     tokenizer_info_.GetVocabSize(),
                     static_cast<int>(options_.any_whitespace),
                     static_cast<int>(options_.strict_mode),
                     std::max(1, options_.max_compiler_threads),
                     static_cast<int>(options_.enable_compiler_cache),
                     static_cast<long long>(options_.compiler_cache_bytes));
}

XGrammarBackend::~XGrammarBackend() = default;

bool XGrammarBackend::isEnabled() const noexcept {
    return tokenizer_info_.GetVocabSize() > 0;
}

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

void XGrammarBackend::setCache(const GrammarKeyCpp& key, std::shared_ptr<xgrammar::CompiledGrammar> compiled) {
    if (!compiled) {
        return;  // use setCacheInvalid for nullptrs.
    }
    const std::string           kid = key.id();
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Drop any invalid entry for this key (it just compiled successfully).
    if (auto iit = invalid_cache_.find(kid); iit != invalid_cache_.end()) {
        invalid_cache_bytes_ -= invalidEntryBytes(kid.size(), iit->second.error_message.size());
        invalid_lru_.erase(iit->second.lru_it);
        invalid_cache_.erase(iit);
    }

    if (auto it = cache_.find(kid); it != cache_.end()) {
        compiled_cache_bytes_ -= compiledEntryBytes(kid.size(), it->second.compiled);
        it->second.compiled = std::move(compiled);
        compiled_cache_bytes_ += compiledEntryBytes(kid.size(), it->second.compiled);
        cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second.lru_it);
    } else {
        cache_lru_.push_front(kid);
        compiled_cache_bytes_ += compiledEntryBytes(kid.size(), compiled);
        cache_[kid] = CompiledEntry{std::move(compiled), cache_lru_.begin()};
    }

    const size_t max_compiled_cache_bytes = maxCompiledCacheBytes();
    while (cache_.size() > kMaxCompiledCacheEntries || compiled_cache_bytes_ > max_compiled_cache_bytes) {
        const std::string victim = cache_lru_.back();  // LRU
        cache_lru_.pop_back();
        auto vit = cache_.find(victim);
        compiled_cache_bytes_ -= compiledEntryBytes(victim.size(), vit->second.compiled);
        cache_.erase(vit);
    }
}

void XGrammarBackend::setCacheInvalid(const GrammarKeyCpp& key, const std::string& error_message) {
    // Mirror getOrCompile: oversize keys MUST NOT enter invalid_cache_ (load-bearing
    // memory invariant; otherwise N distinct >64KB blobs spike memory before LRU evicts).
    if (key.key_string.size() > kMaxKeyStringBytes) {
        return;
    }
    const std::string kid = key.id();
    std::string       err = error_message.size() > kMaxErrorMessageBytes ?
                                error_message.substr(0, kMaxErrorMessageBytes) + "...[truncated]" :
                                error_message;

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Drop any compiled entry for this key (now known invalid).
    if (auto cit = cache_.find(kid); cit != cache_.end()) {
        compiled_cache_bytes_ -= compiledEntryBytes(kid.size(), cit->second.compiled);
        cache_lru_.erase(cit->second.lru_it);
        cache_.erase(cit);
    }

    if (auto it = invalid_cache_.find(kid); it != invalid_cache_.end()) {
        invalid_cache_bytes_ -= invalidEntryBytes(kid.size(), it->second.error_message.size());
        it->second.error_message = std::move(err);
        invalid_cache_bytes_ += invalidEntryBytes(kid.size(), it->second.error_message.size());
        invalid_lru_.splice(invalid_lru_.begin(), invalid_lru_, it->second.lru_it);
    } else {
        invalid_lru_.push_front(kid);
        invalid_cache_bytes_ += invalidEntryBytes(kid.size(), err.size());
        invalid_cache_[kid] = InvalidEntry{std::move(err), invalid_lru_.begin()};
    }

    // Evict on EITHER cap; bytes is the load-bearing one for memory safety.
    while (invalid_cache_.size() > kMaxInvalidCacheEntries || invalid_cache_bytes_ > kMaxInvalidCacheBytes) {
        const std::string victim = std::move(invalid_lru_.back());
        invalid_lru_.pop_back();
        auto vit = invalid_cache_.find(victim);
        invalid_cache_bytes_ -= invalidEntryBytes(victim.size(), vit->second.error_message.size());
        invalid_cache_.erase(vit);
    }
}

// Thread-safe via xgrammar::GrammarCompiler's internal cache.
CompileResult XGrammarBackend::compileNow(const GrammarKeyCpp& key) {
    const auto    t_start = std::chrono::steady_clock::now();
    const auto&   s       = key.key_string;
    CompileResult result;

    // Pick the underlying xgrammar call once; one try/catch covers all four key_types.
    // Schema rejection → is_invalid=true (cacheable); system errors retain retry semantics.
    std::function<xgrammar::CompiledGrammar()> do_compile;
    if (key.key_type == "json") {
        // "$$ANY$$" → any JSON value (response_format=json_object).
        do_compile = [&] {
            return s == "$$ANY$$" ? compiler_.CompileBuiltinJSONGrammar() :
                                    compiler_.CompileJSONSchema(
                                        s, options_.any_whitespace, std::nullopt, std::nullopt, options_.strict_mode);
        };
    } else if (key.key_type == "regex") {
        do_compile = [&] { return compiler_.CompileRegex(s); };
    } else if (key.key_type == "ebnf") {
        do_compile = [&] { return compiler_.CompileGrammar(s); };
    } else if (key.key_type == "structural_tag") {
        do_compile = [&] { return compiler_.CompileStructuralTag(sanitizeStructuralTag(s)); };
    } else {
        result.is_invalid    = true;
        result.error_message = "Unknown grammar key_type: " + key.key_type;
    }

    if (do_compile) {
        try {
            result.compiled = std::make_shared<xgrammar::CompiledGrammar>(do_compile());
        } catch (const std::bad_alloc& e) {
            result.is_invalid    = false;
            result.error_message = std::string("system error (retryable): ") + e.what();
        } catch (const std::runtime_error& e) {
            result.is_invalid    = true;
            result.error_message = e.what();
        } catch (const std::exception& e) {
            result.is_invalid    = false;
            result.error_message = std::string("unexpected error (retryable): ") + e.what();
        }
    }

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();
    if (result.compiled) {
        RTP_LLM_LOG_INFO("XGrammarBackend compile OK: type=%s, len=%zu, elapsed_ms=%lld, bytes=%zu",
                         key.key_type.c_str(),
                         key.key_string.size(),
                         static_cast<long long>(elapsed_ms),
                         result.compiled->MemorySizeBytes());
    } else {
        RTP_LLM_LOG_WARNING("XGrammarBackend compile FAIL: type=%s, len=%zu, elapsed_ms=%lld, invalid=%d, err=%s",
                            key.key_type.c_str(),
                            key.key_string.size(),
                            static_cast<long long>(elapsed_ms),
                            static_cast<int>(result.is_invalid),
                            result.error_message.c_str());
    }
    return result;
}

// Synchronous; same-key races dedup inside xgrammar::GrammarCompiler (last-writer-wins).
CompileResult XGrammarBackend::getOrCompile(const GrammarKeyCpp& key) {
    // Reject oversize payloads before cache/compiler; do NOT cache (memory amplification).
    if (key.key_string.size() > kMaxKeyStringBytes) {
        CompileResult result;
        result.is_invalid    = true;
        result.error_message = "grammar key_string too large: " + std::to_string(key.key_string.size())
                               + " bytes (limit " + std::to_string(kMaxKeyStringBytes) + ")";
        return result;
    }

    const std::string kid = key.id();
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        // Cache hit (touch LRU).
        if (auto it = cache_.find(kid); it != cache_.end()) {
            cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second.lru_it);
            CompileResult result;
            result.compiled = it->second.compiled;
            return result;
        }
        // Invalid cache hit (touch LRU).
        if (auto it = invalid_cache_.find(kid); it != invalid_cache_.end()) {
            invalid_lru_.splice(invalid_lru_.begin(), invalid_lru_, it->second.lru_it);
            CompileResult result;
            result.is_invalid    = true;
            result.error_message = it->second.error_message;
            return result;
        }
    }

    // Compile outside cache_mutex_ — would block cache hits during slow compiles.
    CompileResult result;
    try {
        result = compileNow(key);
    } catch (const std::exception& e) {
        result.compiled      = nullptr;
        result.is_invalid    = false;
        result.error_message = std::string("getOrCompile: unexpected throw: ") + e.what();
    } catch (...) {
        result.compiled      = nullptr;
        result.is_invalid    = false;
        result.error_message = "getOrCompile: unknown throw";
    }

    if (result.compiled) {
        setCache(key, result.compiled);
    } else if (result.is_invalid) {
        setCacheInvalid(key, result.error_message);
    }
    return result;
}

std::shared_ptr<RtpGrammarMatcher> XGrammarBackend::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                                  bool                            require_reasoning,
                                                                  std::optional<std::vector<int>> think_end_token_ids,
                                                                  bool terminate_without_stop_token) {
    RTP_LLM_CHECK_WITH_INFO(compiled != nullptr, "createMatcher requires a non-null CompiledGrammar");
    return std::make_shared<RtpGrammarMatcher>(std::move(compiled),
                                               require_reasoning,
                                               std::move(think_end_token_ids),
                                               options_.override_stop_tokens,
                                               terminate_without_stop_token);
}

void XGrammarBackend::clear() {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
        invalid_cache_.clear();
        cache_lru_.clear();
        invalid_lru_.clear();
        compiled_cache_bytes_ = 0;
        invalid_cache_bytes_ = 0;
    }
    compiler_.ClearCache();
    RTP_LLM_LOG_INFO("XGrammarBackend clear: caches dropped");
}

}  // namespace rtp_llm
