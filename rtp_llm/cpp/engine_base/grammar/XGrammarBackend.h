#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/grammar.h>
#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class RtpGrammarMatcher;

// Grammar request key shared by scheduler and backend.
struct GrammarKeyCpp {
    std::string key_type;    // "json" / "regex" / "ebnf" / "structural_tag"
    std::string key_string;  // schema / pattern / EBNF / structural tag JSON

    bool empty() const noexcept {
        return key_type.empty();
    }
    bool operator==(const GrammarKeyCpp& other) const noexcept {
        return key_type == other.key_type && key_string == other.key_string;
    }

    // 0x1f (US) cannot appear in a grammar body, so the join is collision-free.
    const std::string& id() const {
        if (cached_id_.empty()) {
            cached_id_.reserve(key_type.size() + 1 + key_string.size());
            cached_id_.append(key_type);
            cached_id_.push_back('\x1f');
            cached_id_.append(key_string);
        }
        return cached_id_;
    }

    std::string brief() const {
        return key_type + "(len=" + std::to_string(key_string.size()) + ")";
    }

    mutable std::string cached_id_;
};

// (compiled != null) ok; is_invalid → cacheable schema rejection; else system
// error (not cached).
struct CompileResult {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    bool                                       is_invalid = false;
    std::string                                error_message;
};

struct XGrammarBackendOptions {
    bool                                any_whitespace        = true;
    bool                                strict_mode           = true;
    int                                 max_compiler_threads  = 8;
    bool                                enable_compiler_cache = true;
    int64_t                             compiler_cache_bytes  = -1;  // unlimited
    std::optional<std::vector<int32_t>> override_stop_tokens;
};

// Owns the xgrammar compiler + LRU caches; thread-safe, no GIL.
//
// Lifecycle: built once at engine startup by LogitsProcessorFactory::init via
// fromConfig(cfg); the factory holds the active instance. Returns nullptr from
// fromConfig when grammar is disabled / tokenizer info empty / build fails.
class XGrammarBackend {
public:
    // tokenizer_info_json: xgrammar::TokenizerInfo::SerializeJSON(). Throws on parse fail.
    XGrammarBackend(const std::string& tokenizer_info_json, const XGrammarBackendOptions& options);
    ~XGrammarBackend();

    XGrammarBackend(const XGrammarBackend&)            = delete;
    XGrammarBackend& operator=(const XGrammarBackend&) = delete;

    // Build a backend from GrammarConfig (parses tokenizer info, applies options).
    // Returns nullptr (not throw) if cfg is empty/invalid -> structured output stays disabled.
    static std::shared_ptr<XGrammarBackend> fromConfig(const GrammarConfig& cfg) noexcept;

    // True iff the tokenizer is non-empty so the backend can compile / mask grammars.
    bool isEnabled() const noexcept;

    // Fills missing json_schema/schema defaults; returns input unchanged on parse failure.
    static std::string sanitizeStructuralTag(const std::string& tag_json);

    // Cache hit returns instantly. On miss the compile runs synchronously on the
    // calling thread; concurrent requests for the same key may each compile (the
    // underlying xgrammar::GrammarCompiler caches identical grammars internally,
    // so the duplicate work is bounded). Populates the success cache on completion
    // and the invalid-cache on schema rejection.
    CompileResult getOrCompile(const GrammarKeyCpp& key);

    std::shared_ptr<xgrammar::CompiledGrammar> getCached(const GrammarKeyCpp& key) const;
    std::string                                getCachedInvalid(const GrammarKeyCpp& key) const;

    // Does NOT consult the cache; getOrCompile() is the public API.
    CompileResult compileNow(const GrammarKeyCpp& key);

    void setCache(const GrammarKeyCpp& key, std::shared_ptr<xgrammar::CompiledGrammar> compiled);
    void setCacheInvalid(const GrammarKeyCpp& key, const std::string& error_message);

    std::shared_ptr<RtpGrammarMatcher> createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                     bool                                       require_reasoning,
                                                     std::optional<std::vector<int>>            think_end_token_ids,
                                                     bool terminate_without_stop_token = false);

    void clear();

private:
    XGrammarBackendOptions    options_;
    xgrammar::TokenizerInfo   tokenizer_info_;
    xgrammar::GrammarCompiler compiler_;

    // LRU-bounded caches; front = MRU, evict from back on overflow.
    static constexpr size_t kMaxCompiledCacheEntries = 1024;
    static constexpr size_t kMaxInvalidCacheEntries  = 4096;
    // Per-request schema/regex/EBNF/structural-tag size cap, enforced at the
    // getOrCompile entry. Real schemas in practice are a few KB; this leaves
    // ~10x headroom while preventing a caller from amplifying the cache to
    // GB-scale by submitting MB-scale payloads.
    static constexpr size_t kMaxKeyStringBytes = 64 * 1024;
    // Hard byte budget for invalid_cache_ payload (kid copies + error_message).
    // Pins worst-case memory regardless of how the entry-count and per-key caps
    // multiply out: with 4096 entries × 64 KB keys × 2 copies (map key + LRU
    // node) the entry-count cap alone admits ~512 MB; this caps it at 32 MB.
    static constexpr size_t kMaxInvalidCacheBytes = 32 * 1024 * 1024;
    // Truncate xgrammar error strings before they enter invalid_cache_; the
    // useful prefix (rule name, position) fits well under 1 KB.
    static constexpr size_t kMaxErrorMessageBytes = 1024;

    struct CompiledEntry {
        std::shared_ptr<xgrammar::CompiledGrammar> compiled;
        std::list<std::string>::iterator           lru_it;
    };
    struct InvalidEntry {
        std::string                      error_message;
        std::list<std::string>::iterator lru_it;
    };

    mutable std::mutex                             cache_mutex_;
    mutable std::list<std::string>                 cache_lru_;  // keys, front = MRU
    std::unordered_map<std::string, CompiledEntry> cache_;
    mutable std::list<std::string>                 invalid_lru_;  // keys, front = MRU
    std::unordered_map<std::string, InvalidEntry>  invalid_cache_;
    // Live byte total for invalid_cache_; counts kid (×2: map key + LRU node)
    // and error_message. Maintained under cache_mutex_.
    size_t invalid_cache_bytes_ = 0;
};

}  // namespace rtp_llm
