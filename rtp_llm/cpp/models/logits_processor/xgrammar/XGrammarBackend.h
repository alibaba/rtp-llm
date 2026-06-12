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

namespace rtp_llm {

class RtpGrammarMatcher;

// Grammar request key shared by scheduler and backend.
struct GrammarKeyCpp {
    std::string key_type;    // "json" / "regex" / "ebnf" / "structural_tag"
    std::string key_string;  // schema / pattern / EBNF / structural tag JSON

    bool empty() const noexcept { return key_type.empty(); }
    bool operator==(const GrammarKeyCpp& other) const noexcept {
        return key_type == other.key_type && key_string == other.key_string;
    }

    // Stable composite identity. 0x1f (US) cannot legally appear in any
    // grammar body, so (key_type + 0x1f + key_string) is collision-free.
    // Lazily cached: submit() consults it via getCached/getCachedInvalid plus its
    // own in-flight lookup, so a large key_string is concatenated at most once.
    // A GrammarKeyCpp is single-threaded within a submit() call (the worker uses
    // its own moved copy), so the mutable cache needs no synchronization.
    const std::string& id() const {
        if (cached_id_.empty()) {
            cached_id_.reserve(key_type.size() + 1 + key_string.size());
            cached_id_.append(key_type);
            cached_id_.push_back('\x1f');
            cached_id_.append(key_string);
        }
        return cached_id_;
    }

    // Log-friendly tag — never dumps the full schema (can be large).
    std::string brief() const {
        return key_type + "(len=" + std::to_string(key_string.size()) + ")";
    }

    // Lazily built by id(); left public so GrammarKeyCpp stays an aggregate
    // (callers brace-init it as {type, string}). Not part of identity/equality.
    mutable std::string cached_id_;
};

// Compile outcome. (compiled != null) ok; (is_invalid) cacheable schema
// rejection; otherwise system error (not cached, retry).
struct CompileResult {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    bool                                       is_invalid = false;
    std::string                                error_message;
};

struct XGrammarBackendOptions {
    bool                                          any_whitespace        = true;
    bool                                          strict_mode           = true;
    int                                           max_compiler_threads  = 8;
    bool                                          enable_compiler_cache = true;
    int64_t                                       compiler_cache_bytes  = -1;  // unlimited
    std::optional<std::vector<int32_t>> override_stop_tokens;
};

// Native-C++ replacement for the trio {base,xgrammar,reasoner}_grammar_backend.py.
// Owns the xgrammar compiler + in-memory LRU cache. Thread-safe; no GIL.
// Lifetime is owned by GrammarCompiler (process singleton); construct directly
// in unit tests.
class XGrammarBackend {
public:
    // tokenizer_info_json must be SerializeJSON() of an xgrammar::TokenizerInfo
    // built on the Python side from the model's HF tokenizer. Throws on
    // deserialize failure — caller treats as unrecoverable.
    XGrammarBackend(const std::string&            tokenizer_info_json,
                       const XGrammarBackendOptions& options);
    ~XGrammarBackend();

    XGrammarBackend(const XGrammarBackend&)            = delete;
    XGrammarBackend& operator=(const XGrammarBackend&) = delete;

    // structural_tag preprocessing: parse, fill missing `json_schema:{}` /
    // `schema:{}` defaults, re-serialize. Returns input unchanged on parse failure.
    static std::string sanitizeStructuralTag(const std::string& tag_json);

    std::shared_ptr<xgrammar::CompiledGrammar> getCached(const GrammarKeyCpp& key) const;
    // Returns the cached error message, or empty string if not cached as invalid.
    std::string                                getCachedInvalid(const GrammarKeyCpp& key) const;

    // Synchronous compile — does NOT consult the in-memory cache. Caller
    // checks getCached / getCachedInvalid first.
    CompileResult compileNow(const GrammarKeyCpp& key);

    void setCache(const GrammarKeyCpp& key, std::shared_ptr<xgrammar::CompiledGrammar> compiled);
    void setCacheInvalid(const GrammarKeyCpp& key, const std::string& error_message);

    // Returns a fresh per-stream matcher; ownership transferred to caller.
    // The shared_ptr<CompiledGrammar> inside keeps the compile alive past
    // any cache eviction.
    std::unique_ptr<RtpGrammarMatcher> createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                     bool terminate_without_stop_token = false);

    void clear();

    // Backend health flag surfaced to the gate factory: false means grammar
    // requests must be rejected at admission instead of failing per stream.
    // With the native CUDA bitmask kernel there is no fallible import step, so
    // this stays true once the backend constructs successfully; the flag is
    // kept as the admission-time health hook for future backends.
    bool                     isEnabled() const noexcept { return enabled_; }

private:
    XGrammarBackendOptions    options_;
    xgrammar::TokenizerInfo   tokenizer_info_;
    xgrammar::GrammarCompiler compiler_;

    // LRU-bounded caches so a client streaming many distinct (or malformed)
    // schemas cannot grow them without limit and OOM the process. The *_lru_
    // lists order keys by recency (front = most-recently-used); get*/set* move
    // the touched key to the front and overflow evicts the back (least-recently
    // used) — hot/reused schemas survive a flood of one-off ones. std::list::
    // splice preserves iterators, so the stored lru_it stays valid and the
    // const get* methods can touch the (mutable) list. Compiled grammars are
    // also held by xgrammar's own compiler cache.
    static constexpr size_t kMaxCompiledCacheEntries = 1024;
    static constexpr size_t kMaxInvalidCacheEntries  = 4096;

    struct CompiledEntry {
        std::shared_ptr<xgrammar::CompiledGrammar> compiled;
        std::list<std::string>::iterator           lru_it;
    };
    struct InvalidEntry {
        std::string                      error_message;
        std::list<std::string>::iterator lru_it;
    };

    mutable std::mutex                             cache_mutex_;
    mutable std::list<std::string>                 cache_lru_;    // keys, front = MRU
    std::unordered_map<std::string, CompiledEntry> cache_;
    mutable std::list<std::string>                 invalid_lru_;  // keys, front = MRU
    std::unordered_map<std::string, InvalidEntry>  invalid_cache_;

    bool              enabled_ = true;
};

}  // namespace rtp_llm
