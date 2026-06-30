#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
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
class TokenizerInfo;

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
    std::string id() const {
        std::string result;
        result.reserve(key_type.size() + 1 + key_string.size());
        result.append(key_type);
        result.push_back('\x1f');
        result.append(key_string);
        return result;
    }

    std::string brief() const {
        return key_type + "(len=" + std::to_string(key_string.size()) + ")";
    }
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
class XGrammarBackend {
public:
    // tokenizer_info_json: xgrammar::TokenizerInfo::SerializeJSON(). Throws on parse fail.
    XGrammarBackend(const std::string& tokenizer_info_json, const XGrammarBackendOptions& options);
    ~XGrammarBackend();

    XGrammarBackend(const XGrammarBackend&)            = delete;
    XGrammarBackend& operator=(const XGrammarBackend&) = delete;

    // Returns nullptr (not throw) when tokenizer is empty / build fails.
    static std::shared_ptr<XGrammarBackend> create(const TokenizerInfo& tokenizer_info,
                                                   const GrammarConfig& cfg) noexcept;

    // True iff the tokenizer is non-empty so the backend can compile / mask grammars.
    bool isEnabled() const noexcept;

    // Fills missing json_schema/schema defaults; returns input unchanged on parse failure.
    static std::string sanitizeStructuralTag(const std::string& tag_json);

    // Synchronous; concurrent same-key compiles dedup inside xgrammar::GrammarCompiler.
    CompileResult getOrCompile(const GrammarKeyCpp& key);

    std::shared_ptr<xgrammar::CompiledGrammar> getCached(const GrammarKeyCpp& key) const;
    std::string                                getCachedInvalid(const GrammarKeyCpp& key) const;

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

    // Does NOT consult the cache; getOrCompile() is the public API.
    CompileResult compileNow(const GrammarKeyCpp& key);

    // LRU-bounded caches; front = MRU, evict from back on overflow.
    static constexpr size_t kMaxCompiledCacheEntries = 1024;
    static constexpr size_t kMaxInvalidCacheEntries  = 4096;
    // Per-request payload size cap; blocks MB-scale schemas amplifying the cache.
    static constexpr size_t kMaxKeyStringBytes = 64 * 1024;
    // Hard byte budget for invalid_cache_ — entry-count cap alone admits ~512 MB.
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
    // Live byte total for cache_; counts kid (x2: map key + LRU node) and
    // CompiledGrammar::MemorySizeBytes(). Maintained under cache_mutex_.
    size_t compiled_cache_bytes_ = 0;
    // Live byte total for invalid_cache_; counts kid (×2: map key + LRU node)
    // and error_message. Maintained under cache_mutex_.
    size_t invalid_cache_bytes_ = 0;

    size_t maxCompiledCacheBytes() const {
        return options_.compiler_cache_bytes > 0 ? static_cast<size_t>(options_.compiler_cache_bytes) :
                                                   std::numeric_limits<size_t>::max();
    }

    static size_t compiledEntryBytes(size_t kid_size, const std::shared_ptr<xgrammar::CompiledGrammar>& compiled) {
        return 2 * kid_size + (compiled ? compiled->MemorySizeBytes() : 0);
    }

    // Each invalid entry contributes one kid copy in the map key and one in the LRU node.
    static size_t invalidEntryBytes(size_t kid_size, size_t msg_size) {
        return 2 * kid_size + msg_size;
    }
};

}  // namespace rtp_llm
