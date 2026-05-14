#pragma once

#include <chrono>
#include <cstdint>
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
    std::string id() const {
        std::string out;
        out.reserve(key_type.size() + 1 + key_string.size());
        out.append(key_type);
        out.push_back('\x1f');
        out.append(key_string);
        return out;
    }

    // Log-friendly tag — never dumps the full schema (can be large).
    std::string brief() const {
        return key_type + "(len=" + std::to_string(key_string.size()) + ")";
    }
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
    std::optional<std::vector<int32_t>>           override_stop_tokens;
    std::optional<int32_t>                        think_end_id;  // set ⇒ reasoner mode
};

// Native-C++ replacement for the trio {base,xgrammar,reasoner}_grammar_backend.py.
// Owns the xgrammar compiler + memory cache. All methods thread-safe; no GIL.
class XGrammarBackendCpp {
public:
    // tokenizer_info_json must be SerializeJSON() of an xgrammar::TokenizerInfo
    // built on the Python side from the model's HF tokenizer. Throws on
    // deserialize failure — caller treats as unrecoverable.
    XGrammarBackendCpp(const std::string&            tokenizer_info_json,
                       const XGrammarBackendOptions& options);
    ~XGrammarBackendCpp();

    XGrammarBackendCpp(const XGrammarBackendCpp&)            = delete;
    XGrammarBackendCpp& operator=(const XGrammarBackendCpp&) = delete;

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
                                                     bool                                       require_reasoning);

    void clear();

    bool                   hasReasoner() const noexcept { return options_.think_end_id.has_value(); }
    std::optional<int32_t> thinkEndId() const noexcept { return options_.think_end_id; }

private:
    // structural_tag preprocessing: parse, fill missing `json_schema:{}` /
    // `schema:{}` defaults, re-serialize. Returns input unchanged on parse
    // failure so xgrammar's own InvalidGrammar path takes over.
    static std::string sanitizeStructuralTag(const std::string& tag_json);

    XGrammarBackendOptions    options_;
    xgrammar::TokenizerInfo   tokenizer_info_;
    xgrammar::GrammarCompiler compiler_;

    mutable std::mutex                                                          cache_mutex_;
    std::unordered_map<std::string, std::shared_ptr<xgrammar::CompiledGrammar>> cache_;
    std::unordered_map<std::string, std::string>                                invalid_cache_;
};

}  // namespace rtp_llm
