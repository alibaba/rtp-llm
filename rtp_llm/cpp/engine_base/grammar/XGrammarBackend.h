#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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

    std::string brief() const {
        return key_type + "(len=" + std::to_string(key_string.size()) + ")";
    }
};

// (compiled != null) ok; is_invalid means user-facing grammar rejection; otherwise
// this is a retryable/system error.
struct CompileResult {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    bool                                       is_invalid = false;
    std::string                                error_message;
};

struct XGrammarBackendOptions {
    bool                                any_whitespace            = true;
    bool                                strict_mode               = true;
    int                                 max_compiler_threads      = 8;
    int64_t                             compiler_cache_bytes      = -1;  // unlimited
    std::optional<std::vector<int32_t>> override_stop_tokens;
};

// Owns the xgrammar compiler; thread-safe, no GIL.
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

    // Synchronous; concurrent same-key compiles dedup inside xgrammar::GrammarCompiler.
    CompileResult getOrCompile(const GrammarKeyCpp& key);

    std::shared_ptr<RtpGrammarMatcher> createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                     bool                                       require_reasoning,
                                                     std::optional<std::vector<int>>            think_end_token_ids,
                                                     bool terminate_without_stop_token = false);

    void clear();

private:
    XGrammarBackendOptions    options_;
    xgrammar::TokenizerInfo   tokenizer_info_;
    xgrammar::GrammarCompiler compiler_;

    CompileResult compileNow(const GrammarKeyCpp& key);
};

}  // namespace rtp_llm
