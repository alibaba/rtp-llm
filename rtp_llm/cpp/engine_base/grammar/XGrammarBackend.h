#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <xgrammar/compiler.h>
#include <xgrammar/grammar.h>
#include <xgrammar/tokenizer_info.h>

#include "absl/status/statusor.h"
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
};

// Owns the xgrammar compiler; thread-safe, no GIL.
class XGrammarBackend {
public:
    ~XGrammarBackend();

    XGrammarBackend(const XGrammarBackend&)            = delete;
    XGrammarBackend& operator=(const XGrammarBackend&) = delete;
    XGrammarBackend(XGrammarBackend&&)                 = delete;
    XGrammarBackend& operator=(XGrammarBackend&&)      = delete;

    // Empty tokenizer info intentionally disables structured output. Non-empty
    // tokenizer info is a startup compatibility contract: deserialization or
    // backend construction failures must propagate and abort engine startup.
    static std::shared_ptr<XGrammarBackend> create(const std::string&   tokenizer_info_json,
                                                   const GrammarConfig& cfg);

    // Creates a fresh per-stream matcher from a grammar key. The compiled grammar is
    // cached by compile(); matcher state itself is never shared across streams.
    absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>> createMatcherFromKey(const GrammarKeyCpp& key);

private:
    struct Options {
        bool    any_whitespace               = true;
        bool    strict_mode                  = true;
        bool    terminate_without_stop_token = false;
        int     max_compiler_threads         = 8;
        int64_t compiler_cache_bytes         = -1;  // unlimited
    };

    XGrammarBackend(const xgrammar::TokenizerInfo& tokenizer_info, const Options& options);

    static Options optionsFromConfig(const GrammarConfig& cfg);

    // Synchronous; cache and concurrent same-key races are handled inside xgrammar::GrammarCompiler.
    // InvalidArgument means user-facing grammar rejection; other failures are system/retryable.
    absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> compile(const GrammarKeyCpp& key);

    absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>>
    createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled);

    Options                   options_;
    xgrammar::TokenizerInfo   tokenizer_info_;
    xgrammar::GrammarCompiler compiler_;
};

}  // namespace rtp_llm
