#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
    bool operator==(const GrammarKeyCpp& other) const noexcept {
        return key_type == other.key_type && key_string == other.key_string;
    }

    std::string brief() const {
        return key_type + "(len=" + std::to_string(key_string.size()) + ")";
    }
};

struct XGrammarBackendOptions {
    bool                                any_whitespace       = true;
    bool                                strict_mode          = true;
    int                                 max_compiler_threads = 8;
    int64_t                             compiler_cache_bytes = -1;  // unlimited
    std::optional<std::vector<int32_t>> override_stop_tokens;
};

// Owns the xgrammar compiler; thread-safe, no GIL.
class XGrammarBackend {
public:
    // Production path: use constructor-built TokenizerInfo so xgrammar's derived indexes are intact.
    XGrammarBackend(const xgrammar::TokenizerInfo& tokenizer_info, const XGrammarBackendOptions& options);

    ~XGrammarBackend();

    XGrammarBackend(const XGrammarBackend&) = delete;
    XGrammarBackend& operator=(const XGrammarBackend&) = delete;

    // Returns nullptr (not throw) when tokenizer info is empty / invalid / build fails.
    static std::shared_ptr<XGrammarBackend> create(const std::string&   tokenizer_info_json,
                                                   const GrammarConfig& cfg) noexcept;

    // True iff the tokenizer is non-empty so the backend can compile / mask grammars.
    bool isEnabled() const noexcept;

    // Synchronous; cache and concurrent same-key races are handled inside xgrammar::GrammarCompiler.
    // InvalidArgument means user-facing grammar rejection; other failures are system/retryable.
    absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> compile(const GrammarKeyCpp& key);

    // Creates a fresh per-stream matcher from a grammar key. The compiled grammar is
    // cached by compile(); matcher state itself is never shared across streams.
    absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>> createMatcherFromKey(const GrammarKeyCpp& key,
                                                                            bool terminate_without_stop_token = false);

    absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>>
    createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled, bool terminate_without_stop_token = false);

    void clear();

private:
    XGrammarBackendOptions    options_;
    xgrammar::TokenizerInfo   tokenizer_info_;
    xgrammar::GrammarCompiler compiler_;
};

}  // namespace rtp_llm
