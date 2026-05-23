#pragma once

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

struct GrammarKeyCpp {
    std::string key_type;
    std::string key_string;

    bool empty() const noexcept {
        return key_type.empty() || key_string.empty();
    }

    std::string id() const {
        std::string out;
        out.reserve(key_type.size() + key_string.size() + 1);
        out.append(key_type);
        out.push_back('\x1f');
        out.append(key_string);
        return out;
    }
};

struct CompileResult {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    bool                                       is_invalid = false;
    std::string                                error_message;
};

struct XGrammarBackendOptions {
    bool                            any_whitespace        = true;
    bool                            strict_mode           = true;
    int                             max_compiler_threads  = 8;
    bool                            enable_compiler_cache = true;
    int64_t                         compiler_cache_bytes  = -1;
    std::optional<std::vector<int>> override_stop_tokens;
};

class XGrammarBackendCpp {
public:
    XGrammarBackendCpp(const std::string& tokenizer_info_json, const XGrammarBackendOptions& options);
    ~XGrammarBackendCpp();

    XGrammarBackendCpp(const XGrammarBackendCpp&)            = delete;
    XGrammarBackendCpp& operator=(const XGrammarBackendCpp&) = delete;

    std::shared_ptr<xgrammar::CompiledGrammar> getCached(const GrammarKeyCpp& key) const;
    std::string                                getCachedInvalid(const GrammarKeyCpp& key) const;
    CompileResult                              compileNow(const GrammarKeyCpp& key);
    void setCache(const GrammarKeyCpp& key, std::shared_ptr<xgrammar::CompiledGrammar> compiled);
    void setCacheInvalid(const GrammarKeyCpp& key, const std::string& error_message);

    std::shared_ptr<RtpGrammarMatcher> createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                     bool                                       require_reasoning,
                                                     std::optional<std::vector<int>>            think_end_token_ids);

    void clear();

private:
    static std::string sanitizeStructuralTag(const std::string& tag_json);

private:
    XGrammarBackendOptions    options_;
    xgrammar::TokenizerInfo   tokenizer_info_;
    xgrammar::GrammarCompiler compiler_;

    mutable std::mutex                                                          cache_mutex_;
    std::unordered_map<std::string, std::shared_ptr<xgrammar::CompiledGrammar>> cache_;
    std::unordered_map<std::string, std::string>                                invalid_cache_;
};

using XGrammarBackendCppPtr = std::shared_ptr<XGrammarBackendCpp>;

}  // namespace rtp_llm
