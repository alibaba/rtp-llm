#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <optional>
#include <utility>
#include <variant>

#include <xgrammar/exception.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/TokenizerInfo.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
XGrammarBackendOptions backendOptionsFromConfig(const GrammarConfig& cfg) {
    XGrammarBackendOptions opts;
    opts.any_whitespace       = !cfg.constrained_json_disable_any_whitespace;
    opts.strict_mode          = true;
    opts.max_compiler_threads = std::max(1, cfg.num_workers);
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
              /*enable_cache=*/true,
              options.compiler_cache_bytes) {
    RTP_LLM_LOG_INFO("XGrammarBackend init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, "
                     "compiler_threads=%d, compiler_cache_bytes=%lld",
                     tokenizer_info_.GetVocabSize(),
                     static_cast<int>(options_.any_whitespace),
                     static_cast<int>(options_.strict_mode),
                     std::max(1, options_.max_compiler_threads),
                     static_cast<long long>(options_.compiler_cache_bytes));
}

XGrammarBackend::~XGrammarBackend() = default;

bool XGrammarBackend::isEnabled() const noexcept {
    return tokenizer_info_.GetVocabSize() > 0;
}

// Thread-safe via xgrammar::GrammarCompiler's internal cache.
CompileResult XGrammarBackend::compileNow(const GrammarKeyCpp& key) {
    const auto    t_start = std::chrono::steady_clock::now();
    const auto&   s       = key.key_string;
    CompileResult result;

    // Pick the underlying xgrammar call once; one try/catch covers all four key_types.
    // User grammar rejection → is_invalid=true; system errors retain retry semantics.
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
        do_compile = [&] { return compiler_.CompileStructuralTag(s); };
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

// Synchronous; caching and same-key races are handled by xgrammar::GrammarCompiler.
CompileResult XGrammarBackend::getOrCompile(const GrammarKeyCpp& key) {
    try {
        return compileNow(key);
    } catch (const std::exception& e) {
        CompileResult result;
        result.compiled      = nullptr;
        result.is_invalid    = false;
        result.error_message = std::string("getOrCompile: unexpected throw: ") + e.what();
        return result;
    } catch (...) {
        CompileResult result;
        result.compiled      = nullptr;
        result.is_invalid    = false;
        result.error_message = "getOrCompile: unknown throw";
        return result;
    }
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
    compiler_.ClearCache();
    RTP_LLM_LOG_INFO("XGrammarBackend clear: compiler cache dropped");
}

}  // namespace rtp_llm
