#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
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

template<typename CompileFn>
absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> compileWithErrorClassification(CompileFn compile_fn) {
    try {
        return std::make_shared<xgrammar::CompiledGrammar>(compile_fn());
    } catch (const std::bad_alloc& e) {
        return absl::ResourceExhaustedError(std::string("system error (retryable): ") + e.what());
    } catch (const std::runtime_error& e) {
        return absl::InvalidArgumentError(e.what());
    } catch (const std::exception& e) {
        return absl::UnknownError(std::string("unexpected error (retryable): ") + e.what());
    }
}

int64_t elapsedMsSince(std::chrono::steady_clock::time_point t_start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();
}

std::string serializationErrorToString(const xgrammar::SerializationError& error) {
    return std::visit([](const auto& e) { return e.GetType() + ": " + std::string(e.what()); }, error);
}

void logCompileResult(const GrammarKeyCpp&                                              key,
                      const absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>>& result,
                      int64_t                                                           elapsed_ms) {
    if (result.ok()) {
        RTP_LLM_LOG_INFO("XGrammarBackend compile OK: type=%s, len=%zu, elapsed_ms=%lld, bytes=%zu",
                         key.key_type.c_str(),
                         key.key_string.size(),
                         static_cast<long long>(elapsed_ms),
                         result.value()->MemorySizeBytes());
        return;
    }

    const std::string error_message = std::string(result.status().message());
    RTP_LLM_LOG_WARNING("XGrammarBackend compile FAIL: type=%s, len=%zu, elapsed_ms=%lld, invalid=%d, err=%s",
                        key.key_type.c_str(),
                        key.key_string.size(),
                        static_cast<long long>(elapsed_ms),
                        static_cast<int>(result.status().code() == absl::StatusCode::kInvalidArgument),
                        error_message.c_str());
}
}  // namespace

std::shared_ptr<XGrammarBackend> XGrammarBackend::create(const std::string&   tokenizer_info_json,
                                                         const GrammarConfig& cfg) noexcept {
    try {
        if (tokenizer_info_json.empty()) {
            RTP_LLM_LOG_INFO("XGrammarBackend::create: structured output disabled (TokenizerInfo empty)");
            return nullptr;
        }
        XGrammarBackendOptions opts   = backendOptionsFromConfig(cfg);
        auto                   result = xgrammar::TokenizerInfo::DeserializeJSON(tokenizer_info_json);
        if (std::holds_alternative<xgrammar::SerializationError>(result)) {
            RTP_LLM_LOG_ERROR("XGrammarBackend::create: tokenizer info deserialize failed (%s); disabling grammar",
                              serializationErrorToString(std::get<xgrammar::SerializationError>(result)).c_str());
            return nullptr;
        }
        const auto& tokenizer_info = std::get<xgrammar::TokenizerInfo>(result);
        auto        backend        = std::make_shared<XGrammarBackend>(tokenizer_info, opts);
        RTP_LLM_LOG_INFO("XGrammarBackend::create: ready with serialized TokenizerInfo "
                         "(json_bytes=%zu, override_stop_tokens=%zu, threads=%d)",
                         tokenizer_info_json.size(),
                         cfg.override_stop_tokens.size(),
                         opts.max_compiler_threads);
        return backend;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("XGrammarBackend::create: build threw (%s); disabling grammar", e.what());
        return nullptr;
    }
}

XGrammarBackend::XGrammarBackend(const xgrammar::TokenizerInfo& tokenizer_info, const XGrammarBackendOptions& options):
    options_(options),
    tokenizer_info_(tokenizer_info),
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
absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> XGrammarBackend::compile(const GrammarKeyCpp& key) {
    const auto  t_start = std::chrono::steady_clock::now();
    const auto& s       = key.key_string;

    absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> result =
        absl::InvalidArgumentError("Unknown grammar key_type: " + key.key_type);
    if (key.key_type == "json") {
        // "$$ANY$$" → any JSON value (response_format=json_object).
        result = compileWithErrorClassification([&] {
            if (s == "$$ANY$$") {
                return compiler_.CompileBuiltinJSONGrammar();
            }
            return compiler_.CompileJSONSchema(
                s, options_.any_whitespace, std::nullopt, std::nullopt, options_.strict_mode);
        });
    } else if (key.key_type == "regex") {
        result = compileWithErrorClassification([&] { return compiler_.CompileRegex(s); });
    } else if (key.key_type == "ebnf") {
        result = compileWithErrorClassification([&] { return compiler_.CompileGrammar(s); });
    } else if (key.key_type == "structural_tag") {
        result = compileWithErrorClassification([&] { return compiler_.CompileStructuralTag(s); });
    }

    logCompileResult(key, result, elapsedMsSince(t_start));
    return result;
}

absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>>
XGrammarBackend::createMatcherFromKey(const GrammarKeyCpp& key, bool terminate_without_stop_token) {
    auto compiled_or = compile(key);
    if (!compiled_or.ok()) {
        const std::string err = compiled_or.status().message().empty() ? "unknown compile error" :
                                                                         std::string(compiled_or.status().message());
        return absl::Status(compiled_or.status().code(), "Failed to compile " + key.key_type + " grammar: " + err);
    }

    auto matcher_or = createMatcher(std::move(compiled_or.value()), terminate_without_stop_token);
    if (!matcher_or.ok()) {
        return matcher_or.status();
    }
    return matcher_or.value();
}

absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>>
XGrammarBackend::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled, bool terminate_without_stop_token) {
    if (!compiled) {
        return absl::InvalidArgumentError("createMatcher requires a non-null CompiledGrammar");
    }
    try {
        return std::make_shared<RtpGrammarMatcher>(
            std::move(compiled), options_.override_stop_tokens, terminate_without_stop_token);
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(std::string("grammar matcher install failed: ") + e.what());
    } catch (...) {
        const auto error = absl::UnknownError("grammar matcher install failed: unknown");
        return error;
    }
}

void XGrammarBackend::clear() {
    compiler_.ClearCache();
    RTP_LLM_LOG_INFO("XGrammarBackend clear: compiler cache dropped");
}

}  // namespace rtp_llm
