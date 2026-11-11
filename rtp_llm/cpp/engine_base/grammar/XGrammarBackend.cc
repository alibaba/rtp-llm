#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "absl/status/status.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

std::string serializeJson(const rapidjson::Value& value) {
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    return {buffer.GetString(), buffer.GetSize()};
}

// CompileStructuralTag does not take CompileJSONSchema's whitespace option.
// Use the same JSON compiler for schemas inside reasoning/tool envelopes. An
// explicit per-node whitespace cap remains a request-level formatting choice.
std::string withFixedJsonWhitespace(const std::string& structural_tag, bool strict_mode) {
    rapidjson::Document document;
    document.Parse<rapidjson::kParseIterativeFlag>(structural_tag.data(), structural_tag.size());
    if (document.HasParseError()) {
        throw std::runtime_error("structural_tag must be valid JSON");
    }
    if (!document.IsObject() || !document.HasMember("format")) {
        // Leave legacy structural tags and input validation to XGrammar.
        return structural_tag;
    }

    bool                           changed = false;
    std::vector<rapidjson::Value*> pending{&document["format"]};
    while (!pending.empty()) {
        auto& node = *pending.back();
        pending.pop_back();
        if (node.IsArray()) {
            for (auto& child : node.GetArray()) {
                pending.push_back(&child);
            }
            continue;
        }
        if (!node.IsObject()) {
            continue;
        }
        const auto type = node.FindMember("type");
        if (type != node.MemberEnd() && type->value.IsString()
            && std::string(type->value.GetString(), type->value.GetStringLength()) == "json_schema") {
            const auto schema = node.FindMember("json_schema");
            const auto style  = node.FindMember("style");
            const auto cap    = node.FindMember("max_whitespace_cnt");
            const auto order  = node.FindMember("any_order");
            if (schema != node.MemberEnd() && (schema->value.IsObject() || schema->value.IsBool())
                && (style == node.MemberEnd()
                    || (style->value.IsString()
                        && std::string(style->value.GetString(), style->value.GetStringLength()) == "json"))
                && (cap == node.MemberEnd() || cap->value.IsNull())
                && (order == node.MemberEnd() || order->value.IsBool())) {
                const auto grammar =
                    xgrammar::Grammar::FromJSONSchema(serializeJson(schema->value),
                                                      /*any_whitespace=*/false,
                                                      std::nullopt,
                                                      std::nullopt,
                                                      strict_mode,
                                                      /*max_whitespace_cnt=*/std::nullopt,
                                                      /*print_converted_ebnf=*/false,
                                                      /*any_order=*/order != node.MemberEnd() && order->value.GetBool())
                        .ToString();
                node.SetObject();
                node.AddMember("type", "grammar", document.GetAllocator());
                node.AddMember("grammar",
                               rapidjson::Value(grammar.data(), grammar.size(), document.GetAllocator()),
                               document.GetAllocator());
                changed = true;
            }
            // JSON Schema is opaque data, not structural-tag format nodes.
            continue;
        }
        for (auto& member : node.GetObject()) {
            if (member.name != "json_schema" && member.name != "schema") {
                pending.push_back(&member.value);
            }
        }
    }
    return changed ? serializeJson(document) : structural_tag;
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
        RTP_LLM_LOG_DEBUG("XGrammarBackend compile OK: type=%s, len=%zu, elapsed_ms=%lld, bytes=%zu",
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
                                                         const GrammarConfig& cfg) {
    try {
        if (tokenizer_info_json.empty()) {
            RTP_LLM_LOG_INFO("XGrammarBackend::create: structured output disabled (TokenizerInfo empty)");
            return nullptr;
        }
        Options opts   = optionsFromConfig(cfg);
        auto    result = xgrammar::TokenizerInfo::DeserializeJSON(tokenizer_info_json);
        if (std::holds_alternative<xgrammar::SerializationError>(result)) {
            throw std::runtime_error("tokenizer info deserialize failed: "
                                     + serializationErrorToString(std::get<xgrammar::SerializationError>(result)));
        }
        const auto& serialized_tokenizer_info = std::get<xgrammar::TokenizerInfo>(result);
        // xgrammar derives its token-id lookup in the constructor but does not serialize it.
        // Rebuild from the already-decoded vocabulary so token-level grammar works after the
        // Python-to-C++ JSON boundary without decoding BYTE_LEVEL/BYTE_FALLBACK tokens twice.
        const xgrammar::TokenizerInfo tokenizer_info(serialized_tokenizer_info.GetDecodedVocab(),
                                                     xgrammar::VocabType::RAW,
                                                     serialized_tokenizer_info.GetVocabSize(),
                                                     serialized_tokenizer_info.GetStopTokenIds(),
                                                     serialized_tokenizer_info.GetAddPrefixSpace());
        if (tokenizer_info.GetVocabSize() <= 0) {
            throw std::runtime_error("tokenizer vocab is empty");
        }
        auto backend = std::shared_ptr<XGrammarBackend>(new XGrammarBackend(tokenizer_info, opts));
        RTP_LLM_LOG_INFO("XGrammarBackend::create: ready with serialized TokenizerInfo "
                         "(json_bytes=%zu, threads=%d)",
                         tokenizer_info_json.size(),
                         opts.max_compiler_threads);
        return backend;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("XGrammarBackend::create: initialization failed (%s); aborting startup", e.what());
        throw;
    } catch (...) {
        RTP_LLM_LOG_ERROR("XGrammarBackend::create: initialization failed (unknown); aborting startup");
        throw;
    }
}

XGrammarBackend::Options XGrammarBackend::optionsFromConfig(const GrammarConfig& cfg) {
    Options opts;
    opts.any_whitespace               = !cfg.constrained_json_disable_any_whitespace;
    opts.strict_mode                  = true;
    opts.terminate_without_stop_token = cfg.terminate_without_stop_token;
    opts.max_compiler_threads         = std::max(1, cfg.num_workers);
    opts.compiler_cache_bytes         = cfg.compiler_cache_bytes > 0 ? cfg.compiler_cache_bytes : -1;
    return opts;
}

XGrammarBackend::XGrammarBackend(const xgrammar::TokenizerInfo&  tokenizer_info,
                                 const XGrammarBackend::Options& options):
    options_(options),
    tokenizer_info_(tokenizer_info),
    compiler_(tokenizer_info_,
              std::max(1, options.max_compiler_threads),
              /*enable_cache=*/true,
              options.compiler_cache_bytes) {
    RTP_LLM_LOG_INFO("XGrammarBackend init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, "
                     "terminate_without_stop_token=%d, compiler_threads=%d, compiler_cache_bytes=%lld",
                     tokenizer_info_.GetVocabSize(),
                     static_cast<int>(options_.any_whitespace),
                     static_cast<int>(options_.strict_mode),
                     static_cast<int>(options_.terminate_without_stop_token),
                     std::max(1, options_.max_compiler_threads),
                     static_cast<long long>(options_.compiler_cache_bytes));
}

XGrammarBackend::~XGrammarBackend() = default;

// Thread-safe via xgrammar::GrammarCompiler's internal cache.
absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> XGrammarBackend::compile(const GrammarKeyCpp& key) {
    const auto  t_start = std::chrono::steady_clock::now();
    const auto& s       = key.key_string;

    absl::StatusOr<std::shared_ptr<xgrammar::CompiledGrammar>> result =
        absl::InvalidArgumentError("Unknown grammar key_type: " + key.key_type);
    if (key.key_type == "json") {
        result = compileWithErrorClassification([&] {
            return compiler_.CompileJSONSchema(
                s, options_.any_whitespace, std::nullopt, std::nullopt, options_.strict_mode);
        });
    } else if (key.key_type == "regex") {
        result = compileWithErrorClassification([&] { return compiler_.CompileRegex(s); });
    } else if (key.key_type == "ebnf") {
        result = compileWithErrorClassification([&] { return compiler_.CompileGrammar(s); });
    } else if (key.key_type == "structural_tag") {
        result = compileWithErrorClassification([&] {
            return compiler_.CompileStructuralTag(
                options_.any_whitespace ? s : withFixedJsonWhitespace(s, options_.strict_mode));
        });
    }

    logCompileResult(key, result, elapsedMsSince(t_start));
    return result;
}

absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>> XGrammarBackend::createMatcherFromKey(const GrammarKeyCpp& key) {
    auto compiled_or = compile(key);
    if (!compiled_or.ok()) {
        const std::string err = compiled_or.status().message().empty() ? "unknown compile error" :
                                                                         std::string(compiled_or.status().message());
        return absl::Status(compiled_or.status().code(), "Failed to compile " + key.key_type + " grammar: " + err);
    }

    auto matcher_or = createMatcher(std::move(compiled_or.value()));
    if (!matcher_or.ok()) {
        return matcher_or.status();
    }
    return matcher_or.value();
}

absl::StatusOr<std::shared_ptr<RtpGrammarMatcher>>
XGrammarBackend::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled) {
    if (!compiled) {
        return absl::InvalidArgumentError("createMatcher requires a non-null CompiledGrammar");
    }
    try {
        return std::make_shared<RtpGrammarMatcher>(std::move(compiled), options_.terminate_without_stop_token);
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(std::string("grammar matcher install failed: ") + e.what());
    } catch (...) {
        const auto error = absl::UnknownError("grammar matcher install failed: unknown");
        return error;
    }
}

}  // namespace rtp_llm
