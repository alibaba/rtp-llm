#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <variant>

#include <xgrammar/exception.h>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

using JsonArray = autil::legacy::json::JsonArray;
using JsonMap   = autil::legacy::json::JsonMap;

void sanitizeStructuralFormat(autil::legacy::Any& any) {
    auto* map = autil::legacy::AnyCast<JsonMap>(&any);
    if (!map) {
        return;
    }

    std::string fmt_type;
    if (auto type_it = map->find("type"); type_it != map->end()) {
        if (auto* s = autil::legacy::AnyCast<std::string>(&type_it->second)) {
            fmt_type = *s;
        }
    }

    if ((fmt_type == "json_schema" || fmt_type == "qwen_xml_parameter") && map->find("json_schema") == map->end()) {
        (*map)["json_schema"] = JsonMap{};
    }

    if (fmt_type == "tag") {
        if (auto it = map->find("content"); it != map->end()) {
            sanitizeStructuralFormat(it->second);
        }
    } else if (fmt_type == "sequence" || fmt_type == "or") {
        if (auto it = map->find("elements"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& el : *arr) {
                    sanitizeStructuralFormat(el);
                }
            }
        }
    } else if (fmt_type == "triggered_tags" || fmt_type == "tags_with_separator") {
        if (auto it = map->find("tags"); it != map->end()) {
            if (auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second)) {
                for (auto& tag : *arr) {
                    sanitizeStructuralFormat(tag);
                }
            }
        }
    }
}

void sanitizeLegacyStructures(JsonMap& root) {
    auto it = root.find("structures");
    if (it == root.end()) {
        return;
    }
    auto* arr = autil::legacy::AnyCast<JsonArray>(&it->second);
    if (!arr) {
        return;
    }
    for (auto& item : *arr) {
        if (auto* map = autil::legacy::AnyCast<JsonMap>(&item)) {
            if (map->find("schema") == map->end()) {
                (*map)["schema"] = JsonMap{};
            }
        }
    }
}

}  // namespace

std::string XGrammarBackendCpp::sanitizeStructuralTag(const std::string& tag_json) {
    autil::legacy::Any any;
    try {
        autil::legacy::json::ParseJson(tag_json, any);
    } catch (...) {
        return tag_json;
    }
    auto* root = autil::legacy::AnyCast<JsonMap>(&any);
    if (!root) {
        return tag_json;
    }
    if (root->find("structures") != root->end()) {
        sanitizeLegacyStructures(*root);
    } else if (auto fmt = root->find("format"); fmt != root->end()) {
        sanitizeStructuralFormat(fmt->second);
    }
    try {
        return autil::legacy::json::ToString(any, true);
    } catch (...) {
        return tag_json;
    }
}

XGrammarBackendCpp::XGrammarBackendCpp(const std::string& tokenizer_info_json, const XGrammarBackendOptions& options):
    options_(options),
    tokenizer_info_([&] {
        auto result = xgrammar::TokenizerInfo::DeserializeJSON(tokenizer_info_json);
        if (std::holds_alternative<xgrammar::TokenizerInfo>(result)) {
            return std::get<xgrammar::TokenizerInfo>(std::move(result));
        }
        auto error = std::get<xgrammar::SerializationError>(result);
        throw std::runtime_error(std::string("XGrammarBackendCpp: failed to deserialize TokenizerInfo: ")
                                 + std::visit([](const auto& e) { return std::string(e.what()); }, error));
    }()),
    compiler_(tokenizer_info_,
              std::max(1, options.max_compiler_threads),
              options.enable_compiler_cache,
              options.compiler_cache_bytes) {
    RTP_LLM_LOG_INFO("XGrammarBackendCpp init: vocab_size=%d, any_whitespace=%d, strict_mode=%d, compiler_threads=%d",
                     tokenizer_info_.GetVocabSize(),
                     static_cast<int>(options_.any_whitespace),
                     static_cast<int>(options_.strict_mode),
                     std::max(1, options_.max_compiler_threads));
}

XGrammarBackendCpp::~XGrammarBackendCpp() = default;

std::shared_ptr<xgrammar::CompiledGrammar> XGrammarBackendCpp::getCached(const GrammarKeyCpp& key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto                        it = cache_.find(key.id());
    return it == cache_.end() ? nullptr : it->second;
}

std::string XGrammarBackendCpp::getCachedInvalid(const GrammarKeyCpp& key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto                        it = invalid_cache_.find(key.id());
    return it == invalid_cache_.end() ? std::string() : it->second;
}

void XGrammarBackendCpp::setCache(const GrammarKeyCpp& key, std::shared_ptr<xgrammar::CompiledGrammar> compiled) {
    if (!compiled) {
        return;
    }
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_[key.id()] = std::move(compiled);
    invalid_cache_.erase(key.id());
}

void XGrammarBackendCpp::setCacheInvalid(const GrammarKeyCpp& key, const std::string& error_message) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    invalid_cache_[key.id()] = error_message;
    cache_.erase(key.id());
}

CompileResult XGrammarBackendCpp::compileNow(const GrammarKeyCpp& key) {
    auto compile_with = [](auto&& fn) -> CompileResult {
        CompileResult out;
        try {
            out.compiled = std::make_shared<xgrammar::CompiledGrammar>(fn());
        } catch (const std::runtime_error& e) {
            out.is_invalid    = true;
            out.error_message = e.what();
        }
        return out;
    };

    const auto    begin = std::chrono::steady_clock::now();
    CompileResult result;
    const auto&   grammar = key.key_string;
    if (key.key_type == "json") {
        result = compile_with([&] {
            return grammar == "$$ANY$$" ?
                       compiler_.CompileBuiltinJSONGrammar() :
                       compiler_.CompileJSONSchema(
                           grammar, options_.any_whitespace, std::nullopt, std::nullopt, options_.strict_mode);
        });
    } else if (key.key_type == "regex") {
        result = compile_with([&] { return compiler_.CompileRegex(grammar); });
    } else if (key.key_type == "ebnf") {
        result = compile_with([&] { return compiler_.CompileGrammar(grammar); });
    } else if (key.key_type == "structural_tag") {
        result = compile_with([&] { return compiler_.CompileStructuralTag(sanitizeStructuralTag(grammar)); });
    } else {
        result.is_invalid    = true;
        result.error_message = "unknown grammar type: " + key.key_type;
    }

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
    if (result.compiled) {
        RTP_LLM_LOG_DEBUG("xgrammar compile ok: type=%s, len=%zu, elapsed_ms=%lld",
                          key.key_type.c_str(),
                          key.key_string.size(),
                          static_cast<long long>(elapsed_ms));
    } else {
        RTP_LLM_LOG_WARNING("xgrammar compile failed: type=%s, len=%zu, elapsed_ms=%lld, err=%s",
                            key.key_type.c_str(),
                            key.key_string.size(),
                            static_cast<long long>(elapsed_ms),
                            result.error_message.c_str());
    }
    return result;
}

std::shared_ptr<RtpGrammarMatcher>
XGrammarBackendCpp::createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                  bool                                       require_reasoning,
                                  std::optional<std::vector<int>>            think_end_token_ids) {
    return std::make_shared<RtpGrammarMatcher>(std::move(compiled),
                                               require_reasoning,
                                               std::move(think_end_token_ids),
                                               options_.override_stop_tokens,
                                               /*max_rollback_tokens=*/200);
}

void XGrammarBackendCpp::clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    invalid_cache_.clear();
    compiler_.ClearCache();
}

}  // namespace rtp_llm
