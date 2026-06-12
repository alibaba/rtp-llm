#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarSchemaValidator.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBackend.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"

#include <functional>
#include <string>
#include <vector>

namespace rtp_llm {

namespace {

static constexpr size_t kMaxSchemaSize = 1024 * 1024;  // 1 MiB
static constexpr int    kMaxRefDepth   = 32;

using JsonMap   = autil::legacy::json::JsonMap;
using JsonArray = autil::legacy::json::JsonArray;

struct UnsupportedFeature {
    std::string path;
    std::string keyword;
};

bool hasKey(const JsonMap& map, const std::string& key) {
    return map.find(key) != map.end();
}

void walkSchema(const autil::legacy::Any& node,
                const std::string& path,
                int depth,
                std::vector<UnsupportedFeature>& issues) {
    if (depth > kMaxRefDepth) {
        issues.push_back({path, "$ref recursion depth exceeded"});
        return;
    }

    auto* map = autil::legacy::AnyCast<JsonMap>(&const_cast<autil::legacy::Any&>(node));
    if (!map) {
        return;
    }

    // number/integer unsupported keywords
    if (hasKey(*map, "multipleOf")) {
        issues.push_back({path, "multipleOf"});
    }

    // array unsupported keywords
    if (hasKey(*map, "uniqueItems")) {
        issues.push_back({path, "uniqueItems"});
    }
    if (hasKey(*map, "contains")) {
        issues.push_back({path, "contains"});
    }
    if (hasKey(*map, "minContains")) {
        issues.push_back({path, "minContains"});
    }
    if (hasKey(*map, "maxContains")) {
        issues.push_back({path, "maxContains"});
    }

    // object unsupported keywords
    if (hasKey(*map, "patternProperties")) {
        issues.push_back({path, "patternProperties"});
    }
    if (hasKey(*map, "propertyNames")) {
        issues.push_back({path, "propertyNames"});
    }

    // Recurse into sub-schemas
    auto recurse = [&](const std::string& key) {
        auto it = map->find(key);
        if (it != map->end()) {
            walkSchema(it->second, path + "/" + key, depth + 1, issues);
        }
    };

    recurse("items");
    recurse("additionalProperties");
    recurse("not");

    // properties
    auto props_it = map->find("properties");
    if (props_it != map->end()) {
        auto* props_map = autil::legacy::AnyCast<JsonMap>(&const_cast<autil::legacy::Any&>(props_it->second));
        if (props_map) {
            for (auto& kv : *props_map) {
                walkSchema(kv.second, path + "/properties/" + kv.first, depth + 1, issues);
            }
        }
    }

    // allOf / anyOf / oneOf
    for (const auto& combiner : {"allOf", "anyOf", "oneOf"}) {
        auto cit = map->find(combiner);
        if (cit != map->end()) {
            auto* arr = autil::legacy::AnyCast<JsonArray>(&const_cast<autil::legacy::Any&>(cit->second));
            if (arr) {
                for (size_t i = 0; i < arr->size(); ++i) {
                    walkSchema((*arr)[i], path + "/" + combiner + "/" + std::to_string(i), depth + 1, issues);
                }
            }
        }
    }

    // $defs / definitions
    for (const auto& defs_key : {"$defs", "definitions"}) {
        auto dit = map->find(defs_key);
        if (dit != map->end()) {
            auto* defs_map = autil::legacy::AnyCast<JsonMap>(&const_cast<autil::legacy::Any&>(dit->second));
            if (defs_map) {
                for (auto& kv : *defs_map) {
                    walkSchema(kv.second, path + "/" + defs_key + "/" + kv.first, depth + 1, issues);
                }
            }
        }
    }
}

static constexpr size_t kMaxRegexEbnfSize = 64 * 1024;  // 64 KiB
// structural_tag is deserialized via autil::legacy::json::ParseJson and then
// recursively walked by sanitizeStructuralFormat; both are recursive and have
// no built-in depth limit. A user-controlled payload of pure-`[` nesting
// within the 64 KiB byte cap can still reach ~32k levels and overflow the
// worker thread's 8 MiB stack, sending SIGSEGV — uncatchable by the worker's
// catch(...). Reject obviously-deep payloads at admission so the worker
// thread never sees them. 256 levels is well above any realistic structural
// schema (sequence/or/tag/triggered_tags trees rarely exceed ~10 levels).
static constexpr int kMaxStructuralTagDepth = 256;

GrammarValidateResult scanStructuralTagDepth(const std::string& s) {
    int depth     = 0;
    int max_depth = 0;
    bool in_str  = false;
    bool escape  = false;
    for (char c : s) {
        if (in_str) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_str = false;
            }
            continue;
        }
        if (c == '"') {
            in_str = true;
        } else if (c == '{' || c == '[') {
            ++depth;
            if (depth > max_depth) max_depth = depth;
            if (depth > kMaxStructuralTagDepth) {
                return {GrammarValidateStatus::TooLarge,
                        "structural_tag nesting exceeds maximum depth ("
                            + std::to_string(kMaxStructuralTagDepth) + ")"};
            }
        } else if (c == '}' || c == ']') {
            if (depth > 0) --depth;
        }
    }
    return {GrammarValidateStatus::Ok, ""};
}

GrammarValidateResult validateRegexEbnfKeyLightweight(const GrammarKeyCpp& key) {
    if (key.key_string.size() > kMaxRegexEbnfSize) {
        return {GrammarValidateStatus::TooLarge,
                "regex/ebnf grammar exceeds maximum size limit (64 KiB)"};
    }
    if (key.key_string.empty()) {
        return {GrammarValidateStatus::InvalidSyntax, "empty grammar string"};
    }
    if (key.key_type == "structural_tag") {
        auto depth_result = scanStructuralTagDepth(key.key_string);
        if (depth_result.status != GrammarValidateStatus::Ok) {
            return depth_result;
        }
    }
    return {GrammarValidateStatus::Ok, ""};
}

}  // namespace

GrammarValidateResult validateGrammarKey(const GrammarKeyCpp& key) {
    if (key.empty()) {
        return {GrammarValidateStatus::Ok, ""};
    }

    // regex/ebnf syntax compilation runs in the async worker (compileNow) so
    // compile_timeout_ms and singleflight protect the request path.
    if (key.key_type != "json") {
        return validateRegexEbnfKeyLightweight(key);
    }

    if (key.key_string.size() > kMaxSchemaSize) {
        RTP_LLM_LOG_INFO("[grammar_schema_validate] rejected: schema too large (%zu bytes)",
                         key.key_string.size());
        return {GrammarValidateStatus::TooLarge,
                "JSON schema exceeds maximum size limit (1 MiB)"};
    }

    // "$$ANY$$" is the response_format=json_object sentinel; not a JSON Schema document.
    if (key.key_string == "$$ANY$$") {
        return {GrammarValidateStatus::Ok, ""};
    }

    // Parse JSON
    autil::legacy::Any root;
    try {
        autil::legacy::FastFromJsonString(root, key.key_string);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_INFO("[grammar_schema_validate] rejected: invalid JSON syntax: %s", e.what());
        return {GrammarValidateStatus::InvalidSyntax,
                std::string("invalid JSON schema syntax: ") + e.what()};
    }

    // Walk for unsupported features
    std::vector<UnsupportedFeature> issues;
    walkSchema(root, "", 0, issues);

    if (!issues.empty()) {
        std::string detail = "unsupported JSON Schema feature(s): ";
        for (size_t i = 0; i < issues.size() && i < 3; ++i) {
            if (i > 0) detail += ", ";
            detail += "'" + issues[i].keyword + "'";
            if (!issues[i].path.empty()) {
                detail += " at " + issues[i].path;
            }
        }
        if (issues.size() > 3) {
            detail += " (and " + std::to_string(issues.size() - 3) + " more)";
        }
        RTP_LLM_LOG_INFO("[grammar_schema_validate] rejected key=%s: %s",
                         key.brief().c_str(), detail.c_str());
        return {GrammarValidateStatus::UnsupportedFeature, detail};
    }

    return {GrammarValidateStatus::Ok, ""};
}

}  // namespace rtp_llm
