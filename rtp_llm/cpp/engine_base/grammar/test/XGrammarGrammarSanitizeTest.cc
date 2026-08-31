#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "autil/legacy/any.h"
#include "autil/legacy/json.h"

namespace rtp_llm {
namespace {

using autil::legacy::Any;
using JsonArray = autil::legacy::json::JsonArray;
using JsonMap   = autil::legacy::json::JsonMap;

// Walks `json` to `path` (a chain of object keys, with an index for each array hop expressed as a key) and
// reports whether the node it lands on carries `key`.
const Any* find(const Any& node, const std::vector<std::string>& path) {
    const Any* cur = &node;
    for (const auto& step : path) {
        if (const auto* arr = autil::legacy::AnyCast<JsonArray>(cur)) {
            const size_t index = std::stoul(step);
            if (index >= arr->size()) {
                return nullptr;
            }
            cur = &(*arr)[index];
            continue;
        }
        const auto* map = autil::legacy::AnyCast<JsonMap>(cur);
        if (!map) {
            return nullptr;
        }
        auto it = map->find(step);
        if (it == map->end()) {
            return nullptr;
        }
        cur = &it->second;
    }
    return cur;
}

bool has(const std::string& json, const std::vector<std::string>& path) {
    Any any;
    autil::legacy::json::ParseJson(json, any);
    return find(any, path) != nullptr;
}

// A structural tag whose json_schema holds string fields with length bounds, one object nested inside another,
// a property literally named `maxLength`, instance data that looks like a bounded string schema, bounds that
// are not numbers, and a same-named key outside any schema context.
const char* kTag = R"JSON({
  "type": "structural_tag",
  "format": {
    "type": "triggered_tags",
    "triggers": ["<tool>"],
    "tags": [{
      "type": "tag",
      "begin": "<tool>",
      "content": {
        "type": "json_schema",
        "json_schema": {
          "type": "object",
          "properties": {
            "action": {"type": "string", "minLength": 1, "maxLength": 8000},
            "nullable": {"type": ["string", "null"], "maxLength": 32},
            "nested": {
              "type": "object",
              "properties": {
                "inner": {"type": "string", "maxLength": 16}
              }
            },
            "keyed": {
              "type": "object",
              "propertyNames": {"maxLength": 24},
              "additionalProperties": {"type": "string"}
            },
            "literal": {"const": {"type": "string", "maxLength": 12}},
            "choice": {"enum": [{"type": "string", "minLength": 3}]},
            "textual": {"type": "string", "maxLength": "8000"},
            "flagged": {"type": "string", "minLength": true},
            "maxLength": {"type": "integer"},
            "count": {"type": "integer", "minimum": 1, "maximum": 8}
          }
        },
        "style": "deepseek_xml"
      },
      "end": "</tool>"
    }]
  },
  "metadata": {"type": "string", "maxLength": 64}
})JSON";

const std::vector<std::string> kSchema = {"format", "tags", "0", "content", "json_schema", "properties"};

TEST(XGrammarGrammarSanitizeTest, StructuralTagLosesStringLengthBoundsAtEveryDepth) {
    const std::string out = XGrammarBackendCpp::sanitizeStructuralTag(kTag);

    auto prop = [&](const char* name, const char* key) {
        auto path = kSchema;
        path.push_back(name);
        path.push_back(key);
        return has(out, path);
    };

    EXPECT_FALSE(prop("action", "minLength"));
    EXPECT_FALSE(prop("action", "maxLength"));
    EXPECT_FALSE(prop("nullable", "maxLength"));

    auto inner = kSchema;
    inner.insert(inner.end(), {"nested", "properties", "inner"});
    auto inner_type  = inner;
    auto inner_bound = inner;
    inner_type.push_back("type");
    inner_bound.push_back("maxLength");
    EXPECT_TRUE(has(out, inner_type));  // the path itself survives, so the absence below is the bound's
    EXPECT_FALSE(has(out, inner_bound));

    // xgrammar infers the string type of a `propertyNames` node instead of reading it, and honours the bound
    // there all the same.
    auto keyed = kSchema;
    keyed.insert(keyed.end(), {"keyed", "propertyNames"});
    auto keyed_bound = keyed;
    keyed_bound.push_back("maxLength");
    EXPECT_TRUE(has(out, keyed));
    EXPECT_FALSE(has(out, keyed_bound));
}

TEST(XGrammarGrammarSanitizeTest, StructuralTagKeepsWhatIsNotAStringLengthBound) {
    const std::string out = XGrammarBackendCpp::sanitizeStructuralTag(kTag);

    auto at = [&](std::vector<std::string> tail) {
        auto path = kSchema;
        path.insert(path.end(), tail.begin(), tail.end());
        return has(out, path);
    };

    // A property whose name happens to be `maxLength`: its value is a schema, not a number, so the numeric
    // gate leaves the field itself declared.
    EXPECT_TRUE(at({"maxLength", "type"}));

    // Same gate, from the other side: a bound that is not a number is not the keyword either.
    EXPECT_TRUE(at({"textual", "maxLength"}));
    EXPECT_TRUE(at({"flagged", "minLength"}));

    // `const` and `enum` carry instances, and xgrammar puts them into the grammar verbatim. Rewriting one
    // would change the literal the model is required to emit, even when it happens to look like a schema.
    EXPECT_TRUE(at({"literal", "const", "maxLength"}));
    EXPECT_TRUE(at({"choice", "enum", "0", "minLength"}));

    // Numeric bounds that are not length bounds stay.
    EXPECT_TRUE(at({"count", "minimum"}));

    // Outside a json_schema value the surrounding DSL owns the names, so nothing there is interpreted as a
    // schema keyword.
    EXPECT_TRUE(has(out, {"metadata", "maxLength"}));
}

TEST(XGrammarGrammarSanitizeTest, LegacyStructureSchemaIsAStringLengthBoundContext) {
    const char* legacy = R"JSON({
      "structures": [{
        "begin": "<tool>",
        "end": "</tool>",
        "schema": {"type": "object", "properties": {"a": {"type": "string", "maxLength": 4}}}
      }]
    })JSON";

    const std::string out = XGrammarBackendCpp::sanitizeStructuralTag(legacy);
    EXPECT_FALSE(has(out, {"structures", "0", "schema", "properties", "a", "maxLength"}));
    EXPECT_TRUE(has(out, {"structures", "0", "schema", "properties", "a", "type"}));
}

TEST(XGrammarGrammarSanitizeTest, MalformedSpecIsHandedToXGrammarUnchanged) {
    // Rejecting a malformed spec is xgrammar's job; sanitizing must not turn a diagnosable compile error into
    // a silently different grammar.
    const std::string broken = "{\"type\": \"structural_tag\", ";
    EXPECT_EQ(XGrammarBackendCpp::sanitizeStructuralTag(broken), broken);
    EXPECT_EQ(XGrammarBackendCpp::sanitizeJsonSchema(broken), broken);
}

TEST(XGrammarGrammarSanitizeTest, JsonSchemaLosesStringLengthBounds) {
    const char* schema = R"JSON({
      "type": "object",
      "properties": {
        "a": {"type": "string", "minLength": 1, "maxLength": 8000},
        "b": {"type": "integer"}
      },
      "required": ["a", "b"]
    })JSON";

    const std::string out = XGrammarBackendCpp::sanitizeJsonSchema(schema);
    EXPECT_FALSE(has(out, {"properties", "a", "minLength"}));
    EXPECT_FALSE(has(out, {"properties", "a", "maxLength"}));
    EXPECT_TRUE(has(out, {"properties", "a", "type"}));
    EXPECT_TRUE(has(out, {"properties", "b", "type"}));
}

TEST(XGrammarGrammarSanitizeTest, UntouchedJsonSchemaKeepsItsOriginalText) {
    // Re-serializing would sort object keys and with it reorder `properties`, which decides the order the
    // schema forces fields to be generated in. A schema with nothing to strip must pass through byte for byte.
    const char* schema = R"JSON({"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}}})JSON";
    EXPECT_EQ(XGrammarBackendCpp::sanitizeJsonSchema(schema), schema);
}

}  // namespace
}  // namespace rtp_llm
