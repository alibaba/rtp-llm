#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class PrefixToCandidateTokensTest: public ::testing::Test {
protected:
    void writeJsonFile(const std::string& path, const std::string& content) {
        std::ofstream f(path);
        f << content;
        f.close();
    }

    std::string tmp_path_ = "/tmp/test_prefix_dict.json";
};

TEST_F(PrefixToCandidateTokensTest, testConfigJsonize) {
    string schemaStr =
        "{\"start_token_id\": 224, \"end_token_id\": 1, \"prefix_dict\": {\"1_2\": [3,4,5], \"1_2_3\": [3,4,5]}}";
    TreeDecodeConfig config;
    autil::legacy::FromJsonString(config, schemaStr);
    ASSERT_EQ(224, config.start_token_id);
    ASSERT_EQ(1, config.end_token_id);
    ASSERT_EQ("_", config.sep);
    ASSERT_EQ((size_t)2, config.prefix_dict.size());
    std::vector<int32_t> vec = {3, 4, 5};
    ASSERT_EQ(vec, config.prefix_dict["1_2"]);
    ASSERT_EQ(vec, config.prefix_dict["1_2_3"]);
}

TEST_F(PrefixToCandidateTokensTest, testReloadFile) {
    std::string file_path = "./rtp_llm/cpp/utils/test/gir_prefix_dict.json";
    PrefixToCandidateTokens::instance()->reloadPrefixDict(file_path);
    ASSERT_EQ(225, PrefixToCandidateTokens::instance()->startTokenId());
    ASSERT_EQ(2, PrefixToCandidateTokens::instance()->endTokenId());
    std::unordered_set<int32_t> vec = {3, 4, 5};
    ASSERT_EQ(vec, PrefixToCandidateTokens::instance()->getCandidateTokens("1_2"));
    ASSERT_EQ(vec, PrefixToCandidateTokens::instance()->getCandidateTokens("1_2_3"));
}

TEST_F(PrefixToCandidateTokensTest, SAXLegacyEquivalence) {
    std::string json = R"({
        "start_token_id": 100, "end_token_id": 200, "sep": "-",
        "prefix_dict": {"a-b": [10,20,30], "a-b-c": [40,50]}
    })";
    writeJsonFile(tmp_path_, json);

    // Load via streaming SAX parser
    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess());
    ASSERT_EQ(100, inst->startTokenId());
    ASSERT_EQ(200, inst->endTokenId());

    std::unordered_set<int32_t> expected_ab  = {10, 20, 30};
    std::unordered_set<int32_t> expected_abc = {40, 50};
    ASSERT_EQ(expected_ab, inst->getCandidateTokens("a-b"));
    ASSERT_EQ(expected_abc, inst->getCandidateTokens("a-b-c"));

    // Load via legacy parser for comparison
    TreeDecodeConfig legacy_config;
    autil::legacy::FromJsonString(legacy_config, json);
    ASSERT_EQ(inst->startTokenId(), legacy_config.start_token_id);
    ASSERT_EQ(inst->endTokenId(), legacy_config.end_token_id);
    for (auto& [key, vec] : legacy_config.prefix_dict) {
        std::unordered_set<int32_t> legacy_set(vec.begin(), vec.end());
        ASSERT_EQ(legacy_set, inst->getCandidateTokens(key)) << "Mismatch at key: " << key;
    }
}

TEST_F(PrefixToCandidateTokensTest, SAXEmptyArray) {
    std::string json = R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"leaf": [], "normal": [10,20]}
    })";
    writeJsonFile(tmp_path_, json);

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess());

    ASSERT_TRUE(inst->isValidStatus("leaf")) << "Empty array key should be a valid status";
    ASSERT_TRUE(inst->getCandidateTokens("leaf").empty());

    std::unordered_set<int32_t> expected = {10, 20};
    ASSERT_EQ(expected, inst->getCandidateTokens("normal"));
}

TEST_F(PrefixToCandidateTokensTest, SAXDuplicateTokens) {
    std::string json = R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"dup": [5, 5, 3, 3, 5]}
    })";
    writeJsonFile(tmp_path_, json);

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess());

    std::unordered_set<int32_t> expected = {3, 5};
    ASSERT_EQ(expected, inst->getCandidateTokens("dup"));
}

TEST_F(PrefixToCandidateTokensTest, SAXInvalidJsonFallback) {
    std::string bad_json = R"({ "start_token_id": 1, "end_token_id": 2, INVALID })";
    writeJsonFile(tmp_path_, bad_json);

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    // SAX fails, legacy also fails on malformed JSON → init should fail
    ASSERT_FALSE(inst->initSuccess());
}

TEST_F(PrefixToCandidateTokensTest, SAXLargeTokenIdOverflow) {
    // Uint64 value exceeding int32_t max should be skipped
    std::string json = R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"overflow": [10, 3000000000, 20]}
    })";
    writeJsonFile(tmp_path_, json);

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess());

    auto tokens = inst->getCandidateTokens("overflow");
    ASSERT_TRUE(tokens.count(10));
    ASSERT_TRUE(tokens.count(20));
    ASSERT_EQ(2u, tokens.size()) << "Overflowed value 3000000000 should be skipped";
}

// Negative candidate token ids are invalid: they would be used directly as logits/vocab indices and
// throw at logits time (DFAUtil::getCandidateTokenIds). The SAX parser must skip them while keeping
// the load successful and the remaining valid ids intact.
TEST_F(PrefixToCandidateTokensTest, SAXNegativeCandidateTokensSkipped) {
    std::string json = R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"neg": [-10, 5, -3, 7], "all_neg": [-1, -2]}
    })";
    writeJsonFile(tmp_path_, json);

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess());

    std::unordered_set<int32_t> expected = {5, 7};
    ASSERT_EQ(expected, inst->getCandidateTokens("neg")) << "negative candidate ids must be skipped";

    // A key whose candidates are all negative remains a valid status with an empty candidate set.
    ASSERT_TRUE(inst->isValidStatus("all_neg"));
    ASSERT_TRUE(inst->getCandidateTokens("all_neg").empty());
}

// The following cases are syntactically valid JSON but structurally invalid for a
// tree_decode_config. The SAX parser must reject them (schema/type violation), and since the
// legacy fallback parser is also unable to map them onto TreeDecodeConfig, init must fail rather
// than silently report success with garbage/partial data.

TEST_F(PrefixToCandidateTokensTest, SAXRootNotObject) {
    writeJsonFile(tmp_path_, R"([1, 2, 3])");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "A non-object root must not be treated as a valid config";
}

TEST_F(PrefixToCandidateTokensTest, SAXRootScalar) {
    writeJsonFile(tmp_path_, R"(42)");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "A scalar root must not be treated as a valid config";
}

TEST_F(PrefixToCandidateTokensTest, SAXPrefixDictNotObject) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": [1, 2, 3]
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "prefix_dict must be an object, not an array";
}

TEST_F(PrefixToCandidateTokensTest, SAXPrefixDictValueNotArray) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"k": 5}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "prefix_dict value must be an array, not a scalar";
}

TEST_F(PrefixToCandidateTokensTest, SAXPrefixDictValueObject) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"k": {"nested": 1}}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "prefix_dict value must be an array, not an object";
}

TEST_F(PrefixToCandidateTokensTest, SAXPrefixDictArrayElementNotInt) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"k": [1, "x", 3]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "prefix_dict array elements must be integers";
}

// Per-key value-type checks: start_token_id/end_token_id must be integers, sep must be a string,
// prefix_dict must be an object. A type mismatch must fail init deterministically (no legacy paper-over).

TEST_F(PrefixToCandidateTokensTest, SAXStartTokenIdString) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": "abc", "end_token_id": 2,
        "prefix_dict": {"k": [1, 2]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "start_token_id must be an integer, not a string";
}

TEST_F(PrefixToCandidateTokensTest, SAXEndTokenIdBool) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": true,
        "prefix_dict": {"k": [1, 2]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "end_token_id must be an integer, not a bool";
}

TEST_F(PrefixToCandidateTokensTest, SAXStartTokenIdObject) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": {"x": 1}, "end_token_id": 2,
        "prefix_dict": {"k": [1, 2]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "start_token_id must be an integer, not an object";
}

TEST_F(PrefixToCandidateTokensTest, SAXSepNotString) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2, "sep": 5,
        "prefix_dict": {"k": [1, 2]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "sep must be a string, not a number";
}

TEST_F(PrefixToCandidateTokensTest, SAXPrefixDictNumber) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": 5
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "prefix_dict must be an object, not a number";
}

// P2: a nested array inside a prefix_dict value is not a valid (flat) candidate-token list.
TEST_F(PrefixToCandidateTokensTest, SAXNestedArray) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2,
        "prefix_dict": {"k": [[1, 2], [3]]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "nested arrays must not be accepted as a candidate token list";
}

// P1: the legacy parser silently ignores unknown root-level fields. The SAX parser must keep that
// backward compatibility by skipping unknown container fields (objects/arrays) instead of failing.

TEST_F(PrefixToCandidateTokensTest, SAXUnknownRootObjectFieldTolerated) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 100, "end_token_id": 200,
        "metadata": {"author": "x", "version": 3, "nested": {"a": [1, 2], "b": "str"}},
        "prefix_dict": {"k": [10, 20]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess()) << "unknown root object field must be skipped, not rejected";
    ASSERT_EQ(100, inst->startTokenId());
    ASSERT_EQ(200, inst->endTokenId());

    std::unordered_set<int32_t> expected = {10, 20};
    ASSERT_EQ(expected, inst->getCandidateTokens("k"));
    // The skipped subtree's inner keys must not leak into the prefix map.
    ASSERT_FALSE(inst->isValidStatus("nested"));
    ASSERT_FALSE(inst->isValidStatus("author"));
}

TEST_F(PrefixToCandidateTokensTest, SAXUnknownRootArrayFieldTolerated) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": 2,
        "extra_list": [1, 2, [3, 4], {"x": 5}],
        "prefix_dict": {"k": [10, 20]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess()) << "unknown root array field must be skipped, not rejected";

    std::unordered_set<int32_t> expected = {10, 20};
    ASSERT_EQ(expected, inst->getCandidateTokens("k"));
}

TEST_F(PrefixToCandidateTokensTest, SAXUnknownRootFieldBeforeAndAfter) {
    // Unknown container fields placed both before and after the known fields must not corrupt parsing.
    writeJsonFile(tmp_path_, R"({
        "lead": {"deep": {"deeper": [1, 2, 3]}},
        "start_token_id": 7, "end_token_id": 8,
        "prefix_dict": {"a": [1], "b": [2, 3]},
        "trail": [{"k": "v"}, [9, [10]]]
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess());
    ASSERT_EQ(7, inst->startTokenId());
    ASSERT_EQ(8, inst->endTokenId());

    std::unordered_set<int32_t> expected_a = {1};
    std::unordered_set<int32_t> expected_b = {2, 3};
    ASSERT_EQ(expected_a, inst->getCandidateTokens("a"));
    ASSERT_EQ(expected_b, inst->getCandidateTokens("b"));
}

// P2: an out-of-range start_token_id/end_token_id must fail deterministically instead of silently
// falling back to the default token id. Candidate token arrays keep the lenient skip behavior
// (covered by SAXLargeTokenIdOverflow above).

TEST_F(PrefixToCandidateTokensTest, SAXStartTokenIdOverflow) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 3000000000, "end_token_id": 2,
        "prefix_dict": {"k": [1, 2]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "out-of-range start_token_id must be a schema error";
}

TEST_F(PrefixToCandidateTokensTest, SAXEndTokenIdOverflow) {
    writeJsonFile(tmp_path_, R"({
        "start_token_id": 1, "end_token_id": -3000000000,
        "prefix_dict": {"k": [1, 2]}
    })");

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_FALSE(inst->initSuccess()) << "out-of-range end_token_id must be a schema error";
}

}  // namespace rtp_llm
