
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

TEST_F(PrefixToCandidateTokensTest, SAXNegativeTokenIds) {
    std::string json = R"({
        "start_token_id": -1, "end_token_id": -2,
        "prefix_dict": {"neg": [-10, 5]}
    })";
    writeJsonFile(tmp_path_, json);

    auto inst = PrefixToCandidateTokens::instance();
    inst->reloadPrefixDict(tmp_path_);
    ASSERT_TRUE(inst->initSuccess());
    ASSERT_EQ(-1, inst->startTokenId());
    ASSERT_EQ(-2, inst->endTokenId());

    std::unordered_set<int32_t> expected = {-10, 5};
    ASSERT_EQ(expected, inst->getCandidateTokens("neg"));
}

}  // namespace rtp_llm
