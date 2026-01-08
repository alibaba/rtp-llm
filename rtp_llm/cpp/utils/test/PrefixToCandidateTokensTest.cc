
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class DFAUtilTest: public ::testing::Test {
protected:
};

TEST_F(DFAUtilTest, testConfigJsonize) {
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

TEST_F(DFAUtilTest, testReloadFile) {
    std::string file_path = "./rtp_llm/cpp/utils/test/gir_prefix_dict.json";
    PrefixToCandidateTokens::instance()->reloadPrefixDict(file_path);
    ASSERT_EQ(225, PrefixToCandidateTokens::instance()->startTokenId());
    ASSERT_EQ(2, PrefixToCandidateTokens::instance()->endTokenId());
    std::unordered_set<int32_t> vec = {3, 4, 5};
    ASSERT_EQ(vec, PrefixToCandidateTokens::instance()->getCandidateTokens("1_2"));
    ASSERT_EQ(vec, PrefixToCandidateTokens::instance()->getCandidateTokens("1_2_3"));

    TokenWeights tokenWeights;
    tokenWeights.token_ids                                         = {3, 4, 5};
    tokenWeights.weights                                           = {0.2f, 0.3f, 0.5f};
    std::unordered_map<std::string, TokenWeights>        weightMap = {{"1_2", tokenWeights}};
    const std::unordered_map<std::string, TokenWeights>& resultMap =
        PrefixToCandidateTokens::instance()->getWeightDict();
    auto it = resultMap.find("1_2");
    ASSERT_TRUE(it != resultMap.end());
    ASSERT_EQ(tokenWeights.token_ids, it->second.token_ids);
    ASSERT_EQ(tokenWeights.weights, it->second.weights);
}

}  // namespace rtp_llm
