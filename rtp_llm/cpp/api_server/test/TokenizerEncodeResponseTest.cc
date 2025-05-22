#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/TokenizerEncodeResponse.h"

using namespace ::testing;
namespace rtp_llm {

TEST(TokenizerEncodeResponseTest, ReturnOffsetMappingIsTrue) {
    TokenizerEncodeResponse response;
    response.offset_mapping        = {{1, 2, 3}, {4, 5, 6}};
    response.tokens                = {"hello", "world"};
    response.token_ids             = {10, 11};
    response.return_offset_mapping = true;

    auto json_str = autil::legacy::ToJsonString(response);
    EXPECT_FALSE(json_str.empty());

    TokenizerEncodeResponse response2;
    EXPECT_NO_THROW(autil::legacy::FromJsonString(response2, json_str));
    EXPECT_EQ(response2.offset_mapping, response.offset_mapping);
    EXPECT_EQ(response2.tokens, response.tokens);
    EXPECT_EQ(response2.token_ids, response.token_ids);
    EXPECT_EQ(response2.error, response.error);
    EXPECT_FALSE(response2.return_offset_mapping);
}

TEST(TokenizerEncodeResponseTest, ReturnOffsetMappingIsFalse) {
    TokenizerEncodeResponse response;
    response.offset_mapping        = {{1, 2, 3}, {4, 5, 6}};
    response.tokens                = {"hello", "world"};
    response.token_ids             = {10, 11};
    response.return_offset_mapping = false;

    auto json_str = autil::legacy::ToJsonString(response);
    EXPECT_FALSE(json_str.empty());

    TokenizerEncodeResponse response2;
    EXPECT_NO_THROW(autil::legacy::FromJsonString(response2, json_str));
    EXPECT_TRUE(response2.offset_mapping.empty());
    EXPECT_EQ(response2.tokens, response.tokens);
    EXPECT_EQ(response2.token_ids, response.token_ids);
    EXPECT_EQ(response2.error, response.error);
    EXPECT_FALSE(response2.return_offset_mapping);
}

}  // namespace rtp_llm