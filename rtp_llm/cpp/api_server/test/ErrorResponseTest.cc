#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/ErrorResponse.h"

using namespace ::testing;
namespace rtp_llm {

TEST(ErrorResponseTest, CreateErrorResponseJsonString) {
    std::string   error_msg = "error_msg";
    std::string   json_str  = ErrorResponse::CreateErrorResponseJsonString(1, error_msg);
    ErrorResponse error_response;
    autil::legacy::FromJsonString(error_response, json_str);
    EXPECT_EQ(error_response.error_code, 1);
    EXPECT_EQ(error_response.error_msg, error_msg);
}

}  // namespace rtp_llm