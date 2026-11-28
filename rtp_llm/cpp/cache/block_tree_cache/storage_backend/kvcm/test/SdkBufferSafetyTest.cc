#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"

#include <gtest/gtest.h>

namespace rtp_llm::kvcm {
namespace {
enum class OldCode {
    ER_OK           = 0,
    ER_SDK_TIMEOUT  = 100,
    VENDOR_CODE_119 = 119,
    VENDOR_CODE_120 = 120
};
enum class NewCode {
    ER_OK                      = 0,
    ER_SDK_TIMEOUT             = 100,
    ER_SDKREAD_ERROR           = 104,
    ER_SDK_TIMEOUT_BUFFER_SAFE = 119,
    ER_SDK_ERROR_BUFFER_SAFE   = 120
};
enum class PartialCode {
    ER_OK                      = 0,
    ER_SDK_TIMEOUT_BUFFER_SAFE = 119
};

TEST(SdkBufferSafetyTest, OlderHeadersCompileAndNeverInterpretVendorNumbersAsSafe) {
    EXPECT_TRUE(sdkCodeProvesBufferSafety(OldCode::ER_OK));
    EXPECT_FALSE(sdkCodeProvesBufferSafety(OldCode::ER_SDK_TIMEOUT));
    EXPECT_FALSE(sdkCodeProvesBufferSafety(OldCode::VENDOR_CODE_119));
    EXPECT_FALSE(sdkCodeProvesBufferSafety(OldCode::VENDOR_CODE_120));
}

TEST(SdkBufferSafetyTest, MatchingNamedContractSeparatesUnknownFromExplicitSafeFailures) {
    EXPECT_TRUE(sdkCodeProvesBufferSafety(NewCode::ER_OK));
    EXPECT_FALSE(sdkCodeProvesBufferSafety(NewCode::ER_SDK_TIMEOUT));
    EXPECT_FALSE(sdkCodeProvesBufferSafety(NewCode::ER_SDKREAD_ERROR));
    EXPECT_TRUE(sdkCodeProvesBufferSafety(NewCode::ER_SDK_TIMEOUT_BUFFER_SAFE));
    EXPECT_TRUE(sdkCodeProvesBufferSafety(NewCode::ER_SDK_ERROR_BUFFER_SAFE));
}

TEST(SdkBufferSafetyTest, PartialUnrecognizedContractStaysConservative) {
    EXPECT_TRUE(sdkCodeProvesBufferSafety(PartialCode::ER_OK));
    EXPECT_FALSE(sdkCodeProvesBufferSafety(PartialCode::ER_SDK_TIMEOUT_BUFFER_SAFE));
}
}  // namespace
}  // namespace rtp_llm::kvcm
