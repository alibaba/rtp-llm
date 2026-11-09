#include <gtest/gtest.h>

#include "rtp_llm/models_py/bindings/rocm/mrope_config_validation.h"

namespace rtp_llm {
namespace {

TEST(MropeConfigValidationTest, AcceptsCapacityBoundaries) {
    EXPECT_TRUE(validateMropeLayout(3, 128, 128, 24, 20, 20, true).empty());
    EXPECT_TRUE(validateMropeLayout(3, 66, 128, 11, 11, 11, true).empty());
    EXPECT_TRUE(validateMropeLayout(3, 64, 128, 11, 11, 10, true).empty());
    EXPECT_TRUE(validateMropeLayout(3, 64, 128, 22, 0, 10, true).empty());
}

TEST(MropeConfigValidationTest, AcceptsContiguousSectionsWithoutInterleavedCapacityLimit) {
    EXPECT_TRUE(validateMropeLayout(3, 128, 128, 16, 24, 24, false).empty());
    EXPECT_TRUE(validateMropeLayout(3, 64, 128, 12, 12, 8, false).empty());
    EXPECT_TRUE(validateMropeLayout(3, 64, 128, 0, 32, 0, false).empty());
    EXPECT_TRUE(validateMropeLayout(3, 64, 128, 32, 0, 0, false).empty());
    EXPECT_TRUE(validateMropeLayout(3, 64, 128, 0, 0, 32, false).empty());
}

TEST(MropeConfigValidationTest, RejectsInvalidShapeContracts) {
    for (bool interleaved : {false, true}) {
        EXPECT_NE(validateMropeLayout(2, 64, 128, 11, 11, 10, interleaved).find("index_factor=3"), std::string::npos);
        EXPECT_NE(validateMropeLayout(3, 63, 128, 11, 11, 10, interleaved).find("positive even"), std::string::npos);
        EXPECT_NE(validateMropeLayout(3, 130, 128, 23, 21, 21, interleaved).find("size_per_head"), std::string::npos);
        EXPECT_NE(validateMropeLayout(3, 64, 128, 10, 10, 10, interleaved).find("section sum"), std::string::npos);
        EXPECT_NE(validateMropeLayout(3, 64, 128, -1, 23, 10, interleaved).find("non-negative"), std::string::npos);
    }
}

TEST(MropeConfigValidationTest, RejectsInterleavedCapacityOverflow) {
    EXPECT_NE(validateMropeLayout(3, 64, 128, 10, 12, 10, true).find("H/W capacity"), std::string::npos);
    EXPECT_NE(validateMropeLayout(3, 64, 128, 11, 10, 11, true).find("H/W capacity"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
