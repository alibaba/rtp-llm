#include <gtest/gtest.h>

#include "rtp_llm/models_py/bindings/rocm/mrope_config_validation.h"

namespace rtp_llm {
namespace {

TEST(MropeConfigValidationTest, AcceptsCapacityBoundaries) {
    EXPECT_TRUE(validateInterleavedMropeConfig(3, 128, 128, 24, 20, 20).empty());
    EXPECT_TRUE(validateInterleavedMropeConfig(3, 66, 128, 11, 11, 11).empty());
    EXPECT_TRUE(validateInterleavedMropeConfig(3, 64, 128, 11, 11, 10).empty());
}

TEST(MropeConfigValidationTest, RejectsInvalidShapeContracts) {
    EXPECT_NE(validateInterleavedMropeConfig(2, 64, 128, 11, 11, 10).find("index_factor=3"), std::string::npos);
    EXPECT_NE(validateInterleavedMropeConfig(3, 63, 128, 11, 11, 10).find("positive even"), std::string::npos);
    EXPECT_NE(validateInterleavedMropeConfig(3, 130, 128, 23, 21, 21).find("size_per_head"), std::string::npos);
    EXPECT_NE(validateInterleavedMropeConfig(3, 64, 128, 10, 10, 10).find("section sum"), std::string::npos);
}

TEST(MropeConfigValidationTest, RejectsInterleavedCapacityOverflow) {
    EXPECT_NE(validateInterleavedMropeConfig(3, 64, 128, 10, 12, 10).find("H/W capacity"), std::string::npos);
    EXPECT_NE(validateInterleavedMropeConfig(3, 64, 128, 11, 10, 11).find("H/W capacity"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
