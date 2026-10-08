#include "rtp_llm/models_py/bindings/OpDefsUtils.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace rtp_llm {
namespace {

TEST(PaddingOffsetTest, ReportsEveryWrittenTokenForUnevenRequests) {
    std::array<int32_t, 3> lengths{5, 2, 3};
    std::array<int32_t, 10> offsets;
    offsets.fill(-7);

    const auto written = getPaddingOffset(offsets.data(), lengths.data(), nullptr, 3, 5);

    EXPECT_EQ(written, offsets.size());
    EXPECT_EQ(offsets, (std::array<int32_t, 10>{0, 0, 0, 0, 0, 0, 0, 3, 3, 3}));
}

}  // namespace
}  // namespace rtp_llm
