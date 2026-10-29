#include "rtp_llm/cpp/utils/DecodeCacheOrigin.h"
#include <gtest/gtest.h>
#include <vector>

namespace rtp_llm {
namespace {
const std::string session(32, 'a');
const std::string valid = "v1." + session + ".7680.7680.8192.8200.8201";

TEST(DecodeCacheOriginTest, RoundtripMatchesManagerCanonicalMetadataContract) {
    const auto origin = DecodeCacheOrigin::parse(valid);
    ASSERT_TRUE(origin);
    EXPECT_EQ(origin->encode(), valid);
    const auto marked = attachDecodeCacheOrigin("pace://allocated?z=unchanged&a=one%20two", *origin);
    ASSERT_TRUE(marked);
    EXPECT_EQ(*marked, "pace://allocated?a=one%20two&dtop_origin=" + valid + "&z=unchanged");
    const auto parsed = decodeCacheOriginFromUri(*marked);
    ASSERT_TRUE(parsed);
    EXPECT_TRUE(*parsed == *origin);
    EXPECT_FALSE(attachDecodeCacheOrigin(*marked, *origin));
}

TEST(DecodeCacheOriginTest, RejectsMalformedOrAmbiguousVersionsWithoutAttributingLegacyData) {
    for (const auto& value : std::vector<std::string>{"",
                                                      "v2." + session + ".7680.7680.8192.8200.8201",
                                                      valid + ".0",
                                                      "v1." + session + ".-1.7680.8192.8200.8201",
                                                      "v1." + session + ".07680.7680.8192.8200.8201",
                                                      "v1." + session + ".7680.8192.8192.8200.8201",
                                                      "v1." + session + ".7680.7680.8201.8200.8201",
                                                      "v1." + session + ".7680.7680.8192.8200.8200",
                                                      "v1." + session + ".7680.7680.8192.9223372036854775807.0",
                                                      "v1." + std::string(31, 'a') + ".7680.7680.8192.8200.8201",
                                                      "v1." + std::string(31, 'a') + "_.7680.7680.8192.8200.8201"}) {
        EXPECT_FALSE(DecodeCacheOrigin::parse(value)) << value;
    }
    EXPECT_FALSE(decodeCacheOriginFromUri("pace://legacy?blkid=1"));
    EXPECT_FALSE(decodeCacheOriginFromUri("pace://bad?dtop_origin=" + valid + "&dtop_origin=" + valid));
    EXPECT_FALSE(decodeCacheOriginFromUri("pace://bad?dtop_origin=" + valid + "&x=1&x=2"));
    EXPECT_FALSE(decodeCacheOriginFromUri("pace://bad?dtop_origin=" + valid + "#ignored"));
}

TEST(DecodeCacheOriginTest, CountsComputedIntervalOnlyOnceIndependentlyOfCpCopies) {
    const auto origin = DecodeCacheOrigin::parse(valid);
    ASSERT_TRUE(origin);
    EXPECT_EQ(origin->logicalDecodeTokens(8192), 512);
    EXPECT_EQ(origin->logicalDecodeTokens(8000), 320);
    EXPECT_EQ(origin->logicalDecodeTokens(9000), 512);
    EXPECT_EQ(origin->logicalDecodeTokens(7600), 0);
    EXPECT_EQ(origin->logicalDecodeTokens(std::numeric_limits<int64_t>::min()), 0);
    auto before  = *origin;
    before.begin = 7168;
    before.end   = 7680;
    EXPECT_EQ(before.logicalDecodeTokens(8192), 0);
    auto crossing  = *origin;
    crossing.begin = 7424;
    crossing.end   = 7936;
    EXPECT_EQ(crossing.logicalDecodeTokens(8192), 256);
    // No cp/rank factor exists in the logical interval contract.
}
}  // namespace
}  // namespace rtp_llm
