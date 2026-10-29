#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DtopOriginMetadata.h"
#include <gtest/gtest.h>

namespace rtp_llm::kvcm {
namespace {
using kv_cache_manager::Locations;
Locations locations() {
    Locations value(2);
    for (size_t key = 0; key < value.size(); ++key)
        for (int cp = 0; cp < 4; ++cp)
            for (const auto* tag : {"full", "state"})
                value[key].push_back({std::string(tag) + "#cp" + std::to_string(cp),
                                      "pace://allocated/" + std::to_string(key * 16 + cp) + "?node_id=1&size=128"});
    return value;
}
const std::string session(32, 'a');

TEST(DtopOriginMetadataTest, BindsUnmaskedCanonicalPositionsWithoutMultiplyingCpCopies) {
    const auto original = locations();
    const auto result   = makeDtopOriginMetadata(original, session, {15, 16}, 512, 8704, 8800, 7680);
    ASSERT_TRUE(result);
    ASSERT_EQ(result->origins.size(), size_t{2});
    EXPECT_EQ(result->origins[0].begin, 7680);
    EXPECT_EQ(result->origins[0].end, 8192);
    EXPECT_EQ(result->origins[1].begin, 8192);
    EXPECT_EQ(result->origins[1].end, 8704);
    EXPECT_EQ(result->origins[0].logicalDecodeTokens(8704) + result->origins[1].logicalDecodeTokens(8704), 1024);
    for (size_t key = 0; key < original.size(); ++key) {
        ASSERT_EQ(result->locations[key].size(), original[key].size());
        for (size_t spec = 0; spec < original[key].size(); ++spec) {
            EXPECT_EQ(result->locations[key][spec].spec_name, original[key][spec].spec_name);
            const auto parsed = decodeCacheOriginFromUri(result->locations[key][spec].uri);
            ASSERT_TRUE(parsed);
            EXPECT_TRUE(*parsed == result->origins[key]);
            EXPECT_EQ(original[key][spec].uri.find("dtop_origin"), std::string::npos);
        }
    }
}

TEST(DtopOriginMetadataTest, MaskedExistingPrefixIsNotClaimedAsNewlyPublishedD) {
    const auto result = makeDtopOriginMetadata(locations(), session, {0, 16}, 512, 8704, 8800, 7680);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->origins[0].logicalDecodeTokens(8704), 0);
    EXPECT_EQ(result->origins[1].logicalDecodeTokens(8704), 512);
}

TEST(DtopOriginMetadataTest, ReadRequiresConsensusAcrossEveryRequiredSpecAndRank) {
    auto result = makeDtopOriginMetadata(locations(), session, {15, 16}, 512, 8704, 8800, 7680);
    ASSERT_TRUE(result);
    auto source = readDtopOriginMetadata(result->locations[0], 15, 512);
    ASSERT_TRUE(source);
    EXPECT_TRUE(*source == result->origins[0]);
    EXPECT_FALSE(readDtopOriginMetadata(result->locations[0], 14, 512));
    EXPECT_FALSE(readDtopOriginMetadata(locations()[0], 15, 512));
    auto mixed       = result->locations[0];
    mixed.back().uri = locations()[0].back().uri;
    EXPECT_FALSE(readDtopOriginMetadata(mixed, 15, 512));
    mixed            = result->locations[0];
    auto changed     = result->origins[0];
    changed.session  = std::string(32, 'b');
    mixed.back().uri = *attachDecodeCacheOrigin(locations()[0].back().uri, changed);
    EXPECT_FALSE(readDtopOriginMetadata(mixed, 15, 512));
    mixed = result->locations[0];
    mixed.push_back(mixed.front());
    EXPECT_FALSE(readDtopOriginMetadata(mixed, 15, 512));
}

TEST(DtopOriginMetadataTest, ReadUsesGlobalOrdinalAfterLocalPrefixAndSparsePublication) {
    auto result = makeDtopOriginMetadata(locations(), session, {0, 16}, 512, 8704, 8800, 7680);
    ASSERT_TRUE(result);
    // StartWrite returned only sparse keys 0 and 16. A later request already
    // owns the first 16 keys locally; its remote row zero is global key 16.
    const auto& remote_row = result->locations[1];
    EXPECT_TRUE(readDtopOriginMetadata(remote_row, 16, 512));
    EXPECT_FALSE(readDtopOriginMetadata(remote_row, 0, 512));
    EXPECT_FALSE(readDtopOriginMetadata(remote_row, 1, 512));
    EXPECT_FALSE(readDtopOriginMetadata(remote_row, 16, 0));
    EXPECT_FALSE(readDtopOriginMetadata(remote_row, std::numeric_limits<size_t>::max(), 512));
    EXPECT_FALSE(readDtopOriginMetadata(remote_row, 16, std::numeric_limits<size_t>::max()));
}

TEST(DtopOriginMetadataTest, RejectsBadScopeMaskAndOverflowWithoutChangingInput) {
    const auto input = locations();
    EXPECT_FALSE(makeDtopOriginMetadata(input, session, {16}, 512, 8704, 8800, 7680));
    EXPECT_FALSE(makeDtopOriginMetadata(input, session, {16, 16}, 512, 8704, 8800, 7680));
    EXPECT_FALSE(makeDtopOriginMetadata(input, session, {16, 15}, 512, 8704, 8800, 7680));
    EXPECT_FALSE(makeDtopOriginMetadata(input, session, {15, 16}, 0, 8704, 8800, 7680));
    EXPECT_FALSE(makeDtopOriginMetadata(input, session, {15, 16}, 512, 8704, 8800, -1));
    EXPECT_FALSE(makeDtopOriginMetadata(input, session, {15, 16}, 512, 8704, 8800, 8801));
    EXPECT_FALSE(makeDtopOriginMetadata(input, session, {15, 17}, 512, 8704, 8800, 7680));
    EXPECT_FALSE(
        makeDtopOriginMetadata(input, session, {15, 16}, 512, 8704, std::numeric_limits<int64_t>::max(), 7680));
    EXPECT_FALSE(makeDtopOriginMetadata(input, "invalid", {15, 16}, 512, 8704, 8800, 7680));
    auto malformed = input;
    malformed.back().push_back(malformed.back().front());
    EXPECT_FALSE(makeDtopOriginMetadata(malformed, session, {15, 16}, 512, 8704, 8800, 7680));
    EXPECT_EQ(malformed.front(), input.front());  // No partial source mutation.
}
}  // namespace
}  // namespace rtp_llm::kvcm
