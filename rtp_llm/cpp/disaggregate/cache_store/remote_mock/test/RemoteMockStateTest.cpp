#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockState.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockFormat.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/remote_mock_control.pb.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <locale>

namespace rtp_llm {
namespace {
RemoteMockLimits limits() {
    RemoteMockLimits l;
    l.source_bytes        = 1024;
    l.write_slots         = 2;
    l.write_slot_bytes    = 64;
    l.read_staging_bytes  = 16384;
    l.write_staging_bytes = 16384;
    l.total_bytes         = l.source_bytes + 128 + 32768 + 8192;
    l.max_rpc_bytes       = 64;
    l.max_blocks_per_rpc  = 4;
    l.max_read_operations = 2;
    l.replay_window       = 8;
    return l;
}
RemoteMockManifest manifest(uint32_t rank = 0, uint32_t ranks = 1, uint64_t generation = 1) {
    RemoteMockManifest m;
    m.session        = {99, 7, generation};
    m.rank           = rank;
    m.expected_ranks = ranks;
    m.layout_hash    = 123;
    RemoteMockLayout g;
    g.group             = 2;
    g.pool              = 3;
    g.payload_bytes     = 9;
    g.storage_bytes     = 16;
    g.stride_bytes      = 32;
    g.crc_enabled       = true;
    g.crc_footer_offset = 12;
    g.source_count      = 2;
    g.member_offsets    = {0, 4};
    g.dtype             = "fp16";
    m.groups            = {g};
    return m;
}
TEST(RemoteMockStateTest, AllRanksRequiredForReady) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest(0, 2);
    ASSERT_TRUE(s.registerRank(m).ok());
    EXPECT_FALSE(s.ready(m.session).ok());
    EXPECT_FALSE(s.sourceKey(m.session, 0, 2, 0, 0, 16).ok());
    ASSERT_TRUE(s.registerRank(manifest(1, 2)).ok());
    EXPECT_TRUE(s.ready(m.session).ok());
    EXPECT_TRUE(s.sourceKey(m.session, 0, 2, 0, 0, 16).ok());
}
TEST(RemoteMockStateTest, PartialRegistrationRollback) {
    RemoteMockState s(limits(), 99);
    ASSERT_TRUE(s.registerRank(manifest(0, 2)).ok());
    auto bad                    = manifest(1, 2);
    bad.groups[0].payload_bytes = 100;
    EXPECT_FALSE(s.registerRank(bad).ok());
    EXPECT_FALSE(s.ready(bad.session).ok());
    EXPECT_EQ(s.allocatedBytes(), 0u);
    EXPECT_FALSE(s.registerRank(manifest()).ok());
    EXPECT_TRUE(s.registerRank(manifest(0, 1, 2)).ok());
}
TEST(RemoteMockStateTest, StaleGenerationNeverResolves) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    auto key = s.sourceKey(m.session, 0, 2, 0, 0, 16);
    ASSERT_TRUE(key.ok());
    auto lease = s.resolve(*key, 16);
    ASSERT_TRUE(lease.ok());
    ASSERT_TRUE(s.close(m.session).ok());
    EXPECT_FALSE(s.resolve(*key, 16).ok());
    EXPECT_GT(s.allocatedBytes(), 0u);
    EXPECT_FALSE(s.registerRank(manifest(0, 1, 2)).ok());
    lease->reset();
    EXPECT_EQ(s.allocatedBytes(), 0u);
    auto next = manifest(0, 1, 2);
    ASSERT_TRUE(s.registerRank(next).ok());
    ASSERT_TRUE(s.ready(next.session).ok());
    EXPECT_FALSE(s.resolve(*key, 16).ok());
}
TEST(RemoteMockStateTest, BudgetBeforeAllocation) {
    auto l         = limits();
    l.source_bytes = 63;
    RemoteMockState s(l, 99);
    EXPECT_FALSE(s.registerRank(manifest()).ok());
    EXPECT_EQ(s.allocatedBytes(), 0u);
    l             = limits();
    l.total_bytes = 100;
    RemoteMockState small(l, 99);
    EXPECT_FALSE(small.registerRank(manifest()).ok());
    EXPECT_EQ(small.allocatedBytes(), 0u);
    auto m                   = manifest();
    m.groups[0].stride_bytes = std::numeric_limits<size_t>::max() - 31;
    RemoteMockState overflow(limits(), 99);
    EXPECT_FALSE(overflow.registerRank(m).ok());
}
TEST(RemoteMockStateTest, SourceImmutableAcrossWrite) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    auto key = s.sourceKey(m.session, 0, 2, 0, 0, 16);
    ASSERT_TRUE(key.ok());
    auto a = s.resolve(*key, 16);
    ASSERT_TRUE(a.ok());
    auto key2 = s.sourceKey(m.session, 0, 2, 1, 0, 16);
    ASSERT_TRUE(key2.ok());
    auto b = s.resolve(*key2, 16);
    ASSERT_TRUE(b.ok());
    EXPECT_NE((*a)->data(), (*b)->data());
    std::vector<uint8_t> before((*a)->data(), (*a)->data() + 16);
    auto                 write = s.reserveWrite({m.session, 123, 0, 0, 1}, 16);
    ASSERT_TRUE(write.ok());
    EXPECT_NE((*a)->data(), (*write)->data());
    std::fill_n((*write)->data(), 16, 0xa5);
    EXPECT_EQ(std::vector<uint8_t>((*a)->data(), (*a)->data() + 16), before);
    ASSERT_TRUE(s.close(m.session).ok());
    a->reset();
    b->reset();
    EXPECT_FALSE(s.registerRank(manifest(0, 1, 2)).ok());
    write->reset();
    EXPECT_TRUE(s.registerRank(manifest(0, 1, 2)).ok());
}
TEST(RemoteMockStateTest, RecordSlicesValidateBounds) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    auto key = s.sourceKey(m.session, 0, 2, 0, 4, 12);
    ASSERT_TRUE(key.ok());
    auto lease = s.resolve(*key, 12);
    ASSERT_TRUE(lease.ok());
    EXPECT_EQ((*lease)->size(), 12u);
    EXPECT_FALSE(s.resolve(*key, 11).ok());
    EXPECT_FALSE(s.sourceKey(m.session, 0, 2, 0, 16, 1).ok());
    EXPECT_FALSE(s.sourceKey(m.session, 0, 2, 0, std::numeric_limits<uint64_t>::max(), 2).ok());
    EXPECT_FALSE(s.sourceKey(m.session, 0, 2, 2, 0, 16).ok());
    EXPECT_FALSE(decodeRemoteMockKey(*key + ":0").ok());
    auto decoded = decodeRemoteMockKey(*key);
    ASSERT_TRUE(decoded.ok());
    decoded->session.server_epoch++;
    EXPECT_FALSE(s.resolve(encodeRemoteMockKey(*decoded), 12).ok());
    decoded = decodeRemoteMockKey(*key);
    decoded->checksum++;
    EXPECT_FALSE(s.resolve(encodeRemoteMockKey(*decoded), 12).ok());
}
TEST(RemoteMockStateTest, OutOfOrderWriteReplayAndSlotsAreBounded) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    auto two = s.reserveWrite({m.session, 123, 0, 0, 2}, 16);
    ASSERT_TRUE(two.ok());
    auto one = s.reserveWrite({m.session, 123, 0, 1, 1}, 16);
    ASSERT_TRUE(one.ok());
    EXPECT_FALSE(s.reserveWrite({m.session, 123, 0, 0, 3}, 16).ok());
    two->reset();
    EXPECT_FALSE(s.reserveWrite({m.session, 123, 0, 0, 2}, 16).ok());
    auto three = s.reserveWrite({m.session, 123, 0, 0, 3}, 16);
    ASSERT_TRUE(three.ok());
    three->reset();
    auto ten = s.reserveWrite({m.session, 123, 0, 0, 10}, 16);
    ASSERT_TRUE(ten.ok());
    ten->reset();
    EXPECT_FALSE(s.reserveWrite({m.session, 123, 0, 0, 2}, 16).ok());
    EXPECT_FALSE(s.reserveWrite({m.session, 999, 0, 0, 11}, 16).ok());
    EXPECT_FALSE(s.reserveWrite({m.session, 123, 0, 0, 11}, 65).ok());
}
TEST(RemoteMockStateTest, ReadWriteReservedProgressAndCloseDrain) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    auto w = s.reserveWrite({m.session, 123, 0, 0, 1}, 16);
    ASSERT_TRUE(w.ok());
    auto r1 = s.reserveRead(m.session, 16, 1);
    ASSERT_TRUE(r1.ok());
    auto r2 = s.reserveRead(m.session, 16, 1);
    ASSERT_TRUE(r2.ok());
    EXPECT_FALSE(s.reserveRead(m.session, 16, 1).ok());
    auto w2 = s.reserveWrite({m.session, 123, 0, 1, 2}, 16);
    ASSERT_TRUE(w2.ok());
    r1->reset();
    EXPECT_TRUE(s.reserveRead(m.session, 16, 1).ok());
    ASSERT_TRUE(s.close(m.session).ok());
    EXPECT_FALSE(s.reserveRead(m.session, 1, 1).ok());
    EXPECT_FALSE(s.reserveWrite({m.session, 123, 0, 0, 3}, 1).ok());
}
TEST(RemoteMockStateTest, BothDirectionsMustHaveMinimumProgressBudget) {
    auto l               = limits();
    l.read_staging_bytes = 1;
    RemoteMockState reads(l, 99);
    EXPECT_FALSE(reads.registerRank(manifest()).ok());
    EXPECT_EQ(reads.allocatedBytes(), 0u);
    l                     = limits();
    l.write_staging_bytes = 1;
    RemoteMockState writes(l, 99);
    EXPECT_FALSE(writes.registerRank(manifest()).ok());
    EXPECT_EQ(writes.allocatedBytes(), 0u);
}
TEST(RemoteMockStateTest, PhysicalRecordLargerThanRpcUsesBoundedSlices) {
    RemoteMockState s(limits(), 99);
    auto            m   = manifest();
    auto&           g   = m.groups[0];
    g.payload_bytes     = 100;
    g.storage_bytes     = 112;
    g.crc_footer_offset = 108;
    g.stride_bytes      = 128;
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    EXPECT_FALSE(s.sourceKey(m.session, 0, 2, 0, 0, 112).ok());
    std::vector<uint8_t> record;
    for (const auto range : {std::pair<uint64_t, uint64_t>{0, 64}, {64, 48}}) {
        auto key = s.sourceKey(m.session, 0, 2, 0, range.first, range.second);
        ASSERT_TRUE(key.ok());
        auto lease = s.resolve(*key, range.second);
        ASSERT_TRUE(lease.ok());
        record.insert(record.end(), (*lease)->data(), (*lease)->data() + (*lease)->size());
    }
    EXPECT_TRUE(validateRemoteMockRecord(g, record.data(), record.size()).ok());
}
TEST(RemoteMockFormatTest, CompleteRpcEnvelopeCannotExceedProtobufLimit) {
    auto l               = limits();
    l.max_rpc_bytes      = UINT32_MAX;
    l.max_blocks_per_rpc = UINT32_MAX;
    EXPECT_FALSE(remoteMockStagingCharge(INT32_MAX, 1, l).ok());
    EXPECT_FALSE(remoteMockStagingCharge(1, UINT32_MAX, l).ok());
    EXPECT_TRUE(remoteMockStagingCharge(64, 4, l).ok());
}
TEST(RemoteMockStateTest, BoundedDescriptorsBeforeAllRanksReady) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest(0, 2);
    EXPECT_FALSE(s.sessionInfo().ok());
    ASSERT_TRUE(s.registerRank(m).ok());
    auto info = s.sessionInfo();
    ASSERT_TRUE(info.ok());
    EXPECT_EQ(info->registered_ranks, 1);
    EXPECT_FALSE(info->ready);
    auto descriptors = s.describeRank(m.session, 0);
    ASSERT_TRUE(descriptors.ok());
    ASSERT_EQ(descriptors->size(), 1u);
    EXPECT_EQ(descriptors->front().source_count, 2);
    EXPECT_FALSE(s.describeRank(m.session, 1).ok());
    ASSERT_TRUE(s.registerRank(manifest(1, 2)).ok());
    for (uint32_t index = 0; index < 2; ++index) {
        auto key = s.sourceKey(m.session, 0, 2, index, 0, 16);
        ASSERT_TRUE(key.ok());
        auto decoded = decodeRemoteMockKey(*key);
        ASSERT_TRUE(decoded.ok());
        EXPECT_EQ(decoded->checksum, descriptors->front().record_checksum);
    }
}
TEST(RemoteMockStateTest, MetadataBudgetBeforeAllocation) {
    auto l        = limits();
    l.total_bytes = 72 + 128 + l.read_staging_bytes + l.write_staging_bytes;
    RemoteMockState s(l, 99);
    EXPECT_FALSE(s.registerRank(manifest()).ok());
    EXPECT_EQ(s.allocatedBytes(), 0u);
}
TEST(RemoteMockStateTest, SliceWriteKeysValidateIdentityAndRanges) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    RemoteMockWriteOperation op{m.session, 123, 0, 0, 1};
    RemoteMockKey            key;
    key.write               = true;
    key.session             = m.session;
    key.layout_hash         = 123;
    key.group               = 2;
    key.pool                = 3;
    key.operation           = 1;
    key.offset              = 4;
    key.length              = 8;
    const uint8_t binary[]  = {0, 1, 255, 3, 8, 2, 0, 9};
    key.checksum            = remoteMockCrc32c(binary, sizeof(binary));
    const auto first_record = encodeRemoteMockKey(key);
    key.record_index        = 1;
    EXPECT_NE(first_record, encodeRemoteMockKey(key));
    EXPECT_TRUE(s.validateWriteKey(op, encodeRemoteMockKey(key), 8).ok());
    key.offset = 15;
    EXPECT_FALSE(s.validateWriteKey(op, encodeRemoteMockKey(key), 8).ok());
    key.offset = 4;
    key.group  = 9;
    EXPECT_FALSE(s.validateWriteKey(op, encodeRemoteMockKey(key), 8).ok());
    key.group     = 2;
    key.operation = 2;
    EXPECT_FALSE(s.validateWriteKey(op, encodeRemoteMockKey(key), 8).ok());
}
TEST(RemoteMockStateTest, ForeignOrStaleRegistrationCannotAbortSession) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest(0, 2);
    ASSERT_TRUE(s.registerRank(m).ok());
    auto foreign = manifest(1, 2);
    foreign.session.session_id++;
    EXPECT_FALSE(s.registerRank(foreign).ok());
    ASSERT_TRUE(s.registerRank(manifest(1, 2)).ok());
    EXPECT_TRUE(s.ready(m.session).ok());
    EXPECT_FALSE(s.close(foreign.session).ok());
    EXPECT_TRUE(s.ready(m.session).ok());
}
TEST(RemoteMockStateTest, ManifestDelayMismatchRollsBack) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest(0, 2);
    ASSERT_TRUE(s.registerRank(m).ok());
    auto other          = manifest(1, 2);
    other.read_delay_ms = 5;
    EXPECT_FALSE(s.registerRank(other).ok());
    EXPECT_EQ(s.allocatedBytes(), 0u);
}
TEST(RemoteMockStateTest, IntegrityWithoutLocalCrcAndSliceReassembly) {
    RemoteMockState s(limits(), 99);
    auto            m   = manifest();
    auto&           g   = m.groups[0];
    g.crc_enabled       = false;
    g.crc_footer_offset = 0;
    g.storage_bytes     = g.payload_bytes;
    ASSERT_TRUE(s.registerRank(m).ok());
    ASSERT_TRUE(s.ready(m.session).ok());
    std::vector<uint8_t> record;
    uint32_t             checksum = 0;
    for (const auto range : {std::pair<uint64_t, uint64_t>{0, 4}, {4, 5}}) {
        auto key = s.sourceKey(m.session, 0, 2, 0, range.first, range.second);
        ASSERT_TRUE(key.ok());
        auto lease = s.resolve(*key, range.second);
        ASSERT_TRUE(lease.ok());
        record.insert(record.end(), (*lease)->data(), (*lease)->data() + (*lease)->size());
        checksum = (*lease)->recordChecksum();
    }
    EXPECT_EQ(remoteMockCrc32c(record.data(), record.size()), checksum);
    record[0] = 0xab;
    EXPECT_NE(remoteMockCrc32c(record.data(), record.size()), checksum);
}
TEST(RemoteMockFormatTest, InitializedPaddingCrcAndBinaryContent) {
    const uint8_t known[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    EXPECT_EQ(remoteMockCrc32c(known, 9), 0xe3069283u);
    auto                 g = manifest().groups[0];
    std::vector<uint8_t> bytes(32, 0xff);
    ASSERT_TRUE(initializeRemoteMockRecord(g, bytes.data(), bytes.size(), 0x31).ok());
    EXPECT_EQ(bytes[0], 0x31);
    EXPECT_EQ(bytes[8], 0x31);
    EXPECT_EQ(bytes[9], 0);
    EXPECT_EQ(bytes[11], 0);
    EXPECT_EQ(bytes[31], 0);
    EXPECT_TRUE(validateRemoteMockRecord(g, bytes.data(), 16).ok());
    bytes[0] ^= 1;
    EXPECT_FALSE(validateRemoteMockRecord(g, bytes.data(), 16).ok());
    EXPECT_FALSE(validateRemoteMockRecord(g, bytes.data(), 8).ok());
}
TEST(RemoteMockFormatTest, CanonicalKeyIgnoresGlobalNumberLocale) {
    class Thousands: public std::numpunct<char> {
        char do_thousands_sep() const override {
            return ',';
        }
        std::string do_grouping() const override {
            return std::string(1, 3);
        }
    };
    RemoteMockKey key;
    key.session         = {1000, 2000, 3000};
    key.layout_hash     = 4000;
    key.length          = 16;
    auto       previous = std::locale::global(std::locale(std::locale::classic(), new Thousands));
    const auto encoded  = encodeRemoteMockKey(key);
    std::locale::global(previous);
    EXPECT_EQ(encoded.find(','), std::string::npos);
    EXPECT_TRUE(decodeRemoteMockKey(encoded).ok());
}
TEST(RemoteMockFormatTest, KeyRoundTripAndControlDescriptor) {
    RemoteMockKey key;
    key.session     = {1, 2, 3};
    key.layout_hash = 4;
    key.rank        = 5;
    key.group       = 6;
    key.pool        = 7;
    key.index       = 8;
    key.offset      = 9;
    key.length      = 10;
    key.checksum    = 0xff0011aa;
    auto encoded    = encodeRemoteMockKey(key);
    auto parsed     = decodeRemoteMockKey(encoded);
    ASSERT_TRUE(parsed.ok());
    EXPECT_EQ(encodeRemoteMockKey(*parsed), encoded);
    EXPECT_FALSE(decodeRemoteMockKey("rm1:r:01:2:3:4:5:6:7:8:0:0:9:10:0").ok());
    EXPECT_FALSE(decodeRemoteMockKey(std::string(1024, '1')).ok());
    EXPECT_EQ(remote_mock::RemoteMockControlService::descriptor()->method_count(), 4);
}
TEST(RemoteMockStateTest, AdditionalReadMetadataIsChargedAndReleased) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    auto first = s.reserveRead(m.session, 16, 1, 10000);
    ASSERT_TRUE(first.ok());
    EXPECT_FALSE(s.reserveRead(m.session, 16, 1, 10000).ok());
    first->reset();
    EXPECT_TRUE(s.reserveRead(m.session, 16, 1, 10000).ok());
}
TEST(RemoteMockStateTest, MetadataRejectionDoesNotConsumeWriteReplay) {
    RemoteMockState s(limits(), 99);
    auto            m = manifest();
    ASSERT_TRUE(s.registerRank(m).ok());
    RemoteMockWriteOperation op{m.session, 123, 0, 0, 1};
    EXPECT_FALSE(s.reserveWrite(op, 16, 1, 12000).ok());
    EXPECT_TRUE(s.reserveWrite(op, 16, 1, 10000).ok());
}
TEST(RemoteMockFormatTest, AdditionalMetadataChargeChecksOverflow) {
    auto l = limits();
    EXPECT_FALSE(remoteMockStagingCharge(16, 1, l, UINT64_MAX).ok());
    auto charge = remoteMockStagingCharge(16, 1, l, 10000);
    ASSERT_TRUE(charge.ok());
    EXPECT_EQ(14640, *charge);
}
}  // namespace
}  // namespace rtp_llm
