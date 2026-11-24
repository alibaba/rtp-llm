#include "rtp_llm/cpp/cache/block_tree_cache/load/MockMemoryCacheSource.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {
namespace {

using namespace block_transfer_engine_test;

GroupSetPtr makeSourceGroup(size_t group_set_id, size_t capacity, bool crc = false, size_t payload_bytes = 9) {
    auto topology =
        makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload_bytes)});
    return makeTestGroupSet(
        group_set_id, topology, {"group0"}, {}, makeHostPool(payload_bytes, capacity, crc), nullptr, crc);
}

TEST(MockMemoryCacheSourceTest, ReservesDedicatedReferencesInEveryHostPoolAndReleasesThem) {
    const std::vector<GroupSetPtr> groups{makeSourceGroup(0, 8), makeSourceGroup(7, 6)};
    auto                           result = MockMemoryCacheSource::create(groups, 3, 42);
    ASSERT_TRUE(result.ok()) << result.status();
    auto source = std::move(result).value();
    for (const auto& group : groups) {
        EXPECT_EQ(source->sourceBlockCount(group->groupSetId()), 3u);
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::MOCK_SOURCE), 3u);
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        EXPECT_EQ(group->hostPool()->usedBlocksNum(), 3u);
        EXPECT_EQ(group->hostPool()->activeBlocksNum(), 3u);
        EXPECT_EQ(group->hostPool()->availableBlocksNum(), group->hostPool()->totalBlocksNum() - 3);
    }
    BlockTreeCacheMetricsReporter reporter(nullptr);
    const auto                    snapshots = reporter.collectPoolMetrics(groups);
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ(snapshots[0].mock_source_ref_blocks, 3u);
    EXPECT_EQ(snapshots[1].mock_source_ref_blocks, 3u);
    EXPECT_EQ(snapshots[0].mock_source_ref_bytes, 12288u);
    EXPECT_EQ(snapshots[1].mock_source_ref_bytes, 12288u);
    source.reset();
    for (const auto& group : groups) {
        EXPECT_EQ(group->hostPool()->freeBlocksNum(), group->hostPool()->totalBlocksNum());
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::MOCK_SOURCE), 0u);
    }
    for (const auto& snapshot : reporter.collectPoolMetrics(groups)) {
        EXPECT_EQ(snapshot.mock_source_ref_bytes, 0u);
    }
}

TEST(MockMemoryCacheSourceTest, SeedAndGroupChooseDeterministicRingsWithoutFurtherAllocations) {
    const std::vector<GroupSetPtr> first_groups{makeSourceGroup(0, 20), makeSourceGroup(7, 20)};
    const std::vector<GroupSetPtr> second_groups{makeSourceGroup(0, 20), makeSourceGroup(7, 20)};
    auto                           first  = MockMemoryCacheSource::create(first_groups, 17, 12345);
    auto                           second = MockMemoryCacheSource::create(second_groups, 17, 12345);
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    for (const size_t id : {0u, 7u}) {
        std::set<BlockIdxType> ring;
        for (size_t logical = 0; logical < 17; ++logical) {
            const auto block = first.value()->blockFor(id, logical);
            EXPECT_EQ(block, second.value()->blockFor(id, logical));
            EXPECT_EQ(block, first.value()->blockFor(id, logical + 17 * 1000));
            ring.insert(block);
        }
        EXPECT_EQ(ring.size(), 17u);
        EXPECT_EQ(first.value()->blockFor(id, std::numeric_limits<size_t>::max()),
                  first.value()->blockFor(id, std::numeric_limits<size_t>::max() % 17));
    }
    // Exercise several seeds to avoid relying on any one hash bucket being distinct.
    std::set<BlockIdxType> seeded_offsets;
    std::set<BlockIdxType> group_offsets;
    for (uint64_t seed = 0; seed < 16; ++seed) {
        auto group  = makeSourceGroup(0, 20);
        auto source = MockMemoryCacheSource::create({group}, 17, seed);
        ASSERT_TRUE(source.ok());
        seeded_offsets.insert(source.value()->blockFor(0, 0));
        auto other        = makeSourceGroup(seed, 20);
        auto other_source = MockMemoryCacheSource::create({other}, 17, 42);
        ASSERT_TRUE(other_source.ok());
        group_offsets.insert(other_source.value()->blockFor(seed, 0));
    }
    EXPECT_GT(seeded_offsets.size(), 1u);
    EXPECT_GT(group_offsets.size(), 1u);
    for (const auto& group : first_groups) {
        EXPECT_EQ(group->hostPool()->usedBlocksNum(), 17u);
    }
}

TEST(MockMemoryCacheSourceTest, ClearsAllPayloadAndPaddingBytesWithAndWithoutCrc) {
    for (const bool crc : {false, true}) {
        auto group = makeSourceGroup(0, 8, crc);
        for (BlockIdxType block = 1; block <= 8; ++block) {
            std::memset(group->hostPool()->blockBuffer(block).addr, 0xa5, group->hostPool()->strideBytes());
        }
        auto source = MockMemoryCacheSource::create({group}, 3, 0);
        ASSERT_TRUE(source.ok()) << source.status();
        EXPECT_EQ(group->storageBytes(), crc ? CrcCopyWorkspace::encodedBytes(9) : 9u);
        for (size_t logical = 0; logical < 3; ++logical) {
            const auto* record =
                static_cast<const uint8_t*>(group->hostPool()->blockBuffer(source.value()->blockFor(0, logical)).addr);
            const size_t footer = crc ? CrcCopyWorkspace::encodedBytes(9) - sizeof(uint32_t) : 0;
            for (size_t byte = 0; byte < group->hostPool()->strideBytes(); ++byte) {
                if (!crc || byte < footer || byte >= footer + sizeof(uint32_t)) {
                    ASSERT_EQ(record[byte], 0) << "crc=" << crc << " byte=" << byte;
                }
            }
        }
    }
}

TEST(MockMemoryCacheSourceTest, CpuCrcUsesStandardCastagnoliInitialAndFinalConvention) {
    const uint8_t known[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    EXPECT_EQ(mockMemoryCacheCrc32c(known, sizeof(known)), 0xe3069283U);
    EXPECT_EQ(mockMemoryCacheCrc32c(nullptr, 0), 0u);
}

TEST(MockMemoryCacheSourceTest, SourceFooterMatchesRealGpuStoreAndPassesCheckedHostToDeviceLoad) {
    if (!CrcCopyWorkspace::available() || !block_tree_cache_test::cudaAvailable()) {
        GTEST_SKIP() << "CUDA CRC backend unavailable";
    }
    constexpr size_t payload_bytes = 64;
    const auto       topology =
        makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload_bytes)});
    auto device = makeTestDevicePool({{payload_bytes, 0}}, 4, "mock_crc_target");
    auto host   = makeHostPool(payload_bytes, 6, true);
    auto group  = makeTestGroupSet(0, topology, {"group0"}, {device}, host, nullptr, true);
    auto source = MockMemoryCacheSource::create({group}, 2, 42);
    ASSERT_TRUE(source.ok()) << source.status();
    const auto target = block_tree_cache_test::allocateDeviceBlocksForTest(*group, 1);
    ASSERT_EQ(target.size(), 1u);
    const auto reference = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::STORE);
    ASSERT_NE(reference, NULL_BLOCK_IDX);
    const auto buffer        = device->convertIndexToBuffer(0, target.front().front()).front();
    auto       target_tensor = torch::from_blob(
        buffer.addr, {static_cast<int64_t>(payload_bytes)}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    target_tensor.zero_();
    torch::cuda::synchronize();
    PerRankBlockTransferEngine engine({group});
    auto stored = engine.execute(makeTransferTask({TransferDescriptor::deviceToHost(0, target.front(), reference)}));
    stored->waitDone();
    ASSERT_TRUE(stored->success()) << stored->errorInfo().ToString();
    const auto source_block = source.value()->blockFor(0, 0);
    EXPECT_EQ(
        std::memcmp(host->blockBuffer(reference).addr, host->blockBuffer(source_block).addr, group->storageBytes()), 0);
    target_tensor.fill_(0xa5);
    torch::cuda::synchronize();
    auto loaded = engine.execute(makeTransferTask({TransferDescriptor::hostToDevice(0, source_block, target.front())}));
    loaded->waitDone();
    ASSERT_TRUE(loaded->success()) << loaded->errorInfo().ToString();
    const auto  observed = target_tensor.cpu();
    const auto* bytes    = observed.data_ptr<uint8_t>();
    EXPECT_TRUE(std::all_of(bytes, bytes + payload_bytes, [](uint8_t byte) { return byte == 0; }));
    group->releaseSingleBlock(Tier::HOST, reference, BlockTreeRefType::STORE);
    block_tree_cache_test::unreferenceDeviceBlocksForTest(*group, target);
}

TEST(MockMemoryCacheSourceTest, CapacityFailureReleasesEarlierGroupReservations) {
    const std::vector<GroupSetPtr> groups{makeSourceGroup(0, 8), makeSourceGroup(1, 2)};
    auto                           result = MockMemoryCacheSource::create(groups, 3, 42);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kResourceExhausted);
    for (const auto& group : groups) {
        EXPECT_EQ(group->hostPool()->freeBlocksNum(), group->hostPool()->totalBlocksNum());
        EXPECT_EQ(group->hostPool()->referencedBlocksNum(BlockTreeRefType::MOCK_SOURCE), 0u);
    }
}

TEST(MockMemoryCacheSourceTest, InvalidInputsDoNotReserveHostBlocks) {
    auto group = makeSourceGroup(0, 8);
    EXPECT_EQ(MockMemoryCacheSource::create({group}, 0, 42).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(MockMemoryCacheSource::create({}, 3, 42).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(MockMemoryCacheSource::create({group, nullptr}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(MockMemoryCacheSource::create({group, group}, 3, 42).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(group->hostPool()->usedBlocksNum(), 0u);
}

TEST(MockMemoryCacheSourceTest, MissingHostOrInvalidRecordLayoutRollsBackEarlierGroups) {
    auto first   = makeSourceGroup(0, 8);
    auto invalid = makeSourceGroup(1, 8);
    invalid->initialize(1, invalid->topologyPtr(), {"group0"}, BackingLayout{4096, 4112, 4096, true});
    EXPECT_EQ(MockMemoryCacheSource::create({first, invalid}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(first->hostPool()->usedBlocksNum(), 0u);
    EXPECT_EQ(invalid->hostPool()->usedBlocksNum(), 0u);
    auto missing = makeTestGroupSet(1, first->topologyPtr(), {"group0"}, {});
    EXPECT_EQ(MockMemoryCacheSource::create({first, missing}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(first->hostPool()->usedBlocksNum(), 0u);
}

}  // namespace
}  // namespace rtp_llm
