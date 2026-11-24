#include "rtp_llm/cpp/cache/block_tree_cache/load/MockDiskCacheSource.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <linux/magic.h>
#include <sys/vfs.h>
#include <unistd.h>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace {

// Observe the source's ordinary aligned allocations at the linker boundary,
// without adding allocator injection or test-only hooks to production APIs.
std::atomic<bool>   audit_scratch{false};
std::atomic<void*>  live_scratch{nullptr};
std::atomic<size_t> scratch_allocations{0};
std::atomic<size_t> scratch_frees{0};
std::atomic<bool>   overlapping_scratch{false};

class ScratchAudit {
public:
    ScratchAudit() {
        live_scratch        = nullptr;
        scratch_allocations = 0;
        scratch_frees       = 0;
        overlapping_scratch = false;
        audit_scratch       = true;
    }
    ~ScratchAudit() {
        audit_scratch = false;
    }
    void expectReleased(size_t allocations) {
        audit_scratch = false;
        EXPECT_EQ(scratch_allocations.load(), allocations);
        EXPECT_EQ(scratch_frees.load(), allocations);
        EXPECT_EQ(live_scratch.load(), nullptr);
        EXPECT_FALSE(overlapping_scratch.load());
    }
};

}  // namespace

extern "C" int  __real_posix_memalign(void**, size_t, size_t);
extern "C" void __real_free(void*);

extern "C" int __wrap_posix_memalign(void** result, size_t alignment, size_t bytes) {
    const int status = __real_posix_memalign(result, alignment, bytes);
    if (status == 0 && audit_scratch.load()) {
        ++scratch_allocations;
        if (live_scratch.exchange(*result) != nullptr) {
            overlapping_scratch = true;
        }
    }
    return status;
}

extern "C" void __wrap_free(void* allocation) {
    void* expected = allocation;
    if (allocation && audit_scratch.load() && live_scratch.compare_exchange_strong(expected, nullptr)) {
        ++scratch_frees;
    }
    __real_free(allocation);
}

namespace rtp_llm {
namespace {

using namespace block_transfer_engine_test;

// These tests perform real O_DIRECT file reads/writes. The ring is a bounded
// working set, files belong to each rank, and O_DIRECT bypasses the OS page cache
// but cannot guarantee misses in a storage controller's cache or durable writes.
class MockDiskCacheSourceTest: public ::testing::Test {
protected:
    GroupSetPtr makeGroup(size_t                       id,
                          size_t                       capacity = 8,
                          bool                         crc      = false,
                          size_t                       payload  = 9,
                          bool                         host     = true,
                          bool                         buffered = false,
                          std::unique_ptr<DiskBlockIO> io       = nullptr) {
        auto topology =
            makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload)});
        auto disk = makeDiskPool(payload,
                                 capacity,
                                 directory_.path,
                                 std::move(io),
                                 "mock_source_" + std::to_string(next_pool_++),
                                 buffered,
                                 crc);
        return makeTestGroupSet(
            id, topology, {"group0"}, {}, host ? makeHostPool(payload, 2, crc) : nullptr, disk, crc);
    }

    static void expectReleased(const std::vector<GroupSetPtr>& groups) {
        for (const auto& group : groups) {
            EXPECT_EQ(group->diskPool()->freeBlocksNum(), group->diskPool()->totalBlocksNum());
            EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::MOCK_SOURCE), 0u);
            if (group->hostPool()) {
                EXPECT_EQ(group->hostPool()->freeBlocksNum(), group->hostPool()->totalBlocksNum());
            }
        }
    }

    TempDirGuard directory_{"mock_disk_source"};
    size_t       next_pool_{0};
};

TEST_F(MockDiskCacheSourceTest, ReservesOnlyDedicatedDiskReferencesAndReleasesThem) {
    const std::vector<GroupSetPtr> groups{makeGroup(0), makeGroup(7)};
    auto                           result = MockDiskCacheSource::create(groups, 3, 42);
    ASSERT_TRUE(result.ok()) << result.status();
    auto source = std::move(result).value();
    for (const auto& group : groups) {
        EXPECT_EQ(source->sourceBlockCount(group->groupSetId()), 3u);
        EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::MOCK_SOURCE), 3u);
        EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
        EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        EXPECT_EQ(group->diskPool()->availableBlocksNum(), 5u);
        EXPECT_EQ(group->hostPool()->availableBlocksNum(), 2u);
        EXPECT_EQ(group->hostPool()->usedBlocksNum(), 0u);
    }
    source.reset();
    expectReleased(groups);
}

TEST_F(MockDiskCacheSourceTest, SeededRingsWrapSafelyAndDoNotAllocateDuringSelection) {
    auto first_group  = makeGroup(7, 20, false, 9, false);
    auto second_group = makeGroup(7, 20, false, 9, false);
    auto first        = MockDiskCacheSource::create({first_group}, 17, 12345);
    auto second       = MockDiskCacheSource::create({second_group}, 17, 12345);
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    std::set<BlockIdxType> ring;
    for (size_t logical = 0; logical < 17; ++logical) {
        const auto block = first.value()->blockFor(7, logical);
        EXPECT_EQ(block, second.value()->blockFor(7, logical));
        EXPECT_EQ(block, first.value()->blockFor(7, logical + 17 * 1000));
        ring.insert(block);
    }
    EXPECT_EQ(ring.size(), 17u);
    EXPECT_EQ(first.value()->blockFor(7, std::numeric_limits<size_t>::max()),
              first.value()->blockFor(7, std::numeric_limits<size_t>::max() % 17));
    EXPECT_EQ(first_group->diskPool()->usedBlocksNum(), 17u);
    EXPECT_THROW(first.value()->blockFor(8, 0), std::out_of_range);
    EXPECT_THROW(first.value()->sourceBlockCount(8), std::out_of_range);
    std::set<BlockIdxType> seeded_offsets;
    std::set<BlockIdxType> group_offsets;
    for (uint64_t seed = 0; seed < 8; ++seed) {
        auto group  = makeGroup(0, 20, false, 9, false);
        auto source = MockDiskCacheSource::create({group}, 17, seed);
        ASSERT_TRUE(source.ok()) << source.status();
        seeded_offsets.insert(source.value()->blockFor(0, 0));
        auto other        = makeGroup(seed, 20, false, 9, false);
        auto other_source = MockDiskCacheSource::create({other}, 17, 42);
        ASSERT_TRUE(other_source.ok()) << other_source.status();
        group_offsets.insert(other_source.value()->blockFor(seed, 0));
    }
    EXPECT_GT(seeded_offsets.size(), 1u);
    EXPECT_GT(group_offsets.size(), 1u);
}

TEST_F(MockDiskCacheSourceTest, RealDirectIoRoundTripsGroupSpecificPayloadPaddingAndCrc) {
    // Literal footer offsets and checksums independently specify GroupSet's
    // encoded layout: CRC32C of zero payload, then pool padding to 4096 bytes.
    struct Layout {
        size_t   payload;
        bool     crc;
        size_t   stride;
        size_t   footer;
        uint32_t checksum;
    };
    const std::vector<Layout> layouts{{9, false, 4096, 0, 0},
                                      {9, true, 4096, 12, 0xbbe568a3U},
                                      {4096, true, 8192, 4108, 0x98f94189U},
                                      {8193, true, 12288, 8204, 0x61006a3dU}};
    std::vector<GroupSetPtr>  groups;
    for (size_t id = 0; id < layouts.size(); ++id) {
        groups.push_back(makeGroup(id, 4, layouts[id].crc, layouts[id].payload, false));
    }
    auto source = MockDiskCacheSource::create(groups, 3, 42);
    ASSERT_TRUE(source.ok()) << source.status();
    alignas(4096) uint8_t buffer[12288];
    for (size_t id = 0; id < layouts.size(); ++id) {
        const auto& layout = layouts[id];
        const auto  disk   = groups[id]->diskPool();
        EXPECT_FALSE(disk->bufferedIO());
        EXPECT_EQ(disk->strideBytes(), layout.stride);
        EXPECT_EQ(disk->writeBytes(), 3 * layout.stride);
        EXPECT_EQ(disk->readBytes(), 0u);
        for (size_t logical = 0; logical < 3; ++logical) {
            std::memset(buffer, 0xa5, sizeof(buffer));
            ASSERT_EQ(disk->read(source.value()->blockFor(id, logical), buffer, layout.stride), BlockIOStatus::OK);
            if (layout.crc) {
                uint32_t checksum = 0;
                std::memcpy(&checksum, buffer + layout.footer, sizeof(checksum));
                EXPECT_EQ(checksum, layout.checksum);
            }
            for (size_t byte = 0; byte < layout.stride; ++byte) {
                if (!layout.crc || byte < layout.footer || byte >= layout.footer + 4) {
                    ASSERT_EQ(buffer[byte], 0) << "group=" << id << " byte=" << byte;
                }
            }
        }
        EXPECT_EQ(disk->readBytes(), 3 * layout.stride);
        EXPECT_EQ(disk->writeBytes(), 3 * layout.stride);
    }
}

TEST_F(MockDiskCacheSourceTest, RejectsZeroCountEmptyGroupsMissingPoolsAndDuplicateGroups) {
    auto group   = makeGroup(0);
    auto missing = makeTestGroupSet(1, group->topologyPtr(), {"group0"}, {});
    for (const auto& groups : std::vector<std::vector<GroupSetPtr>>{{}, {nullptr}, {group, group}, {group, missing}}) {
        EXPECT_EQ(MockDiskCacheSource::create(groups, 3, 42).status().code(), absl::StatusCode::kInvalidArgument);
    }
    EXPECT_EQ(MockDiskCacheSource::create({group}, 0, 42).status().code(), absl::StatusCode::kInvalidArgument);
    auto duplicate_id = makeGroup(0);
    EXPECT_EQ(MockDiskCacheSource::create({group, duplicate_id}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    auto shared_pool = makeTestGroupSet(1, group->topologyPtr(), {"group0"}, {}, nullptr, group->diskPool());
    EXPECT_EQ(MockDiskCacheSource::create({group, shared_pool}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    expectReleased({group, duplicate_id});
}

TEST_F(MockDiskCacheSourceTest, RejectsBufferedIoAndInconsistentLayoutsBeforeWriting) {
    auto first    = makeGroup(0);
    auto buffered = makeGroup(1, 8, false, 9, false, true);
    EXPECT_EQ(MockDiskCacheSource::create({first, buffered}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    auto bad_layout = makeGroup(1);
    bad_layout->initialize(1, bad_layout->topologyPtr(), {"group0"}, BackingLayout{9, 32, 4096, true});
    EXPECT_EQ(MockDiskCacheSource::create({first, bad_layout}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    auto bad_payload = makeGroup(2);
    bad_payload->initialize(2, bad_payload->topologyPtr(), {"group0"}, BackingLayout{17, 17, 4096, false});
    EXPECT_EQ(MockDiskCacheSource::create({first, bad_payload}, 3, 42).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(first->diskPool()->writeBytes(), 0u);
    expectReleased({first, buffered, bad_layout, bad_payload});
}

TEST_F(MockDiskCacheSourceTest, RejectsRamBackedFilesEvenWhenDirectIoIsConfigured) {
    struct statfs filesystem{};
    if (::statfs("/dev/shm", &filesystem) != 0 || filesystem.f_type != TMPFS_MAGIC) {
        GTEST_SKIP() << "tmpfs fixture unavailable";
    }
    char memory_path[] = "/dev/shm/mock_disk_source_XXXXXX";
    ASSERT_NE(::mkdtemp(memory_path), nullptr);
    ASSERT_EQ(::rmdir(directory_.path.c_str()), 0);
    directory_.path = memory_path;
    // Exercise the source's statfs guard on kernels that reject O_DIRECT on
    // tmpfs at open. Model a filesystem accepting and ignoring the flag.
    class IgnoreDirectIo: public BlockTreePosixDiskBlockIO {
    public:
        DiskBlockIOStatus openAndPreallocate(const std::string& path, size_t bytes, bool buffered) override {
            EXPECT_FALSE(buffered);
            return BlockTreePosixDiskBlockIO::openAndPreallocate(path, bytes, true);
        }
    };
    auto group = makeGroup(0, 4, false, 9, false, false, std::make_unique<IgnoreDirectIo>());
    EXPECT_FALSE(group->diskPool()->bufferedIO());
    EXPECT_EQ(MockDiskCacheSource::create({group}, 3, 42).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(group->diskPool()->writeBytes(), 0u);
    expectReleased({group});
}

TEST_F(MockDiskCacheSourceTest, InsufficientCapacityInLaterGroupRollsBackEveryReservation) {
    const std::vector<GroupSetPtr> groups{makeGroup(0), makeGroup(1, 2)};
    ScratchAudit                   audit;
    EXPECT_EQ(MockDiskCacheSource::create(groups, 3, 42).status().code(), absl::StatusCode::kResourceExhausted);
    audit.expectReleased(1);
    EXPECT_EQ(groups.front()->diskPool()->writeBytes(), 3 * 4096u);
    expectReleased(groups);
}

class FailSecondWriteIO: public BlockTreePosixDiskBlockIO {
public:
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override {
        if (++writes_ == 2) {
            return DiskBlockIOStatus::IO_ERROR;
        }
        return BlockTreePosixDiskBlockIO::write(offset, src, bytes);
    }

private:
    size_t writes_{0};
};

class ScratchObservingIO: public BlockTreePosixDiskBlockIO {
public:
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override {
        if (writes_++ == 0) {
            scratch_ = src;
        }
        reused_  = reused_ && scratch_ == src;
        aligned_ = aligned_ && reinterpret_cast<uintptr_t>(src) % 4096 == 0 && bytes == 4096;
        return BlockTreePosixDiskBlockIO::write(offset, src, bytes);
    }
    const void* scratch_{nullptr};
    size_t      writes_{0};
    bool        reused_{true};
    bool        aligned_{true};
};

TEST_F(MockDiskCacheSourceTest, InitializationReusesOneAlignedScratchRecordWithoutHostPoolReservations) {
    auto         io       = std::make_unique<ScratchObservingIO>();
    const auto*  observed = io.get();
    auto         group    = makeGroup(0, 8, false, 9, true, false, std::move(io));
    ScratchAudit audit;
    auto         source = MockDiskCacheSource::create({group}, 8, 42);
    audit.expectReleased(1);
    ASSERT_TRUE(source.ok()) << source.status();
    EXPECT_EQ(observed->writes_, 8u);
    EXPECT_TRUE(observed->reused_);
    EXPECT_TRUE(observed->aligned_);
    EXPECT_EQ(group->hostPool()->usedBlocksNum(), 0u);
}

TEST_F(MockDiskCacheSourceTest, WriteFailureRollsBackEarlierGroupsAndPartlyWrittenGroup) {
    const std::vector<GroupSetPtr> groups{
        makeGroup(0), makeGroup(1, 8, false, 9, true, false, std::make_unique<FailSecondWriteIO>())};
    ScratchAudit audit;
    auto         result = MockDiskCacheSource::create(groups, 3, 42);
    audit.expectReleased(2);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
    EXPECT_EQ(groups[0]->diskPool()->writeBytes(), 3 * 4096u);
    EXPECT_EQ(groups[1]->diskPool()->writeBytes(), 4096u);
    expectReleased(groups);
    // The same pools remain usable after the failed initialization.
    auto retry = MockDiskCacheSource::create(groups, 3, 42);
    ASSERT_TRUE(retry.ok()) << retry.status();
    retry.value().reset();
    expectReleased(groups);
}

TEST_F(MockDiskCacheSourceTest, SharedReadersKeepRecordsReservedUntilLastReaderReleasesSource) {
    auto group  = makeGroup(0);
    auto result = MockDiskCacheSource::create({group}, 3, 42);
    ASSERT_TRUE(result.ok()) << result.status();
    auto       owner  = std::move(result).value();
    auto       reader = owner;
    const auto block  = reader->blockFor(0, 0);
    owner.reset();
    EXPECT_EQ(group->diskPool()->referencedBlocksNum(BlockTreeRefType::MOCK_SOURCE), 3u);
    // Distinct ordinary write destinations cannot allocate a source record.
    auto writes = group->allocateBlocks(5, Tier::DISK, BlockTreeRefType::STORE);
    ASSERT_TRUE(writes);
    for (size_t logical = 0; logical < 3; ++logical) {
        EXPECT_EQ(std::count(writes->begin(), writes->end(), reader->blockFor(0, logical)), 0);
    }
    alignas(4096) uint8_t buffer[4096];
    ASSERT_EQ(group->diskPool()->read(block, buffer, sizeof(buffer)), BlockIOStatus::OK);
    EXPECT_TRUE(std::all_of(buffer, buffer + sizeof(buffer), [](uint8_t byte) { return byte == 0; }));
    group->releaseBlocks(Tier::DISK, *writes, BlockTreeRefType::STORE);
    reader.reset();
    expectReleased({group});
}

}  // namespace
}  // namespace rtp_llm
