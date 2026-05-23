#include "rtp_llm/cpp/cache/connector/memory/DiskSpillFileManager.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class DiskSpillFileManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        const auto stamp =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        base_path_ = (std::filesystem::temp_directory_path()
                      / ("rtp_llm_disk_spill_fm_" + std::to_string(::getpid()) + "_" + std::to_string(stamp)))
                         .string();
        std::error_code ec;
        std::filesystem::create_directories(base_path_, ec);
        EXPECT_FALSE(ec) << ec.message();
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(base_path_, ec);
    }

    DiskSpillFileManager::Config makeConfig(size_t disk_id        = 0,
                                            size_t slot_stride    = 4096,
                                            size_t capacity_bytes = 4096UL * 1024,
                                            size_t segment_bytes  = 4096UL * 256) const {
        DiskSpillFileManager::Config c;
        c.base_path         = base_path_;
        c.disk_id           = disk_id;
        c.capacity_bytes    = capacity_bytes;
        c.segment_bytes     = segment_bytes;
        c.slot_stride_bytes = slot_stride;
        c.align_bytes       = 4096;
        c.direct_io         = false;  // many CI tmpfs don't support O_DIRECT
        c.schema_hash       = "deadbeef";
        c.world_rank        = 0;
        c.startup_uuid      = "test_uuid";
        c.hostname          = "testhost";
        c.max_staging_buffers = 4;
        c.cleanup_old_startup_dirs = false;
        return c;
    }

    std::string base_path_;
};

TEST_F(DiskSpillFileManagerTest, BasicInitAndDirectoryLayout) {
    auto cfg = makeConfig();
    DiskSpillFileManager fm(cfg);
    ASSERT_TRUE(fm.init());
    EXPECT_GT(fm.slotCount(), 0u);
    EXPECT_EQ(fm.alignBytes(), 4096u);
    // verify path layout contains schema_, rank_, host_pid_uuid_
    bool found = false;
    for (auto& p : std::filesystem::recursive_directory_iterator(base_path_)) {
        const auto path = p.path().string();
        if (path.find("schema_deadbeef") != std::string::npos
            && path.find("rank_0") != std::string::npos
            && path.find("host_testhost") != std::string::npos
            && path.find("uuid_test_uuid") != std::string::npos) {
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(DiskSpillFileManagerTest, BasicWriteReadRoundtrip) {
    auto cfg = makeConfig();
    DiskSpillFileManager fm(cfg);
    ASSERT_TRUE(fm.init());
    std::vector<char> data(cfg.slot_stride_bytes, 'x');
    ASSERT_TRUE(fm.pwriteSlot(0, data.data(), data.size()));
    std::vector<char> read_back(cfg.slot_stride_bytes, 0);
    ASSERT_TRUE(fm.preadSlot(0, read_back.data(), read_back.size()));
    EXPECT_EQ(read_back, data);
}

TEST_F(DiskSpillFileManagerTest, SegmentBoundary) {
    // 2 slots per segment, 5 total slots → 3 segments (2+2+1)
    auto cfg = makeConfig(/*disk_id=*/0,
                          /*slot_stride=*/4096,
                          /*capacity_bytes=*/4096 * 5,
                          /*segment_bytes=*/4096 * 2);
    DiskSpillFileManager fm(cfg);
    ASSERT_TRUE(fm.init());
    EXPECT_EQ(fm.slotCount(), 5u);
    // write different patterns to each slot to ensure offsets are correct
    for (int s = 0; s < 5; ++s) {
        std::vector<char> data(4096, static_cast<char>('a' + s));
        ASSERT_TRUE(fm.pwriteSlot(s, data.data(), data.size()));
    }
    for (int s = 0; s < 5; ++s) {
        std::vector<char> data(4096, 0);
        ASSERT_TRUE(fm.preadSlot(s, data.data(), data.size()));
        EXPECT_EQ(data[0], static_cast<char>('a' + s));
    }
}

TEST_F(DiskSpillFileManagerTest, RejectSlotLargerThanSegment) {
    auto cfg = makeConfig(0, /*slot_stride=*/8192, /*capacity_bytes=*/8192, /*segment_bytes=*/4096);
    DiskSpillFileManager fm(cfg);
    EXPECT_FALSE(fm.init());
}

TEST_F(DiskSpillFileManagerTest, RejectInvalidCapacity) {
    auto cfg = makeConfig(0, /*slot_stride=*/4096, /*capacity_bytes=*/1024);
    DiskSpillFileManager fm(cfg);
    EXPECT_FALSE(fm.init());
}

TEST_F(DiskSpillFileManagerTest, RejectEmptySchemaHash) {
    auto cfg = makeConfig();
    cfg.schema_hash.clear();
    DiskSpillFileManager fm(cfg);
    EXPECT_FALSE(fm.init());
}

TEST_F(DiskSpillFileManagerTest, FlockPreventsSecondInitOnSameRunDir) {
    auto cfg = makeConfig();
    cfg.cleanup_on_destroy = false;
    DiskSpillFileManager fm1(cfg);
    ASSERT_TRUE(fm1.init());
    DiskSpillFileManager fm2(cfg);
    // Same pid+uuid+host -> same dir; second flock should fail.
    EXPECT_FALSE(fm2.init());
}

TEST_F(DiskSpillFileManagerTest, StagingBufferPoolAcquireRelease) {
    auto cfg = makeConfig();
    cfg.max_staging_buffers = 2;
    DiskSpillFileManager fm(cfg);
    ASSERT_TRUE(fm.init());

    auto b1 = fm.acquireStagingBuffer();
    ASSERT_NE(b1, nullptr);
    EXPECT_TRUE(b1->valid());
    auto b2 = fm.acquireStagingBuffer();
    ASSERT_NE(b2, nullptr);
    auto b3 = fm.acquireStagingBuffer();
    EXPECT_EQ(b3, nullptr) << "pool exhausted should return nullptr";

    fm.releaseStagingBuffer(b1);
    b3 = fm.acquireStagingBuffer();
    EXPECT_NE(b3, nullptr) << "released buffer should be re-acquirable";
}

TEST_F(DiskSpillFileManagerTest, StatsTrackBasics) {
    auto cfg = makeConfig();
    DiskSpillFileManager fm(cfg);
    ASSERT_TRUE(fm.init());
    const auto stats = fm.getStats();
    EXPECT_GT(stats.slot_count, 0u);
    EXPECT_GE(stats.segment_count, 1u);
    EXPECT_EQ(stats.io_mode, DiskSpillFileManager::IoMode::BUFFERED);
    EXPECT_FALSE(stats.unhealthy);
    EXPECT_EQ(stats.staging_used, 0u);
    EXPECT_GT(stats.staging_total, 0u);
}

TEST_F(DiskSpillFileManagerTest, UnhealthyAfterRepeatedFailures) {
    auto cfg = makeConfig();
    DiskSpillFileManager fm(cfg);
    ASSERT_TRUE(fm.init());
    // try writing to an out-of-range slot 5 times -> mark unhealthy
    const int bad_slot = static_cast<int>(fm.slotCount()) + 10;
    std::vector<char> data(cfg.slot_stride_bytes, 'x');
    for (int i = 0; i < 6; ++i) {
        fm.pwriteSlot(bad_slot, data.data(), data.size());
    }
    EXPECT_TRUE(fm.isUnhealthy());
}

TEST_F(DiskSpillFileManagerTest, ProbeHealthRecovers) {
    auto cfg = makeConfig();
    DiskSpillFileManager fm(cfg);
    ASSERT_TRUE(fm.init());
    fm.forceUnhealthy_TestOnly();
    EXPECT_TRUE(fm.isUnhealthy());
    EXPECT_TRUE(fm.probeHealth());
    EXPECT_FALSE(fm.isUnhealthy());
}

TEST_F(DiskSpillFileManagerTest, CleanupOnDestroyRemovesRunDir) {
    auto cfg = makeConfig();
    cfg.cleanup_on_destroy = true;
    std::string run_dir_observed;
    {
        DiskSpillFileManager fm(cfg);
        ASSERT_TRUE(fm.init());
        // Walk base_path to find the run dir
        for (auto& p : std::filesystem::recursive_directory_iterator(base_path_)) {
            const auto path = p.path().string();
            if (path.find("host_testhost_pid_") != std::string::npos) {
                run_dir_observed = path;
            }
        }
        ASSERT_FALSE(run_dir_observed.empty());
    }
    EXPECT_FALSE(std::filesystem::exists(run_dir_observed));
}

}  // namespace
}  // namespace rtp_llm
