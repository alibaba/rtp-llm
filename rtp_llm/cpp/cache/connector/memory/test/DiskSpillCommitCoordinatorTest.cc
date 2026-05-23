#include "rtp_llm/cpp/cache/connector/memory/DiskSpillCommitCoordinator.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class DiskSpillCommitCoordinatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        const auto stamp =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        base_path_ = (std::filesystem::temp_directory_path()
                      / ("rtp_llm_cc_test_" + std::to_string(::getpid()) + "_" + std::to_string(stamp)))
                         .string();
        std::error_code ec;
        std::filesystem::create_directories(base_path_, ec);
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(base_path_, ec);
    }

    std::shared_ptr<DiskSpillBlockCache> makeCache(size_t block_size = 4096) {
        DiskSpillBlockCache::InitConfig cfg;
        cfg.disks.push_back(DiskSpillBlockCache::DiskConfig{base_path_, /*capacity_mb=*/1});
        cfg.block_size              = block_size;
        cfg.align_bytes             = 4096;
        cfg.segment_bytes           = 1024UL * 1024UL;
        cfg.direct_io               = false;
        cfg.schema_hash             = "cct";
        cfg.startup_uuid            = "u_" + std::to_string(uuid_++);
        cfg.hostname                = "h";
        cfg.io_threads_per_disk     = 1;
        cfg.io_queue_size           = 32;
        cfg.cleanup_old_startup_dirs = false;
        auto cache = DiskSpillBlockCache::create(cfg);
        EXPECT_TRUE(cache->init());
        return cache;
    }

    std::string base_path_;
    int         uuid_{0};
};

TEST_F(DiskSpillCommitCoordinatorTest, StartStop) {
    auto cache = makeCache();
    DiskSpillCommitCoordinator c(cache, {}, /*worker_count=*/0, nullptr, nullptr, nullptr);
    ASSERT_TRUE(c.start());
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, SubmitSpillTp1Commits) {
    auto cache = makeCache();
    auto slot  = cache->reserve(/*key=*/100, 4096, true);
    ASSERT_TRUE(slot.has_value());

    DiskSpillCommitCoordinator c(cache, {}, /*worker_count=*/0, nullptr, nullptr, nullptr);
    ASSERT_TRUE(c.start());

    std::promise<SpillStageState> done;
    auto                          fut = done.get_future();
    std::vector<char>             data(4096, 'X');
    const auto id =
        c.submitSpill(*slot, data, [&](SpillJobId, SpillStageState s) { done.set_value(s); });
    EXPECT_NE(id, 0u);

    const auto state = fut.wait_for(std::chrono::seconds(2));
    ASSERT_EQ(state, std::future_status::ready);
    EXPECT_EQ(fut.get(), SpillStageState::COMMITTED);
    EXPECT_TRUE(cache->contains(100));
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, SubmitSpillTp2WaitsForWorkerAck) {
    auto cache = makeCache();
    auto slot  = cache->reserve(/*key=*/101, 4096, true);
    ASSERT_TRUE(slot.has_value());

    DiskSpillCommitCoordinator::Config cfg;
    cfg.commit_timeout_ms = 1000;
    cfg.poll_interval_ms  = 20;

    std::atomic<int>           poll_called{0};
    std::atomic<SpillWriteStatus> reported_status{SpillWriteStatus::PENDING};
    DiskSpillCommitCoordinator c(
        cache,
        cfg,
        /*worker_count=*/1,
        [](SpillJobId, const DiskSpillBlockCache::DiskItem&) { return true; },
        nullptr,
        [&](int /*rank*/, SpillJobId) {
            poll_called.fetch_add(1);
            return reported_status.load();
        });
    ASSERT_TRUE(c.start());

    std::promise<SpillStageState> done;
    auto                          fut = done.get_future();
    std::vector<char>             data(4096, 'Y');
    c.submitSpill(*slot, data, [&](SpillJobId, SpillStageState s) { done.set_value(s); });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_NE(fut.wait_for(std::chrono::milliseconds(50)), std::future_status::ready)
        << "should still be PENDING until worker reports success";
    reported_status.store(SpillWriteStatus::SUCCESS);

    const auto status = fut.wait_for(std::chrono::seconds(2));
    ASSERT_EQ(status, std::future_status::ready);
    EXPECT_EQ(fut.get(), SpillStageState::COMMITTED);
    EXPECT_TRUE(cache->contains(101));
    EXPECT_GT(poll_called.load(), 0);
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, WorkerFailureTriggersAbortAndDeleteBroadcast) {
    auto cache = makeCache();
    auto slot  = cache->reserve(/*key=*/102, 4096, true);
    ASSERT_TRUE(slot.has_value());

    std::atomic<int>           delete_calls{0};
    DiskSpillCommitCoordinator c(
        cache,
        {},
        /*worker_count=*/1,
        [](SpillJobId, const DiskSpillBlockCache::DiskItem&) { return true; },
        [&](const DiskSpillBlockCache::DiskItem&) {
            delete_calls.fetch_add(1);
            return true;
        },
        [](int, SpillJobId) { return SpillWriteStatus::FAILED; });
    ASSERT_TRUE(c.start());

    std::promise<SpillStageState> done;
    auto                          fut = done.get_future();
    std::vector<char>             data(4096, 'Z');
    c.submitSpill(*slot, data, [&](SpillJobId, SpillStageState s) { done.set_value(s); });

    const auto s = fut.wait_for(std::chrono::seconds(2));
    ASSERT_EQ(s, std::future_status::ready);
    EXPECT_EQ(fut.get(), SpillStageState::FREE);
    EXPECT_FALSE(cache->contains(102));
    EXPECT_GE(delete_calls.load(), 1);
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, BroadcastFailureAborts) {
    auto cache = makeCache();
    auto slot  = cache->reserve(/*key=*/103, 4096, true);
    ASSERT_TRUE(slot.has_value());

    DiskSpillCommitCoordinator c(
        cache,
        {},
        /*worker_count=*/1,
        [](SpillJobId, const DiskSpillBlockCache::DiskItem&) { return false; },  // broadcast fails
        nullptr,
        nullptr);
    ASSERT_TRUE(c.start());

    std::promise<SpillStageState> done;
    auto                          fut = done.get_future();
    std::vector<char>             data(4096, 'W');
    c.submitSpill(*slot, data, [&](SpillJobId, SpillStageState s) { done.set_value(s); });

    const auto s = fut.wait_for(std::chrono::seconds(2));
    ASSERT_EQ(s, std::future_status::ready);
    EXPECT_NE(fut.get(), SpillStageState::COMMITTED);
    EXPECT_FALSE(cache->contains(103));
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, CommitTimeoutAborts) {
    auto cache = makeCache();
    auto slot  = cache->reserve(/*key=*/104, 4096, true);
    ASSERT_TRUE(slot.has_value());

    DiskSpillCommitCoordinator::Config cfg;
    cfg.commit_timeout_ms = 200;
    cfg.stage_ack_timeout_ms = 50;
    cfg.poll_interval_ms  = 20;
    DiskSpillCommitCoordinator c(
        cache,
        cfg,
        /*worker_count=*/1,
        [](SpillJobId, const DiskSpillBlockCache::DiskItem&) { return true; },
        nullptr,
        [](int, SpillJobId) { return SpillWriteStatus::PENDING; });
    ASSERT_TRUE(c.start());

    std::promise<SpillStageState> done;
    auto                          fut = done.get_future();
    std::vector<char>             data(4096, 'T');
    c.submitSpill(*slot, data, [&](SpillJobId, SpillStageState s) { done.set_value(s); });

    const auto s = fut.wait_for(std::chrono::seconds(2));
    ASSERT_EQ(s, std::future_status::ready);
    EXPECT_NE(fut.get(), SpillStageState::COMMITTED);
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, NotifyWorkerStatusPushPath) {
    auto cache = makeCache();
    auto slot  = cache->reserve(/*key=*/105, 4096, true);
    ASSERT_TRUE(slot.has_value());

    DiskSpillCommitCoordinator c(
        cache,
        {},
        /*worker_count=*/1,
        [](SpillJobId, const DiskSpillBlockCache::DiskItem&) { return true; },
        nullptr,
        [](int, SpillJobId) { return SpillWriteStatus::PENDING; });
    ASSERT_TRUE(c.start());

    std::promise<SpillStageState> done;
    auto                          fut = done.get_future();
    std::vector<char>             data(4096, 'N');
    const auto                    id = c.submitSpill(*slot, data, [&](SpillJobId, SpillStageState s) { done.set_value(s); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    c.notifyWorkerStatus(id, 0, SpillWriteStatus::SUCCESS);

    const auto s = fut.wait_for(std::chrono::seconds(2));
    ASSERT_EQ(s, std::future_status::ready);
    EXPECT_EQ(fut.get(), SpillStageState::COMMITTED);
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, BackpressureWhenMaxInflight) {
    auto cache = makeCache();

    DiskSpillCommitCoordinator::Config cfg;
    cfg.max_inflight_jobs = 2;
    cfg.poll_interval_ms  = 1000;  // slow down so jobs queue up
    DiskSpillCommitCoordinator c(
        cache,
        cfg,
        /*worker_count=*/1,
        [](SpillJobId, const DiskSpillBlockCache::DiskItem&) { return true; },
        nullptr,
        [](int, SpillJobId) { return SpillWriteStatus::PENDING; });  // never resolve
    ASSERT_TRUE(c.start());

    int rejected = 0;
    std::vector<SpillJobId> accepted_ids;
    for (int i = 0; i < 5; ++i) {
        auto slot = cache->reserve(static_cast<CacheKeyType>(200 + i), 4096, true);
        ASSERT_TRUE(slot.has_value());
        std::vector<char> data(4096, static_cast<char>('a' + i));
        const auto id = c.submitSpill(*slot, data, [](SpillJobId, SpillStageState) {});
        if (id == 0) {
            ++rejected;
        } else {
            accepted_ids.push_back(id);
        }
    }
    EXPECT_GE(rejected, 1);
    EXPECT_LE(accepted_ids.size(), 2u);
    c.stop();
}

TEST_F(DiskSpillCommitCoordinatorTest, GetJobStateForUnknownReturnsFree) {
    auto cache = makeCache();
    DiskSpillCommitCoordinator c(cache, {}, /*worker_count=*/0, nullptr, nullptr, nullptr);
    EXPECT_EQ(c.getJobState(99999), SpillStageState::FREE);
    EXPECT_EQ(c.getJobWriteStatus(99999), SpillWriteStatus::UNKNOWN_JOB);
}

}  // namespace
}  // namespace rtp_llm
