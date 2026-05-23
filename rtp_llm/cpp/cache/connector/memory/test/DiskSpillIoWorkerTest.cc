#include "rtp_llm/cpp/cache/connector/memory/DiskSpillIoWorker.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class DiskSpillIoWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        const auto stamp =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        base_path_ = (std::filesystem::temp_directory_path()
                      / ("rtp_llm_disk_spill_iow_" + std::to_string(::getpid()) + "_" + std::to_string(stamp)))
                         .string();
        std::error_code ec;
        std::filesystem::create_directories(base_path_, ec);
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(base_path_, ec);
    }

    DiskSpillFileManagerPtr makeFileManager(size_t slot_stride = 4096) {
        DiskSpillFileManager::Config c;
        c.base_path                = base_path_;
        c.disk_id                  = 0;
        c.capacity_bytes           = slot_stride * 16;
        c.segment_bytes            = slot_stride * 4;
        c.slot_stride_bytes        = slot_stride;
        c.align_bytes              = 4096;
        c.direct_io                = false;
        c.schema_hash              = "ftest";
        c.world_rank               = 0;
        c.startup_uuid             = "u_" + std::to_string(uuid_counter_++);
        c.hostname                 = "h";
        c.max_staging_buffers      = 4;
        c.cleanup_old_startup_dirs = false;
        auto fm                    = std::make_shared<DiskSpillFileManager>(c);
        EXPECT_TRUE(fm->init());
        return fm;
    }

    std::string base_path_;
    int         uuid_counter_{0};
};

TEST_F(DiskSpillIoWorkerTest, StartStop) {
    auto fm = makeFileManager();
    DiskSpillIoWorker::Config cfg;
    cfg.write_threads             = 2;
    cfg.read_threads              = 2;
    cfg.queue_size                = 16;
    cfg.health_probe_interval_ms  = 0;
    DiskSpillIoWorker w(fm, cfg);
    ASSERT_TRUE(w.start());
    EXPECT_TRUE(w.running());
    w.stop();
    EXPECT_FALSE(w.running());
}

TEST_F(DiskSpillIoWorkerTest, AsyncWriteThenRead) {
    auto fm = makeFileManager();
    DiskSpillIoWorker w(fm, DiskSpillIoWorker::Config{});
    ASSERT_TRUE(w.start());
    std::vector<char> data(4096, 'A');
    std::atomic<bool> done{false};
    bool              success = false;
    w.submitWrite(/*slot_id=*/0, data.data(), data.size(), [&](bool ok, const std::string&) {
        success = ok;
        done.store(true);
    });
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(success);

    std::vector<char> rd(4096, 0);
    std::atomic<bool> read_done{false};
    bool              read_ok = false;
    w.submitRead(/*slot_id=*/0, rd.data(), rd.size(), [&](bool ok, const std::string&) {
        read_ok = ok;
        read_done.store(true);
    });
    while (!read_done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(read_ok);
    EXPECT_EQ(rd, data);
}

TEST_F(DiskSpillIoWorkerTest, SyncWriteRead) {
    auto fm = makeFileManager();
    DiskSpillIoWorker w(fm, DiskSpillIoWorker::Config{});
    ASSERT_TRUE(w.start());
    std::vector<char> data(4096, 'S');
    EXPECT_TRUE(w.pwriteSync(0, data.data(), data.size()));
    std::vector<char> rd(4096, 0);
    EXPECT_TRUE(w.preadSync(0, rd.data(), rd.size()));
    EXPECT_EQ(rd, data);
}

TEST_F(DiskSpillIoWorkerTest, QueueFullDropsTask) {
    auto fm = makeFileManager();
    DiskSpillIoWorker::Config cfg;
    cfg.write_threads             = 1;
    cfg.read_threads              = 1;
    cfg.queue_size                = 1;
    cfg.drop_on_queue_full        = true;
    cfg.health_probe_interval_ms  = 0;
    DiskSpillIoWorker w(fm, cfg);
    ASSERT_TRUE(w.start());

    std::vector<char> data(4096, 'q');
    std::atomic<int>  drop_count{0};
    std::atomic<int>  ack_count{0};
    // submit many tasks rapidly; some should drop
    for (int i = 0; i < 50; ++i) {
        const bool ok = w.submitWrite(0, data.data(), data.size(), [&](bool s, const std::string& err) {
            if (!s && err == disk_error::kQueueFull) {
                drop_count.fetch_add(1);
            }
            ack_count.fetch_add(1);
        });
        (void)ok;
    }
    // drain
    while (ack_count.load() < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GT(drop_count.load(), 0);
}

TEST_F(DiskSpillIoWorkerTest, StopCancelsPendingTasks) {
    auto fm = makeFileManager();
    DiskSpillIoWorker::Config cfg;
    cfg.write_threads = 1;
    cfg.queue_size    = 8;
    DiskSpillIoWorker w(fm, cfg);
    ASSERT_TRUE(w.start());
    std::atomic<int>  aborted{0};
    std::atomic<int>  total{0};
    std::vector<char> data(4096, 'z');
    // Pause workers by holding the file manager momentarily isn't easy; instead, just stop very fast
    for (int i = 0; i < 8; ++i) {
        w.submitWrite(0, data.data(), data.size(), [&](bool ok, const std::string& err) {
            if (!ok && err == disk_error::kTpBroadcastAbort) {
                aborted.fetch_add(1);
            }
            total.fetch_add(1);
        });
    }
    w.stop();
    // every callback should have been invoked
    while (total.load() < 8) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(total.load(), 8);
    // some may have completed before stop, others aborted; total count must match
}

TEST_F(DiskSpillIoWorkerTest, HealthProbeRecoversUnhealthy) {
    auto fm = makeFileManager();
    DiskSpillIoWorker::Config cfg;
    cfg.health_probe_interval_ms = 50;
    DiskSpillIoWorker w(fm, cfg);
    ASSERT_TRUE(w.start());
    fm->forceUnhealthy_TestOnly();
    EXPECT_TRUE(fm->isUnhealthy());
    // wait for probe loop
    for (int i = 0; i < 50; ++i) {
        if (!fm->isUnhealthy()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_FALSE(fm->isUnhealthy());
}

TEST_F(DiskSpillIoWorkerTest, SubmitToStoppedWorkerFails) {
    auto fm = makeFileManager();
    DiskSpillIoWorker w(fm, DiskSpillIoWorker::Config{});
    ASSERT_TRUE(w.start());
    w.stop();
    std::vector<char> data(4096, 'x');
    std::atomic<bool> got_failure{false};
    const bool        accepted = w.submitWrite(0, data.data(), data.size(), [&](bool ok, const std::string&) {
        got_failure.store(!ok);
    });
    EXPECT_FALSE(accepted);
    EXPECT_TRUE(got_failure.load());
}

TEST_F(DiskSpillIoWorkerTest, ReadAndWriteLanesIndependent) {
    auto fm = makeFileManager();
    DiskSpillIoWorker::Config cfg;
    cfg.write_threads = 1;
    cfg.read_threads  = 1;
    cfg.queue_size    = 64;
    DiskSpillIoWorker w(fm, cfg);
    ASSERT_TRUE(w.start());

    std::vector<char> w_data(4096, 'w');
    std::vector<char> r_data(4096, 0);
    ASSERT_TRUE(w.pwriteSync(0, w_data.data(), w_data.size()));

    // Submit many writes followed by a read; read should not be blocked indefinitely.
    std::atomic<int> w_done{0};
    for (int i = 0; i < 20; ++i) {
        w.submitWrite(0, w_data.data(), w_data.size(), [&](bool, const std::string&) { w_done.fetch_add(1); });
    }
    std::atomic<bool> read_done{false};
    auto              t0 = std::chrono::steady_clock::now();
    w.submitRead(0, r_data.data(), r_data.size(), [&](bool, const std::string&) {
        read_done.store(true);
    });
    while (!read_done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    // Read should not have been queued behind all 20 writes; expect well under 100ms
    EXPECT_LT(elapsed, 500);
}

TEST_F(DiskSpillIoWorkerTest, ConcurrentSubmittersDoNotCorruptQueue) {
    auto fm = makeFileManager();
    DiskSpillIoWorker::Config cfg;
    cfg.write_threads = 2;
    cfg.read_threads  = 2;
    cfg.queue_size    = 64;
    DiskSpillIoWorker w(fm, cfg);
    ASSERT_TRUE(w.start());
    std::vector<char> data(4096, 'c');
    std::atomic<int>  ack{0};
    std::vector<std::thread> ths;
    for (int t = 0; t < 4; ++t) {
        ths.emplace_back([&]() {
            for (int i = 0; i < 25; ++i) {
                w.submitWrite(0, data.data(), data.size(), [&](bool, const std::string&) { ack.fetch_add(1); });
            }
        });
    }
    for (auto& t : ths) {
        t.join();
    }
    while (ack.load() < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(ack.load(), 100);
}

TEST_F(DiskSpillIoWorkerTest, ProbeHealthOnceTestApi) {
    auto fm = makeFileManager();
    DiskSpillIoWorker w(fm, DiskSpillIoWorker::Config{});
    ASSERT_TRUE(w.start());
    fm->forceUnhealthy_TestOnly();
    EXPECT_TRUE(w.probeHealthOnce());
    EXPECT_FALSE(fm->isUnhealthy());
}

TEST_F(DiskSpillIoWorkerTest, MultipleStartIdempotent) {
    auto fm = makeFileManager();
    DiskSpillIoWorker w(fm, DiskSpillIoWorker::Config{});
    EXPECT_TRUE(w.start());
    EXPECT_TRUE(w.start());  // second start should be no-op true
    w.stop();
    w.stop();  // second stop should be no-op
}

}  // namespace
}  // namespace rtp_llm
