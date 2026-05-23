#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskSpillFileManager.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillTypes.h"

namespace rtp_llm {

// DiskSpillIoWorker: per-disk worker thread pool that owns the BLOCKING IO calls
// against a DiskSpillFileManager. Adds:
//   - bounded queue (with optional drop_on_queue_full)
//   - separate read lane (read_workers) so spill writes can't starve disk reads
//   - EINTR retry is done inside FileManager; this layer handles enqueue + dispatch
//   - periodic health probe to recover unhealthy disks
//   - graceful drain + cancel on stop()
class DiskSpillIoWorker {
public:
    using IoCallback = std::function<void(bool success, const std::string& error_type)>;

    struct Config {
        int  write_threads{2};
        int  read_threads{2};
        int  queue_size{1024};
        bool drop_on_queue_full{true};
        int  health_probe_interval_ms{30000};
    };

public:
    DiskSpillIoWorker(DiskSpillFileManagerPtr file_manager, Config config);
    ~DiskSpillIoWorker();

    DiskSpillIoWorker(const DiskSpillIoWorker&)            = delete;
    DiskSpillIoWorker& operator=(const DiskSpillIoWorker&) = delete;

    bool start();
    void stop();

    // Submit an async pwrite. Buffer must remain valid until callback fires.
    // Returns true if accepted; false if queue full and drop_on_queue_full=true.
    bool submitWrite(int slot_id, const void* data, size_t bytes, IoCallback cb);
    bool submitRead(int slot_id, void* data, size_t bytes, IoCallback cb);

    // Synchronous variants — block calling thread; useful for init/handshake or
    // single-issue tests. NOT routed through queue.
    bool pwriteSync(int slot_id, const void* data, size_t bytes);
    bool preadSync(int slot_id, void* data, size_t bytes);

    // Stats
    size_t writeQueueDepth() const;
    size_t readQueueDepth() const;
    size_t writeInflight() const;
    size_t readInflight() const;
    bool   running() const {
        return running_.load();
    }

    // Test-only: trigger one health probe synchronously.
    bool probeHealthOnce();

private:
    enum class Op : uint8_t {
        WRITE = 0,
        READ  = 1,
    };

    struct Task {
        Op          op;
        int         slot_id;
        const void* read_or_write_buf;
        void*       read_dst_buf;
        size_t      bytes;
        IoCallback  cb;
    };

    void writeLoop();
    void readLoop();
    void healthLoop();
    void runTask(const Task& task);

    bool pushWrite(Task&& t);
    bool pushRead(Task&& t);

    DiskSpillFileManagerPtr           file_manager_;
    Config                            config_;
    std::atomic<bool>                 running_{false};

    mutable std::mutex                write_mutex_;
    std::condition_variable           write_cv_;
    std::deque<Task>                  write_queue_;
    std::atomic<size_t>               write_inflight_{0};

    mutable std::mutex                read_mutex_;
    std::condition_variable           read_cv_;
    std::deque<Task>                  read_queue_;
    std::atomic<size_t>               read_inflight_{0};

    std::vector<std::thread>          write_threads_;
    std::vector<std::thread>          read_threads_;
    std::thread                       health_thread_;
    std::mutex                        health_mutex_;
    std::condition_variable           health_cv_;
};

using DiskSpillIoWorkerPtr = std::shared_ptr<DiskSpillIoWorker>;

}  // namespace rtp_llm
