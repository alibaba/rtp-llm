#include "rtp_llm/cpp/cache/connector/memory/DiskSpillIoWorker.h"

#include <chrono>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

DiskSpillIoWorker::DiskSpillIoWorker(DiskSpillFileManagerPtr file_manager, Config config)
    : file_manager_(std::move(file_manager)), config_(config) {}

DiskSpillIoWorker::~DiskSpillIoWorker() {
    stop();
}

bool DiskSpillIoWorker::start() {
    if (!file_manager_) {
        return false;
    }
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }
    const int write_n = std::max(1, config_.write_threads);
    const int read_n  = std::max(1, config_.read_threads);
    write_threads_.reserve(write_n);
    read_threads_.reserve(read_n);
    for (int i = 0; i < write_n; ++i) {
        write_threads_.emplace_back([this]() { writeLoop(); });
    }
    for (int i = 0; i < read_n; ++i) {
        read_threads_.emplace_back([this]() { readLoop(); });
    }
    if (config_.health_probe_interval_ms > 0) {
        health_thread_ = std::thread([this]() { healthLoop(); });
    }
    RTP_LLM_LOG_INFO("disk spill io worker started, disk_id=%d write_threads=%d read_threads=%d queue=%d",
                     file_manager_->diskId(),
                     write_n,
                     read_n,
                     config_.queue_size);
    return true;
}

void DiskSpillIoWorker::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return;
    }
    write_cv_.notify_all();
    read_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lk(health_mutex_);
        health_cv_.notify_all();
    }
    for (auto& t : write_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    for (auto& t : read_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (health_thread_.joinable()) {
        health_thread_.join();
    }
    write_threads_.clear();
    read_threads_.clear();

    // cancel pending tasks
    std::deque<Task> pending_w;
    std::deque<Task> pending_r;
    {
        std::lock_guard<std::mutex> lk(write_mutex_);
        pending_w.swap(write_queue_);
    }
    {
        std::lock_guard<std::mutex> lk(read_mutex_);
        pending_r.swap(read_queue_);
    }
    for (auto& t : pending_w) {
        if (t.cb) {
            t.cb(false, disk_error::kTpBroadcastAbort);
        }
    }
    for (auto& t : pending_r) {
        if (t.cb) {
            t.cb(false, disk_error::kTpBroadcastAbort);
        }
    }
}

bool DiskSpillIoWorker::pushWrite(Task&& t) {
    std::unique_lock<std::mutex> lk(write_mutex_);
    if (static_cast<int>(write_queue_.size()) >= config_.queue_size) {
        if (config_.drop_on_queue_full) {
            return false;
        }
        write_cv_.wait(lk, [this]() {
            return !running_.load() || static_cast<int>(write_queue_.size()) < config_.queue_size;
        });
        if (!running_.load()) {
            return false;
        }
    }
    write_queue_.push_back(std::move(t));
    write_cv_.notify_one();
    return true;
}

bool DiskSpillIoWorker::pushRead(Task&& t) {
    std::unique_lock<std::mutex> lk(read_mutex_);
    if (static_cast<int>(read_queue_.size()) >= config_.queue_size) {
        if (config_.drop_on_queue_full) {
            return false;
        }
        read_cv_.wait(lk,
                      [this]() { return !running_.load() || static_cast<int>(read_queue_.size()) < config_.queue_size; });
        if (!running_.load()) {
            return false;
        }
    }
    read_queue_.push_back(std::move(t));
    read_cv_.notify_one();
    return true;
}

bool DiskSpillIoWorker::submitWrite(int slot_id, const void* data, size_t bytes, IoCallback cb) {
    if (!running_.load()) {
        if (cb) {
            cb(false, disk_error::kTpBroadcastAbort);
        }
        return false;
    }
    Task t;
    t.op                = Op::WRITE;
    t.slot_id           = slot_id;
    t.read_or_write_buf = data;
    t.read_dst_buf      = nullptr;
    t.bytes             = bytes;
    t.cb                = cb;  // intentional copy: we still need cb on drop fallback
    if (pushWrite(std::move(t))) {
        return true;
    }
    if (cb) {
        cb(false, disk_error::kQueueFull);
    }
    return false;
}

bool DiskSpillIoWorker::submitRead(int slot_id, void* data, size_t bytes, IoCallback cb) {
    if (!running_.load()) {
        if (cb) {
            cb(false, disk_error::kTpBroadcastAbort);
        }
        return false;
    }
    Task t;
    t.op                = Op::READ;
    t.slot_id           = slot_id;
    t.read_or_write_buf = nullptr;
    t.read_dst_buf      = data;
    t.bytes             = bytes;
    t.cb                = cb;
    if (pushRead(std::move(t))) {
        return true;
    }
    if (cb) {
        cb(false, disk_error::kQueueFull);
    }
    return false;
}

bool DiskSpillIoWorker::pwriteSync(int slot_id, const void* data, size_t bytes) {
    return file_manager_->pwriteSlot(slot_id, data, bytes);
}

bool DiskSpillIoWorker::preadSync(int slot_id, void* data, size_t bytes) {
    return file_manager_->preadSlot(slot_id, data, bytes);
}

void DiskSpillIoWorker::runTask(const Task& task) {
    bool ok = false;
    if (task.op == Op::WRITE) {
        write_inflight_.fetch_add(1);
        ok = file_manager_->pwriteSlot(task.slot_id, task.read_or_write_buf, task.bytes);
        write_inflight_.fetch_sub(1);
    } else {
        read_inflight_.fetch_add(1);
        ok = file_manager_->preadSlot(task.slot_id, task.read_dst_buf, task.bytes);
        read_inflight_.fetch_sub(1);
    }
    if (task.cb) {
        const std::string err = ok ? "ok" : (task.op == Op::WRITE ? disk_error::kPwrite : disk_error::kPread);
        try {
            task.cb(ok, err);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("disk spill io callback threw: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("disk spill io callback threw unknown");
        }
    }
}

void DiskSpillIoWorker::writeLoop() {
    while (running_.load()) {
        Task t;
        {
            std::unique_lock<std::mutex> lk(write_mutex_);
            write_cv_.wait(lk, [this]() { return !running_.load() || !write_queue_.empty(); });
            if (!running_.load() && write_queue_.empty()) {
                return;
            }
            if (write_queue_.empty()) {
                continue;
            }
            t = std::move(write_queue_.front());
            write_queue_.pop_front();
        }
        write_cv_.notify_one();
        runTask(t);
    }
}

void DiskSpillIoWorker::readLoop() {
    while (running_.load()) {
        Task t;
        {
            std::unique_lock<std::mutex> lk(read_mutex_);
            read_cv_.wait(lk, [this]() { return !running_.load() || !read_queue_.empty(); });
            if (!running_.load() && read_queue_.empty()) {
                return;
            }
            if (read_queue_.empty()) {
                continue;
            }
            t = std::move(read_queue_.front());
            read_queue_.pop_front();
        }
        read_cv_.notify_one();
        runTask(t);
    }
}

void DiskSpillIoWorker::healthLoop() {
    while (running_.load()) {
        {
            std::unique_lock<std::mutex> lk(health_mutex_);
            health_cv_.wait_for(lk,
                                std::chrono::milliseconds(config_.health_probe_interval_ms),
                                [this]() { return !running_.load(); });
        }
        if (!running_.load()) {
            return;
        }
        if (file_manager_ && file_manager_->isUnhealthy()) {
            const bool ok = file_manager_->probeHealth();
            if (ok) {
                RTP_LLM_LOG_INFO("disk spill probe recovered disk_id=%d path_hash=%s",
                                 file_manager_->diskId(),
                                 file_manager_->pathHash().c_str());
            }
        }
    }
}

bool DiskSpillIoWorker::probeHealthOnce() {
    if (!file_manager_) {
        return false;
    }
    return file_manager_->probeHealth();
}

size_t DiskSpillIoWorker::writeQueueDepth() const {
    std::lock_guard<std::mutex> lk(write_mutex_);
    return write_queue_.size();
}

size_t DiskSpillIoWorker::readQueueDepth() const {
    std::lock_guard<std::mutex> lk(read_mutex_);
    return read_queue_.size();
}

size_t DiskSpillIoWorker::writeInflight() const {
    return write_inflight_.load();
}

size_t DiskSpillIoWorker::readInflight() const {
    return read_inflight_.load();
}

}  // namespace rtp_llm
