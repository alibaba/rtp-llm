// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0

#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlReadWorker.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <future>
#include <utility>

namespace rtp_llm {

HixlLoadScheduler::HixlLoadScheduler(size_t max_inflight, size_t max_pending):
    max_inflight_(max_inflight), max_pending_(max_pending) {}

HixlLoadScheduler::~HixlLoadScheduler() {
    stop();
}

void HixlLoadScheduler::start() {
    thread_ = std::thread([this] { run(); });
}

bool HixlLoadScheduler::submit(HixlPendingLoad load) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || pending_.size() >= max_pending_) {
            return false;
        }
        pending_.push_back(std::move(load));
    }
    changed_.notify_all();
    return true;
}

bool HixlLoadScheduler::Peer::hasSequenceSpace() const {
    return active.empty() || next_sequence - (*active.begin() - 1) < kHixlTransferWindow;
}

void HixlLoadScheduler::retired(const std::string& endpoint, uint64_t sequence) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (peers_.at(endpoint).active.erase(sequence) == 0) {
            return;
        }
        --active_;
    }
    changed_.notify_all();
}

void HixlLoadScheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool HixlLoadScheduler::waitForIdle(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, timeout, [this] { return active_ == 0 && pending_.empty(); });
}

size_t HixlLoadScheduler::inflight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
}

size_t HixlLoadScheduler::pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
}

uint64_t HixlLoadScheduler::retiredThrough(const std::string& endpoint) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto&                 peer = peers_.at(endpoint);
    return peer.active.empty() ? peer.next_sequence : *peer.active.begin() - 1;
}

void HixlLoadScheduler::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        const auto now      = std::chrono::steady_clock::now();
        auto       wake_at  = std::chrono::steady_clock::time_point::max();
        auto       chosen   = pending_.end();
        uint64_t   sequence = 0;
        for (auto it = pending_.begin(); it != pending_.end(); ++it) {
            if (stopping_ || it->deadline <= now) {
                chosen = it;
                break;
            }
            auto& peer = peers_[it->endpoint];
            if (active_ < max_inflight_ && peer.hasSequenceSpace()) {
                sequence = ++peer.next_sequence;
                peer.active.insert(sequence);
                ++active_;
                chosen = it;
                break;
            }
            wake_at = std::min(wake_at, it->deadline);
        }
        if (chosen != pending_.end()) {
            auto load = std::move(*chosen);
            pending_.erase(chosen);
            const auto reason = stopping_ ? HixlLoadRejection::STOPPED : HixlLoadRejection::TIMEOUT;
            lock.unlock();
            if (sequence) {
                load.start(sequence);
            } else {
                load.reject(reason);
            }
            lock.lock();
            changed_.notify_all();
        } else if (stopping_) {
            return;
        } else if (wake_at == std::chrono::steady_clock::time_point::max()) {
            changed_.wait(lock);
        } else {
            changed_.wait_until(lock, wake_at);
        }
    }
}

HixlReadProgress::HixlReadProgress(std::unique_ptr<HixlReadDriver> driver): driver_(std::move(driver)) {}

HixlReadProgress::~HixlReadProgress() {
    stop();
}

bool HixlReadProgress::start() {
    std::promise<bool> ready;
    auto               result = ready.get_future();
    thread_                   = std::thread([this, ready = std::move(ready)]() mutable {
        const bool active = driver_->activate();
        ready.set_value(active);
        if (active) {
            run();
        }
    });
    return result.get();
}

bool HixlReadProgress::submit(HixlReadJob job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return false;
        }
        incoming_.push_back(std::move(job));
    }
    wake_.notify_one();
    return true;
}

void HixlReadProgress::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void HixlReadProgress::finish(Operation operation, HixlReadResult result) {
    // Operation holds the leases until the callback itself has returned.
    operation.job.done(result);
}

void HixlReadProgress::accept() {
    std::deque<HixlReadJob> incoming;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        incoming.swap(incoming_);
        shutdown_ = stopping_;
    }
    for (auto& job : incoming) {
        if (shutdown_ || std::chrono::steady_clock::now() >= job.deadline) {
            finish({std::move(job)}, shutdown_ ? HixlReadResult::STOPPED : HixlReadResult::TIMEOUT);
            continue;
        }
        auto previous = peers_.find(job.peer);
        if (previous != peers_.end() && previous->second.phase == Peer::Phase::CLOSED) {
            // CLOSED is reached only after the previous connection physically drains.
            peers_.erase(previous);
        }
        auto  inserted = peers_.try_emplace(job.peer);
        auto& peer     = inserted.first->second;
        if (inserted.second) {
            peer.source_instance = job.source_instance;
        }
        if (inserted.second && !driver_->connect(job.peer)) {
            peer.operations.push_back({std::move(job)});
            drain(inserted.first->first, peer, HixlReadResult::CONNECT_ERROR);
        } else if (peer.phase == Peer::Phase::DRAINING) {
            finish({std::move(job)}, peer.result);
        } else if (peer.source_instance != job.source_instance) {
            // A new source process invalidates the old remote registrations.
            // Reject this load after draining; the next load establishes a fresh link.
            peer.operations.push_back({std::move(job)});
            drain(inserted.first->first, peer, HixlReadResult::CONNECT_ERROR);
        } else {
            peer.operations.push_back({std::move(job)});
        }
    }
}

void HixlReadProgress::drain(const std::string& name, Peer& peer, HixlReadResult result) {
    if (peer.phase == Peer::Phase::DRAINING || peer.phase == Peer::Phase::CLOSED) {
        return;
    }
    peer.phase          = Peer::Phase::DRAINING;
    peer.result         = result;
    peer.drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    // Even a failed submit can have reached the device. No callback or lease
    // release is allowed until disconnect reports a physical drain.
    if (!driver_->disconnect(name)) {
        fatalDrain(name);
    }
}

void HixlReadProgress::progress(const std::string& name, Peer& peer) {
    if (shutdown_) {
        drain(name, peer, HixlReadResult::STOPPED);
    }
    if (peer.phase == Peer::Phase::CLOSED) {
        return;
    }
    if (peer.phase == Peer::Phase::DRAINING) {
        if (driver_->connection(name) == HixlLinkState::CLOSED) {
            peer.phase      = Peer::Phase::CLOSED;
            auto operations = std::move(peer.operations);
            for (auto& op : operations) {
                finish(std::move(op), peer.result);
            }
        } else if (std::chrono::steady_clock::now() >= peer.drain_deadline) {
            fatalDrain(name);
        }
        return;
    }
    if (peer.phase == Peer::Phase::CONNECTING) {
        const auto state = driver_->connection(name);
        if (state == HixlLinkState::READY) {
            peer.phase = Peer::Phase::READY;
        } else if (state == HixlLinkState::FAILED || state == HixlLinkState::CLOSED) {
            drain(name, peer, HixlReadResult::CONNECT_ERROR);
            return;
        }
    }
    progressReads(name, peer);
}

void HixlReadProgress::progressReads(const std::string& name, Peer& peer) {
    for (auto it = peer.operations.begin(); it != peer.operations.end();) {
        if (std::chrono::steady_clock::now() >= it->job.deadline) {
            drain(name, peer, HixlReadResult::TIMEOUT);
            return;
        }
        if (peer.phase != Peer::Phase::READY) {
            ++it;
            continue;
        }
        if (!it->handle) {
            if (!driver_->read(name, it->job.slices, it->handle)) {
                drain(name, peer, HixlReadResult::TRANSFER_ERROR);
                return;
            }
        }
        const auto state = driver_->poll(it->handle);
        if (state == HixlReadState::FAILED) {
            drain(name, peer, HixlReadResult::TRANSFER_ERROR);
            return;
        }
        if (state == HixlReadState::COMPLETE) {
            auto op = std::move(*it);
            it      = peer.operations.erase(it);
            finish(std::move(op), HixlReadResult::OK);
        } else {
            ++it;
        }
    }
}

bool HixlReadProgress::idle() const {
    for (const auto& item : peers_) {
        if (!item.second.operations.empty() || item.second.phase == Peer::Phase::DRAINING) {
            return false;
        }
    }
    return true;
}

void HixlReadProgress::run() {
    for (;;) {
        accept();
        for (auto& item : peers_) {
            progress(item.first, item.second);
        }
        if (shutdown_ && idle()) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        if (idle()) {
            wake_.wait(lock, [this] { return stopping_ || !incoming_.empty(); });
        } else {
            wake_.wait_for(lock, std::chrono::microseconds(100), [this] { return !incoming_.empty(); });
        }
    }
}

[[noreturn]] void HixlReadProgress::fatalDrain(const std::string& peer) {
    // Continuing would allow an allocator to recycle memory still in DMA use.
    // Escalate an unrecoverable device failure to the owning model process.
    std::fprintf(
        stderr, "HIXL physical drain failed for %s; terminating before releasing device leases\n", peer.c_str());
    std::terminate();
}

}  // namespace rtp_llm
