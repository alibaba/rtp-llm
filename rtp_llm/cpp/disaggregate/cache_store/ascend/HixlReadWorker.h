// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlProtocol.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

enum class HixlLoadRejection {
    TIMEOUT,
    STOPPED
};

struct HixlPendingLoad {
    std::string                            endpoint;
    std::chrono::steady_clock::time_point  deadline;
    std::function<void(uint64_t)>          start;
    std::function<void(HixlLoadRejection)> reject;
};

// Queue before manifest RPCs, so waiting work has neither a sequence nor a
// remote DMA grant. One instance thread dispatches without holding the mutex.
class HixlLoadScheduler {
public:
    HixlLoadScheduler(size_t max_inflight, size_t max_pending);
    ~HixlLoadScheduler();
    void start();
    bool submit(HixlPendingLoad load);
    // Called once after the source acknowledges retirement, not at DMA completion.
    void     retired(const std::string& endpoint, uint64_t sequence);
    void     stop();
    bool     waitForIdle(std::chrono::milliseconds timeout);
    size_t   inflight() const;
    size_t   pending() const;
    uint64_t retiredThrough(const std::string& endpoint) const;

private:
    struct Peer {
        uint64_t           next_sequence{0};
        std::set<uint64_t> active;
        bool               hasSequenceSpace() const;
    };
    void run();

    const size_t                          max_inflight_;
    const size_t                          max_pending_;
    mutable std::mutex                    mutex_;
    std::condition_variable               changed_;
    std::list<HixlPendingLoad>            pending_;
    std::unordered_map<std::string, Peer> peers_;
    size_t                                active_{0};
    bool                                  stopping_{false};
    std::thread                           thread_;
};

enum class HixlLinkState {
    CONNECTING,
    READY,
    CLOSED,
    FAILED
};
enum class HixlReadState {
    WAITING,
    COMPLETE,
    FAILED
};
enum class HixlReadResult {
    OK,
    TIMEOUT,
    CONNECT_ERROR,
    TRANSFER_ERROR,
    STOPPED
};

// All methods run on one progress thread, including context activation.
// CLOSED after disconnect is a physical drain, not a logical cancellation.
class HixlReadDriver {
public:
    virtual ~HixlReadDriver()                                                                               = default;
    virtual bool          activate()                                                                        = 0;
    virtual bool          connect(const std::string& peer)                                                  = 0;
    virtual HixlLinkState connection(const std::string& peer)                                               = 0;
    virtual bool read(const std::string& peer, const std::vector<HixlTransferSlice>& slices, void*& handle) = 0;
    virtual HixlReadState poll(void* handle)                                                                = 0;
    virtual bool          disconnect(const std::string& peer)                                               = 0;
};

struct HixlReadJob {
    std::string                           peer;
    std::vector<HixlTransferSlice>        slices;
    std::chrono::steady_clock::time_point deadline;
    // Includes the actual destination allocator leases and source-grant owner.
    std::shared_ptr<void>               keepalive;
    std::function<void(HixlReadResult)> done;
    std::string                         source_instance;
};

// Reads already admitted by HixlLoadScheduler; no thread or connection per request.
// Callbacks run after physical completion/drain, without the queue mutex.
// The owner must stop admission and destroy this object outside its callbacks.
class HixlReadProgress {
public:
    explicit HixlReadProgress(std::unique_ptr<HixlReadDriver> driver);
    ~HixlReadProgress();
    bool start();
    bool submit(HixlReadJob job);
    void stop();

private:
    struct Operation {
        HixlReadJob job;
        void*       handle{nullptr};
    };
    struct Peer {
        enum class Phase {
            CONNECTING,
            READY,
            DRAINING,
            CLOSED
        };
        Phase                                 phase{Phase::CONNECTING};
        std::string                           source_instance;
        HixlReadResult                        result{HixlReadResult::CONNECT_ERROR};
        std::chrono::steady_clock::time_point drain_deadline;
        std::deque<Operation>                 operations;
    };
    void                     run();
    void                     accept();
    void                     progress(const std::string& name, Peer& peer);
    void                     progressReads(const std::string& name, Peer& peer);
    void                     drain(const std::string& name, Peer& peer, HixlReadResult result);
    void                     finish(Operation operation, HixlReadResult result);
    bool                     idle() const;
    [[noreturn]] static void fatalDrain(const std::string& peer);

    std::unique_ptr<HixlReadDriver> driver_;
    mutable std::mutex              mutex_;
    std::condition_variable         wake_;
    std::deque<HixlReadJob>         incoming_;
    bool                            stopping_{false};
    std::thread                     thread_;
    // Progress-thread-owned state.
    bool                                  shutdown_{false};
    std::unordered_map<std::string, Peer> peers_;
};

}  // namespace rtp_llm
