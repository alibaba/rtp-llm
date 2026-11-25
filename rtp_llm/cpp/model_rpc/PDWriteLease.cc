#include "rtp_llm/cpp/model_rpc/PDWriteLease.h"

#include "autil/Scope.h"

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>

namespace rtp_llm {
namespace {
std::string newWorkerEpoch() {
    std::random_device random;
    return std::to_string(random()) + "-" + std::to_string(random()) + "-" + std::to_string(random()) + "-"
           + std::to_string(random());
}
}  // namespace

bool samePDWriteLease(const PDWriteLeasePB& expected, const PDWriteLeasePB& actual) {
    return expected.token() != 0 && !expected.worker_epoch().empty() && expected.token() == actual.token()
           && expected.worker_epoch() == actual.worker_epoch();
}

grpc::Status preparePDWriteLeases(std::vector<PDWriteLeasePeer>&        peers,
                                  const PreparePDWriteLeaseRequestPB&   request,
                                  std::chrono::system_clock::time_point deadline,
                                  const std::function<bool()>&          cancelled) {
    struct Call {
        grpc::ClientContext                                              context;
        grpc::Status                                                     status;
        std::unique_ptr<grpc::ClientAsyncResponseReader<PDWriteLeasePB>> reader;
        bool                                                             finished = false;
    };
    std::vector<Call>     calls(peers.size());
    grpc::CompletionQueue queue;
    autil::ScopeGuard     drain([&] {
        for (auto& call : calls) {
            if (call.reader && !call.finished)
                call.context.TryCancel();
        }
        queue.Shutdown();
        void* tag;
        bool  ok;
        while (queue.Next(&tag, &ok)) {}
    });
    for (size_t i = 0; i < peers.size(); ++i) {
        auto& call = calls[i];
        call.context.set_deadline(deadline);
        call.reader = peers[i].stub->AsyncPreparePDWriteLease(&call.context, request, &queue);
        if (!call.reader)
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "could not start lease preparation");
        call.reader->Finish(&peers[i].lease, &call.status, &call);
    }
    for (size_t received = 0; received < calls.size();) {
        if (cancelled())
            return grpc::Status(grpc::StatusCode::CANCELLED, "P/D write lease prepare cancelled");
        const auto now = std::chrono::system_clock::now();
        if (now >= deadline)
            return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "P/D write lease prepare timed out");
        void*      tag   = nullptr;
        bool       ok    = false;
        const auto event = queue.AsyncNext(&tag, &ok, std::min(deadline, now + std::chrono::milliseconds(10)));
        if (event == grpc::CompletionQueue::TIMEOUT)
            continue;
        if (event != grpc::CompletionQueue::GOT_EVENT || !ok) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "P/D write lease prepare reply lost");
        }
        ++received;
        auto& call    = *static_cast<Call*>(tag);
        call.finished = true;
        if (!call.status.ok())
            return call.status;
    }
    for (const auto& peer : peers) {
        if (peer.lease.worker_epoch().empty() || peer.lease.token() == 0) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "worker returned no P/D write lease");
        }
        if (!peer.lease.supports_protected_destinations()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "worker cannot validate protected P/D targets");
        }
    }
    return grpc::Status::OK;
}

PDWriteLeaseRegistry::PDWriteLeaseRegistry(): epoch_(newWorkerEpoch()) {}

PDWriteLeasePB PDWriteLeaseRegistry::prepare(const PreparePDWriteLeaseRequestPB& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_)
        return {};
    const auto now = Clock::now();
    while (!expiry_.empty() && expiry_.begin()->first <= now) {
        entries_.erase(expiry_.begin()->second);
        expiry_.erase(expiry_.begin());
    }
    if (issued_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("P/D write lease tokens exhausted");
    }
    PDWriteLeasePB lease;
    lease.set_worker_epoch(epoch_);
    lease.set_token(++issued_);
    lease.set_supports_protected_destinations(true);
    const auto ttl    = std::chrono::milliseconds(std::clamp<int64_t>(request.timeout_ms(), 1, INT32_MAX));
    auto       expiry = expiry_.emplace(now + ttl, lease.token());
    try {
        entries_.emplace(lease.token(),
                         Entry{request.request_id(), request.request_key(), request.dp_rank(), expiry, false, {}});
    } catch (...) {
        expiry_.erase(expiry);
        throw;
    }
    return lease;
}

std::shared_ptr<void> PDWriteLeaseRegistry::claim(const BroadcastLoadRequestPB& request) {
    const auto& lease = request.write_lease();
    if (lease.worker_epoch() != epoch_)
        return {};
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = entries_.find(lease.token());
    if (it == entries_.end())
        return {};
    auto& entry = it->second;
    if (entry.expiry == expiry_.end() || entry.expiry->first <= Clock::now() || entry.request_id != request.request_id()
        || entry.request_key != request.request_key() || entry.dp_rank != request.dp_rank()) {
        return {};
    }
    auto writer   = std::make_shared<Writer>();
    writer->owner = shared_from_this();
    writer->token = lease.token();
    expiry_.erase(entry.expiry);
    entry.expiry = expiry_.end();
    return writer;
}

PDWriteLeaseRegistry::Writer::~Writer() {
    if (owner) {
        std::lock_guard<std::mutex> lock(owner->mutex_);
        owner->entries_.erase(token);
    }
}

void PDWriteLeaseRegistry::closeAdmission() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    for (const auto& unclaimed : expiry_)
        entries_.erase(unclaimed.second);
    expiry_.clear();
}

size_t PDWriteLeaseRegistry::activeWriters() {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size() - expiry_.size();
}

// These callbacks arm transport recovery. Only Writer destruction certifies that the DMA has stopped.
void PDWriteLeaseRegistry::startRecovery(uint64_t token, Recoveries& recoveries) {
    for (auto it = recoveries.begin(); it != recoveries.end();) {
        try {
            (*it)();
        } catch (...) {
            ++it;
            continue;
        }
        it = recoveries.erase(it);
    }
    if (!recoveries.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        entry = entries_.find(token);
        if (entry != entries_.end()) {
            entry->second.recoveries.splice(entry->second.recoveries.end(), recoveries);
        }
    }
}

void PDWriteLeaseRegistry::registerRecovery(const PDWriteLeasePB& lease, std::function<void()> recover) {
    if (lease.worker_epoch() != epoch_ || !recover)
        return;
    Recoveries pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = entries_.find(lease.token());
        if (it == entries_.end() || it->second.expiry != expiry_.end())
            return;
        auto& entry = it->second;
        entry.recoveries.emplace_back(std::move(recover));
        if (entry.abandoned)
            pending.splice(pending.end(), entry.recoveries);
    }
    startRecovery(lease.token(), pending);
}

bool PDWriteLeaseRegistry::seal(const PDWriteLeasePB& lease, bool& stopped) {
    std::unique_lock<std::mutex> lock(mutex_);
    stopped = false;
    if (lease.worker_epoch() != epoch_ || lease.token() == 0 || lease.token() > issued_) {
        return false;
    }
    const auto it = entries_.find(lease.token());
    if (it != entries_.end()) {
        if (it->second.expiry == expiry_.end()) {
            it->second.abandoned = true;
            Recoveries pending;
            pending.splice(pending.end(), it->second.recoveries);
            lock.unlock();
            startRecovery(lease.token(), pending);
            return true;
        }
        expiry_.erase(it->second.expiry);
        entries_.erase(it);  // Close admission even if RemoteLoad is still in flight.
    }
    stopped = true;
    return true;
}

PDWriteLeaseRecovery& PDWriteLeaseRecovery::instance() {
    // Only a matching worker can release these pins. This process-lifetime
    // service must outlive RPC server/engine replacement and channel recovery.
    static auto* recovery = new PDWriteLeaseRecovery;
    return *recovery;
}

PDWriteLeaseRecovery::PDWriteLeaseRecovery() {
    std::thread([this] { run(); }).detach();
}

void PDWriteLeaseRecovery::adopt(std::list<PDWriteLeaseDrain>& pending) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.splice(pending_.end(), pending);
    }
    changed_.notify_one();
}

namespace {
constexpr size_t kConcurrentSeals = 32;

// Each peer owns one copy of the same target hold. Drop its copy as soon as
// its matching proof arrives, even while other workers are unreachable.
void sealBatch(std::list<PDWriteLeaseDrain>& batch) {
    struct Call {
        grpc::ClientContext                                                          context;
        grpc::Status                                                                 status;
        SealPDWriteLeaseResponsePB                                                   response;
        std::unique_ptr<grpc::ClientAsyncResponseReader<SealPDWriteLeaseResponsePB>> reader;
        std::list<PDWriteLeaseDrain>::iterator                                       record;
        bool                                                                         finished = false;
    };
    std::array<Call, kConcurrentSeals> calls;
    grpc::CompletionQueue              queue;
    autil::ScopeGuard                  drain([&] {
        for (auto& call : calls) {
            if (call.reader && !call.finished)
                call.context.TryCancel();
        }
        queue.Shutdown();
        void* tag;
        bool  ok;
        while (queue.Next(&tag, &ok)) {}
    });
    size_t                             index       = 0;
    size_t                             outstanding = 0;
    const auto                         deadline    = std::chrono::system_clock::now() + std::chrono::milliseconds(250);
    for (auto it = batch.begin(); it != batch.end(); ++it) {
        auto& call  = calls[index++];
        call.record = it;
        call.context.set_deadline(deadline);
        call.reader = it->peer.stub->AsyncSealPDWriteLease(&call.context, it->peer.lease, &queue);
        if (call.reader) {
            call.reader->Finish(&call.response, &call.status, &call);
            ++outstanding;
        }
    }
    while (outstanding > 0) {
        void* tag;
        bool  ok;
        if (!queue.Next(&tag, &ok))
            return;
        auto& call    = *static_cast<Call*>(tag);
        call.finished = true;
        --outstanding;
        if (ok && call.status.ok() && call.response.stopped()
            && samePDWriteLease(call.record->peer.lease, call.response.lease())) {
            batch.erase(call.record);
        }
    }
}
}  // namespace

void PDWriteLeaseRecovery::run() {
    using Clock = std::chrono::steady_clock;
    std::list<PDWriteLeaseDrain> retries;
    for (;;) {
        std::list<PDWriteLeaseDrain> batch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (pending_.empty() && (retries.empty() || retries.front().next_poll > Clock::now())) {
                if (retries.empty()) {
                    changed_.wait(lock);
                } else {
                    changed_.wait_until(lock, retries.front().next_poll);
                }
            }
            const auto now = Clock::now();
            for (size_t i = 0; i < kConcurrentSeals; ++i) {
                const bool retry_due = !retries.empty() && retries.front().next_poll <= now;
                if (!pending_.empty() && (!retry_due || i % 2 == 0)) {
                    batch.splice(batch.end(), pending_, pending_.begin());
                } else if (retry_due) {
                    batch.splice(batch.end(), retries, retries.begin());
                } else {
                    break;
                }
            }
        }
        try {
            sealBatch(batch);
        } catch (...) {
            // No exception or lost response supplies proof. Keep the remaining pins.
        }
        const auto next_poll = Clock::now() + std::chrono::milliseconds(100);
        for (auto& peer : batch)
            peer.next_poll = next_poll;
        retries.splice(retries.end(), batch);
    }
}

}  // namespace rtp_llm
