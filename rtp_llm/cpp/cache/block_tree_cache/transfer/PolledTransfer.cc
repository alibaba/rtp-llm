#include "PolledTransfer.h"

#include <algorithm>
#include <condition_variable>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace rtp_llm {
namespace {
bool remoteWrite(btc_work_kind kind) {
    return kind == BTC_WORK_HOST_TO_REMOTE || kind == BTC_WORK_DEVICE_TO_REMOTE;
}

PolledOutcome::Objects readObjects(const btc_receipt& receipt, const PolledWork& work) {
    PolledOutcome::Objects objects;
    if (!receipt.object_count) {
        return objects;
    }
    if (work.kind != BTC_WORK_REMOTE_QUERY || !receipt.objects) {
        throw std::invalid_argument("Remote locations require a query and an object array");
    }
    std::set<std::pair<uint32_t, uint64_t>> expected;
    for (const auto& page : work.pages) {
        expected.emplace(page.position.group, page.position.end_token);
    }
    for (size_t i = 0; i < receipt.object_count; ++i) {
        const auto& item     = receipt.objects[i];
        const auto  position = std::make_pair(item.position.group, item.position.end_token);
        if (!expected.count(position) || !item.descriptor.data || !item.descriptor.size
            || !objects
                    .emplace(position,
                             std::string(static_cast<const char*>(item.descriptor.data), item.descriptor.size))
                    .second) {
            throw std::invalid_argument("Unexpected remote location in transfer receipt");
        }
    }
    return objects;
}
}  // namespace

class PolledRankBroadcast final: public RankTransferBroadcast {
public:
    PolledRankBroadcast(std::shared_ptr<TransferBufferSafety> proof, std::function<void(bool)> on_safe):
        proof_(std::move(proof)), on_safe_(std::move(on_safe)) {}

    void waitDone() override {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return done_; });
    }
    void onDone(DoneCallback callback) override {
        {
            std::lock_guard lock(mutex_);
            if (!done_) {
                callback_ = std::move(callback);
                return;
            }
        }
        callback();
    }
    bool done() const override {
        std::lock_guard lock(mutex_);
        return done_;
    }
    std::string failureSummary() const override {
        std::lock_guard lock(mutex_);
        return unknown_ ? "worker copy has no final buffer-safety proof" : std::string{};
    }
    std::vector<RankTransferReply> replies() const override {
        std::lock_guard lock(mutex_);
        return replies_;
    }

    void deliver(std::vector<RankTransferReply> replies, bool safe) {
        DoneCallback callback;
        bool         late;
        {
            std::lock_guard lock(mutex_);
            late = done_;
            if (!done_) {
                replies_ = std::move(replies);
                unknown_ = !safe;
                done_    = true;
                callback = std::move(callback_);
            }
        }
        if (safe && proof_)
            proof_->report(true);
        ready_.notify_all();
        if (callback)
            callback();
        if (safe && on_safe_)
            on_safe_(late);
    }

private:
    const std::shared_ptr<TransferBufferSafety> proof_;
    const std::function<void(bool)>             on_safe_;
    mutable std::mutex                          mutex_;
    std::condition_variable                     ready_;
    DoneCallback                                callback_;
    std::vector<RankTransferReply>              replies_;
    bool                                        done_    = false;
    bool                                        unknown_ = false;
};

namespace {
RankTransferReply copyReply(uint32_t actor, btc_result code, bool safe, bool accepted) {
    RankTransferReply reply;
    auto&             snapshot = reply.snapshot;
    snapshot.rank              = actor;
    snapshot.server_address    = "worker/" + std::to_string(actor);
    snapshot.dispatch_state    = accepted ? TransferRpcDispatchState::SUBMITTED :
                                            (code == BTC_REJECTED ? TransferRpcDispatchState::NOT_SUBMITTED :
                                                                    TransferRpcDispatchState::POSSIBLY_SUBMITTED);
    if (!code)
        return reply;
    snapshot.response_received = snapshot.response_valid = true;
    auto& result                                         = reply.result.emplace();
    if (code != BTC_SUCCESS) {
        result = TransferResult::executionException(safe, "worker copy failed");
        if (!safe)
            result.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        else if (code == BTC_TIMEOUT)
            result.status.setErrorCode(ErrorCode::DEADLINE_EXCEEDED);
        else if (code == BTC_REJECTED)
            result.copy_error = CacheCopyError::NOT_EXECUTED;
    }
    result.buffer_safe         = safe;
    snapshot.worker_error_code = static_cast<int64_t>(result.status.code());
    snapshot.worker_copy_error = result.copy_error;
    snapshot.buffer_safe       = safe;
    return reply;
}
}  // namespace

PolledTransfer::PolledTransfer(uint64_t instance, Routes routes): instance_(instance), routes_(std::move(routes)) {
    for (const auto& [_, actors] : routes_) {
        if (actors.empty() || std::set<uint32_t>(actors.begin(), actors.end()).size() != actors.size()) {
            throw std::invalid_argument("Each polled transfer route requires distinct participants");
        }
    }
}

uint64_t PolledTransfer::submit(btc_work_kind kind, uint64_t request, std::vector<PolledPage> pages, Done done) {
    return enqueue(kind, request, std::move(pages), std::move(done), nullptr);
}

RankTransferRequest PolledTransfer::prepareCopy(btc_work_kind                         kind,
                                                uint64_t                              request,
                                                std::vector<PolledPage>               pages,
                                                std::shared_ptr<TransferBufferSafety> proof,
                                                std::function<void(bool)>             on_safe) {
    RankTransferRequest call;
    for (auto actor : routes_.at(kind)) {
        call.peers.push_back({actor, "worker/" + std::to_string(actor)});
    }
    auto broadcast = std::make_shared<PolledRankBroadcast>(std::move(proof), std::move(on_safe));
    call.send      = [this, kind, request, pages = std::move(pages), broadcast](std::chrono::milliseconds) mutable {
        enqueue(kind, request, std::move(pages), {}, broadcast);
        return broadcast;
    };
    return call;
}

uint64_t PolledTransfer::enqueue(btc_work_kind                        kind,
                                 uint64_t                             request,
                                 std::vector<PolledPage>              pages,
                                 Done                                 done,
                                 std::shared_ptr<PolledRankBroadcast> ranks) {
    const auto&     actors = routes_.at(kind);
    std::lock_guard lock(mutex_);
    const auto      id = ++next_;
    Pending         p{{id, kind, request, actors, std::move(pages)}, {}, std::move(done), {}};
    p.ranks = std::move(ranks);
    for (auto actor : actors) {
        p.participants.emplace(actor, Participant{});
    }
    pending_.emplace(id, std::move(p));
    return id;
}

std::vector<PolledWork> PolledTransfer::poll() {
    std::lock_guard         lock(mutex_);
    std::vector<PolledWork> result;
    result.reserve(pending_.size());
    for (auto& entry : pending_) {
        auto& p = entry.second;
        result.push_back(p.work);
        auto& actors = result.back().participants;
        actors.erase(
            std::remove_if(actors.begin(), actors.end(), [&](uint32_t actor) { return p.participants.at(actor).safe; }),
            actors.end());
        p.exposed = true;
    }
    return result;
}

size_t PolledTransfer::size() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
}

std::vector<uint64_t> PolledTransfer::findLoads(const std::vector<PolledLoadLookup>& loads) const {
    std::vector<uint64_t> result(loads.size());
    if (loads.empty()) {
        return result;
    }
    const auto same_page = [](const btc_page& left, const btc_page& right) {
        return left.pool == right.pool && left.slot == right.slot && left.generation == right.generation
               && left.hold == right.hold;
    };
    std::unordered_multimap<uint64_t, size_t> by_hold;
    by_hold.reserve(loads.size());
    for (size_t i = 0; i < loads.size(); ++i) {
        if (!loads[i].target->empty()) {
            by_hold.emplace(loads[i].target->front().hold, i);
        }
    }
    std::lock_guard lock(mutex_);
    for (const auto& [operation, pending] : pending_) {
        if (pending.failed_reported) {
            continue;
        }
        for (const auto& page : pending.work.pages) {
            if (page.target.empty()) {
                continue;
            }
            const auto [begin, end] = by_hold.equal_range(page.target.front().hold);
            for (auto it = begin; it != end; ++it) {
                const auto& lookup = loads[it->second];
                if (pending.work.kind == lookup.kind && page.position.group == lookup.position.group
                    && page.position.end_token == lookup.position.end_token
                    && std::equal(page.target.begin(),
                                  page.target.end(),
                                  lookup.target->begin(),
                                  lookup.target->end(),
                                  same_page)) {
                    result[it->second] = operation;
                }
            }
        }
    }
    return result;
}

void PolledTransfer::accept(btc_operation id, uint32_t actor) {
    std::lock_guard lock(mutex_);
    auto            it = pending_.find(id.operation);
    if (id.instance == instance_ && it != pending_.end()) {
        auto found = it->second.participants.find(actor);
        if (found == it->second.participants.end()) {
            throw std::invalid_argument("Unknown transfer participant");
        }
        it->second.exposed     = true;
        found->second.accepted = true;
    }
}

std::function<void()> PolledTransfer::Pending::notification(bool all_safe) {
    if (ranks) {
        std::vector<RankTransferReply> replies;
        replies.reserve(participants.size());
        for (const auto& [id, actor] : participants)
            replies.push_back(copyReply(id, actor.result, actor.safe, actor.accepted));
        return [broadcast = ranks, replies = std::move(replies), all_safe]() mutable {
            broadcast->deliver(std::move(replies), all_safe);
        };
    }
    // Storage commands additionally carry commit evidence and remote locations.
    PolledOutcome outcome{BTC_SUCCESS, all_safe, std::move(objects), remoteWrite(work.kind)};
    for (const auto& [_, actor] : participants) {
        outcome.committed &= actor.committed;
        if (actor.result && actor.result != BTC_SUCCESS) outcome.result = actor.result;
    }
    return [notify = done, outcome = std::move(outcome)]() mutable { notify(std::move(outcome)); };
}

void PolledTransfer::complete(const btc_receipt* receipts, size_t count) {
    if (count && !receipts) {
        throw std::invalid_argument("Missing transfer receipt array");
    }
    std::unique_lock                           lock(mutex_);
    std::map<Participant*, Participant>        staged;
    std::map<uint64_t, PolledOutcome::Objects> locations;
    std::set<uint64_t>                         changed;
    for (size_t i = 0; i < count; ++i) {
        const auto& receipt = receipts[i];
        auto        it      = pending_.find(receipt.id.operation);
        if (receipt.id.instance != instance_ || it == pending_.end())
            continue;
        auto& pending = it->second;
        auto  actor   = pending.participants.find(receipt.participant);
        if (actor == pending.participants.end())
            throw std::invalid_argument("Unexpected transfer participant");
        auto& update = staged.try_emplace(&actor->second, actor->second).first->second;
        if (update.safe)
            continue;
        if (receipt.result < BTC_SUCCESS || receipt.result > BTC_REJECTED || receipt.safe > 1 || receipt.committed > 1
            || (pending.disk_running && receipt.safe)
            || (receipt.result == BTC_REJECTED && (update.accepted || !receipt.safe))) {
            throw std::invalid_argument("Invalid transfer receipt");
        }
        auto result = remoteWrite(pending.work.kind) && receipt.result == BTC_SUCCESS && !receipt.committed ?
                          static_cast<btc_result>(BTC_FAILED) :
                          receipt.result;
        update.accepted |= result != BTC_REJECTED;
        update.safe = receipt.safe;
        update.committed |= receipt.committed != 0;
        if (!update.result || update.result == BTC_SUCCESS)
            update.result = result;
        if (receipt.object_count) {
            auto  objects = readObjects(receipt, pending.work);
            auto& added   = locations[it->first];
            for (auto& [position, descriptor] : objects) {
                auto prior = pending.objects.find(position);
                if (prior != pending.objects.end() && prior->second != descriptor)
                    throw std::invalid_argument("Conflicting remote replica descriptors");
                auto [entry, fresh] = added.try_emplace(position, std::move(descriptor));
                if (!fresh && entry->second != descriptor)
                    throw std::invalid_argument("Conflicting remote replicas in a batch");
            }
        }
        changed.insert(it->first);
    }
    for (const auto& [actor, update] : staged)
        *actor = update;
    for (auto& [id, objects] : locations)
        pending_.at(id).objects.merge(objects);
    std::vector<std::function<void()>> callbacks;
    for (auto id : changed) {
        auto& p = pending_.at(id);
        p.exposed = true;
        const bool safe = std::all_of(p.participants.begin(), p.participants.end(),
                                     [](const auto& actor) { return actor.second.safe; });
        const bool failed = std::any_of(p.participants.begin(), p.participants.end(), [](const auto& actor) {
            return actor.second.result && actor.second.result != BTC_SUCCESS;
        });
        if (safe || (failed && !p.failed_reported)) {
            callbacks.emplace_back(p.notification(safe));
            p.failed_reported = failed;
        }
        if (safe) {
            pending_.erase(id);
        }
    }
    lock.unlock();
    for (auto& callback : callbacks)
        callback();
}

void PolledTransfer::cancelUnexposed() {
    std::vector<std::function<void()>> callbacks;
    {
        std::lock_guard lock(mutex_);
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->second.exposed || it->second.work.kind == BTC_WORK_RELEASE || it->second.disk_running) {
                ++it;
            } else {
                auto& pending = it->second;
                for (auto& [_, actor] : pending.participants) {
                    actor.result = BTC_REJECTED;
                    actor.safe = true;
                }
                callbacks.emplace_back(pending.notification(true));
                it = pending_.erase(it);
            }
        }
    }
    for (auto& callback : callbacks)
        callback();
}

std::vector<PolledPage> PolledTransfer::beginDisk(btc_operation id, btc_work_kind& kind) {
    std::lock_guard lock(mutex_);
    auto            it = pending_.find(id.operation);
    if (id.instance != instance_ || it == pending_.end() || it->second.disk_running) {
        throw std::invalid_argument("Unknown or active disk operation");
    }
    auto& p = it->second;
    if (p.work.kind != BTC_WORK_DISK_TO_HOST && p.work.kind != BTC_WORK_HOST_TO_DISK) {
        throw std::invalid_argument("This operation is not a local disk copy");
    }
    auto pages     = p.work.pages;
    p.exposed      = true;
    p.disk_running = true;
    kind           = p.work.kind;
    return pages;
}

void PolledTransfer::endDisk(btc_operation id) {
    std::lock_guard lock(mutex_);
    pending_.at(id.operation).disk_running = false;
}
}  // namespace rtp_llm
