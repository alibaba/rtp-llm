#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"

#include <algorithm>
#include <climits>
#include <exception>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
constexpr size_t kQueryBatchSize = 64;
constexpr size_t kForgetBatchSize = 1024;

void encodeCompletionKey(const TransferCompletionKey& key, TransferCompletionKeyPB& wire) {
    wire.set_trace_id(key.trace_id);
    switch (key.legacy_wire_field) {
        case TransferCompletionKey::LegacyWireField::TRANSFER_TRACE_ID:
            wire.set_transfer_trace_id(std::stoull(key.trace_id));
            break;
        case TransferCompletionKey::LegacyWireField::KVCM_TRACE_ID:
            wire.set_kvcm_trace_id(key.trace_id);
            break;
        case TransferCompletionKey::LegacyWireField::NONE:
            wire.clear_trace();
            break;
    }
}

std::string decodeCompletionTraceId(const TransferCompletionKeyPB& wire) {
    std::string legacy_id;
    switch (wire.trace_case()) {
        case TransferCompletionKeyPB::kTransferTraceId:
            if (wire.transfer_trace_id() == 0)
                return {};
            legacy_id = std::to_string(wire.transfer_trace_id());
            break;
        case TransferCompletionKeyPB::kKvcmTraceId:
            legacy_id = wire.kvcm_trace_id();
            break;
        default:
            return wire.trace_id();
    }
    if (!wire.trace_id().empty() && wire.trace_id() != legacy_id)
        return {};
    return legacy_id;
}

auto callWorkers(BroadcastManager& manager, const std::vector<size_t>& workers, const FunctionRequestPB& request,
                 std::chrono::steady_clock::time_point deadline, const char* operation) {
    using namespace std::chrono;
    const auto remaining = [&] { return ceil<milliseconds>(deadline - steady_clock::now()).count(); };
    const auto timeout_ms = remaining();
    using Result = std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>;
    if (timeout_ms <= 0)
        return Result{};
    const auto rpc_deadline = system_clock::now() + milliseconds(timeout_ms);
    auto rpc = manager.broadcastToRanks<FunctionRequestPB, FunctionResponsePB>(
        workers, request, static_cast<int>(timeout_ms),
        [rpc_deadline](const auto& stub, const auto& context, const auto& message, auto* queue) {
            context->set_deadline(rpc_deadline);
            return stub->AsyncExecuteFunction(context.get(), message, queue);
        }, std::nullopt, operation);
    const auto wait_ms = remaining();
    if (rpc && !rpc->done() && wait_ms > 0)
        rpc->waitDone(static_cast<int>(wait_ms), false);
    return rpc;
}

}

BlockTreeReclaimer::BlockTreeReclaimer(std::shared_ptr<BlockTreeCacheMetricsReporter> metrics):
    metrics_(std::move(metrics)) {}

BlockTreeReclaimer::~BlockTreeReclaimer() {
    stop();
}

void BlockTreeReclaimer::configure(int rank, int64_t interval, int64_t timeout, int64_t force_timeout,
                                   std::shared_ptr<BroadcastManager> broadcast) {
    rank_ = rank;
    interval_ms_ = interval;
    timeout_ms_ = timeout;
    force_release_timeout_ms_ = force_timeout;
    broadcast_ = std::move(broadcast);
}

bool BlockTreeReclaimer::start() {
    if (interval_ms_ < 0 || interval_ms_ > INT_MAX || timeout_ms_ <= 0 || timeout_ms_ > INT_MAX
        || force_release_timeout_ms_ < 0 || force_release_timeout_ms_ > INT_MAX)
        return false;
    if (!threads_.empty())
        return true;
    stopped_ = false;
    try {
        auto start_worker = [this](auto method, int64_t interval) {
            threads_.emplace_back(&BlockTreeReclaimer::periodic, this, method, interval);
        };
        start_worker(&BlockTreeReclaimer::cleanupExpiredMissing, 60000);
        if (rank_ == 0) {
            start_worker(&BlockTreeReclaimer::drainForget, 100);
            if (interval_ms_ > 0)
                start_worker(&BlockTreeReclaimer::queryFrozen, interval_ms_);
            if (interval_ms_ > 0 && force_release_timeout_ms_ > 0)
                start_worker(&BlockTreeReclaimer::releaseExpired, std::min(interval_ms_, force_release_timeout_ms_));
        }
    } catch (...) {
        stop();
        return false;
    }
    return true;
}

void BlockTreeReclaimer::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    cv_.notify_all();
    for (auto& worker : threads_)
        worker.join();
    threads_.clear();
}

void BlockTreeReclaimer::periodic(void (BlockTreeReclaimer::*work)(), int64_t interval) {
    while (!stopped_) {
        try {
            (this->*work)();
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("block tree reclamation failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("block tree reclamation failed");
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(interval), [this] { return stopped_.load(); });
    }
}

bool BlockTreeReclaimer::registerTask(TransferCompletionKey trace, Participants participants,
                                      std::vector<TransferTargetBlock> targets, std::string backend) {
    if (trace.trace_id.empty() || participants.ranks == 0 || (!participants.remote && participants.ranks != 1))
        return false;
    Task task;
    task.forget.push_back({std::move(trace), participants});
    task.targets = std::move(targets);
    task.backend = std::move(backend);
    const auto& id = task.forget.front().trace.trace_id;
    std::lock_guard<std::mutex> lock(mutex_);
    return !frozen_.count(id) && registered_.try_emplace(id, std::move(task)).second;
}

void BlockTreeReclaimer::unregisterTask(const TransferCompletionKey& trace) {
    TaskMap::node_type task;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        task = registered_.extract(trace.trace_id);
    }
    if (task)
        unregisterTask(std::move(task), false);
}

void BlockTreeReclaimer::freezeTargetBlocksWithTraceIDAndStillIOWorkerID(const TransferCompletionKey& trace,
                                                                         const std::vector<size_t>& workers) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = registered_.find(trace.trace_id);
    if (it == registered_.end())
        return;
    auto& task = it->second;
    const auto ranks = task.forget.front().participants.ranks;
    auto& pending = task.pending;
    pending = workers;
    if (pending.empty()) {
        pending.resize(ranks);
        std::iota(pending.begin(), pending.end(), size_t{0});
    } else {
        std::sort(pending.begin(), pending.end());
        pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
        RTP_LLM_CHECK(pending.back() < ranks);
    }
    std::map<IBlockPool*, std::set<BlockIdxType>> groups;
    for (const auto& target : task.targets) {
        RTP_LLM_CHECK(target.pool != nullptr);
        groups[target.pool.get()].insert(target.block);
    }
    std::vector<std::function<void()>> release;
    for (const auto& [pool, blocks] : groups)
        release.push_back(pool->prepareFrozenReferenceRelease({blocks.begin(), blocks.end()}));
    task.release = std::move(release);
    task.frozen_at = Clock::now();
    for (const auto& [pool, blocks] : groups)
        for (const auto block : blocks)
            pool->freezeBlock(block);
    frozen_.insert(registered_.extract(it));
}

bool BlockTreeReclaimer::markWorkerSafe(const TransferCompletionKey& trace, size_t worker) {
    TaskMap::node_type task;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = frozen_.find(trace.trace_id);
        if (it == frozen_.end())
            return false;
        auto& pending = it->second.pending;
        const auto pos = std::lower_bound(pending.begin(), pending.end(), worker);
        if (pos == pending.end() || *pos != worker)
            return false;
        pending.erase(pos);
        if (!pending.empty())
            return false;
        task = frozen_.extract(it);
    }
    unregisterTask(std::move(task), false);
    return true;
}

bool BlockTreeReclaimer::forceRelease(const TransferCompletionKey& trace) {
    TaskMap::node_type task;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        task = frozen_.extract(trace.trace_id);
    }
    if (!task)
        return false;
    unregisterTask(std::move(task), true);
    return true;
}

void BlockTreeReclaimer::unregisterTask(TaskMap::node_type task, bool forced) {
    auto& entry = task.mapped();
    for (auto& release_blocks : entry.release)
        release_blocks();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entry.forget.front().forced = forced;
        forget_queue_.splice(forget_queue_.end(), entry.forget);
    }
    if (metrics_ && entry.frozen_at != Clock::time_point{}) {
        try {
            metrics_->reportUnfreezeMetrics(entry.targets, entry.backend, forced);
        } catch (...) {
            RTP_LLM_LOG_WARNING("failed to report unfreeze metrics for trace=%s", task.key().c_str());
        }
    }
}

void BlockTreeReclaimer::releaseExpired() {
    if (force_release_timeout_ms_ == 0)
        return;
    std::vector<TransferCompletionKey> expired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto cutoff = Clock::now() - std::chrono::milliseconds(force_release_timeout_ms_);
        for (const auto& [id, task] : frozen_)
            if (task.frozen_at <= cutoff)
                expired.push_back(task.forget.front().trace);
    }
    for (const auto& trace : expired) {
        if (stopped_)
            break;
        if (forceRelease(trace))
            RTP_LLM_LOG_WARNING("forced release of frozen transfer trace=%s", trace.trace_id.c_str());
    }
}

void BlockTreeReclaimer::queryFrozen() {
    std::vector<TransferCompletionKey> local;
    std::map<std::vector<size_t>, std::vector<TransferCompletionKey>> remote;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [id, task] : frozen_) {
            const auto& route = task.forget.front();
            if (!route.participants.remote)
                local.push_back(route.trace);
            else if (broadcast_ && broadcast_->workerNum() == route.participants.ranks)
                remote[task.pending].push_back(route.trace);
        }
    }
    for (const auto& trace : local) {
        if (stopped_)
            return;
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms_);
        try {
            const auto state = queryOne(trace.trace_id, deadline);
            if (!stopped_ && Clock::now() <= deadline && (state == State::BUFFER_SAFE || state == State::REJECTED))
                markWorkerSafe(trace, 0);
        } catch (...) {
            continue;
        }
    }
    for (const auto& [workers, traces] : remote) {
        for (size_t begin = 0; begin < traces.size(); begin += kQueryBatchSize) {
            if (stopped_)
                return;
            try {
                queryWorkers(workers, traces.data() + begin, std::min(kQueryBatchSize, traces.size() - begin));
            } catch (...) {
                continue;
            }
        }
    }
}

void BlockTreeReclaimer::queryWorkers(const std::vector<size_t>& workers, const TransferCompletionKey* traces,
                                      size_t count) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms_);
    FunctionRequestPB request;
    for (size_t j = 0; j < count; ++j)
        encodeCompletionKey(traces[j], *request.mutable_completion_query()->add_keys());
    auto rpc = callWorkers(*broadcast_, workers, request, deadline, "completion_query");
    if (!rpc)
        return;
    for (size_t i = 0; i < workers.size(); ++i) {
        auto reply = rpc->responseIfSucceeded(i, deadline);
        if (!reply || !reply->has_completion_query())
            continue;
        const auto& response = reply->completion_query();
        const auto& entries = response.entries();
        if (response.worker_case() != TransferCompletionQueryResponsePB::kRank || response.rank() != workers[i]
            || static_cast<size_t>(entries.size()) > count
            || !std::equal(entries.begin(), entries.end(), traces, [](const auto& entry, const auto& trace) {
                   return decodeCompletionTraceId(entry.key()) == trace.trace_id;
               }))
            continue;
        for (int j = 0; j < entries.size() && !stopped_; ++j) {
            const auto& entry = entries.Get(j);
            if ((entry.state() == TRANSFER_COMPLETION_BUFFER_SAFE && !entry.admission_closed())
                || (entry.state() == TRANSFER_COMPLETION_UNKNOWN && entry.admission_closed()))
                markWorkerSafe(traces[j], workers[i]);
        }
    }
}

std::vector<bool> BlockTreeReclaimer::forget(const std::vector<TransferCompletionKey>& traces, Participants participants, bool forced) {
    const auto deadline = Clock::now() + std::chrono::seconds(1);
    const auto ranks = participants.ranks;
    std::vector<bool> done(traces.size(), false);
    if (!participants.remote) {
        for (size_t i = 0; i < traces.size() && Clock::now() < deadline && !stopped_; ++i)
            done[i] = forgetOne(traces[i].trace_id, forced) == ForgetResult::DONE;
        return done;
    }
    if (!broadcast_ || broadcast_->workerNum() != ranks)
        return done;
    FunctionRequestPB request;
    request.mutable_completion_forget()->set_transfer_released(forced);
    request.mutable_completion_forget()->set_kvcm_read_released(forced);
    for (const auto& trace : traces)
        encodeCompletionKey(trace, *request.mutable_completion_forget()->add_keys());
    std::vector<size_t> workers(ranks);
    std::iota(workers.begin(), workers.end(), size_t{0});
    auto rpc = callWorkers(*broadcast_, workers, request, deadline, "completion_forget");
    if (!rpc)
        return done;
    std::fill(done.begin(), done.end(), true);
    for (size_t i = 0; i < ranks; ++i) {
        auto reply = rpc->responseIfSucceeded(i, deadline);
        if (!reply || !reply->has_completion_forget())
            return std::vector<bool>(traces.size(), false);
        const auto& response = reply->completion_forget();
        if (response.worker_case() != TransferCompletionForgetResponsePB::kRank || response.rank() != i
            || static_cast<size_t>(response.entries_size()) != traces.size())
            return std::vector<bool>(traces.size(), false);
        for (size_t j = 0; j < traces.size(); ++j) {
            const auto& entry = response.entries(j);
            done[j] = done[j] && decodeCompletionTraceId(entry.key()) == traces[j].trace_id
                      && entry.status() == TRANSFER_FORGET_DONE;
        }
    }
    return done;
}

void BlockTreeReclaimer::drainForget() {
    std::list<Route> work;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        work.swap(forget_queue_);
    }
    while (!work.empty() && !stopped_) {
        const auto& route = work.front();
        auto end = work.begin();
        std::vector<TransferCompletionKey> traces;
        try {
            while (end != work.end() && end->forced == route.forced && end->participants.remote == route.participants.remote
                   && end->participants.ranks == route.participants.ranks && traces.size() < kForgetBatchSize) {
                traces.push_back(end->trace);
                ++end;
            }
            const auto done = forget(traces, route.participants, route.forced);
            auto it = work.begin();
            for (bool acknowledged : done) {
                auto current = it++;
                if (acknowledged)
                    work.erase(current);
            }
        } catch (...) {
            if (end == work.begin())
                break;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        forget_queue_.splice(forget_queue_.end(), work, work.begin(), end);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    forget_queue_.splice(forget_queue_.end(), work);
}

bool BlockTreeReclaimer::registerOnWorkerTaskWithTraceID(const std::string& id, QueryCallback offeredQuery, ForgetCallback offeredForget) {
    if (id.empty())
        return false;
    std::lock_guard<std::mutex> lock(worker_mutex_);
    const auto [it, inserted] = records_.try_emplace(id);
    if (inserted) {
        it->second.query = std::move(offeredQuery);
        it->second.forget = std::move(offeredForget);
    } else if (it->second.state == State::REJECTED) {
        expirations_.erase(it->second.expiry);
        records_.erase(it);
    }
    return inserted;
}

bool BlockTreeReclaimer::ioFinishedOnWorkerWithTraceID(const std::string& id) {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    const auto it = records_.find(id);
    if (it == records_.end())
        return false;
    if (it->second.state == State::REJECTED)
        extendMissing(it->second);
    else
        it->second.state = State::BUFFER_SAFE;
    return true;
}

std::vector<BlockTreeReclaimer::State> BlockTreeReclaimer::queryIfIoFinishedInOneWorkerWithTraceIDs(const std::vector<std::string>& ids) {
    std::vector<State> result;
    result.reserve(ids.size());
    for (const auto& id : ids)
        result.push_back(queryOne(id));
    return result;
}

std::vector<BlockTreeReclaimer::ForgetResult> BlockTreeReclaimer::forgetTaskInfoOnWorker(const std::vector<std::string>& ids, bool forced) {
    std::vector<ForgetResult> result;
    result.reserve(ids.size());
    for (const auto& id : ids)
        result.push_back(forgetOne(id, forced));
    return result;
}

void BlockTreeReclaimer::setRecoveryMaintenance(std::function<void()> maintenance) {
    std::lock_guard<std::mutex> lock(mutex_);
    recovery_maintenance_ = std::move(maintenance);
}

void BlockTreeReclaimer::cleanupExpiredMissing() {
    std::function<void()> maintenance;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        maintenance = recovery_maintenance_;
    }
    if (maintenance)
        maintenance();
    std::unique_lock<std::mutex> lock(worker_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
        return;
    const auto now = now_();
    size_t removed = 0;
    while (removed < 128 && !expirations_.empty() && expirations_.begin()->first <= now) {
        records_.erase(expirations_.begin()->second);
        expirations_.erase(expirations_.begin());
        ++removed;
    }
}

BlockTreeReclaimer::State BlockTreeReclaimer::localState(const std::string& id) const {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    const auto it = records_.find(id);
    return it == records_.end() ? State::UNKNOWN : it->second.state;
}

void BlockTreeReclaimer::extendMissing(Record& record) {
    const auto hour = std::chrono::hours(1);
    const auto base = std::max(now_(), record.expiry->first);
    auto node = expirations_.extract(record.expiry);
    node.key() = std::min(base, Clock::time_point::max() - hour) + hour;
    record.expiry = expirations_.insert(std::move(node));
}

BlockTreeReclaimer::Record& BlockTreeReclaimer::findOrRejectMissing(const std::string& id) {
    const auto [it, inserted] = records_.try_emplace(id);
    auto& record = it->second;
    if (inserted) {
        try {
            record.expiry = expirations_.emplace(Clock::time_point::min(), id);
        } catch (...) {
            records_.erase(it);
            throw;
        }
        record.state = State::REJECTED;
    }
    return record;
}

BlockTreeReclaimer::State BlockTreeReclaimer::queryOne(const std::string& id, Clock::time_point deadline) {
    if (id.empty())
        return State::UNKNOWN;
    std::unique_lock<std::mutex> lock(worker_mutex_);
    auto& record = findOrRejectMissing(id);
    if (record.state == State::REJECTED) {
        extendMissing(record);
    } else if (record.state == State::RUNNING && record.query && !record.forgetting) {
        ++record.queries;
        lock.unlock();
        try {
            if (record.query(deadline))
                ioFinishedOnWorkerWithTraceID(id);
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("worker buffer safety query failed: trace=%s, reason=%s", id.c_str(), error.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("worker buffer safety query failed: trace=%s, reason=unknown exception", id.c_str());
        }
        lock.lock();
        --record.queries;
    }
    return record.state;
}

BlockTreeReclaimer::ForgetResult BlockTreeReclaimer::forgetOne(const std::string& id, bool forced) {
    if (id.empty())
        return ForgetResult::INVALID;
    std::unique_lock<std::mutex> lock(worker_mutex_);
    const bool missing = records_.find(id) == records_.end();
    auto& record = findOrRejectMissing(id);
    if (record.state == State::REJECTED) {
        if (missing)
            extendMissing(record);
        return ForgetResult::DONE;
    }
    if ((!forced && record.state != State::BUFFER_SAFE) || record.queries || record.forgetting)
        return ForgetResult::RUNNING;
    if (record.forget) {
        record.forgetting = true;
        lock.unlock();
        try {
            record.forget(forced);
        } catch (...) {
            lock.lock();
            record.forgetting = false;
            return ForgetResult::FAILED;
        }
        lock.lock();
    }
    auto removed = records_.extract(id);
    lock.unlock();
    return ForgetResult::DONE;
}

bool BlockTreeReclaimer::executeFunction(BlockTreeReclaimer* owner, int rank, const FunctionRequestPB& request, FunctionResponsePB& response, Clock::time_point deadline) {
    if (request.has_completion_forget()) {
        auto* forget_response = response.mutable_completion_forget();
        forget_response->Clear();
        forget_response->set_rank(static_cast<uint32_t>(rank));
        const auto& keys = request.completion_forget().keys();
        if (static_cast<size_t>(keys.size()) > kForgetBatchSize) {
            RTP_LLM_LOG_WARNING("completion forget batch exceeds limit: %d", keys.size());
            return false;
        }
        for (const auto& key : keys) {
            auto* entry = forget_response->add_entries();
            entry->mutable_key()->CopyFrom(key);
            const auto trace_id = decodeCompletionTraceId(key);
            auto result = ForgetResult::INVALID;
            if (!trace_id.empty())
                result = owner ? owner->forgetOne(trace_id, request.completion_forget().transfer_released() || request.completion_forget().kvcm_read_released()) : ForgetResult::FAILED;
            switch (result) {
                case ForgetResult::DONE:
                    entry->set_status(TRANSFER_FORGET_DONE);
                    break;
                case ForgetResult::RUNNING:
                    entry->set_status(TRANSFER_FORGET_RUNNING);
                    break;
                case ForgetResult::INVALID:
                    entry->set_status(TRANSFER_FORGET_INVALID);
                    break;
                default:
                    entry->set_status(TRANSFER_FORGET_FAILED);
                    break;
            }
        }
        return true;
    }
    if (request.has_completion_query()) {
        auto* query_response = response.mutable_completion_query();
        query_response->Clear();
        query_response->set_rank(static_cast<uint32_t>(rank));
        const auto& keys = request.completion_query().keys();
        if (static_cast<size_t>(keys.size()) > kQueryBatchSize)
            return false;
        const bool batch = owner && keys.size() > 1;
        if (batch)
            deadline = std::min(deadline, Clock::now() + std::chrono::milliseconds(owner->timeout_ms_));
        for (int i = 0; i < keys.size(); ++i) {
            auto key_deadline = deadline;
            if (batch) {
                const auto now = Clock::now();
                if (now >= deadline)
                    break;
                key_deadline = now + (deadline - now) / (keys.size() - i);
            }
            const auto& key = keys.Get(i);
            const auto trace_id = decodeCompletionTraceId(key);
            const auto state = owner ? owner->queryOne(trace_id, key_deadline) : State::UNKNOWN;
            auto* entry = query_response->add_entries();
            entry->mutable_key()->CopyFrom(key);
            switch (state) {
                case State::REJECTED:
                    entry->set_state(TRANSFER_COMPLETION_UNKNOWN);
                    entry->set_admission_closed(true);
                    break;
                case State::RUNNING:
                    entry->set_state(TRANSFER_COMPLETION_RUNNING);
                    break;
                case State::BUFFER_SAFE:
                    entry->set_state(TRANSFER_COMPLETION_BUFFER_SAFE);
                    break;
                default:
                    entry->set_state(TRANSFER_COMPLETION_UNKNOWN);
                    break;
            }
        }
        return true;
    }
    return false;
}

}
