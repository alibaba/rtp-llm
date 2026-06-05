#include "rtp_llm/cpp/model_rpc/RequestSession.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

SessionManager::SessionManager(int64_t default_payload_ttl_us,
                               int64_t default_attach_deadline_us,
                               int64_t default_tombstone_ttl_us):
    default_payload_ttl_us_(default_payload_ttl_us),
    default_attach_deadline_us_(default_attach_deadline_us),
    default_tombstone_ttl_us_(default_tombstone_ttl_us) {}

SessionManager::~SessionManager() {
    stopGc();
}

CreateResult SessionManager::create(const SessionCreateOptions& options) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = sessions_.find(options.request_id);
    if (it != sessions_.end()) {
        auto& existing = it->second;
        if (existing->samePayload(options.payload_hash)) {
            return CreateResult{BatchEnqueueStatus::ALREADY_ADMITTED,
                                existing,
                                existing->sessionEpoch(),
                                grpc::Status::OK};
        }
        return CreateResult{BatchEnqueueStatus::CONFLICT_PAYLOAD,
                            nullptr, 0,
                            grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "payload hash mismatch")};
    }

    auto tomb_it = tombstones_.find(options.request_id);
    if (tomb_it != tombstones_.end()) {
        return CreateResult{BatchEnqueueStatus::CONFLICT_PAYLOAD,
                            nullptr, 0,
                            grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "request_id in tombstone")};
    }

    auto opts = options;
    if (opts.attach_deadline_us == 0) {
        opts.attach_deadline_us = default_attach_deadline_us_;
    }
    if (opts.payload_ttl_us == 0) {
        opts.payload_ttl_us = default_payload_ttl_us_;
    }
    if (opts.tombstone_ttl_us == 0) {
        opts.tombstone_ttl_us = default_tombstone_ttl_us_;
    }

    int64_t epoch = next_session_epoch_++;
    auto session = std::make_shared<RequestSession>(opts, epoch);
    sessions_.emplace(options.request_id, session);

    return CreateResult{BatchEnqueueStatus::ADMITTED,
                        session,
                        epoch,
                        grpc::Status::OK};
}

LookupResult SessionManager::lookup(int64_t request_id,
                                     int64_t session_epoch,
                                     int64_t now_us) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = sessions_.find(request_id);
    if (it != sessions_.end()) {
        auto& session = it->second;
        if (session_epoch != 0 && session->sessionEpoch() != session_epoch) {
            return LookupResult{AttachState::EPOCH_MISMATCH, nullptr, nullptr, std::nullopt};
        }
        auto result = session->buildLookup(now_us);
        result.session = session;
        return result;
    }

    auto tomb_it = tombstones_.find(request_id);
    if (tomb_it != tombstones_.end()) {
        auto& tomb = tomb_it->second;
        if (session_epoch != 0 && tomb.session_epoch != session_epoch) {
            return LookupResult{AttachState::EPOCH_MISMATCH, nullptr, nullptr, std::nullopt};
        }
        TerminalInfo ti;
        ti.reason = tomb.terminal_reason;
        ti.status = tomb.final_status;
        ti.terminal_time_us = tomb.terminal_time_us;
        return LookupResult{AttachState::GONE, nullptr, nullptr, ti};
    }

    return LookupResult{AttachState::NOT_FOUND, nullptr, nullptr, std::nullopt};
}

bool SessionManager::cancelSession(int64_t request_id,
                                    int64_t session_epoch,
                                    CancelReason reason,
                                    int64_t now_us) {
    std::shared_ptr<RequestSession> session;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(request_id);
        if (it == sessions_.end()) {
            return false;
        }
        session = it->second;
        if (session_epoch != 0 && session->sessionEpoch() != session_epoch) {
            return false;
        }
    }
    return session->cancel(reason, now_us);
}

void SessionManager::startGc() {
    gc_stop_.store(false);
    gc_thread_ = std::thread([this] {
        while (!gc_stop_.load()) {
            {
                std::unique_lock<std::mutex> lock(gc_mu_);
                gc_cv_.wait_for(lock, std::chrono::seconds(5), [this] { return gc_stop_.load(); });
            }
            if (gc_stop_.load()) {
                break;
            }
            auto now_us = autil::TimeUtility::currentTimeInMicroSeconds();
            auto swept = gcOnce();
            auto reaped = reapTimeouts(now_us);
            if (swept > 0 || reaped > 0) {
                RTP_LLM_LOG_INFO("SessionManager GC swept %zu, reaped %zu timeouts", swept, reaped);
            }
        }
    });
}

void SessionManager::stopGc() {
    gc_stop_.store(true);
    gc_cv_.notify_all();
    if (gc_thread_.joinable()) {
        gc_thread_.join();
    }
}

size_t SessionManager::gcOnce() {
    const int64_t now_us = autil::TimeUtility::currentTimeInMicroSeconds();
    size_t swept = 0;

    std::lock_guard<std::mutex> lock(mu_);

    for (auto it = sessions_.begin(); it != sessions_.end();) {
        auto& session = it->second;
        if (session->payloadExpired(now_us)) {
            auto ti = session->terminalInfo();
            TombstoneRecord tomb;
            tomb.request_id = session->requestId();
            tomb.session_epoch = session->sessionEpoch();
            tomb.terminal_reason = ti.reason;
            tomb.final_status = ti.status;
            tomb.terminal_time_us = ti.terminal_time_us;
            tomb.tombstone_expire_time_us = ti.tombstone_expire_time_us;
            tombstones_[it->first] = std::move(tomb);
            it = sessions_.erase(it);
            ++swept;
        } else {
            ++it;
        }
    }

    for (auto it = tombstones_.begin(); it != tombstones_.end();) {
        if (now_us >= it->second.tombstone_expire_time_us) {
            it = tombstones_.erase(it);
            ++swept;
        } else {
            ++it;
        }
    }

    return swept;
}

size_t SessionManager::reapTimeouts(int64_t now_us) {
    std::vector<std::shared_ptr<RequestSession>> to_cancel;

    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& [request_id, session] : sessions_) {
            if (session->isTerminal()) {
                continue;
            }
            if (session->hasConsumer()) {
                continue;
            }
            if ((now_us - session->admittedAtUs()) >= session->attachDeadlineUs()) {
                to_cancel.push_back(session);
            }
        }
    }

    for (auto& session : to_cancel) {
        session->cancel(CancelReason::ATTACH_DEADLINE, now_us);
    }
    return to_cancel.size();
}

void SessionManager::shutdown(int64_t now_us) {
    std::vector<std::shared_ptr<RequestSession>> all_sessions;
    {
        std::lock_guard<std::mutex> lock(mu_);
        all_sessions.reserve(sessions_.size());
        for (auto& [id, session] : sessions_) {
            all_sessions.push_back(session);
        }
    }
    for (auto& session : all_sessions) {
        session->cancel(CancelReason::EXPLICIT_CANCEL, now_us);
    }
}

size_t SessionManager::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return sessions_.size();
}

size_t SessionManager::tombstoneCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return tombstones_.size();
}

}  // namespace rtp_llm
