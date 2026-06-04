#include "rtp_llm/cpp/model_rpc/RequestSession.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

SessionManager::SessionManager(int64_t terminal_ttl_us, int64_t attach_deadline_us):
    terminal_ttl_us_(terminal_ttl_us), attach_deadline_us_(attach_deadline_us) {}

SessionManager::~SessionManager() {
    stopGc();
}

bool SessionManager::registerSession(int64_t request_id, std::shared_ptr<RequestSession> session) {
    std::lock_guard<std::mutex> lock(mu_);
    if (sessions_.count(request_id) || tombstones_.count(request_id)) {
        return false;
    }
    sessions_.emplace(request_id, std::move(session));
    return true;
}

std::pair<LookupResult, std::shared_ptr<RequestSession>> SessionManager::lookup(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(request_id);
    if (it != sessions_.end()) {
        auto& session = it->second;
        if (session->isTerminal()) {
            return {LookupResult::FINISHED_IN_TTL, session};
        }
        return {LookupResult::RUNNING, session};
    }
    auto tomb_it = tombstones_.find(request_id);
    if (tomb_it != tombstones_.end()) {
        return {LookupResult::GONE, nullptr};
    }
    return {LookupResult::NOT_FOUND, nullptr};
}

bool SessionManager::cancelSession(int64_t request_id, CancelReason reason) {
    std::shared_ptr<RequestSession> session;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(request_id);
        if (it == sessions_.end()) {
            return false;
        }
        session = it->second;
    }
    session->cancel(reason);
    return true;
}

void SessionManager::removeSession(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    sessions_.erase(request_id);
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
            auto swept = gcOnce();
            auto reaped = reapAttachDeadline();
            if (swept > 0 || reaped > 0) {
                RTP_LLM_LOG_INFO("SessionManager GC swept %zu, reaped %zu attach-deadline", swept, reaped);
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
    size_t        swept  = 0;

    std::lock_guard<std::mutex> lock(mu_);

    for (auto it = sessions_.begin(); it != sessions_.end();) {
        auto& session = it->second;
        if (session->isTerminal() && (now_us - session->finishedAtUs()) >= terminal_ttl_us_) {
            tombstones_[it->first] = session->finishedAtUs();
            it = sessions_.erase(it);
            ++swept;
        } else {
            ++it;
        }
    }

    for (auto it = tombstones_.begin(); it != tombstones_.end();) {
        if ((now_us - it->second) >= terminal_ttl_us_ * 2) {
            it = tombstones_.erase(it);
            ++swept;
        } else {
            ++it;
        }
    }

    return swept;
}

size_t SessionManager::reapAttachDeadline() {
    const int64_t now_us = autil::TimeUtility::currentTimeInMicroSeconds();
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
            if ((now_us - session->admittedAtUs()) >= attach_deadline_us_) {
                to_cancel.push_back(session);
            }
        }
    }

    for (auto& session : to_cancel) {
        session->cancel(CancelReason::ATTACH_DEADLINE);
    }
    return to_cancel.size();
}

size_t SessionManager::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return sessions_.size();
}

void SessionManager::cancelAll() {
    std::vector<std::shared_ptr<RequestSession>> all_sessions;
    {
        std::lock_guard<std::mutex> lock(mu_);
        all_sessions.reserve(sessions_.size());
        for (auto& [id, session] : sessions_) {
            all_sessions.push_back(session);
        }
    }
    for (auto& session : all_sessions) {
        session->cancel(CancelReason::EXPLICIT_CANCEL);
    }
}

}  // namespace rtp_llm
