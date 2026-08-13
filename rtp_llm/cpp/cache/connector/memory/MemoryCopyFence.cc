#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyFence.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

struct MemoryCopyFence::Entry {
    std::string                           operation_id;
    bool                                  sealed{false};
    bool                                  expiry_elapsed{false};
    size_t                                active_operations{0};
    uint64_t                              expiry_version{0};
    std::chrono::steady_clock::time_point expires_at{std::chrono::steady_clock::time_point::min()};
    std::condition_variable               changed;
};

struct MemoryCopyFence::State {
    struct Expiry {
        std::chrono::steady_clock::time_point expires_at;
        std::string                           operation_id;
        uint64_t                              version{0};

        bool operator>(const Expiry& other) const {
            return expires_at > other.expires_at;
        }
    };

    mutable std::mutex                                                        mutex;
    std::unordered_map<std::string, std::shared_ptr<Entry>>                   entries;
    std::priority_queue<Expiry, std::vector<Expiry>, std::greater<Expiry>> expiries;
    std::condition_variable                                                   changed;
    bool                                                                      stopped{false};
    size_t                                                                    prune_candidate_checks{0};

    void pruneExpiredLocked(std::chrono::steady_clock::time_point now) {
        while (!expiries.empty()) {
            ++prune_candidate_checks;
            const auto expiry = expiries.top();
            if (expiry.expires_at > now) {
                break;
            }
            expiries.pop();

            const auto it = entries.find(expiry.operation_id);
            if (it == entries.end()) {
                continue;
            }
            const auto& entry = it->second;
            if (entry->expiry_version != expiry.version || entry->expires_at != expiry.expires_at) {
                continue;
            }
            if (entry->active_operations == 0) {
                entries.erase(it);
            } else {
                entry->expiry_elapsed = true;
            }
        }
    }

    void scheduleExpiryLocked(const std::shared_ptr<Entry>&     entry,
                              std::chrono::steady_clock::time_point expires_at) {
        entry->expires_at = expires_at;
        entry->expiry_elapsed = false;
        ++entry->expiry_version;
        expiries.push(Expiry{expires_at, entry->operation_id, entry->expiry_version});
    }

    bool allQuiescedLocked() const {
        return std::all_of(entries.begin(), entries.end(), [](const auto& item) {
            return item.second->active_operations == 0;
        });
    }
};

MemoryCopyFence::OperationGuard::OperationGuard(std::shared_ptr<State> state, std::shared_ptr<Entry> entry):
    state_(std::move(state)), entry_(std::move(entry)) {}

MemoryCopyFence::OperationGuard::~OperationGuard() {
    if (state_ == nullptr || entry_ == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (entry_->active_operations > 0) {
            --entry_->active_operations;
        }
        if (entry_->active_operations == 0
            && (entry_->expiry_elapsed || entry_->expires_at <= std::chrono::steady_clock::now())) {
            const auto it = state_->entries.find(entry_->operation_id);
            if (it != state_->entries.end() && it->second == entry_) {
                state_->entries.erase(it);
            }
        }
    }
    entry_->changed.notify_all();
    state_->changed.notify_all();
}

MemoryCopyFence::MemoryCopyFence(): state_(std::make_shared<State>()) {}

MemoryCopyFence::BeginResult MemoryCopyFence::begin(const std::string&         operation_id,
                                                    std::chrono::milliseconds retention) {
    if (operation_id.empty()) {
        return {nullptr, "memory copy operation id is empty"};
    }
    if (retention.count() <= 0) {
        return {nullptr, "memory copy retention is invalid"};
    }

    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(state_->mutex);
    return beginLocked(operation_id, retention, now);
}

MemoryCopyFence::BeginResult MemoryCopyFence::beginBeforeDeadline(
    const std::string&         operation_id,
    std::chrono::milliseconds retention,
    int64_t                   operation_deadline_unix_ms) {
    if (operation_id.empty()) {
        return {nullptr, "memory copy operation id is empty"};
    }
    if (retention.count() <= 0) {
        return {nullptr, "memory copy retention is invalid"};
    }
    if (operation_deadline_unix_ms <= 0) {
        return {nullptr, "memory copy operation deadline is invalid"};
    }

    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto steady_now = std::chrono::steady_clock::now();
    state_->pruneExpiredLocked(steady_now);
    if (state_->stopped) {
        return {nullptr, "memory copy fence is stopped"};
    }
    const auto existing = state_->entries.find(operation_id);
    if (existing != state_->entries.end()) {
        return {nullptr,
                existing->second->sealed ? "memory copy operation is sealed" :
                                           "memory copy operation has already begun"};
    }

    const auto unix_now = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
    if (unix_now >= operation_deadline_unix_ms) {
        return {nullptr, "memory copy operation deadline has expired"};
    }
    return beginLocked(operation_id, retention, steady_now);
}

MemoryCopyFence::BeginResult MemoryCopyFence::beginLocked(const std::string&                    operation_id,
                                                          std::chrono::milliseconds             retention,
                                                          std::chrono::steady_clock::time_point now) {
    state_->pruneExpiredLocked(now);
    if (state_->stopped) {
        return {nullptr, "memory copy fence is stopped"};
    }
    const auto existing = state_->entries.find(operation_id);
    if (existing != state_->entries.end()) {
        return {nullptr,
                existing->second->sealed ? "memory copy operation is sealed" :
                                           "memory copy operation has already begun"};
    }

    auto entry               = std::make_shared<Entry>();
    entry->operation_id      = operation_id;
    entry->active_operations = 1;
    state_->scheduleExpiryLocked(entry, now + retention);
    state_->entries.emplace(operation_id, entry);
    return {Operation(new OperationGuard(state_, entry)), {}};
}

bool MemoryCopyFence::sealAndWait(const std::string&         operation_id,
                                  std::chrono::milliseconds wait_timeout,
                                  std::chrono::milliseconds retention) {
    if (operation_id.empty() || wait_timeout.count() <= 0 || retention.count() <= 0) {
        return false;
    }

    const auto wait_deadline = std::chrono::steady_clock::now() + wait_timeout;
    std::unique_lock<std::mutex> lock(state_->mutex);
    const auto retention_now = std::chrono::steady_clock::now();
    state_->pruneExpiredLocked(retention_now);

    auto it = state_->entries.find(operation_id);
    if (it == state_->entries.end()) {
        auto entry          = std::make_shared<Entry>();
        entry->operation_id = operation_id;
        entry->sealed       = true;
        state_->scheduleExpiryLocked(entry, retention_now + retention);
        it = state_->entries.emplace(operation_id, std::move(entry)).first;
    }
    auto entry    = it->second;
    entry->sealed = true;
    const auto extended_expiry = std::max(entry->expires_at, retention_now + retention);
    if (extended_expiry != entry->expires_at) {
        state_->scheduleExpiryLocked(entry, extended_expiry);
    }

    if (entry->active_operations == 0) {
        return true;
    }
    return entry->changed.wait_until(lock, wait_deadline, [&entry]() { return entry->active_operations == 0; });
}

bool MemoryCopyFence::stopAndWait(std::chrono::milliseconds wait_timeout) {
    if (wait_timeout.count() <= 0) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->stopped = true;
    for (const auto& item : state_->entries) {
        item.second->sealed = true;
    }
    return state_->changed.wait_until(lock, deadline, [this]() { return state_->allQuiescedLocked(); });
}

size_t MemoryCopyFence::entryCountForTest() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->entries.size();
}

size_t MemoryCopyFence::pruneCandidateChecksForTest() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->prune_candidate_checks;
}

void MemoryCopyFence::pruneExpiredAtForTest(std::chrono::steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->pruneExpiredLocked(now);
}

void MemoryCopyFence::withStateLockForTest(const std::function<void()>& callback) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    callback();
}

}  // namespace rtp_llm
