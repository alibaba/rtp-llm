#include "rtp_llm/cpp/model_rpc/RemoteLoadFence.h"

#include <algorithm>
#include <charconv>
#include <condition_variable>
#include <mutex>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/model_rpc/RemoteLoadBudget.h"

namespace rtp_llm {

namespace {

int64_t currentUnixMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::chrono::steady_clock::time_point localExpiryFromProtocolDeadline(int64_t protocol_deadline_unix_ms) {
    const auto steady_now   = std::chrono::steady_clock::now();
    const auto remaining_ms = protocol_deadline_unix_ms - currentUnixMs();
    return saturatingSteadyDeadline(steady_now, remaining_ms);
}

std::chrono::steady_clock::time_point resolveLocalExpiry(
    int64_t protocol_deadline_unix_ms, std::chrono::steady_clock::time_point local_expiry) {
    return local_expiry == std::chrono::steady_clock::time_point::max() ?
               localExpiryFromProtocolDeadline(protocol_deadline_unix_ms) :
               local_expiry;
}

}  // namespace

struct RemoteLoadFenceRegistry::Entry {
    bool    sealed{false};
    size_t  active_operations{0};
    std::chrono::steady_clock::time_point expires_at{std::chrono::steady_clock::time_point::min()};
    std::condition_variable changed;
};

struct RemoteLoadFenceRegistry::State {
    mutable std::mutex                                     mutex;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries;
    std::chrono::steady_clock::time_point next_expiry{std::chrono::steady_clock::time_point::max()};

    void pruneExpiredLocked(std::chrono::steady_clock::time_point now) {
        if (now < next_expiry) {
            return;
        }
        next_expiry = std::chrono::steady_clock::time_point::max();
        for (auto it = entries.begin(); it != entries.end();) {
            const auto& entry = it->second;
            if (entry->active_operations == 0 && entry->expires_at <= now) {
                it = entries.erase(it);
            } else {
                if (entry->expires_at > now) {
                    next_expiry = std::min(next_expiry, entry->expires_at);
                }
                ++it;
            }
        }
    }

    void recordExpiryLocked(std::chrono::steady_clock::time_point expiry) {
        next_expiry = std::min(next_expiry, expiry);
    }
};

RemoteLoadFenceRegistry::OperationGuard::OperationGuard(std::shared_ptr<State> state, std::shared_ptr<Entry> entry):
    state_(std::move(state)), entry_(std::move(entry)) {}

RemoteLoadFenceRegistry::OperationGuard::~OperationGuard() {
    if (state_ == nullptr || entry_ == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (entry_->active_operations > 0) {
            --entry_->active_operations;
        }
        if (entry_->active_operations == 0) {
            state_->recordExpiryLocked(entry_->expires_at);
        }
    }
    entry_->changed.notify_all();
}

RemoteLoadFenceRegistry::RemoteLoadFenceRegistry() {
    constexpr size_t kShardCount = 64;
    states_.reserve(kShardCount);
    for (size_t i = 0; i < kShardCount; ++i) {
        states_.push_back(std::make_shared<State>());
    }
}

std::shared_ptr<RemoteLoadFenceRegistry::State>
RemoteLoadFenceRegistry::stateFor(const std::string& token) const {
    return states_[std::hash<std::string>{}(token) % states_.size()];
}

absl::StatusOr<std::string> makeRemoteLoadAllocationToken(const std::string& owner,
                                                          const std::string& unique_id,
                                                          int64_t            request_deadline_unix_ms) {
    if (owner.empty()) {
        return absl::InvalidArgumentError("remote load allocation owner is empty");
    }
    if (unique_id.empty()) {
        return absl::InvalidArgumentError("remote load allocation identity is empty");
    }
    if (request_deadline_unix_ms <= 0) {
        return absl::InvalidArgumentError("remote load allocation deadline is invalid");
    }
    return std::to_string(request_deadline_unix_ms) + ":" + std::to_string(owner.size()) + ":" + owner + ":"
           + unique_id;
}

absl::StatusOr<int64_t> remoteLoadAllocationDeadline(const std::string& token) {
    const auto separator = token.find(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= token.size()) {
        return absl::InvalidArgumentError("remote load allocation token is malformed");
    }

    int64_t deadline_unix_ms = 0;
    const auto result =
        std::from_chars(token.data(), token.data() + separator, deadline_unix_ms);
    if (result.ec != std::errc() || result.ptr != token.data() + separator || deadline_unix_ms <= 0) {
        return absl::InvalidArgumentError("remote load allocation token has an invalid deadline");
    }
    return deadline_unix_ms;
}

absl::Status validateRemoteLoadAllocationOwner(const std::string& token, const std::string& expected_owner) {
    if (expected_owner.empty()) {
        return absl::InvalidArgumentError("remote load allocation owner is empty");
    }
    const auto deadline_separator = token.find(':');
    if (deadline_separator == std::string::npos) {
        return absl::InvalidArgumentError("remote load allocation token is malformed");
    }
    const auto owner_length_separator = token.find(':', deadline_separator + 1);
    if (owner_length_separator == std::string::npos || owner_length_separator == deadline_separator + 1) {
        return absl::InvalidArgumentError("remote load allocation token has no owner length");
    }

    size_t owner_length = 0;
    const auto owner_length_result = std::from_chars(token.data() + deadline_separator + 1,
                                                     token.data() + owner_length_separator,
                                                     owner_length);
    if (owner_length_result.ec != std::errc()
        || owner_length_result.ptr != token.data() + owner_length_separator || owner_length == 0) {
        return absl::InvalidArgumentError("remote load allocation token has an invalid owner length");
    }
    const auto owner_begin = owner_length_separator + 1;
    if (owner_length > token.size() - owner_begin || token.size() == owner_begin + owner_length
        || token[owner_begin + owner_length] != ':') {
        return absl::InvalidArgumentError("remote load allocation token has a malformed owner");
    }
    if (owner_length != expected_owner.size()
        || token.compare(owner_begin, owner_length, expected_owner) != 0) {
        return absl::FailedPreconditionError("remote load allocation belongs to a different server instance");
    }
    return absl::OkStatus();
}

absl::StatusOr<RemoteLoadFenceRegistry::Operation>
RemoteLoadFenceRegistry::begin(const std::string&                    token,
                               int64_t                               request_deadline_unix_ms,
                               std::chrono::steady_clock::time_point local_expiry) {
    if (token.empty()) {
        return absl::InvalidArgumentError("remote load allocation token is empty");
    }
    const auto token_deadline = remoteLoadAllocationDeadline(token);
    if (!token_deadline.ok()) {
        return token_deadline.status();
    }
    if (*token_deadline != request_deadline_unix_ms) {
        return absl::InvalidArgumentError("remote load request deadline does not match its allocation token");
    }

    local_expiry = resolveLocalExpiry(request_deadline_unix_ms, local_expiry);
    auto state   = stateFor(token);
    auto entry   = std::make_shared<Entry>();
    entry->active_operations  = 1;
    entry->expires_at         = local_expiry;
    Operation operation(new OperationGuard(state, entry));

    std::lock_guard<std::mutex> lock(state->mutex);
    const auto                  now = std::chrono::steady_clock::now();
    if (local_expiry <= now) {
        return absl::DeadlineExceededError("remote load request deadline has expired");
    }
    state->pruneExpiredLocked(now);

    const auto existing = state->entries.find(token);
    if (existing != state->entries.end()) {
        if (existing->second->sealed) {
            return absl::FailedPreconditionError("remote load allocation token is sealed");
        }
        return absl::AlreadyExistsError("remote load allocation token has already begun");
    }

    state->entries.emplace(token, entry);
    state->recordExpiryLocked(local_expiry);
    return operation;
}

absl::Status RemoteLoadFenceRegistry::sealAndWait(const std::string&                           token,
                                                  int64_t                                      request_deadline_unix_ms,
                                                  std::chrono::steady_clock::time_point        wait_deadline,
                                                  UnseenTokenPolicy unseen_token_policy,
                                                  std::chrono::steady_clock::time_point local_expiry) {
    if (token.empty()) {
        return absl::InvalidArgumentError("remote load allocation token is empty");
    }
    const auto token_deadline = remoteLoadAllocationDeadline(token);
    if (!token_deadline.ok()) {
        return token_deadline.status();
    }
    if (*token_deadline != request_deadline_unix_ms) {
        return absl::InvalidArgumentError("remote load request deadline does not match its allocation token");
    }

    local_expiry = resolveLocalExpiry(request_deadline_unix_ms, local_expiry);
    auto                         state = stateFor(token);
    std::unique_lock<std::mutex> lock(state->mutex);
    const auto                   now = std::chrono::steady_clock::now();
    state->pruneExpiredLocked(now);

    auto it = state->entries.find(token);
    if (it == state->entries.end() && unseen_token_policy == UnseenTokenPolicy::Reject) {
        if (local_expiry <= now) {
            return absl::OkStatus();
        }
        return absl::NotFoundError("remote load allocation token is unknown to this server instance");
    }
    if (it == state->entries.end()) {
        auto new_entry = std::make_shared<Entry>();
        it = state->entries.emplace(token, std::move(new_entry)).first;
    }
    auto entry        = it->second;
    entry->sealed     = true;
    entry->expires_at = std::max(entry->expires_at, local_expiry);
    state->recordExpiryLocked(entry->expires_at);

    if (entry->active_operations == 0) {
        return absl::OkStatus();
    }
    if (!entry->changed.wait_until(lock, wait_deadline, [&entry]() { return entry->active_operations == 0; })) {
        return absl::DeadlineExceededError("remote load operation did not quiesce before the wait deadline");
    }
    return absl::OkStatus();
}

void RemoteLoadFenceRegistry::pruneExpired() {
    for (const auto& state : states_) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->pruneExpiredLocked(std::chrono::steady_clock::now());
    }
}

size_t RemoteLoadFenceRegistry::entryCountForTest() const {
    size_t count = 0;
    for (const auto& state : states_) {
        std::lock_guard<std::mutex> lock(state->mutex);
        count += state->entries.size();
    }
    return count;
}

}  // namespace rtp_llm
