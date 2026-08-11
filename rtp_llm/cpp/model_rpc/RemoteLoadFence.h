#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace rtp_llm {

absl::StatusOr<std::string> makeRemoteLoadAllocationToken(const std::string& owner,
                                                          const std::string& unique_id,
                                                          int64_t            request_deadline_unix_ms);
absl::StatusOr<int64_t>     remoteLoadAllocationDeadline(const std::string& token);
absl::Status                validateRemoteLoadAllocationOwner(const std::string& token,
                                                              const std::string& expected_owner);

class RemoteLoadFenceRegistry {
private:
    struct Entry;
    struct State;

public:
    enum class UnseenTokenPolicy {
        Seal,
        Reject,
    };

    class OperationGuard {
    public:
        ~OperationGuard();

        OperationGuard(const OperationGuard&)            = delete;
        OperationGuard& operator=(const OperationGuard&) = delete;

    private:
        friend class RemoteLoadFenceRegistry;

        OperationGuard(std::shared_ptr<State> state, std::shared_ptr<Entry> entry);

    private:
        std::shared_ptr<State> state_;
        std::shared_ptr<Entry> entry_;
    };

    using Operation = std::shared_ptr<OperationGuard>;

    RemoteLoadFenceRegistry();
    ~RemoteLoadFenceRegistry() = default;

    absl::StatusOr<Operation>
    begin(const std::string&                    token,
          int64_t                               request_deadline_unix_ms,
          std::chrono::steady_clock::time_point local_expiry = std::chrono::steady_clock::time_point::max());

    absl::Status sealAndWait(const std::string&                           token,
                             int64_t                                      request_deadline_unix_ms,
                             std::chrono::steady_clock::time_point        wait_deadline,
                             UnseenTokenPolicy unseen_token_policy = UnseenTokenPolicy::Seal,
                             std::chrono::steady_clock::time_point local_expiry =
                                 std::chrono::steady_clock::time_point::max());

    void   pruneExpired();
    size_t entryCountForTest() const;

private:
    std::shared_ptr<State> stateFor(const std::string& token) const;

private:
    std::vector<std::shared_ptr<State>> states_;
};

}  // namespace rtp_llm
