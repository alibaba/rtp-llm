#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/statusor.h"

namespace rtp_llm {

class RemoteLoadLeaseRetainer {
private:
    struct State;

public:
    struct Config {
        size_t                    max_jobs{64};
        std::chrono::milliseconds initial_backoff{50};
        std::chrono::milliseconds max_backoff{5000};
        std::chrono::milliseconds stop_grace{5000};
        size_t                    worker_count{8};
    };

    using Quiesce = std::function<bool()>;
    // Invoked only when final destruction cannot quiesce unresolved leases.
    using FailurePolicy = std::function<void(const std::string&)>;

    // Tickets are thread-confined. A ticket's methods and destructor must not
    // execute concurrently. markStarted() must succeed before any transfer can
    // become visible to a remote peer.
    class Ticket {
    public:
        ~Ticket();

        Ticket(const Ticket&)            = delete;
        Ticket& operator=(const Ticket&) = delete;

        bool markStarted();
        bool complete();

    private:
        friend class RemoteLoadLeaseRetainer;

        Ticket(std::weak_ptr<State> state, std::string token);
        void abandon() noexcept;

    private:
        std::weak_ptr<State> state_;
        std::string          token_;
        bool                 finished_{false};
    };

    RemoteLoadLeaseRetainer();
    explicit RemoteLoadLeaseRetainer(Config config, FailurePolicy failure_policy = {});
    ~RemoteLoadLeaseRetainer();

    RemoteLoadLeaseRetainer(const RemoteLoadLeaseRetainer&)            = delete;
    RemoteLoadLeaseRetainer& operator=(const RemoteLoadLeaseRetainer&) = delete;

    absl::StatusOr<std::unique_ptr<Ticket>>
    reserve(const std::string& token, std::shared_ptr<void> lease, Quiesce quiesce);

    // A bounded timeout closes admission but leaves jobs, workers, and leases owned
    // for a later retry, then returns false. Destruction remains fail-closed.
    bool   stop(std::chrono::milliseconds grace);
    size_t activeJobsForTest() const;

private:
    static void runWorker(const std::shared_ptr<State>& state);

private:
    std::shared_ptr<State> state_;
    std::vector<std::thread> workers_;
    std::mutex               lifecycle_mutex_;
    bool                     stopped_{false};
};

}  // namespace rtp_llm
