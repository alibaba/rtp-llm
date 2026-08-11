#include "rtp_llm/cpp/model_rpc/RemoteLoadLeaseRetainer.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include "absl/status/status.h"

namespace rtp_llm {

namespace {

thread_local const void* current_retainer_state = nullptr;

std::chrono::steady_clock::time_point deadlineAfter(std::chrono::milliseconds delay) {
    const auto now = std::chrono::steady_clock::now();
    if (delay <= std::chrono::milliseconds::zero()) {
        return now;
    }
    const auto max_delay =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::time_point::max() - now);
    return delay >= max_delay ? std::chrono::steady_clock::time_point::max() : now + delay;
}

std::chrono::milliseconds doubledBackoff(std::chrono::milliseconds current,
                                         std::chrono::milliseconds maximum) {
    if (current >= maximum || current > maximum - current) {
        return maximum;
    }
    return current + current;
}

}  // namespace

struct RemoteLoadLeaseRetainer::State {
    struct Job {
        std::string                 token;
        std::shared_ptr<void>       lease;
        Quiesce                     quiesce;
        bool                        started{false};
        bool                        background{false};
        bool                        in_flight{false};
        uint64_t                    attempt{0};
        std::chrono::steady_clock::time_point next_retry;
        std::chrono::milliseconds  backoff;
    };

    explicit State(Config config, FailurePolicy failure_policy):
        config(std::move(config)), failure_policy(std::move(failure_policy)) {}

    mutable std::mutex                                    mutex;
    std::condition_variable                               changed;
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs;
    Config                                                config;
    FailurePolicy                                         failure_policy;
    bool                                                  accepting{true};
    bool                                                  shutdown{false};
};

RemoteLoadLeaseRetainer::Ticket::Ticket(std::weak_ptr<State> state, std::string token):
    state_(std::move(state)), token_(std::move(token)) {}

RemoteLoadLeaseRetainer::Ticket::~Ticket() {
    abandon();
}

bool RemoteLoadLeaseRetainer::Ticket::markStarted() {
    if (finished_) {
        return false;
    }
    auto state = state_.lock();
    if (state == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const auto                  it = state->jobs.find(token_);
    if (it == state->jobs.end()) {
        finished_ = true;
        return false;
    }
    if (!state->accepting) {
        return false;
    }
    it->second->started = true;
    return true;
}

bool RemoteLoadLeaseRetainer::Ticket::complete() {
    if (finished_) {
        return false;
    }
    auto state = state_.lock();
    if (state == nullptr) {
        finished_ = true;
        return false;
    }

    std::shared_ptr<State::Job> retired_job;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        const auto                  it = state->jobs.find(token_);
        if (it == state->jobs.end()) {
            finished_ = true;
            return false;
        }
        retired_job = it->second;
        state->jobs.erase(it);
        finished_ = true;
    }
    state->changed.notify_all();
    retired_job.reset();
    return true;
}

void RemoteLoadLeaseRetainer::Ticket::abandon() noexcept {
    if (finished_) {
        return;
    }
    auto state = state_.lock();
    if (state == nullptr) {
        finished_ = true;
        return;
    }

    std::shared_ptr<State::Job> retired_job;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        const auto                  it = state->jobs.find(token_);
        if (it == state->jobs.end()) {
            finished_ = true;
            return;
        }
        if (!it->second->started) {
            retired_job = it->second;
            state->jobs.erase(it);
        } else {
            it->second->background = true;
            it->second->next_retry = std::chrono::steady_clock::now();
        }
        finished_ = true;
    }
    state->changed.notify_all();
    retired_job.reset();
}

RemoteLoadLeaseRetainer::RemoteLoadLeaseRetainer(): RemoteLoadLeaseRetainer(Config{}, {}) {}

RemoteLoadLeaseRetainer::RemoteLoadLeaseRetainer(Config config, FailurePolicy failure_policy) {
    if (config.max_jobs == 0) {
        config.max_jobs = 1;
    }
    if (config.initial_backoff <= std::chrono::milliseconds::zero()) {
        config.initial_backoff = std::chrono::milliseconds(1);
    }
    if (config.max_backoff < config.initial_backoff) {
        config.max_backoff = config.initial_backoff;
    }
    config.worker_count = std::max<size_t>(1, std::min(config.worker_count, config.max_jobs));
    if (!failure_policy) {
        failure_policy = [](const std::string&) { std::abort(); };
    }
    state_ = std::make_shared<State>(std::move(config), std::move(failure_policy));
    workers_.reserve(state_->config.worker_count);
    try {
        for (size_t index = 0; index < state_->config.worker_count; ++index) {
            workers_.emplace_back([state = state_]() { runWorker(state); });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->accepting = false;
            state_->shutdown  = true;
        }
        state_->changed.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
    }
}

RemoteLoadLeaseRetainer::~RemoteLoadLeaseRetainer() {
    if (!stopped_ && !stop(state_->config.stop_grace)) {
        state_->failure_policy("remote load leases remain unresolved at shutdown");
        std::abort();
    }
}

absl::StatusOr<std::unique_ptr<RemoteLoadLeaseRetainer::Ticket>>
RemoteLoadLeaseRetainer::reserve(const std::string& token, std::shared_ptr<void> lease, Quiesce quiesce) {
    if (token.empty()) {
        return absl::InvalidArgumentError("remote load lease token is empty");
    }
    if (lease == nullptr) {
        return absl::InvalidArgumentError("remote load lease is empty");
    }
    if (!quiesce) {
        return absl::InvalidArgumentError("remote load quiesce callback is empty");
    }

    auto job        = std::make_shared<State::Job>();
    job->token      = token;
    job->lease      = std::move(lease);
    job->quiesce    = std::move(quiesce);
    job->next_retry = std::chrono::steady_clock::now();
    job->backoff    = state_->config.initial_backoff;
    std::unique_ptr<Ticket> ticket;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->accepting) {
            return absl::FailedPreconditionError("remote load lease retainer is stopping");
        }
        if (state_->jobs.size() >= state_->config.max_jobs) {
            return absl::ResourceExhaustedError("remote load lease retainer is full");
        }
        if (state_->jobs.find(token) != state_->jobs.end()) {
            return absl::AlreadyExistsError("remote load lease token already exists");
        }
        ticket.reset(new Ticket(state_, token));
        state_->jobs.emplace(token, job);
    }
    return ticket;
}

bool RemoteLoadLeaseRetainer::stop(std::chrono::milliseconds grace) {
    if (current_retainer_state == state_.get()) {
        return false;
    }

    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (stopped_) {
        return true;
    }

    const auto deadline = deadlineAfter(grace);
    {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->accepting = false;
        state_->changed.notify_all();
        if (!state_->changed.wait_until(lock, deadline, [this]() { return state_->jobs.empty(); })) {
            return false;
        }
        state_->shutdown = true;
    }
    state_->changed.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    stopped_ = true;
    return true;
}

size_t RemoteLoadLeaseRetainer::activeJobsForTest() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->jobs.size();
}

void RemoteLoadLeaseRetainer::runWorker(const std::shared_ptr<State>& state) {
    current_retainer_state = state.get();
    while (true) {
        std::shared_ptr<State::Job> job;
        uint64_t                    attempt = 0;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            while (true) {
                if (state->shutdown) {
                    current_retainer_state = nullptr;
                    return;
                }

                const auto now = std::chrono::steady_clock::now();
                auto       due = state->jobs.end();
                auto       next_wakeup = std::chrono::steady_clock::time_point::max();
                for (auto it = state->jobs.begin(); it != state->jobs.end(); ++it) {
                    const auto& candidate = it->second;
                    if (!candidate->background || candidate->in_flight) {
                        continue;
                    }
                    if (candidate->next_retry <= now) {
                        due = it;
                        break;
                    }
                    next_wakeup = std::min(next_wakeup, candidate->next_retry);
                }

                if (due != state->jobs.end()) {
                    job            = due->second;
                    job->in_flight = true;
                    attempt        = ++job->attempt;
                    break;
                }
                if (next_wakeup == std::chrono::steady_clock::time_point::max()) {
                    state->changed.wait(lock);
                } else {
                    state->changed.wait_until(lock, next_wakeup);
                }
            }
        }

        bool quiesced = false;
        try {
            quiesced = job->quiesce();
        } catch (...) {
            quiesced = false;
        }

        std::shared_ptr<State::Job> retired_job;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            const auto                  it = state->jobs.find(job->token);
            if (it == state->jobs.end() || it->second != job || job->attempt != attempt) {
                continue;
            }
            if (quiesced) {
                retired_job = it->second;
                state->jobs.erase(it);
            } else {
                job->in_flight  = false;
                job->next_retry = deadlineAfter(job->backoff);
                job->backoff    = doubledBackoff(job->backoff, state->config.max_backoff);
            }
        }
        state->changed.notify_all();
        retired_job.reset();
    }
}

}  // namespace rtp_llm
