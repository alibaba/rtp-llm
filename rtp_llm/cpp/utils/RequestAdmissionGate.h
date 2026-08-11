#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace rtp_llm {

class RequestAdmissionGate {
private:
    struct State {
        std::mutex              mutex;
        std::condition_variable drained;
        bool                    closed = false;
        size_t                  active = 0;
    };

public:
    class Permit {
    public:
        Permit() noexcept = default;

        Permit(const Permit&)            = delete;
        Permit& operator=(const Permit&) = delete;

        Permit(Permit&& other) noexcept: state_(std::move(other.state_)) {}

        Permit& operator=(Permit&& other) noexcept {
            if (this != &other) {
                reset();
                state_ = std::move(other.state_);
            }
            return *this;
        }

        ~Permit() {
            reset();
        }

        explicit operator bool() const noexcept {
            return state_ != nullptr;
        }

        void reset() noexcept {
            auto state = std::move(state_);
            if (!state) {
                return;
            }

            bool notify = false;
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (state->active == 0) {
                    return;
                }
                --state->active;
                notify = state->closed && state->active == 0;
            }
            if (notify) {
                state->drained.notify_all();
            }
        }

    private:
        friend class RequestAdmissionGate;

        explicit Permit(std::shared_ptr<State> state) noexcept: state_(std::move(state)) {}

        std::shared_ptr<State> state_;
    };

    RequestAdmissionGate(): state_(std::make_shared<State>()) {}

    RequestAdmissionGate(const RequestAdmissionGate&)            = delete;
    RequestAdmissionGate& operator=(const RequestAdmissionGate&) = delete;
    RequestAdmissionGate(RequestAdmissionGate&&)                 = delete;
    RequestAdmissionGate& operator=(RequestAdmissionGate&&)      = delete;

    [[nodiscard]] Permit tryAcquire() {
        auto state = state_;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->closed) {
            return {};
        }
        ++state->active;
        return Permit(std::move(state));
    }

    void close() {
        auto state  = state_;
        bool notify = false;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->closed) {
                return;
            }
            state->closed = true;
            notify        = state->active == 0;
        }
        if (notify) {
            state->drained.notify_all();
        }
    }

    bool isClosed() const {
        auto state = state_;
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->closed;
    }

    void wait() const {
        auto state = state_;
        std::unique_lock<std::mutex> lock(state->mutex);
        state->drained.wait(lock, [&state]() { return state->closed && state->active == 0; });
    }

    template<typename Clock, typename Duration>
    bool waitUntil(const std::chrono::time_point<Clock, Duration>& deadline) const {
        auto state = state_;
        std::unique_lock<std::mutex> lock(state->mutex);
        return state->drained.wait_until(
            lock, deadline, [&state]() { return state->closed && state->active == 0; });
    }

    template<typename Rep, typename Period>
    bool waitFor(const std::chrono::duration<Rep, Period>& timeout) const {
        auto state = state_;
        std::unique_lock<std::mutex> lock(state->mutex);
        return state->drained.wait_for(
            lock, timeout, [&state]() { return state->closed && state->active == 0; });
    }

private:
    std::shared_ptr<State> state_;
};

}  // namespace rtp_llm
