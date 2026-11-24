#pragma once

#include <functional>
#include <mutex>
#include <utility>

namespace rtp_llm {

// Owns an exact registration. Borrowed BlockBuffer owners remain valid after reset.
class UserBufferRegistration {
public:
    explicit UserBufferRegistration(std::function<void()> release): release_(std::move(release)) {}
    ~UserBufferRegistration() {
        reset();
    }
    UserBufferRegistration(const UserBufferRegistration&)            = delete;
    UserBufferRegistration& operator=(const UserBufferRegistration&) = delete;

    void reset() {
        std::function<void()> release;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            release = std::move(release_);
        }
        if (release) {
            release();
        }
    }

private:
    std::mutex            mutex_;
    std::function<void()> release_;
};

}  // namespace rtp_llm
