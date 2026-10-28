#pragma once
#include <exception>

namespace rtp_llm::kvcm {

// Aggregate source-buffer safety independently of transfer success. Every
// worker must explicitly prove safety; legacy/missing responses are unknown.
class WorkerPayloadProof {
public:
    enum class Result { SUCCESS, BUFFER_SAFE_FAILURE, UNKNOWN };

    void observe(bool explicit_buffer_safe, bool success, bool failed) noexcept {
        all_safe_ = all_safe_ && explicit_buffer_safe && (success || failed);
        all_success_ = all_success_ && success;
    }

    Result result() const noexcept {
        return !all_safe_ ? Result::UNKNOWN :
               all_success_ ? Result::SUCCESS : Result::BUFFER_SAFE_FAILURE;
    }

    std::exception_ptr failure(bool any_timeout,
                               const std::exception_ptr& safe_failure,
                               const std::exception_ptr& unknown_failure,
                               const std::exception_ptr& timeout_failure) const noexcept {
        switch (result()) {
            case Result::SUCCESS: return {};
            case Result::BUFFER_SAFE_FAILURE: return safe_failure;
            case Result::UNKNOWN: return any_timeout ? timeout_failure : unknown_failure;
        }
        return unknown_failure;
    }

private:
    bool all_safe_ = true;
    bool all_success_ = true;
};

}  // namespace rtp_llm::kvcm
