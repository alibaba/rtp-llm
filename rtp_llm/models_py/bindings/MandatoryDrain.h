#pragma once

#include <cstdlib>
#include <exception>
#include <utility>

namespace rtp_llm {

// Runs an asynchronous submission under a fail-closed completion contract. The
// caller may return or propagate the submission error only after drain proves
// that all previously submitted work is terminal.
template<typename Submit, typename Drain, typename FailStop>
void runWithMandatoryDrain(Submit&& submit, Drain&& drain, FailStop&& fail_stop) {
    std::exception_ptr submit_error;
    try {
        std::forward<Submit>(submit)();
    } catch (...) {
        submit_error = std::current_exception();
    }

    bool drained = false;
    try {
        drained = std::forward<Drain>(drain)();
    } catch (...) {
        drained = false;
    }
    if (!drained) {
        try {
            std::forward<FailStop>(fail_stop)();
        } catch (...) {
        }
        std::abort();
    }

    if (submit_error) {
        std::rethrow_exception(submit_error);
    }
}

}  // namespace rtp_llm
