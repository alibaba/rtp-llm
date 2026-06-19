#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Carries an ErrorCode through a throw boundary so LogitsProcessorStates /
// GenerateStream can route processor failures to the owning stream's
// error_info_ without collapsing the original code to EXECUTION_EXCEPTION.
class LogitsProcessorException: public std::runtime_error {
public:
    LogitsProcessorException(ErrorCode code, std::string message):
        std::runtime_error(std::move(message)), code_(code) {}

    ErrorCode code() const noexcept {
        return code_;
    }

private:
    ErrorCode code_;
};

}  // namespace rtp_llm
