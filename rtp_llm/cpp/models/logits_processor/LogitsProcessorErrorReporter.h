#pragma once

#include <functional>
#include <string>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// stream_lock_held picks reportError vs reportErrorWithoutLock at the call site.
using LogitsProcessorErrorReporter = std::function<void(ErrorCode, const std::string&, bool stream_lock_held)>;

}  // namespace rtp_llm
