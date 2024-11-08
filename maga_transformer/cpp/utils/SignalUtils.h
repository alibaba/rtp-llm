#pragma once

#include <cstring>
#include <string>
#include <signal.h>

#include "absl/debugging/symbolize.h"
#include "absl/debugging/stacktrace.h"

namespace rtp_llm {

bool installSighandler();

};  // namespace rtp_llm
