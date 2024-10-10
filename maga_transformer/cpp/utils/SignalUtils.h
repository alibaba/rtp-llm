#pragma once

#include <cstring>
#include <string>
#include <signal.h>

#include "absl/debugging/symbolize.h"
#include "absl/debugging/stacktrace.h"

namespace rtp_llm {

void getStackTraceSighandler(int signum, siginfo_t* siginfo, void* ucontext);

bool installSighandler();

};  // namespace rtp_llm
