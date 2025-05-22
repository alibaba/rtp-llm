#pragma once

#include <cstring>
#include <string>
#include <signal.h>

namespace rtp_llm {

bool installSighandler();

void printSignalStackTrace(int signum, siginfo_t* siginfo, void* ucontext);

};  // namespace rtp_llm
