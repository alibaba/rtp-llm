#pragma once

#include <cstdio>
#include <stdexcept>

namespace rtp_llm::btc_host_standalone {

[[noreturn]] inline void failCheck(const char* message) {
    throw std::logic_error(message);
}

template<typename... Args>
[[noreturn]] void failCheck(const char* format, Args... args) {
    char message[512];
    std::snprintf(message, sizeof(message), format, args...);
    throw std::logic_error(message);
}

}  // namespace rtp_llm::btc_host_standalone

// Keep standalone BTC bridge builds independent of RTP's logger and backtrace machinery.
#define RTP_LLM_CHECK_WITH_INFO(condition, ...)                                                                         \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            rtp_llm::btc_host_standalone::failCheck(__VA_ARGS__);                                                        \
        }                                                                                                              \
    } while (false)
#define RTP_LLM_CHECK(condition) RTP_LLM_CHECK_WITH_INFO(condition, "%s", #condition)
#define RTP_LLM_LOG_INFO(...) ((void)0)
#define RTP_LLM_LOG_DEBUG(...) ((void)0)
#define RTP_LLM_LOG_ERROR(...)                                                                                         \
    do {                                                                                                               \
        std::fprintf(stderr, __VA_ARGS__);                                                                               \
        std::fputc('\n', stderr);                                                                                       \
    } while (false)
#define RTP_LLM_LOG_WARNING(...) RTP_LLM_LOG_ERROR(__VA_ARGS__)
