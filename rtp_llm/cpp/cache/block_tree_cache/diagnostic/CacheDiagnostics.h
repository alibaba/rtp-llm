#pragma once

#include <chrono>
#include <atomic>
#include <cstdint>

namespace rtp_llm {

enum class CacheLogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error
};

// An engine can install a process-lifetime sink before constructing the cache.
// Check failures must throw even when the sink has disabled logging.
struct CacheDiagnosticSink {
    bool (*enabled)(CacheLogLevel);
    void (*write)(CacheLogLevel, const char* file, int line, const char* function, const char* message);
    void (*fail)(const char* file, int line, const char* message);
};

void setCacheDiagnosticSink(const CacheDiagnosticSink& sink) noexcept;
bool cacheLogEnabled(CacheLogLevel level);
void cacheLog(CacheLogLevel level, const char* file, int line, const char* function, const char* format, ...);
[[noreturn]] void cacheCheckFailed(const char* file, int line, const char* format, ...);

inline int64_t cacheNowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
inline int64_t cacheNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace rtp_llm

#define BTC_LOG(level, ...)                                                                                            \
    do {                                                                                                               \
        if (::rtp_llm::cacheLogEnabled(::rtp_llm::CacheLogLevel::level)) {                                             \
            ::rtp_llm::cacheLog(::rtp_llm::CacheLogLevel::level, __FILE__, __LINE__, __func__, __VA_ARGS__);           \
        }                                                                                                              \
    } while (false)
#define BTC_LOG_TRACE(...) BTC_LOG(Trace, __VA_ARGS__)
#define BTC_LOG_DEBUG(...) BTC_LOG(Debug, __VA_ARGS__)
#define BTC_LOG_INFO(...) BTC_LOG(Info, __VA_ARGS__)
#define BTC_LOG_WARNING(...) BTC_LOG(Warning, __VA_ARGS__)
#define BTC_LOG_ERROR(...) BTC_LOG(Error, __VA_ARGS__)
#define BTC_INTERVAL_LOG_WARNING(seconds, ...)                                                                         \
    do {                                                                                                               \
        static std::atomic<int64_t> next_log_us{0};                                                                    \
        const auto                  now =                                                                              \
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()) \
                .count();                                                                                              \
        auto next = next_log_us.load(std::memory_order_relaxed);                                                       \
        if (now >= next                                                                                                \
            && next_log_us.compare_exchange_strong(                                                                    \
                next, now + int64_t(seconds) * 1000000, std::memory_order_relaxed)) {                                  \
            BTC_LOG_WARNING(__VA_ARGS__);                                                                              \
        }                                                                                                              \
    } while (false)
#define BTC_CHECK_WITH_INFO(expr, ...)                                                                                 \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            ::rtp_llm::cacheCheckFailed(__FILE__, __LINE__, __VA_ARGS__);                                              \
        }                                                                                                              \
    } while (false)
#define BTC_CHECK(expr) BTC_CHECK_WITH_INFO(expr, "%s", #expr)
#define BTC_FAIL(...) ::rtp_llm::cacheCheckFailed(__FILE__, __LINE__, __VA_ARGS__)
