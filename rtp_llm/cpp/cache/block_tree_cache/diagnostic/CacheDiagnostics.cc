#include "CacheDiagnostics.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace rtp_llm {
namespace {
std::atomic<const CacheDiagnosticSink*> current_sink{nullptr};

std::string message(const char* format, va_list args) {
    va_list copy;
    va_copy(copy, args);
    const int length = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (length < 0) {
        return format;
    }
    std::string result(static_cast<size_t>(length) + 1, '\0');
    std::vsnprintf(result.data(), result.size(), format, args);
    result.resize(static_cast<size_t>(length));
    return result;
}
}  // namespace

void setCacheDiagnosticSink(const CacheDiagnosticSink& sink) noexcept {
    current_sink.store(&sink, std::memory_order_release);
}

bool cacheLogEnabled(CacheLogLevel level) {
    const auto* sink = current_sink.load(std::memory_order_acquire);
    return sink ? sink->enabled(level) : level >= CacheLogLevel::Warning;
}

void cacheLog(CacheLogLevel level, const char* file, int line, const char* function, const char* format, ...) {
    va_list args;
    va_start(args, format);
    const auto text = message(format, args);
    va_end(args);
    if (const auto* sink = current_sink.load(std::memory_order_acquire)) {
        sink->write(level, file, line, function, text.c_str());
    } else {
        std::fprintf(stderr, "%s:%d %s: %s\n", file, line, function, text.c_str());
    }
}

[[noreturn]] void cacheCheckFailed(const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);
    const auto text = message(format, args);
    va_end(args);
    if (const auto* sink = current_sink.load(std::memory_order_acquire)) {
        sink->fail(file, line, text.c_str());
    }
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + text);
}
}  // namespace rtp_llm
