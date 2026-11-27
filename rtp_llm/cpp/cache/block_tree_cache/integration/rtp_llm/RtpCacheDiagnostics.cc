#include "RtpCacheDiagnostics.h"

#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {
uint32_t nativeLevel(CacheLogLevel level) {
    switch (level) {
        case CacheLogLevel::Trace:
            return alog::LOG_LEVEL_TRACE1;
        case CacheLogLevel::Debug:
            return alog::LOG_LEVEL_DEBUG;
        case CacheLogLevel::Info:
            return alog::LOG_LEVEL_INFO;
        case CacheLogLevel::Warning:
            return alog::LOG_LEVEL_WARN;
        case CacheLogLevel::Error:
            return alog::LOG_LEVEL_ERROR;
    }
    return alog::LOG_LEVEL_ERROR;
}
}  // namespace

void installRtpCacheDiagnostics() {
    static const CacheDiagnosticSink sink{
        [](CacheLogLevel level) { return Logger::getEngineLogger().isLevelEnabled(nativeLevel(level)); },
        [](CacheLogLevel level, const char* file, int line, const char* fn, const char* text) {
            Logger::getEngineLogger().log(nativeLevel(level), file, line, fn, "%s", text);
        },
        [](const char* file, int line, const char* text) { myAssert(file, line, text); },
    };
    setCacheDiagnosticSink(sink);
}
}  // namespace rtp_llm
