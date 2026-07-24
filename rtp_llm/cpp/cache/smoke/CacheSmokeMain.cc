#include <gtest/gtest.h>

#include <string>

#include "autil/Log.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace {

const std::string kCacheSmokeLogConfig = R"conf(
alog.rootLogger=INFO, cacheSmokeAppender
alog.max_msg_len=4096
alog.appender.cacheSmokeAppender=ConsoleAppender
alog.appender.cacheSmokeAppender.flush=true
alog.appender.cacheSmokeAppender.layout=PatternLayout
alog.appender.cacheSmokeAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] [%%m]
alog.logger.arpc=WARN
alog.logger.kmonitor=WARN
)conf";

}  // namespace

int main(int argc, char** argv) {
    rtp_llm::initKmonitorFactory();
    AUTIL_LOG_CONFIG_FROM_STRING(kCacheSmokeLogConfig.c_str());
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    rtp_llm::stopKmonitorFactory();
    return result;
}
