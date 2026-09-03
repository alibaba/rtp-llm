#include <gtest/gtest.h>
#include "autil/Log.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

const std::string UNITTEST_DEFAULT_LOG_CONF = R"conf(
alog.rootLogger=INFO, unittestAppender
alog.max_msg_len=4096
alog.appender.unittestAppender=ConsoleAppender
alog.appender.unittestAppender.flush=true
alog.appender.unittestAppender.layout=PatternLayout
alog.appender.unittestAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] [%%m]
alog.logger.arpc=WARN
alog.logger.kmonitor=WARN
)conf";

int main(int argc, char** argv) {
    rtp_llm::initKmonitorFactory();
    AUTIL_LOG_CONFIG_FROM_STRING(UNITTEST_DEFAULT_LOG_CONF.c_str());
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    rtp_llm::stopKmonitorFactory();
    return ret;
}