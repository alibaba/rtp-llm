#include <gtest/gtest.h>
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
class CountingAppender: public alog::Appender {
public:
    int count = 0;
    int append(alog::LoggingEvent&) override {
        return ++count;
    }
    void flush() override {}
};
}  // namespace

TEST(AccessLoggerTest, DisableAccessLogKeepsEngineLogs) {
    const char*       original = std::getenv("DISABLE_ACCESS_LOG");
    const std::string saved    = original ? original : "";
    for (const char* value : {"0", "1"}) {
        setenv("DISABLE_ACCESS_LOG", value, 1);
        Logger logger("access_test");
        auto*  sink =
            alog::Logger::getLogger(autil::EnvUtil::getEnv("FT_SERVER_TEST", 0) == 1 ? "console" : "access_test");
        auto* appender = new CountingAppender();
        sink->setAppender(appender);
        sink->setLevel(alog::LOG_LEVEL_INFO);
        logger.log_access(alog::LOG_LEVEL_INFO, "%s", "request and response");
        EXPECT_EQ(appender->count, std::string(value) == "1" ? 0 : 1);
        const int before = appender->count;
        logger.log(alog::LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, "%s", "engine diagnostic");
        EXPECT_EQ(appender->count, before + 1);
        sink->removeAllAppenders();
    }
    if (original) {
        setenv("DISABLE_ACCESS_LOG", saved.c_str(), 1);
    } else {
        unsetenv("DISABLE_ACCESS_LOG");
    }
}
}  // namespace rtp_llm
