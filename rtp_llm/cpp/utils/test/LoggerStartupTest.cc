#include "rtp_llm/cpp/utils/Logger.h"
#include "gtest/gtest.h"
#include <fstream>
#include <unistd.h>

namespace rtp_llm {
namespace {

const char* configuredName(const char* name) {
    // Command-line --test_env can override this target's BUILD environment.
    return autil::EnvUtil::getEnv("FT_SERVER_TEST", 0) == 1 ? "console" : name;
}

}  // namespace

TEST(LoggerStartupTest, BindPreservesIdentityAndDoesNotChangeBackendLevel) {
    Logger logger("startup_test_original");
    logger.setRank(37);
    const auto ip          = logger.ip_;
    auto*      original    = logger.logger_;
    auto*      replacement = alog::Logger::getLogger("startup_test_replacement");
    replacement->setLevel(alog::LOG_LEVEL_DEBUG);

    logger.bindBackendForStartup(replacement, alog::LOG_LEVEL_DEBUG);

    EXPECT_EQ(logger.logger_, replacement);
    EXPECT_EQ(logger.base_log_level_, alog::LOG_LEVEL_DEBUG);
    EXPECT_EQ(logger.rank_, 37);
    EXPECT_EQ(logger.ip_, ip);
    EXPECT_EQ(replacement->getLevel(), alog::LOG_LEVEL_DEBUG);
    EXPECT_EQ(alog::Logger::getLogger(configuredName("startup_test_original")), original);
}

TEST(LoggerStartupTest, SuccessfulInitRebindsAllLoggersAndFailedInitDoesNot) {
    struct Entry {
        Logger&     logger;
        const char* name;
    };
    Entry entries[] = {{Logger::getEngineLogger(), "engine"},
                       {Logger::getAccessLogger(), "access"},
                       {Logger::getQueryAccessLogger(), "query_access"},
                       {Logger::getStackTraceLogger(), "stack_trace"}};
    auto* stale     = alog::Logger::getLogger("startup_test_stale");
    for (auto& entry : entries) {
        entry.logger.bindBackendForStartup(stale, alog::LOG_LEVEL_DEBUG);
    }

    char      path[] = "/tmp/rtp_logger_startup_XXXXXX";
    const int fd     = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    const std::string missing_path = std::string(path) + ".missing";
    EXPECT_FALSE(initLogger(missing_path));
    for (auto& entry : entries) {
        EXPECT_EQ(entry.logger.logger_, stale);
    }
    {
        std::ofstream config(path);
        config << "alog.rootLogger=INFO\n";
    }
    // Repeated init also re-applies the startup environment override.
    for (int attempt = 0; attempt < 2; ++attempt) {
        EXPECT_TRUE(initLogger(path));
        for (auto& entry : entries) {
            EXPECT_EQ(entry.logger.logger_, alog::Logger::getLogger(configuredName(entry.name)));
            EXPECT_EQ(entry.logger.base_log_level_, alog::LOG_LEVEL_INFO);
            EXPECT_EQ(entry.logger.logger_->getLevel(), alog::LOG_LEVEL_INFO);
        }
        // In console mode all four entries alias one backend. Verify them all
        // before changing its level for the next initialization attempt.
        for (auto& entry : entries) {
            entry.logger.logger_->setLevel(alog::LOG_LEVEL_DEBUG);
        }
    }
    unlink(path);
    EXPECT_EQ(alog::Logger::getLogger("startup_test_stale"), stale);
}

}  // namespace rtp_llm
