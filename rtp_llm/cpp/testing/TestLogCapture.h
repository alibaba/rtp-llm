#pragma once

#include <atomic>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::test {

class TestLogCapture {
public:
    explicit TestLogCapture(const std::string& name) {
        const auto id       = counter_.fetch_add(1);
        const auto base     = "/tmp/rtp_llm_" + name + "_" + std::to_string(getpid()) + "_" + std::to_string(id);
        const auto appender = "captureAppender" + std::to_string(id);
        config_path_        = base + ".conf";
        log_path_           = base + ".log";

        std::ofstream config(config_path_);
        config << "alog.rootLogger=INFO, " << appender << "\n"
               << "alog.logger.engine=INFO, " << appender << "\n"
               << "inherit.engine=false\n"
               << "alog.appender." << appender << "=FileAppender\n"
               << "alog.appender." << appender << ".fileName=" << log_path_ << "\n"
               << "alog.appender." << appender << ".layout=PatternLayout\n"
               << "alog.appender." << appender << ".layout.LogPattern=%%m\n"
               << "alog.appender." << appender << ".async_flush=false\n"
               << "alog.appender." << appender << ".flush=true\n";
        config.close();
        if (!initLogger(config_path_)) {
            throw std::runtime_error("failed to configure test log capture: " + config_path_);
        }
    }

    ~TestLogCapture() {
        Logger::getEngineLogger().flush();
        initLogger();
    }

    std::string content() const {
        Logger::getEngineLogger().flush();
        std::ifstream      input(log_path_);
        std::ostringstream content;
        content << input.rdbuf();
        return content.str();
    }

private:
    inline static std::atomic<size_t> counter_{0};
    std::string                       config_path_;
    std::string                       log_path_;
};

}  // namespace rtp_llm::test
