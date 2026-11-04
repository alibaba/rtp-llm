#include "navi/log/NaviLoggerProvider.h"
#include "navi/log/NaviLogger.h"
#include "navi/log/NaviLogManager.h"
#include "navi/log/TraceAppender.h"
#include "navi/log/ConsoleAppender.h"

namespace navi {

NaviLoggerProvider::NaviLoggerProvider(const std::string &logLevel,
                                       const std::string &prefix,
                                       bool alogToConsole)
    : _logManager(new NaviLogManager(0))
{
    LogConfig config;
    config.maxMessageLength = 1024 * 1024;
    _logManager->init(config);
    auto logger = _logManager->createLogger();
    if (alogToConsole) {
        auto consoleAppender = std::make_shared<ConsoleAppender>(logLevel);
        auto traceAppender = std::make_shared<TraceAppender>(logLevel);
        traceAppender->setSubAppender(std::move(consoleAppender));
        logger->setTraceAppender(traceAppender);
    } else {
        logger->setTraceAppender(std::make_shared<TraceAppender>(logLevel));
        logger->addAppender(std::make_shared<ConsoleAppender>(logLevel));
    }
    _logger = new NaviObjectLogger(this, prefix.c_str(), logger);
    _scope = new NaviLoggerScope(*_logger);
}

NaviLoggerProvider::~NaviLoggerProvider() {
    delete _scope;
    delete _logger;
}

std::string NaviLoggerProvider::getErrorMessage() const {
    auto errorEvent = _logger->logger->firstErrorEvent();
    if (errorEvent) {
        return errorEvent->message;
    } else {
        return std::string();
    }
}

std::vector<std::string> NaviLoggerProvider::getTrace(
        const std::string &filter,
        const std::string &formatPattern) const
{
    TraceCollector collector;
    collector.setFormatPattern(formatPattern);
    _logger->logger->collectTrace(collector);
    std::vector<std::string> traceVec;
    collector.format(traceVec);
    std::vector<std::string> ret;
    ret.reserve(traceVec.size());
    for (const auto &msg : traceVec) {
        if (!filter.empty()) {
            if (msg.find(filter) != std::string::npos) {
                ret.push_back(msg);
            }
        } else {
            ret.push_back(msg);
        }
    }
    return ret;
}
}
