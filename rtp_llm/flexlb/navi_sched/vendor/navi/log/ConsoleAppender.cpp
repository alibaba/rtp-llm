#include "navi/log/ConsoleAppender.h"
#include "navi/log/NaviLogger.h"
#include <iostream>

namespace navi {

ConsoleAppender::ConsoleAppender(const std::string &levelStr)
    : Appender(levelStr, nullptr)
{
    auto patternLayout = std::make_shared<PatternLayout>();
    patternLayout->setLogPattern(DEFAULT_LOG_PATTERN);
    setLayout(patternLayout);
}

ConsoleAppender::~ConsoleAppender() {
}

int ConsoleAppender::append(LoggingEvent& event) {
    std::string formatedStr = m_layout->format(event);
    if (formatedStr.length() > 0) {
        autil::ScopedLock lock(_lock);
        std::cout << formatedStr << std::flush;
    }
    return formatedStr.length();
}

}

