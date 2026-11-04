#pragma once

#include "navi/log/AlogAppender.h"
#include "autil/Lock.h"

namespace navi {

class ConsoleAppender : public Appender
{
public:
    ConsoleAppender(const std::string &levelStr);
    ~ConsoleAppender();
private:
    ConsoleAppender(const ConsoleAppender &);
    ConsoleAppender &operator=(const ConsoleAppender &);
public:
    int append(LoggingEvent& event) override;
    void flush() override {
    }
private:
    autil::ThreadMutex _lock;
};

}
