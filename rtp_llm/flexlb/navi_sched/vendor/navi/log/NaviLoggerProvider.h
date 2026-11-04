#pragma once

#include "navi/common.h"
#include <vector>

namespace navi {

class NaviLogManager;
class NaviLogger;
class NaviLoggerScope;
class NaviObjectLogger;

class NaviLoggerProvider
{
public:
    NaviLoggerProvider(const std::string &logLevel = "ERROR",
                       const std::string &prefix = "",
                       bool alogToConsole = false);
    ~NaviLoggerProvider();
private:
    NaviLoggerProvider(const NaviLoggerProvider &);
    NaviLoggerProvider &operator=(const NaviLoggerProvider &);
public:
    std::string getErrorMessage() const;
    std::vector<std::string> getTrace(
            const std::string &filter = "",
            const std::string &formatPattern = "%%m") const;
    void setFormatPattern(const std::string &pattern);
private:
    std::shared_ptr<NaviLogManager> _logManager;
    NaviObjectLogger *_logger;
    NaviLoggerScope *_scope;
};

}
