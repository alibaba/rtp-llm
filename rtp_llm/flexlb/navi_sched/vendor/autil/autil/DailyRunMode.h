#pragma once

#include <string>

namespace autil {

class DailyRunMode {
private:
    DailyRunMode();
    ~DailyRunMode();

public:
    static bool enable();

public:
    static bool _isEnable;
    static const std::string DAILY_RUN_MODE;
};

} // namespace autil
