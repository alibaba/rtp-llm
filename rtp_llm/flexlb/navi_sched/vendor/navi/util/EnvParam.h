#pragma once

#include <stdint.h>

namespace navi {

struct EnvParam {
public:
    EnvParam();
    ~EnvParam();
    EnvParam(const EnvParam &) = default;
    EnvParam &operator=(const EnvParam &) = default;

public:
    static const EnvParam &inst();

public:
    int64_t forceWaitNoQueryTime;
};

} // namespace navi
