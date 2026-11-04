#pragma once

#include <memory>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"
#include "autil/Log.h"

namespace autil {

class NumberDivHashFunction : public HashFunctionBase {
public:
    NumberDivHashFunction(const std::string &hashFunction,
                          const std::map<std::string, std::string> &params,
                          uint32_t partitionCount);
    virtual ~NumberDivHashFunction();

public:
    bool init() override;
    uint32_t getHashId(const std::string &str) const override;

private:
    static int64_t hashString(const std::string &str);

private:
    AUTIL_LOG_DECLARE();

private:
    std::map<std::string, std::string> _params;
    size_t _divisor;
};

} // namespace autil
