#pragma once

#include <memory>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"

namespace autil {

class NumberHashFunction : public HashFunctionBase {
public:
    NumberHashFunction(const std::string &hashFunction, uint32_t partitionCount)
        : HashFunctionBase(hashFunction, partitionCount) {}
    virtual ~NumberHashFunction() {}

public:
    uint32_t getHashId(const std::string &str) const override;

private:
    static int64_t hashString(const std::string &str);
};

typedef std::shared_ptr<NumberHashFunction> NumberHashFunctionPtr;
} // namespace autil
