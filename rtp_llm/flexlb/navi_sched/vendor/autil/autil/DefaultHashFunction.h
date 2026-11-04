#pragma once

#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"

namespace autil {

class DefaultHashFunction : public HashFunctionBase {
public:
    DefaultHashFunction(const std::string &hashFunction, uint32_t partitionCount)
        : HashFunctionBase(hashFunction, partitionCount) {}
    virtual ~DefaultHashFunction() {}

public:
    /*override*/ virtual uint32_t getHashId(const std::string &str) const;
    /* override */ uint32_t getHashId(const char *buf, size_t len) const;

public:
    int hashString(const std::string &str) const;
};

typedef std::shared_ptr<DefaultHashFunction> DefaultHashFunctionPtr;

} // namespace autil
