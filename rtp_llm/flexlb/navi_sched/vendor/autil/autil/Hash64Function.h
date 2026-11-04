#pragma once

#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"

namespace autil {

class Hash64Function : public HashFunctionBase {
public:
    Hash64Function(const std::string &hashFunction, uint32_t partitionCount)
        : HashFunctionBase(hashFunction, partitionCount) {}
    virtual ~Hash64Function() {}

public:
    /*override*/ virtual uint32_t getHashId(const std::string &str) const;
    /* override */ uint32_t getHashId(const char *buf, size_t len) const;
};

typedef std::shared_ptr<Hash64Function> Hash64FunctionPtr;

} // namespace autil
