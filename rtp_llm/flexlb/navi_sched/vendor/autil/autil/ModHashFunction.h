#pragma once

#include <memory>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"

namespace autil {

class ModHashFunction : public HashFunctionBase {
public:
    ModHashFunction(const std::string &hashFunction, uint32_t partitionCount)
        : HashFunctionBase(hashFunction, partitionCount) {}
    virtual ~ModHashFunction() {}

public:
    /*override*/ virtual uint32_t getHashId(const std::string &str) const;
};

typedef std::shared_ptr<ModHashFunction> ModHashFunctionPtr;

} // namespace autil
