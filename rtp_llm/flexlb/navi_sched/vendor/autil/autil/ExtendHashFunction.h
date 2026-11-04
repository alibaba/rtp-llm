#pragma once

#include <memory>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"
#include "autil/KeyRangeManager.h"

namespace autil {

class ExtendHashFunction : public HashFunctionBase {
public:
    ExtendHashFunction(const std::string &hashFunction, uint32_t partitionCount)
        : HashFunctionBase(hashFunction, partitionCount), _hasher(partitionCount) {}
    ~ExtendHashFunction() {}

public:
    virtual uint32_t getHashId(const std::string &str) const;

public:
    static std::string hashString(const std::string &str);

protected:
    static std::string extendMd5(uint8_t md[16]);

private:
    AutoHashKeyRangeManager _hasher;
};

typedef std::shared_ptr<ExtendHashFunction> ExtendHashFunctionPtr;

} // namespace autil
