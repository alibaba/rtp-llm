#pragma once

#include <memory>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"

namespace autil {

class KsHashFunction : public HashFunctionBase {
public:
    KsHashFunction(const std::string &hashFunction, uint32_t partitionCout = 65536, uint32_t ksHashRange = 720)
        : HashFunctionBase(hashFunction, partitionCout), _ksHashRange(ksHashRange) {}
    virtual ~KsHashFunction() {}

public:
    virtual uint32_t getHashId(const std::string &str) const;
    uint32_t getKsHashRange() const { return _ksHashRange; }

public:
    static uint32_t getHashId(uint64_t key, uint32_t rangeBefore, uint32_t rangeAfter);
    static uint32_t getHashId(uint32_t key, uint32_t rangeBefore, uint32_t rangeAfter);

private:
    uint32_t _ksHashRange;
};

typedef std::shared_ptr<KsHashFunction> KsHashFunctionPtr;

} // namespace autil
