#pragma once

#include <map>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"

namespace autil {

class MurmurHash3Function : public HashFunctionBase {
public:
    static constexpr uint32_t DEFAULT_HASH_SEED = 104729;

    MurmurHash3Function(const std::string &hashFunction,
                        const std::map<std::string, std::string> &param,
                        uint32_t partitionCount);

    virtual ~MurmurHash3Function() {}

    static const char *name() { return "MURMUR_HASH3"; }

public:
    bool init() override;
    uint32_t getHashId(const std::string &str) const override;

private:
    std::map<std::string, std::string> _params;
    uint32_t _seed;
};

typedef std::shared_ptr<MurmurHash3Function> MurmurHash3FunctionPtr;

} // namespace autil
