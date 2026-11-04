#pragma once

#include <map>
#include <memory>
#include <stdint.h>
#include <string>

#include "autil/HashFunctionBase.h"

namespace autil {

class HashFuncFactory {
public:
    HashFuncFactory();
    ~HashFuncFactory();

public:
    static HashFunctionBasePtr createHashFunc(const std::string &funcName, uint32_t partitionCount = 65536);
    static HashFunctionBasePtr createHashFunc(const std::string &funcName,
                                              const std::map<std::string, std::string> &params,
                                              uint32_t partitionCount = 65536);
};

typedef std::shared_ptr<HashFuncFactory> HashFuncFactoryPtr;

} // namespace autil
