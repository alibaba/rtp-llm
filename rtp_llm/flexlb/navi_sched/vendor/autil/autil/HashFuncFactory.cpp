#include "autil/HashFuncFactory.h"

#include <iosfwd>

#include "autil/DefaultHashFunction.h"
#include "autil/ExtendHashFunction.h"
#include "autil/Hash64Function.h"
#include "autil/KsHashFunction.h"
#include "autil/Log.h"
#include "autil/MurmurHash3Function.h"
#include "autil/NumberDivHashFunction.h"
#include "autil/NumberHashFunction.h"
#include "autil/RoutingHashFunction.h"
#include "autil/SmartWaveHashFunction.h"
#include "autil/StringTokenizer.h"
#include "autil/StringUtil.h"

using namespace std;

namespace autil {
AUTIL_DECLARE_AND_SETUP_LOGGER(autil, HashFuncFactory);

const std::string SEP = "#";
HashFuncFactory::HashFuncFactory() {}

HashFuncFactory::~HashFuncFactory() {}

HashFunctionBasePtr HashFuncFactory::createHashFunc(const string &funcName, uint32_t partitionCount) {
    static map<string, string> params;
    return HashFuncFactory::createHashFunc(funcName, params, partitionCount);
}

HashFunctionBasePtr
HashFuncFactory::createHashFunc(const string &funcName, const map<string, string> &params, uint32_t partitionCount) {
    StringTokenizer tokenizer(funcName, SEP, StringTokenizer::TOKEN_IGNORE_EMPTY | StringTokenizer::TOKEN_TRIM);
    if (tokenizer.getNumTokens() <= 0) {
        return HashFunctionBasePtr();
    }
    string realFuncName = tokenizer[0];
    HashFunctionBasePtr func;
    if (realFuncName == "HASH") {
        func.reset(new DefaultHashFunction(realFuncName, partitionCount));
    } else if (realFuncName == "HASH64") {
        func.reset(new Hash64Function(realFuncName, partitionCount));
    } else if (realFuncName == "GALAXY_HASH") {
        func.reset(new ExtendHashFunction(realFuncName, partitionCount));
    } else if (realFuncName == "NUMBER_HASH") {
        func.reset(new NumberHashFunction(realFuncName, partitionCount));
    } else if (realFuncName == "NUMBER_DIV_HASH") {
        func.reset(new NumberDivHashFunction(realFuncName, params, partitionCount));
    } else if (realFuncName == "KINGSO_HASH") {
        uint32_t ksHashRange = 720;
        if (tokenizer.getNumTokens() == 2) {
            uint32_t value = 0;
            bool ret = StringUtil::fromString(tokenizer[1], value);
            if (ret && value != 0) {
                ksHashRange = value;
            } else {
                AUTIL_LOG(WARN, "parameter %s is invalid.", funcName.c_str());
            }
        }
        func.reset(new KsHashFunction(realFuncName, partitionCount, ksHashRange));
    } else if (realFuncName == "ROUTING_HASH") {
        func.reset(new RoutingHashFunction(realFuncName, params, partitionCount));
    } else if (realFuncName == "SMARTWAVE_HASH") {
        func.reset(new SmartWaveHashFunction(realFuncName, params, partitionCount));
    } else if (realFuncName == MurmurHash3Function::name()) {
        // MURMUR_HASH3
        func.reset(new MurmurHash3Function(realFuncName, params, partitionCount));
    } else {
        AUTIL_LOG(ERROR, "unsupported hash function[%s], real[%s]", funcName.c_str(), realFuncName.c_str());
        return nullptr;
    }
    if (!func->init()) {
        AUTIL_LOG(ERROR,
                  "init hash function failed, real[%s] name[%s]",
                  realFuncName.c_str(),
                  func->getFunctionName().c_str());
        return nullptr;
    }
    return func;
}

} // namespace autil
