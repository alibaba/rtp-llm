#include "autil/NumberDivHashFunction.h"

#include <iosfwd>

#include "autil/HashAlgorithm.h"
#include "autil/StringUtil.h"

using namespace std;

namespace autil {

AUTIL_LOG_SETUP(autil, NumberDivHashFunction);

static const string DIVISOR_KEY = "divisor";

NumberDivHashFunction::NumberDivHashFunction(const std::string &hashFunction,
                                             const std::map<std::string, std::string> &params,
                                             uint32_t partitionCount)
    : HashFunctionBase(hashFunction, partitionCount), _params(params) {}
NumberDivHashFunction::~NumberDivHashFunction() {}

bool NumberDivHashFunction::init() {
    if (auto it = _params.find(DIVISOR_KEY); it != _params.end()) {
        if (!StringUtil::fromString(it->second, _divisor)) {
            AUTIL_LOG(ERROR, "parse param [%s] value [%s] failed", it->first.c_str(), it->second.c_str());
            return false;
        }
        if (_divisor == 0) {
            AUTIL_LOG(ERROR, "divisor value must not be zero");
            return false;
        }
    } else {
        AUTIL_LOG(ERROR, "param [%s] not set", DIVISOR_KEY.c_str());
        return false;
    }
    return true;
}

int64_t NumberDivHashFunction::hashString(const string &str) {
    int64_t num;
    if (StringUtil::fromString(str, num)) {
        return num;
    } else {
        return 0;
    }
}

uint32_t NumberDivHashFunction::getHashId(const std::string &str) const {
    if (_hashSize == 0) {
        return 0;
    }
    int64_t val = hashString(str) / _divisor;
    return (uint32_t)(HashAlgorithm::hashString((char *)&val, sizeof(val), 0) % _hashSize);
}

} // namespace autil
