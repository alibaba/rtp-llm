#include "autil/MurmurHash3Function.h"

#include <algorithm>
#include <cstddef>

#include "autil/Log.h"
#include "autil/MurmurHash3.h"
#include "autil/StringUtil.h"

using namespace std;

namespace autil {

AUTIL_DECLARE_AND_SETUP_LOGGER(autil, MurmurHash3Function);

MurmurHash3Function::MurmurHash3Function(const std::string &hashFunction,
                                         const std::map<std::string, std::string> &params,
                                         uint32_t partitionCount)
    : HashFunctionBase(hashFunction, partitionCount), _params(params), _seed(DEFAULT_HASH_SEED) {}

bool MurmurHash3Function::init() {
    auto it = _params.find("seed");
    if (it != _params.end()) {
        if (!StringUtil::fromString(it->second, _seed)) {
            AUTIL_LOG(WARN, "parse seed [%s] failed", it->second.c_str());
            return false;
        }
    }
    return true;
}

uint32_t MurmurHash3Function::getHashId(const std::string &str) const {
    if (_hashSize == 0) {
        return 0;
    }

    uint32_t h = 0;
    MurmurHash3_x86_32(str.c_str(), str.length(), _seed, &h);
    return h % _hashSize;
}

} // namespace autil
