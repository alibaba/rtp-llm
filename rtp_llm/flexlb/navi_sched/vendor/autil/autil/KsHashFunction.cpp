#include "autil/KsHashFunction.h"

#include <iosfwd>

#include "autil/Log.h"
#include "autil/StringUtil.h"

using namespace std;
namespace autil {
AUTIL_DECLARE_AND_SETUP_LOGGER(autil, KsHashFunction);

uint32_t KsHashFunction::getHashId(const std::string &str) const {
    uint64_t value = 0;
    bool ret = StringUtil::fromString(str, value);
    if (!ret) {
        AUTIL_LOG(WARN, "%s is not number!", str.c_str());
        return 0;
    }
    return getHashId(value, _ksHashRange, _hashSize);
}

uint32_t KsHashFunction::getHashId(uint64_t key, uint32_t rangeBefore, uint32_t rangeAfter) {
    uint64_t tempHashId = key % rangeBefore;
    return (2 * tempHashId + 1) * rangeAfter / rangeBefore / 2;
}

// fix for symbol compability
uint32_t KsHashFunction::getHashId(uint32_t key, uint32_t rangeBefore, uint32_t rangeAfter) {
    uint64_t tempHashId = key % rangeBefore;
    return (2 * tempHashId + 1) * rangeAfter / rangeBefore / 2;
}

} // namespace autil
