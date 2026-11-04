#include "autil/NumberHashFunction.h"

#include <iosfwd>

#include "autil/StringUtil.h"

using namespace std;

namespace autil {

int64_t NumberHashFunction::hashString(const string &str) {
    int64_t num;
    if (StringUtil::fromString(str, num)) {
        return num;
    } else {
        return 0;
    }
}

uint32_t NumberHashFunction::getHashId(const std::string &str) const {
    if (_hashSize == 0) {
        return 0;
    }
    return (uint32_t)((uint64_t)this->hashString(str) % _hashSize);
}

} // namespace autil
