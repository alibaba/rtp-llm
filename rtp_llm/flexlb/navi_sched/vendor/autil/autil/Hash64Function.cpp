#include "autil/Hash64Function.h"

#include <assert.h>
#include <cstddef>

#include "autil/HashAlgorithm.h"

using namespace std;
namespace autil {

uint32_t Hash64Function::getHashId(const std::string &str) const {
    int64_t h = HashAlgorithm::hashString64Deprecated(str.c_str(), str.length(), 1) & INT64_MAX;
    return h % _hashSize;
}

uint32_t Hash64Function::getHashId(const char *buf, size_t len) const {
    assert(buf);
    int64_t h = HashAlgorithm::hashString64Deprecated(buf, len, 1) & INT64_MAX;
    return h % _hashSize;
}

} // namespace autil
