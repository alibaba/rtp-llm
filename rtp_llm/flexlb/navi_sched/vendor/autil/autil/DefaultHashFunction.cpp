#include "autil/DefaultHashFunction.h"

#include <cstddef>

#include "autil/HashAlgorithm.h"

using namespace std;
namespace autil {

int DefaultHashFunction::hashString(const string &str) const {
    int h = (int)HashAlgorithm::hashString(str.c_str(), str.length(), 0);
    return h;
}

uint32_t DefaultHashFunction::getHashId(const std::string &str) const {
    uint32_t h = (uint32_t)HashAlgorithm::hashString(str.c_str(), str.length(), 0);
    return h % _hashSize;
}

uint32_t DefaultHashFunction::getHashId(const char *buf, size_t len) const {
    uint32_t h = (uint32_t)HashAlgorithm::hashString(buf, len, 0);
    return h % _hashSize;
}

} // namespace autil
