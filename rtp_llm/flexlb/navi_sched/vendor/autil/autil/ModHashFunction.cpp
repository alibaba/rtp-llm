#include "autil/ModHashFunction.h"

#include <iosfwd>

#include "autil/HashAlgorithm.h"

using namespace std;
namespace autil {

uint32_t ModHashFunction::getHashId(const std::string &str) const {
    int64_t h = HashAlgorithm::hashString64(str.c_str(), 1) & INT64_MAX;
    return h % _hashSize;
}

} // namespace autil
