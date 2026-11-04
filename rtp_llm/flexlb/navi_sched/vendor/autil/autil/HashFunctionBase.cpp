#include "autil/HashFunctionBase.h"

#include <cstdint>
#include <iosfwd>

using namespace std;
namespace autil {

string HashFunctionBase::getFunctionName() const { return _hashFunction; }

uint32_t HashFunctionBase::getHashSize() const { return _hashSize; }

uint32_t HashFunctionBase::getHashId(const vector<string> &strVec) const {
    if (strVec.size() == 0) {
        return 0;
    } else {
        return getHashId(strVec[0]);
    }
}

vector<pair<uint32_t, uint32_t>> HashFunctionBase::getHashRange(uint32_t partId) const {
    if (partId >= _hashSize) {
        return {};
    }
    vector<pair<uint32_t, uint32_t>> range = {{partId, partId}};
    return range;
}

vector<pair<uint32_t, uint32_t>> HashFunctionBase::getHashRange(uint32_t partId, float ratio) const {
    return getHashRange(partId);
}

vector<pair<uint32_t, uint32_t>> HashFunctionBase::getHashRange(const vector<string> &strVec) const {
    uint32_t hashId = getHashId(strVec);
    vector<pair<uint32_t, uint32_t>> rangeVec;
    rangeVec.emplace_back(hashId, hashId);
    return rangeVec;
}

} // namespace autil
