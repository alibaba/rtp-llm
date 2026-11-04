#include "autil/MultiValueCreator.h"

#include <iosfwd>

using namespace std;
using namespace autil::mem_pool;

namespace autil {

template <typename StringType>
char *createMultiStringBufferT(const StringType *data, size_t tokenNum, PoolBase *pool) {
    size_t offsetLen = 0;
    size_t bufLen = MultiValueFormatter::calculateMultiStringBufferLen(data, tokenNum, offsetLen);
    char *buffer = POOL_COMPATIBLE_NEW_VECTOR(pool, char, bufLen);
    MultiValueFormatter::formatMultiStringToBuffer(buffer, bufLen, offsetLen, data, tokenNum);
    return buffer;
}

char *MultiValueCreator::createMultiStringBuffer(const std::vector<std::string> &strVec, mem_pool::PoolBase *pool) {
    return createMultiStringBufferT(strVec.data(), strVec.size(), pool);
}

char *MultiValueCreator::createMultiStringBuffer(const std::vector<StringView> &strVec, mem_pool::PoolBase *pool) {
    return createMultiStringBufferT(strVec.data(), strVec.size(), pool);
}

char *MultiValueCreator::createMultiValueBuffer(const std::string *data, size_t size, mem_pool::PoolBase *pool) {
    return createMultiStringBufferT(data, size, pool);
}

char *MultiValueCreator::createMultiValueBuffer(const MultiChar *data, size_t size, mem_pool::PoolBase *pool) {
    return createMultiStringBufferT(data, size, pool);
}

char *MultiValueCreator::createMultiValueBuffer(const StringView *data, size_t size, mem_pool::PoolBase *pool) {
    return createMultiStringBufferT(data, size, pool);
}

char *MultiValueCreator::createNullMultiValueBuffer(mem_pool::PoolBase *pool) {
    size_t bufLen = MultiValueFormatter::getEncodedCountLength(MultiValueFormatter::VAR_NUM_NULL_FIELD_VALUE_COUNT);
    char *buffer = POOL_COMPATIBLE_NEW_VECTOR(pool, char, bufLen);
    char *writeBuf = buffer;
    assert(buffer);
    MultiValueFormatter::encodeCount(MultiValueFormatter::VAR_NUM_NULL_FIELD_VALUE_COUNT, writeBuf, bufLen);
    return buffer;
}

} // namespace autil
