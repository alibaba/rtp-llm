#include "autil/LongHashValue.h"

#include <string.h>

#include "autil/StringUtil.h"

namespace autil {

template <>
std::string LongHashValue<>::toString() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

template <>
bool uint128_t::hexStrToUint128(const char *str) {
    char buf[17];
    buf[16] = 0;
    for (int i = 0; i < 2; i++) {
        memcpy(buf, str + i * 16, 16);
        StringUtil::hexStrToUint64(buf, this->value[i]);
    }
    return true;
}

template <>
void uint128_t::uint128ToHexStr(char *hexStr, int len) {
    assert(len > 32);
    for (int i = 0; i < 2; i++) {
        StringUtil::uint64ToHexStr(this->value[i], hexStr, 17);
        hexStr += 16;
    }
}

std::ostream &operator<<(std::ostream &stream, uint128_t v) {
    char tmp[17] = {0};
    for (int i = 0; i < v.count(); i++) {
        StringUtil::uint64ToHexStr(v.value[i], tmp, 17);
        stream << tmp;
    }
    return stream;
}

std::ostream &operator<<(std::ostream &stream, uint256_t v) {
    char tmp[17] = {0};
    for (int i = 0; i < v.count(); i++) {
        StringUtil::uint64ToHexStr(v.value[i], tmp, 17);
        stream << tmp;
    }
    return stream;
}

std::istream &operator>>(std::istream &stream, uint128_t &v) {
    char tmp[17] = {0};
    for (int i = 0; i < v.count(); i++) {
        stream.read(tmp, 16);
        StringUtil::hexStrToUint64(tmp, v.value[i]);
    }
    return stream;
}

std::istream &operator>>(std::istream &stream, uint256_t &v) {
    char tmp[17] = {0};
    for (int i = 0; i < v.count(); i++) {
        stream.read(tmp, 16);
        StringUtil::hexStrToUint64(tmp, v.value[i]);
    }
    return stream;
}

} // namespace autil
