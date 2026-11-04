#include "autil/ShortString.h"

namespace autil {

ShortString &ShortString::operator+=(const ShortString &addStr) {
    size_t addSize = addStr.size();
    if (_size + addSize >= BUF_SIZE) {
        char *newBuf = new char[_size + addSize + 1];
        memcpy(newBuf, _data, _size);
        memcpy(newBuf + _size, addStr.c_str(), addSize);
        if (_size >= BUF_SIZE) {
            delete[] _data;
        }
        _size += addSize;
        newBuf[_size] = '\0';
        _data = newBuf;
    } else {
        memcpy(_buf + _size, addStr.c_str(), addSize);
        _size += addSize;
        _buf[_size] = '\0';
    }
    return *this;
}

} // namespace autil
