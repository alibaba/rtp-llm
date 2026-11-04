#pragma once

#include <cstdint>

namespace alog {

class LogBuf {
public:
    static constexpr size_t STACK_BUF_SIZE = 4096 - sizeof(uint32_t);

public:
    ~LogBuf() {
        if (_size > STACK_BUF_SIZE) {
            delete[] _repr.heapBuf;
        }
    }

public:
    char *getBuf() {
        if (_size <= STACK_BUF_SIZE) {
            return _repr.stackBuf;
        } else {
            return _repr.heapBuf;
        }
    }

    void resize(size_t size) {
        if (size > STACK_BUF_SIZE) {
            if (_size > STACK_BUF_SIZE) {
                delete[] _repr.heapBuf;
            }
            _repr.heapBuf = new char[size];
            _size = size;
        }
    }

    size_t size() const { return _size; }

private:
    uint32_t _size = STACK_BUF_SIZE; // 4 bytes

    union {
        char stackBuf[STACK_BUF_SIZE];
        char *heapBuf;
    } _repr;
};

} // namespace alog
