#pragma once

#include <cstdint>

namespace anet {

class DirectPlaceholder {
public:
    DirectPlaceholder() : _addr(nullptr), _len(0U) {}
    DirectPlaceholder(char *addr, uint32_t len) : _addr(addr), _len(len) {}
    char *getAddr() const { return _addr; }
    uint32_t getLen() const { return _len; }

private:
    char *_addr;
    uint32_t _len;
};

} // namespace anet

