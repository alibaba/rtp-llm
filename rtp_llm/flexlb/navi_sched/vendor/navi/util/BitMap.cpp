#include "navi/util/BitMap.h"

#include <string.h>

using namespace autil;

namespace navi {

std::ostream &operator<<(std::ostream &os, const BitMap16 &bitMap16) { return os << bitMap16.toString(); }

BitMap::BitMap() {}

BitMap::BitMap(uint32_t size) { init(size); }

BitMap::~BitMap() {}

void BitMap::init(uint32_t size) {
    _size = size;
    _bitMaps.reserve((size + BIT_UNIT - 1) >> BIT_POWER);
    uint32_t i = 0;
    while (size > BIT_UNIT) {
        _bitMaps.emplace_back(BIT_UNIT);
        size -= BIT_UNIT;
        ++i;
    }
    if (size > 0) {
        _bitMaps.emplace_back(size);
    }
}

bool BitMap::all() const {
    for (const auto &bitMap : _bitMaps) {
        if (!bitMap.all()) {
            return false;
        }
    }
    return true;
}

bool BitMap::any() const {
    for (const auto &bitMap : _bitMaps) {
        if (bitMap.any()) {
            return true;
        }
    }
    return false;
}

bool BitMap::none() const {
    for (const auto &bitMap : _bitMaps) {
        if (!bitMap.none()) {
            return false;
        }
    }
    return true;
}

size_t BitMap::size() const { return _size; }

size_t BitMap::count() const {
    size_t count = 0;
    for (const auto &bitMap : _bitMaps) {
        count += bitMap.count();
    }
    return count;
}

void BitMap::set() {
    for (auto &bitMap : _bitMaps) {
        bitMap.set();
    }
}

void BitMap::reset() {
    for (auto &bitMap : _bitMaps) {
        bitMap.reset();
    }
}

void BitMap::set(uint32_t pos, bool value) {
    uint32_t index;
    uint8_t bitPos;
    calcIndexAndPos(pos, index, bitPos);
    _bitMaps[index].set(bitPos, value);
}

bool BitMap::operator[](std::size_t pos) const {
    uint32_t index;
    uint8_t bitPos;
    calcIndexAndPos(pos, index, bitPos);
    return _bitMaps[index][bitPos];
}

std::string BitMap::toString() const {
    std::string s;
    for (auto it = _bitMaps.rbegin(); it != _bitMaps.rend(); ++it) {
        s.append(std::move(it->toString()));
    }
    return s;
}

void BitMap::calcIndexAndPos(uint32_t bit, uint32_t &index, uint8_t &pos) const {
    index = bit >> BIT_POWER;
    pos = bit & (BIT_UNIT - 1);
}

std::ostream &operator<<(std::ostream &os, const BitMap &bitMap) {
    os << "size[" << bitMap._size << "]," << std::endl;
    for (const auto &map : bitMap._bitMaps) {
        os << map << std::endl;
    }
    return os;
}

} // namespace navi
