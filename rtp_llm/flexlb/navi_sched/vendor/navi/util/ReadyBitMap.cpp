#include "navi/util/ReadyBitMap.h"

#include <string.h>

#include "autil/StringUtil.h"

using namespace autil;

namespace navi {

std::ostream &operator<<(std::ostream &os, const ReadyBitMap &readyBitMap)
{
    os << (void *)&readyBitMap; // avoid infinite recursion with `os << const ReadyBitMap *`
    os << ", OhRF:" << (int)readyBitMap.isOk() << (int)readyBitMap.hasReady() << (int)readyBitMap.isReady()
       << (int)readyBitMap.isFinish();
    os << " bitSize:" << readyBitMap.getBitSize();
    os << std::hex;
#define CASE(name)                                      \
    os << " "#name":";                                 \
    for (size_t i = readyBitMap._size; i > 0; --i) {    \
        os << (i == readyBitMap._size ? "" : ",");      \
        os << readyBitMap._bitMaps[i-1].bitMap.name;    \
    }
    CASE(readyFinish.finish);
    CASE(readyFinish.ready);
    CASE(optional);
    CASE(mask);
#undef CASE
    return os << std::dec;
}

std::ostream &operator<<(std::ostream &os, const InlineReadyBitMap &readyBitMap)
{
    if (readyBitMap.isInline()) {
        os << std::hex;
#define CASE(name)                                                             \
    os << " " #name ":" << readyBitMap._bitMapOrPtr.bitMap.bitMap.name;
        CASE(readyFinish.finish);
        CASE(readyFinish.ready);
        CASE(optional);
        CASE(mask);
#undef CASE
        return os << std::dec;
    } else {
        return os << *readyBitMap.getPtr();
    }
}

void BitState16::setValid(uint32_t index, bool value) {
    if (value) {
        bitMap.mask |= 1u << index;
    } else {
        bitMap.mask &= ~(1u << index);
    }
}

bool BitState16::isFinish(uint32_t index) const {
    assert(bitMap.mask & (1u << index));
    return bitMap.readyFinish.finish & (1u << index);
}

bool BitState16::isOptional(uint32_t index) const {
    assert(bitMap.mask & (1u << index));
    return bitMap.optional & (1u << index);
}

bool BitState16::isValid(uint32_t index) const {
    return bitMap.mask & (1u << index);
}

bool BitState16::hasReady() const {
    return (bitMap.readyFinish.ready & bitMap.mask) != 0;
}

bool BitState16::isOk() const {
    auto mask = (BitType)((~bitMap.readyFinish.finish) & (~bitMap.optional) & bitMap.mask);
    return (mask & bitMap.readyFinish.ready) == mask;
}

bool BitState16::isReady(uint32_t index) const {
    assert(bitMap.mask & (1u << index));
    return bitMap.readyFinish.ready & (1u << index);
}

bool BitState16::isReady() const {
    return (bitMap.readyFinish.ready & bitMap.mask) == bitMap.mask;
}

bool BitState16::isFinish() const {
    return (bitMap.readyFinish.finish & bitMap.mask) == (BIT_TYPE_MASK & bitMap.mask);
}

bool BitState16::hasFinishBit() const {
    return (bitMap.readyFinish.finish & bitMap.mask) !=
           (BitType(~BIT_TYPE_MASK) & bitMap.mask);
}

ReadyBitMap::ReadyBitMap(uint32_t bitSize)
    : _bitSize(bitSize | (1u << 31))
    , _size(ReadyBitMap::getBitMapSize(bitSize))
{
    initBitMaps();
}

ReadyBitMap::~ReadyBitMap() {
}

void ReadyBitMap::initBitMaps() {
    uint32_t i = 0;
    auto bitSize = getBitSize();
    while (bitSize > UNIT) {
        new (&_bitMaps[i]) BitState16(UNIT);
        bitSize -= UNIT;
        i++;
    }
    if (bitSize > 0) {
        new (&_bitMaps[i]) BitState16(bitSize);
    }
}

ReadyBitMap *ReadyBitMap::createReadyBitMap(uint32_t bitSize) {
    auto size = ReadyBitMap::getBitMapSize(bitSize);
    size_t allocSize = sizeof(ReadyBitMap) + sizeof(BitState16) * size;
    void  *mem = malloc(allocSize);
    new (mem) ReadyBitMap(bitSize);
    return (ReadyBitMap *)mem;
}

void ReadyBitMap::setReady(uint32_t bit, bool value) {
    assert(bit < getBitSize());
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    _bitMaps[index].setReady(offset, value);
}

void ReadyBitMap::setReady(bool value) {
    for (size_t i = 0; i < _size; i++) {
        auto &bitMap = _bitMaps[i];
        bitMap.setReady(value);
    }
}

void ReadyBitMap::setFinish(uint32_t bit, bool value) {
    assert(bit < getBitSize());
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    _bitMaps[index].setFinish(offset, value);
}

void ReadyBitMap::setFinish(bool value) {
    for (size_t i = 0; i < _size; i++) {
        auto &bitMap = _bitMaps[i];
        bitMap.setFinish(value);
    }
}

void ReadyBitMap::setReadyFinish(uint32_t bit, bool ready, bool finish) {
    assert(bit < getBitSize());
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    _bitMaps[index].setReadyFinish(offset, ready, finish);
}

void ReadyBitMap::setOptional(uint32_t bit, bool value) {
    assert(bit < getBitSize());
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    _bitMaps[index].setOptional(offset, value);
}

void ReadyBitMap::setOptional(bool value) {
    for (size_t i = 0; i < _size; i++) {
        auto &bitMap = _bitMaps[i];
        bitMap.setOptional(value);
    }
}

void ReadyBitMap::setValid(uint32_t bit, bool value) {
    // ATTENTION: no thread-safety provided, only used while creating
    assert(bit < getBitSize());
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    _bitMaps[index].setValid(offset, value);
}

void ReadyBitMap::setValid(bool value)
{
    // ATTENTION: no thread-safety provided, only used while creating
    auto bitSize = getBitSize();
    for (size_t i = 0; i < bitSize; i++) {
        setValid(i, value);
    }
}

bool ReadyBitMap::isReady(uint32_t bit) const {
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    return _bitMaps[index].isReady(offset);
}

bool ReadyBitMap::isFinish(uint32_t bit) const {
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    return _bitMaps[index].isFinish(offset);
}

bool ReadyBitMap::isOptional(uint32_t bit) const {
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    return _bitMaps[index].isOptional(offset);
}

bool ReadyBitMap::isValid(uint32_t bit) const {
    uint32_t index;
    uint8_t offset;
    getIndexAndBitMask(bit, index, offset);
    return _bitMaps[index].isValid(offset);
}

bool ReadyBitMap::isOk() const {
    for (size_t i = 0; i < _size; i++) {
        const auto &bitMap = _bitMaps[i];
        if (!bitMap.isOk()) {
            return false;
        }
    }
    return true;
}

bool ReadyBitMap::hasReady() const {
    for (size_t i = 0; i < _size; i++) {
        const auto &bitMap = _bitMaps[i];
        if (bitMap.hasReady()) {
            return true;
        }
    }
    return false;
}

bool ReadyBitMap::isReady() const {
    for (size_t i = 0; i < _size; i++) {
        const auto &bitMap = _bitMaps[i];
        if (!bitMap.isReady()) {
            return false;
        }
    }
    return true;
}

bool ReadyBitMap::isFinish() const {
    for (size_t i = 0; i < _size; i++) {
        const auto &bitMap = _bitMaps[i];
        if (!bitMap.isFinish()) {
            return false;
        }
    }
    return true;
}

bool ReadyBitMap::hasFinishBit() const {
    for (size_t i = 0; i < _size; i++) {
        const auto &bitMap = _bitMaps[i];
        if (bitMap.hasFinishBit()) {
            return true;
        }
    }
    return false;
}

size_t ReadyBitMap::getBitMapSize(size_t bitSize) {
    return (bitSize + UNIT - 1) >> POWER;
}

void ReadyBitMap::getIndexAndBitMask(uint32_t bit, uint32_t &index, uint8_t &offset) {
    index = bit >> POWER;
    offset = bit & MASK;
}

}
