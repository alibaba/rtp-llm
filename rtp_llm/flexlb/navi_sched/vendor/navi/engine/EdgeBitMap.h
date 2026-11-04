#include "navi/common.h"

namespace navi {

class EdgeBitInfo {
private:
    static constexpr uint8_t ALL_EOF = ((1u << 8) - 1u);
public:
    EdgeBitInfo() : _bitMap(0) {}
    EdgeBitInfo(size_t value)
        : _bitMap(value)
    {
    }
public:
    void init(IndexType size) {
        assert(size <= 8);
        auto mask = ~((1ul << size) - 1ul);
        _hasData = mask;
        _eof = mask;
        _slotFinish = mask;
        _optional = mask;
        _mask = mask;
    }
    bool isInputOk() const {
        auto mask = (_eof | _slotFinish);
        return ((_hasData | mask) == mask);
    }
    bool isInputOk(IndexType index) const {
        auto mask = ((_eof | _slotFinish) & (1u << index));
        return (((_hasData & (1u << index)) | mask) == mask);
    }
    bool isOutputOk(IndexType index) const {
        return (_eof | _slotFinish | _hasData | _optional) & (1u << index);
    }
    bool isEofOrFinish() const {
        return ALL_EOF == (_eof | _slotFinish);
    }
    bool isEofOrFinish(IndexType index) const {
        return (_eof | _slotFinish) & (1u << index);
    }
    void setOptional(IndexType index, bool value) {
        if (value) {
            _optional |= (1u << index);
        } else {
            _optional &= (~(1u << index));
        }
    }
    bool isOptional(IndexType index) const {
        return _optional & (1u << index);
    }
    bool isEof(IndexType index) const {
        return _eof & (1u << index);
    }
    bool isEof() const {
        return ALL_EOF == _eof;
    }
    bool isSlotFinished(IndexType index) const {
        return _slotFinish & (1u << index);
    }
    bool slotHasData(IndexType index) const {
        return (_hasData & (1u << index));
    }
private:
    friend class EdgeBitMap;
private:
    void setSlotHasData(bool eof) {
        _hasData = ALL_EOF;
        if (eof) {
            _eof = ALL_EOF;
        }
    }
    bool clearSlotHasData(IndexType index) {
        _hasData &= (~(1u << index));
        return isAllClear(_hasData);
    }
    void setSlotFinish(IndexType index) {
        _slotFinish |= (1u << index);
    }
    bool isAllClear(uint8_t value) const {
        return (((value & (~_slotFinish)) | _mask) == _mask);
    }
    bool isFinish() const {
        return _slotFinish == ALL_EOF;
    }
    bool isOk() const {
        auto notFinish = (~_slotFinish);
        return ((notFinish & (~_hasData)) == notFinish);
    }
    bool cmpxchgUpdateHasData(uint8_t currentValue, uint8_t newValue) {
        return cmpxchg(&_hasData, currentValue, newValue);
    }
    bool cmpxchgUpdateFinish(uint8_t currentValue, uint8_t newValue) {
        return cmpxchg(&_slotFinish, currentValue, newValue);
    }
private:
    union {
        struct {
            uint8_t _hasData;
            uint8_t _eof;
            uint8_t _slotFinish;
            uint8_t _optional;
            uint8_t _mask;
        };
        uint64_t _bitMap;
    };
};

class EdgeBitMap {
public:
    EdgeBitMap() {
    }
public:
    void init(size_t size) {
        _info.init(size);
    }
    void setOptional(IndexType index, bool value) {
        _info.setOptional(index, value);
    }
    void setSlotHasData(bool eof) {
        _info.setSlotHasData(eof);
    }
    bool clearSlotHasData(IndexType index, bool atomicClear) {
        if (!atomicClear) {
            return _info.clearSlotHasData(index);
        } else {
            uint8_t currentValue;
            uint8_t newValue;
            do {
                auto currentBitmap = _info;
                currentValue = currentBitmap._hasData;
                currentBitmap.clearSlotHasData(index);
                newValue = currentBitmap._hasData;
            } while (!_info.cmpxchgUpdateHasData(currentValue, newValue));
            return !_info.isAllClear(currentValue) && _info.isAllClear(newValue);
        }
    }
    bool setSlotFinish(IndexType index) {
        uint64_t currentValue;
        uint64_t newValue;
        do {
            auto currentBitmap = _info;
            currentValue = currentBitmap._slotFinish;
            currentBitmap.setSlotFinish(index);
            newValue = currentBitmap._slotFinish;
        } while (!_info.cmpxchgUpdateFinish(currentValue, newValue));
        return (EdgeBitInfo::ALL_EOF != currentValue) &&
               (EdgeBitInfo::ALL_EOF == newValue);
    }
    bool isInputOk() const {
        return _info.isInputOk();
    }
    bool isInputOk(IndexType index) const {
        return _info.isInputOk(index);
    }
    bool isOutputOk(IndexType index) const {
        return _info.isOutputOk(index);
    }
    bool isEofOrFinish() const {
        return _info.isEofOrFinish();
    }
    bool isEofOrFinish(IndexType index) const {
        return _info.isEofOrFinish(index);
    }
    bool isOptional(IndexType index) const {
        return _info.isOptional(index);
    }
    bool isEof(IndexType index) const {
        return _info.isEof(index);
    }
    bool isEof() const {
        return _info.isEof();
    }
    bool isSlotFinished(IndexType index) const {
        return _info.isSlotFinished(index);
    }
    bool slotHasData(IndexType index) const {
        return _info.slotHasData(index);
    }
    bool allSlotEmpty() const {
        return _info.isOk();
    }
    bool isAllSlotFinished() const {
        return _info.isFinish();
    }
private:
    EdgeBitInfo _info;
};

class EdgeBitMapTree {
public:
    static EdgeBitMapTree *createTree(size_t size) {
        auto leafCount = ((size + 7) >> 3);
        auto count = leafCount - 1;
        auto layer = 0;
        auto layerCount = 1;
        auto s = 0;
        while (true) {
            count = (count >> 3);
            if (count > 0) {
                layer++;
                s += layerCount;
                layerCount = (layerCount << 3);
            } else {
                break;
            }
        }
        auto layerBitCount = (layerCount << 3);
        size_t remain = 0;
        if (size > layerBitCount) {
            remain = size - layerBitCount;
        }
        auto w = (remain + 6) / 7;
        auto mallocCount = s + layerCount + w;
        auto start = s + (w >> 3);
        auto offset = (w & 7);
        auto ret =
            malloc(sizeof(EdgeBitMapTree) + sizeof(EdgeBitMap) * mallocCount);
        new (ret) EdgeBitMapTree(start, offset, size, mallocCount);
        return (EdgeBitMapTree *)ret;
    }
    static void freeTree(EdgeBitMapTree *tree) {
        if (tree) {
            free(tree);
        }
    }
private:
    EdgeBitMapTree(size_t start, size_t offset, size_t size, size_t count) {
        _start = start;
        _offset = offset;
        _count = count;
        size_t retOffset;
        auto index = getIndex(size - 1, retOffset);
        for (size_t i = 0; i < index; i++) {
            _maps[i].init(8);
        }
        _maps[index].init(retOffset + 1);
    }
public:
    void setOptional(IndexType index, bool value) {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        return _maps[arrayIndex].setOptional(retOffset, value);
    }
    void setSlotHasData(bool eof) {
        for (size_t i = 0; i < _count; i++) {
            _maps[i].setSlotHasData(eof);
        }
    }
    bool clearSlotHasData(IndexType index, bool atomicClear) {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        bool currentClear;
        do {
            currentClear =
                _maps[arrayIndex].clearSlotHasData(retOffset, atomicClear);
            if (0 == arrayIndex) {
                break;
            }
            retOffset = (arrayIndex - 1) & 7;
            arrayIndex = (arrayIndex - 1) >> 3;
        } while (currentClear);
        return currentClear;
    }
    bool setSlotFinish(IndexType index) {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        bool currentFinish;
        do {
            currentFinish = _maps[arrayIndex].setSlotFinish(retOffset);
            if (0 == arrayIndex) {
                break;
            }
            retOffset = (arrayIndex - 1) & 7;
            arrayIndex = (arrayIndex - 1) >> 3;
        } while (currentFinish);
        return currentFinish;
    }
    bool isInputOk() const {
        return _maps[0].isInputOk();
    }
    bool isInputOk(IndexType index) const {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        return _maps[arrayIndex].isInputOk(retOffset);
    }
    bool isEof() const {
        return _maps[0].isEof();
    }
    bool isEof(IndexType index) const {
        return _maps[0].isEof(index);
    }
    bool isEofOrFinish() const {
        return _maps[0].isEofOrFinish();
    }
    bool isOptional(IndexType index) const {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        return _maps[arrayIndex].isOptional(retOffset);
    }
    bool isEofOrFinish(IndexType index) const {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        return _maps[arrayIndex].isEofOrFinish(retOffset);
    }
    bool isOutputOk(IndexType index) const {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        return _maps[arrayIndex].isOutputOk(retOffset);
    }
    bool isSlotFinished(IndexType index) const {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        return _maps[arrayIndex].isSlotFinished(retOffset);
    }
    bool slotHasData(IndexType index) const {
        size_t retOffset;
        auto arrayIndex = getIndex(index, retOffset);
        return _maps[arrayIndex].slotHasData(retOffset);
    }
    bool allSlotEmpty() const {
        return _maps[0].allSlotEmpty();
    }
    bool isAllSlotFinished() const {
        return _maps[0].isAllSlotFinished();
    }
private:
    size_t getIndex(size_t bit, size_t &retOffset) const {
        auto bitOffset = bit + _offset;
        retOffset = bitOffset & 7;
        return _start + (bitOffset >> 3);
    }
private:
    uint64_t _count;
    uint64_t _start;
    uint64_t _offset;
    EdgeBitMap _maps[0];
};

}
