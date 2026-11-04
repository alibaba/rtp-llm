#pragma once

#include "navi/common.h"

namespace navi {

class ReadyBitMap;
class InlineReadyBitMap;

constexpr uint32_t POWER = 4u;
constexpr uint32_t UNIT = 1u << POWER;
constexpr uint32_t MASK = UNIT - 1;

typedef uint16_t BitType;
constexpr BitType BIT_TYPE_MASK = ~((BitType)0);

class BitState16 {
public:
    BitState16(uint32_t size)
    {
        init(size);
    }
public:
    friend std::ostream &operator<<(std::ostream &os, const ReadyBitMap &readyBitMap);
    friend std::ostream &operator<<(std::ostream &os,
                                    const InlineReadyBitMap &readyBitMap);
private:
    BitState16() {}
private:
    union ReadyFinish {
        struct {
            BitType finish;
            BitType ready;
        };
        uint32_t readyFinish;
    };
    struct BitState {
        ReadyFinish readyFinish;
        BitType optional;
        BitType mask;
    };
private:
    friend class InlineReadyBitMap;
private:
    BitState bitMap;
public:
    inline void init(uint32_t size) {
        bitMap.readyFinish.finish = BIT_TYPE_MASK;
        bitMap.readyFinish.ready = 0;
        bitMap.optional = ~((1u << size) - 1u);
        bitMap.mask = (1u << size) - 1u;
        assert(size <= 16);
    }
    inline void setReady(bool value) {
        if (value) {
            setBitByMask(&bitMap.readyFinish.ready, BIT_TYPE_MASK);
        } else {
            clearBitByMask(&bitMap.readyFinish.ready, (BitType)~BIT_TYPE_MASK);
        }
    }
    inline void setReady(uint32_t index, bool value) {
        if (value) {
            setBit(&bitMap.readyFinish.ready, index);
        } else {
            clearBit(&bitMap.readyFinish.ready, index);
        }
    }
    inline void setFinish(bool value) {
        if (value) {
            setBitByMask(&bitMap.readyFinish.finish, BIT_TYPE_MASK);
        } else {
            clearBitByMask(&bitMap.readyFinish.finish, (BitType)~BIT_TYPE_MASK);
        }
    }
    inline void setFinish(uint32_t index, bool value) {
        if (value) {
            setBit(&bitMap.readyFinish.finish, index);
        } else {
            clearBit(&bitMap.readyFinish.finish, index);
        }
    }
    // do not use setReadyFinish when copy finish bit
    inline void setReadyFinish(uint32_t index, bool ready, bool finish) {
        auto bit = (1u << index);
        ReadyFinish currentValue;
        ReadyFinish newValue;
        do {
            currentValue = bitMap.readyFinish;
            if (finish) {
                newValue.finish = (currentValue.finish | bit);
            } else {
                newValue.finish = currentValue.finish;
            }
            if (ready) {
                newValue.ready = (currentValue.ready | bit);
            } else {
                newValue.ready = (currentValue.ready & (~bit));
            }
        } while (!cmpxchg(&bitMap.readyFinish.readyFinish,
                          currentValue.readyFinish,
                          newValue.readyFinish));
    }
    inline void setOptional(bool value) {
        if (value) {
            setBitByMask(&bitMap.optional, BIT_TYPE_MASK);
        } else {
            clearBitByMask(&bitMap.optional, (BitType)~BIT_TYPE_MASK);
        }
    }

    inline void setOptional(uint32_t index, bool value) {
        if (value) {
            setBit(&bitMap.optional, index);
        } else {
            clearBit(&bitMap.optional, index);
        }
    }

    void setValid(uint32_t index, bool value);

    // input: init(false), graph-input(true)
    // output: init(false), linked(true), graph-output(true)
    bool isReady(uint32_t index) const;
    bool isReady() const;
    bool hasReady() const;

    // input: init(true), linked(false)
    // output : init(true),linked(false)
    bool isFinish(uint32_t index) const;
    bool isOptional(uint32_t index) const;
    bool isValid(uint32_t index) const;

    bool isOk() const;
    bool isFinish() const;
    bool hasFinishBit() const;
private:
    inline void setBit(BitType *bits, uint32_t index) {
        BitType mask = 1u << index;
        setBitByMask(bits, mask);
    }
    inline void setBitByMask(BitType *bits, BitType mask) {
        mask &= bitMap.mask; // exclude invalid bit
        BitType currentValue;
        BitType newValue;
        do {
            currentValue = *bits;
            newValue = currentValue | mask;
        } while (!cmpxchg(bits, currentValue, newValue));
    }
    inline void clearBit(BitType *bits, uint32_t index) {
        BitType mask = ~(1u << index);
        clearBitByMask(bits, mask);
    }
    inline void clearBitByMask(BitType *bits, BitType mask) {
        mask |= ~bitMap.mask; // exclude invaild bit
        BitType currentValue;
        BitType newValue;
        do {
            currentValue = *bits;
            newValue = currentValue & mask;
        } while (!cmpxchg(bits, currentValue, newValue));
    }
};

#define ALIGN_16(size) ((size + 15) & (~15))

class ReadyBitMap
{
public:
    ReadyBitMap(uint32_t bitSize);
    ~ReadyBitMap();
private:
    ReadyBitMap(const ReadyBitMap &);
    ReadyBitMap& operator=(const ReadyBitMap &);
public:
    friend std::ostream& operator<<(std::ostream& os,
                                    const ReadyBitMap &readyBitMap);
public:
    static ReadyBitMap *createReadyBitMap(uint32_t size);
    static inline void freeReadyBitMap(ReadyBitMap *readyBitMap) {
        if (readyBitMap) {
            free(readyBitMap);
        }
    }
    static size_t getBitMapSize(size_t bitSize);
public:
    void setReady(uint32_t bit, bool value);
    void setReady(bool value);
    void setFinish(uint32_t bit, bool value);
    void setFinish(bool value);
    void setReadyFinish(uint32_t bit, bool ready, bool finish);
    void setOptional(uint32_t bit, bool value);
    void setOptional(bool value);
    void setValid(uint32_t bit, bool value);
    void setValid(bool value);

    bool isReady(uint32_t bit) const;
    bool hasReady() const;
    bool isReady() const;
    bool isFinish(uint32_t bit) const;
    bool isOptional(uint32_t bit) const;
    bool isValid(uint32_t bit) const;
    bool isOk() const;
    bool isFinish() const;
    bool hasFinishBit() const;
    inline uint32_t getBitSize() const {
        return _bitSize & (0x7fffffff);
    }
    inline void setBitSize(size_t bitSize) {
        _bitSize = bitSize;
    }
private:
    void initBitMaps();
private:
    inline static void getIndexAndBitMask(uint32_t bit, uint32_t &index, uint8_t &offset);
private:
    uint32_t _bitSize;
    uint32_t _size;
    BitState16 _bitMaps[0]; // finish map first, then ready
};

NAVI_TYPEDEF_PTR(ReadyBitMap);

class InlineReadyBitMap {
public:
    InlineReadyBitMap()
        : _value(0)
    {
    }
public:
    static InlineReadyBitMap createReadyBitMap(uint32_t size) {
        InlineReadyBitMap bitMap;
        if (likely(size <= 15)) {
            bitMap._bitMapOrPtr.bitMap.init(size);
        } else {
            auto ptr = ReadyBitMap::createReadyBitMap(size);
            bitMap.setPtr(ptr);
        }
        return bitMap;
    }
    static void freeReadyBitMap(const InlineReadyBitMap &bitMap) {
        auto ptr = bitMap.getPtr();
        if (ptr) {
            ReadyBitMap::freeReadyBitMap(ptr);
        }
    }
public:
    bool isEmpty() const {
        auto mask = _bitMapOrPtr.bitMap.bitMap.mask;
        return (0 == mask);
    }
    inline operator bool() const { return (_value != 0); }
    inline bool isInline() const { return !(_bitMapOrPtr.bitMap.bitMap.mask & (1u << 15)); }
    inline void setPtr(ReadyBitMap *ptr) { _bitMapOrPtr.readyBitMap = (ReadyBitMap *)(((size_t)ptr) | (1lu << 63)); }
    inline void setReadyFinish(uint32_t bit, bool ready, bool finish) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setReadyFinish(bit, ready, finish);
        } else {
            getPtr()->setReadyFinish(bit, ready, finish);
        }
    }
    inline void setOptional(bool value) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setOptional(value);
        } else {
            getPtr()->setOptional(value);
        }
    }
    inline void setOptional(uint32_t bit, bool value) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setOptional(bit, value);
        } else {
            getPtr()->setOptional(bit, value);
        }
    }
    inline void setReady(uint32_t bit, bool value) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setReady(bit, value);
        } else {
            getPtr()->setReady(bit, value);
        }
    }
    inline void setReady(bool value) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setReady(value);
        } else {
            getPtr()->setReady(value);
        }
    }
    inline void setValid(uint32_t bit, bool value) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setValid(bit, value);
        } else {
            getPtr()->setValid(bit, value);
        }
    }
    inline void setInValid() {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.bitMap.mask = 0;
        } else {
            getPtr()->setValid(false);
        }
    }
    inline void setFinish(uint32_t bit, bool value) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setFinish(bit, value);
        } else {
            getPtr()->setFinish(bit, value);
        }
    }
    inline void setFinish(bool value) {
        if (likely(isInline())) {
            _bitMapOrPtr.bitMap.setFinish(value);
        } else {
            getPtr()->setFinish(value);
        }
    }
    inline bool isOk() const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.isOk();
        } else {
            return getPtr()->isOk();
        }
    }
    inline bool isFinish() const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.isFinish();
        } else {
            return getPtr()->isFinish();
        }
    }
    inline bool isFinish(uint32_t bit) const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.isFinish(bit);
        } else {
            return getPtr()->isFinish(bit);
        }
    }
    inline bool hasReady() const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.hasReady();
        } else {
            return getPtr()->hasReady();
        }
    }
    inline bool isReady() const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.isReady();
        } else {
            return getPtr()->isReady();
        }
    }
    inline bool isReady(uint32_t bit) const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.isReady(bit);
        } else {
            return getPtr()->isReady(bit);
        }
    }
    inline bool isOptional(uint32_t bit) const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.isOptional(bit);
        } else {
            return getPtr()->isOptional(bit);
        }
    }
    inline bool hasFinishBit() const {
        if (likely(isInline())) {
            return _bitMapOrPtr.bitMap.hasFinishBit();
        } else {
            return getPtr()->hasFinishBit();
        }
    }
public:
    friend std::ostream &operator<<(std::ostream &os,
                                    const InlineReadyBitMap &readyBitMap);
private:
    inline ReadyBitMap *getPtr() const {
        if (likely(isInline())) {
            return nullptr;
        } else {
            return (ReadyBitMap *)(((size_t)_bitMapOrPtr.readyBitMap) & (~(1lu << 63)));
        }
    }
private:
    union BitMapOrPtr {
        BitState16 bitMap;
        ReadyBitMap *readyBitMap;
    };
    union {
        BitMapOrPtr _bitMapOrPtr;
        uint64_t _value;
    };
};

extern std::ostream &operator<<(std::ostream &os, const ReadyBitMap &readyBitMap);
extern std::ostream &operator<<(std::ostream &os, const InlineReadyBitMap &readyBitMap);

}
