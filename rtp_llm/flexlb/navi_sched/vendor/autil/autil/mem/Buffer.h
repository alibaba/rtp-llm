#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>

#include "autil/NoCopyable.h"
#include "autil/mem/QuotaController.h"
#include "autil/result/Errors.h"
#include "autil/result/Result.h"

namespace autil {
namespace mem {

class Buffer;
typedef std::unique_ptr<Buffer> BufferPtr;

class Buffer : public autil::NoMoveable {
public:
    static constexpr size_t DEFAULT_BUFFER_CHUNK_SIZE = 4 * 1024;
    static constexpr size_t DEFAULT_ALIGN_SIZE = sizeof(char *);
    static constexpr size_t DEFAULT_ALIGN_SIZE_BYTES = sizeof(char *);

public:
    ~Buffer();

    void *allocate(size_t numBytes, size_t alignment);

    inline size_t getTotalBytes() const { return _size; }

    inline size_t getUsedSize() const { return _pos; }

    static inline autil::Result<BufferPtr> createBuffer(QuotaController *qc,
                                                        size_t numBytes);

    static inline size_t alignBytes(size_t numBytes, size_t alignSize);

private:
    Buffer(QuotaController *qc, size_t allocSize);

    inline void *doAllocate(size_t nBytes)
    {
        if (_size - _pos < nBytes) {
            return nullptr;
        }
        void *ptr = (void *)(_data + _pos);
        _pos += nBytes;
        return ptr;
    }

    inline void *doAllocate(size_t nBytes, size_t align)
    {
        auto ptr = (void *)((_data) + _pos);
        auto space = _size - _pos;
        auto alignedPtr = std::align(align, nBytes, ptr, space);
        if (alignedPtr) {
            _pos = (char *)alignedPtr - _data + nBytes;
        }
        return alignedPtr;
    }

private:
    char *_data = nullptr;
    size_t _pos = 0;
    size_t _size = 0;
    QuotaController *_qc;
};

inline autil::Result<BufferPtr> Buffer::createBuffer(QuotaController *qc, size_t numBytes)
{
    size_t allocSize =
        alignBytes(std::max(DEFAULT_BUFFER_CHUNK_SIZE, numBytes), DEFAULT_ALIGN_SIZE);
    if (!qc->allocate(allocSize)) {
        return autil::result::RuntimeError::make(
            "Allocate memory chunk of size %ld failed", (int64_t)allocSize);
    }
    return BufferPtr(new Buffer(qc, allocSize));
}

inline size_t Buffer::alignBytes(size_t numBytes, size_t alignSize)
{
    return ((numBytes + alignSize - 1) & ~(alignSize - 1));
}

} // namespace mem
} // namespace autil
