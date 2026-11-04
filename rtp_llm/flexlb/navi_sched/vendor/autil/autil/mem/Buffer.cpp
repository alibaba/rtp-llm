#include "autil/mem/Buffer.h"

namespace autil {
namespace mem {
Buffer::Buffer(QuotaController *qc, size_t allocSize)
    : _data(
          reinterpret_cast<char *>(new uint64_t[(allocSize + DEFAULT_ALIGN_SIZE_BYTES - 1) / DEFAULT_ALIGN_SIZE_BYTES]))
    , _size(allocSize)
    , _qc(qc) {}

Buffer::~Buffer() {
    if (_data) {
        delete[] reinterpret_cast<uint64_t *>(_data);
        _data = nullptr;
    }
    _qc->deallocate(_size);
    _pos = 0;
    _size = 0;
    _qc = nullptr;
}

void *Buffer::allocate(size_t numBytes, size_t alignment) { return doAllocate(numBytes, alignment); }

} // namespace mem
} // namespace autil
