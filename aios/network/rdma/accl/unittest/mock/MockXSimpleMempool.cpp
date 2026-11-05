#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

namespace accl {
namespace barex {

MockXSimpleMempool::~MockXSimpleMempool() {
    for (auto& iter : _buffers) {
        free(iter.first);
    }
}

const int kAlignSize = 64;
void*     alignAlloc(int len) {
    int realSize = (len % kAlignSize == 0) ? len : (len / kAlignSize * kAlignSize + kAlignSize);
    return aligned_alloc(kAlignSize, realSize);
}

void MockXSimpleMempool::MockOnCallAllocAndReleaseBuffer(int len, device_type device, int deviceId) {
    auto buf = alignAlloc(len);
    if (!buf) {
        return;
    }
    _buffers[buf] = device;

    EXPECT_CALL(*this, AllocBuffer(len, device, deviceId, nullptr)).WillOnce(Return(buf));
    EXPECT_CALL(*this, ReleaseBuffer(buf, device)).WillOnce(Return(BAREX_SUCCESS));
}

void MockXSimpleMempool::MockOnAnyCallAllocAndReleaseBuffer() {
    auto& buffers = _buffers;
    ON_CALL(*this, AllocBuffer(_, _, 0, nullptr))
        .WillByDefault(Invoke([this, &buffers](int len, device_type device, int deviceId, void* attr) -> void* {
            auto buf = alignAlloc(len);
            if (!buf) {
                return nullptr;
            }
            buffers[buf] = device;
            EXPECT_CALL(*this, ReleaseBuffer(buf, device)).WillOnce(Return(BAREX_SUCCESS));
            return buf;
        }));
    EXPECT_CALL(*this, AllocBuffer(_, _, _, nullptr)).Times(::testing::AtLeast(1));
}

void MockXSimpleMempool::MockOnAnyCallIsValid() {
    auto& buffers = _buffers;
    ON_CALL(*this, IsValid(_, _, _))
        .WillByDefault(Invoke([&buffers](void* buffer, device_type device, bool adopt) -> bool {
            auto iter = buffers.find(buffer);
            if (iter == buffers.end()) {
                return false;
            }
            return iter->second == device;
        }));
    EXPECT_CALL(*this, IsValid(_, _, _)).Times(::testing::AtLeast(1));
}

}  // namespace barex
}  // namespace accl