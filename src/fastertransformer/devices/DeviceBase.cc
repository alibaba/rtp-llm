#include "src/fastertransformer/devices/DeviceBase.h"

using namespace std;

namespace fastertransformer {

DeviceBase::DeviceBase() {}

void DeviceBase::init() {
    buffer_manager_.reset(new BufferManager(getAllocator(), getHostAllocator()));
}

CloneOutput DeviceBase::clone(const CloneParams& params) {
    const auto& src = params.input;
    auto dst = allocateBuffer({src.type(), src.shape(), params.alloc_type});
    copy({*dst, src});
    return move(dst);
}

unique_ptr<Buffer> DeviceBase::allocateBuffer(const BufferParams& params, const BufferHints& hints) {
    return buffer_manager_->allocate(params, hints);
}

unique_ptr<Buffer> DeviceBase::allocateBufferLike(const Buffer& buffer, const BufferHints& hints) {
    return allocateBuffer({buffer.type(), buffer.shape()}, hints);
}

void DeviceBase::syncAndCheck() {
    return;
}

TransposeOutput DeviceBase::transpose(const TransposeParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

}; // namespace fastertransformer

