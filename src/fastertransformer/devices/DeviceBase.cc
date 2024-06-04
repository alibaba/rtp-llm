#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/core/TrackerAllocator.h"

using namespace std;

namespace fastertransformer {

DeviceBase::DeviceBase(const DeviceInitParams& params)
    : device_id_(params.device_id)
    , init_params_(params)
    {}

void DeviceBase::init() {
    buffer_manager_.reset(new BufferManager(getAllocator(), getHostAllocator()));
}

DeviceStatus DeviceBase::getDeviceStatus() {
    return DeviceStatus();
}

void DeviceBase::traceMemoryUsage() {
    FT_LOG_INFO("Device Memory: %s", buffer_manager_->printAllocationRecords(getAllocator()).c_str());
    FT_LOG_INFO("Host Memory: %s", buffer_manager_->printAllocationRecords(getHostAllocator()).c_str());
    return;
}

AllocationType DeviceBase::getMemAllocationType(const MemoryType type) {
    return (type == getAllocator()->memoryType()) ? AllocationType::DEVICE : AllocationType::HOST;
}

BufferStatus DeviceBase::queryBufferStatus() {
    return buffer_manager_->queryStatus();
}

BufferPtr DeviceBase::allocateBuffer(const BufferParams& params, const BufferHints& hints) {
    return buffer_manager_->allocate(params, hints);
}

BufferPtr DeviceBase::allocateBufferLike(const Buffer& buffer, const BufferHints& hints) {
    return allocateBuffer({buffer.type(), buffer.shape(), getMemAllocationType(buffer.where())}, hints);
}

void DeviceBase::syncAndCheck() {
    return;
}

void DeviceBase::syncCommunication() {
    return;
}

CloneOutput DeviceBase::clone(const CloneParams& params) {
    const auto& src = params.input;
    auto dst = allocateBuffer({src.type(), src.shape(), params.alloc_type}, params.hints);
    copy({*dst, src});
    return move(dst);
}

}; // namespace fastertransformer

