#include "src/fastertransformer/devices/DeviceBase.h"

using namespace std;

namespace fastertransformer {

DeviceBase::DeviceBase() {}

void DeviceBase::init() {
    buffer_manager_.reset(new BufferManager(getAllocator(), getHostAllocator()));
}

std::unique_ptr<Buffer> DeviceBase::allocateBuffer(const BufferParams& params, const BufferHints& hints) {
    return buffer_manager_->allocate(params, hints);
}

}; // namespace fastertransformer

