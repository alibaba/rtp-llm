#include "src/fastertransformer/devices/DeviceBase.h"

using namespace std;

namespace fastertransformer {

DeviceBase::DeviceBase() {}

IAllocator* DeviceBase::getAllocator() {
    return allocator_.get();
}

}; // namespace fastertransformer

