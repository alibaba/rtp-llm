#include "src/fastertransformer/devices/DeviceOps.h"

namespace fastertransformer {

DeviceOps::DeviceOps() {}

DeviceOps::~DeviceOps() {}

size_t DeviceOps::getKvCacheBlockSize() const {
    return 0;
};

} // namespace fastertransformer

