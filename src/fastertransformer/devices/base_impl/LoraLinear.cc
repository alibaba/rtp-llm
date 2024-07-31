#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

namespace fastertransformer {

LoraLinearOutput DeviceBase::loraLinear(const LoraLinearParams& params) {
    auto output = gemm(params.gemm_params);
    return LoraLinearOutput({std::move(output)});
}

}; // namespace fastertransformer

