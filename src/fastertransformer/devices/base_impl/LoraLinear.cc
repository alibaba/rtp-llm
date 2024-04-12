#include "src/fastertransformer/devices/DeviceBase.h"

using namespace std;

namespace fastertransformer {

LoraLinearOutput DeviceBase::loraLinear(const LoraLinearParams& params) {
    auto output = gemm({params.input, *(params.weight.kernel)});

    if (params.lora_ids != std::nullopt) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    return LoraLinearOutput({std::move(output)});
}

}; // namespace fastertransformer

