#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/cuda/cutlass_utils.h"

using namespace std;

namespace fastertransformer {

bool smoothQuantGemmSupportFuseBiasActivation(const LoraLinearWithActivationParams& params, const trt_plugins::SmoothQuantGemmPlugin& plugin) {
    if (params.lora_linear_params.lora_input != nullptr) {
        return false;
    }
    // gate act not support fuse for now
    if (params.activation_params.gate != std::nullopt || params.activation_params.act_scale != std::nullopt || params.activation_params.gate_bias != std::nullopt) {
        return false;
    }
    auto gemm_type = params.lora_linear_params.gemm_params.dispatch();
    // only smooth quant support fuse
    if (gemm_type != GemmType::QBufferA_QBufferB_BufferC_2DGemm) {
        return false;
    }
    // check activation type supported;
    return plugin.addBiasActivationEpilogueSupported(CutlassUtils::ActivationToCutlassType(params.activation_params.atype));
}


BufferPtr CudaDevice::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
    if (smoothQuantGemmSupportFuseBiasActivation(params, *smooth_quant_plugin_)) {        
        auto fuse_gemm_params = params.lora_linear_params.gemm_params;
        fuse_gemm_params.activationType = params.activation_params.atype;
        fuse_gemm_params.C = params.activation_params.bias;
        return gemm(fuse_gemm_params);
    }
    return DeviceBase::loraLinearWithActivation(params);
}
}  // namespace fastertransformer