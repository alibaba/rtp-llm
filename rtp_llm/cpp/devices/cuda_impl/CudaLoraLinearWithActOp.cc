#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;

namespace rtp_llm {

bool smoothQuantGemmSupportFuseBiasActivation(const LoraLinearWithActivationParams&     params,
                                              const trt_plugins::SmoothQuantGemmPlugin& plugin) {
    if (params.lora_linear_params.lora_input != nullptr) {
        return false;
    }
    // gate act not support fuse for now
    if (params.activation_params.gate != std::nullopt || params.activation_params.act_scale != std::nullopt
        || params.activation_params.gate_bias != std::nullopt) {
        return false;
    }
    auto gemm_type = params.lora_linear_params.gemm_params.dispatch();
    // only smooth quant support fuse
    if (gemm_type != GemmType::QBufferA_QBufferB_BufferC_2DGemm) {
        return false;
    }
    // check activation type supported;
    return plugin.addBiasActivationEpilogueSupported(params.activation_params.atype);
}

BufferPtr CudaDevice::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
    if (smoothQuantGemmSupportFuseBiasActivation(params, *smooth_quant_plugin_)) {
        auto fuse_gemm_params           = params.lora_linear_params.gemm_params;
        fuse_gemm_params.activationType = params.activation_params.atype;
        fuse_gemm_params.C              = params.activation_params.bias;

        if (initParams().profile_debug_logging_config.check_nan) {
            if (fuse_gemm_params.A.isQBuffer()) {
                const auto& qbuffer = reinterpret_cast<const QBuffer&>(fuse_gemm_params.A);
                checkNAN(qbuffer.kernel(), "loraLinearWithAct_fused_A_kernel_dump", nullptr, true);
                checkNAN(qbuffer.scales(), "loraLinearWithAct_fused_A_scales_dump", nullptr, true);
            } else {
                checkNAN(fuse_gemm_params.A, "loraLinearWithAct_fused_A_dump", nullptr, true);
            }
            if (fuse_gemm_params.B.isQBuffer()) {
                const auto& qbuffer = reinterpret_cast<const QBuffer&>(fuse_gemm_params.B);
                checkNAN(qbuffer.kernel(), "loraLinearWithAct_fused_B_kernel_dump", nullptr, true);
                checkNAN(qbuffer.scales(), "loraLinearWithAct_fused_B_scales_dump", nullptr, true);
            } else {
                checkNAN(fuse_gemm_params.B, "loraLinearWithAct_fused_B_dump", nullptr, true);
            }
            if (fuse_gemm_params.C.has_value()) {
                checkNAN(fuse_gemm_params.C.value().get(), "loraLinearWithAct_fused_C_dump", nullptr, true);
            }
        }

        auto output = gemm(fuse_gemm_params);

        if (initParams().profile_debug_logging_config.check_nan) {
            if (output->isQBuffer()) {
                const auto& qbuffer = reinterpret_cast<const QBuffer&>(*output);
                checkNAN(qbuffer.kernel(), "loraLinearWithAct_fused_output_kernel_dump", nullptr, true);
                checkNAN(qbuffer.scales(), "loraLinearWithAct_fused_output_scales_dump", nullptr, true);
            } else {
                checkNAN(*output, "loraLinearWithAct_fused_output_dump", nullptr, true);
            }
        }

        return output;
    }
    return DeviceBase::loraLinearWithActivation(params);
}
}  // namespace rtp_llm