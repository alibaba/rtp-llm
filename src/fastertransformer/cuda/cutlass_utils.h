#pragma once

#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"

#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

class CutlassUtils {
public:
    static tkc::CutlassActivationType ActivationToCutlassType(fastertransformer::ActivationType act_type) {
        switch(act_type) {
            case fastertransformer::ActivationType::Identity:
                return tkc::CutlassActivationType::IDENTITY;
            case fastertransformer::ActivationType::Geglu:
            case fastertransformer::ActivationType::Gelu:
                return tkc::CutlassActivationType::GELU_FAST;
            case fastertransformer::ActivationType::GeluNoneApproximate:
            case fastertransformer::ActivationType::GeGluNoneApproximate:
                return tkc::CutlassActivationType::GELU;
            case fastertransformer::ActivationType::Relu:
                return tkc::CutlassActivationType::RELU;
            case fastertransformer::ActivationType::Sigmoid:
                return tkc::CutlassActivationType::SIGMOID;
                case fastertransformer::ActivationType::Silu:
            case fastertransformer::ActivationType::Swiglu:
                return tkc::CutlassActivationType::SILU;
            default:
                FT_CHECK_WITH_INFO(false, "ERROR ACTIVATION TYPE");            
        }
        return tkc::CutlassActivationType::INVALID;
    }
};