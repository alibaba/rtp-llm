#include "rtp_llm/cpp/model_utils/QuantInfo.h"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace rtp_llm {

void QuantAlgo::setQuantAlgo(const std::string& quant_method, int64_t bits, int64_t group_size) {
    if (quant_method == "gptq") {
        quant_method_ = GptQ;
        weight_bits_  = bits;
        group_size_   = group_size;
    } else if (quant_method == "awq") {
        quant_method_ = Awq;
        weight_bits_  = bits;
        group_size_   = group_size;
    } else if (quant_method == "weight_only_per_col") {
        quant_method_ = WeightOnlyPerCol;
        weight_bits_  = bits;
        if (weight_bits_ != 8) {
            throw std::invalid_argument("invalid weight_bits: " + std::to_string(weight_bits_));
        }
    } else if (quant_method == "smooth_quant") {
        quant_method_ = SmoothQuant;
        weight_bits_  = 8;
    } else if (quant_method == "omni_quant") {
        quant_method_ = OmniQuant;
        weight_bits_  = 8;
    } else if (quant_method == "pertensor_quant") {
        quant_method_ = PerTensorQuant;
        weight_bits_  = 8;
    } else if (quant_method == "fp8" || quant_method == "fp8_dynamic_per_tensor") {
        quant_method_ = FP8Quant;
        weight_bits_  = 8;
        group_size_   = group_size;
    } else if (quant_method == "fp8-perchannel-compressed-tensors") {
        quant_method_ = FP8PTPC;
        weight_bits_  = 8;
    } else if (quant_method == "fp8-perchannel-quark") {
        quant_method_ = FP8PTPC;
        weight_bits_  = 8;
    } else if (quant_method == "w4a8_int4_per_channel") {
        quant_method_ = W4A8INT4PTPC;
        weight_bits_  = 4;
        group_size_   = group_size;
    } else {
        throw std::invalid_argument("unknown quant_method: " + quant_method);
    }
    if (weight_bits_ != 4 && weight_bits_ != 8) {
        throw std::invalid_argument("invalid weight_bits: " + std::to_string(weight_bits_));
    }
    if (group_size_ != 0 && group_size_ != 8 && group_size_ != 16 && group_size_ != 32 && group_size_ != 64
        && group_size_ != 128) {
        throw std::invalid_argument("invalid group_size: " + std::to_string(group_size_));
    }
}

}  // namespace rtp_llm
