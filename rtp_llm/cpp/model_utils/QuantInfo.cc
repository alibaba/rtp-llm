#include "rtp_llm/cpp/model_utils/QuantInfo.h"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace rtp_llm {

void QuantAlgo::setQuantAlgo(const std::string& method, int64_t bits, int64_t group_size) {
    std::string method_lower = method;
    std::transform(method_lower.begin(), method_lower.end(), method_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (method_lower == "gptq") {
        quant_method_ = GptQ;
        weight_bits_  = bits;
        group_size_   = group_size;
    } else if (method_lower == "awq") {
        quant_method_ = Awq;
        weight_bits_  = bits;
        group_size_   = group_size;
    } else if (method_lower == "weight_only_per_col") {
        quant_method_ = WeightOnlyPerCol;
        weight_bits_  = bits;
        if (weight_bits_ != 8) {
            throw std::invalid_argument("invalid weight_bits: " + std::to_string(weight_bits_));
        }
    } else if (method_lower == "smooth_quant") {
        quant_method_ = SmoothQuant;
        weight_bits_  = 8;
    } else if (method_lower == "omni_quant") {
        quant_method_ = OmniQuant;
        weight_bits_  = 8;
    } else if (method_lower == "per_tensor_quant") {
        quant_method_ = PerTensorQuant;
        weight_bits_  = bits;
    } else if (method_lower == "fp8" || method_lower == "fp8_quant") {
        quant_method_ = FP8Quant;
        weight_bits_  = 8;
    } else if (method_lower == "fp8ptpc") {
        quant_method_ = FP8PTPC;
        weight_bits_  = 8;
    } else if (method_lower == "none" || method_lower == "") {
        quant_method_ = None;
        weight_bits_  = 16;
        group_size_   = 0;
    } else {
        throw std::invalid_argument("unknown quant method: " + method);
    }
}

}  // namespace rtp_llm










