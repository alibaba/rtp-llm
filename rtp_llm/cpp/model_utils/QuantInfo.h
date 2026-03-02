#pragma once

#include <string>
namespace rtp_llm {

enum QuantMethod {
    None             = 0,
    WeightOnlyPerCol = 1,
    GptQ             = 2,
    Awq              = 3,
    SmoothQuant      = 4,
    OmniQuant        = 5,
    PerTensorQuant   = 6,
    FP8Quant         = 7,
    FP8PTPC          = 8,
    ModelOptFP4      = 9,
};

struct QuantAlgo {
public:
    QuantAlgo() = default;
    QuantAlgo(QuantMethod method, int bits, int group_size):
        quant_method_(method), weight_bits_(bits), group_size_(group_size) {}
    bool isWeightOnlyPerCol() const {
        return quant_method_ == WeightOnlyPerCol;
    }
    bool isPerTensorQuant() const {
        return quant_method_ == PerTensorQuant;
    }
    bool isGptq() const {
        return quant_method_ == GptQ;
    }
    bool isAwq() const {
        return quant_method_ == Awq;
    }
    bool isSmoothQuant() const {
        return quant_method_ == SmoothQuant;
    }
    bool isOmniQuant() const {
        return quant_method_ == OmniQuant;
    }
    bool isFp8() const {
        return quant_method_ == FP8Quant;
    }
    bool isFp8PTPC() const {
        return quant_method_ == FP8PTPC;
    }
    bool isQuant() const {
        return quant_method_ != None;
    }
    bool isGroupwise() const {
        return group_size_ > 0;
    }
    bool isModelOptFP4() const {
        return group_size_ > 0 && quant_method_ == ModelOptFP4;
    }
    QuantMethod getQuantMethod() const {
        return quant_method_;
    }
    int64_t getGroupSize() const {
        return group_size_;
    }
    int64_t getWeightBits() const {
        return weight_bits_;
    }
    int64_t getActivationBits() const {
        if (quant_method_ == None || quant_method_ == WeightOnlyPerCol || quant_method_ == Awq
            || quant_method_ == GptQ) {
            return 16;
        } else {
            return weight_bits_;
        }
    }
    void setQuantAlgo(const std::string& method, int64_t bits, int64_t group_size);

private:
    QuantMethod quant_method_ = None;
    int64_t     weight_bits_  = 16;
    int64_t     group_size_   = 0;
};

}  // namespace rtp_llm
