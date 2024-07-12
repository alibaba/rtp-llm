#pragma once

#include <cmath>
#include <vector>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace ft = fastertransformer;

namespace rtp_llm {

std::vector<float> get_slopes_power_of_2(int n) {
    double start = std::pow(2, -std::pow(2, -(std::log2(n) - 3)));
    double ratio = start;
    std::vector<float> slopes;
    slopes.reserve(n);

    for (int i = 0; i < n; ++i) {
        slopes.push_back(start * std::pow(ratio, i));
    }
    
    return slopes;
}

std::vector<float> get_slopes(int n) {
    if (std::floor(std::log2(n)) == std::log2(n)) {
        return get_slopes_power_of_2(n);
    } else {
        int closest_power_of_2 = std::pow(2, std::floor(std::log2(n)));
        auto slopes = get_slopes_power_of_2(closest_power_of_2);
        auto extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2);
        
        for (size_t i = 0; i < extra_slopes.size(); i += 2) {
            if (slopes.size() >= static_cast<size_t>(n)) {
                break;
            }
            slopes.push_back(extra_slopes[i]);
        }
        return slopes;
    }
}

ft::BufferPtr splitSlopesTP(const ft::GptInitParameter& gpt_init_params, std::vector<float> slopes, ft::DeviceBase* device) {
    int local_head_num = (gpt_init_params.head_num_ == 1) ? 1 : gpt_init_params.head_num_ / gpt_init_params.tp_size_;
    int start_pos = local_head_num * gpt_init_params.tp_rank_;
    auto tp_slopes = std::vector<float>(slopes.begin() + start_pos, slopes.begin() + start_pos + local_head_num);
    auto tp_slopes_buffer = ft::vector2Buffer(tp_slopes);
    auto convert_slopes = tp_slopes_buffer ? device->convert({tp_slopes_buffer, ft::getDataType(gpt_init_params.data_type_)}) : nullptr;
    auto linear_bias_slopes = convert_slopes ? device->clone({*convert_slopes, ft::AllocationType::DEVICE}) : nullptr;
    return linear_bias_slopes;
}

ft::BufferPtr createLinearBias(const ft::GptInitParameter& gpt_init_params, ft::DeviceBase* device) {
    if (gpt_init_params.use_attention_linear_bias_) {
        auto vec = get_slopes(gpt_init_params.head_num_);
        return splitSlopesTP(gpt_init_params, vec, device);
    }
    return nullptr;
}

}
