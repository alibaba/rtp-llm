#pragma once

#include <sstream>
#include <string>
#include <algorithm>
#include <torch/extension.h>

namespace rtp_llm {

// Torch-native debug string utilities (no Buffer dependency)
inline std::string tensorDebugString(const torch::Tensor& t) {
    if (!t.defined())
        return "(undefined)";
    std::string s = "Tensor(dtype=" + std::string(c10::toString(t.scalar_type())) + ", shape=[";
    for (int64_t i = 0; i < t.dim(); i++) {
        if (i)
            s += ", ";
        s += std::to_string(t.size(i));
    }
    s += "], device=" + t.device().str() + ")";
    return s;
}

template<typename T>
std::string tensorDebugStringWithData(const torch::Tensor& t, size_t count = 0) {
    if (!t.defined())
        return "(undefined)";
    auto meta = tensorDebugString(t);
    if (t.is_cuda())
        return meta + ", Device tensor data can NOT be dumped";
    auto cpu_t = t.contiguous();
    auto base  = cpu_t.data_ptr<T>();
    auto total = static_cast<size_t>(cpu_t.numel());
    if (count == 0)
        count = total;
    auto               data_size = std::min(count, total);
    std::ostringstream oss;
    for (size_t i = 0; i < data_size; i++)
        oss << base[i] << ", ";
    if (data_size != total) {
        oss << "...... ";
        for (size_t i = total - data_size; i < total; i++)
            oss << base[i] << ", ";
    }
    return meta + ", Data(" + oss.str() + ")";
}

}  // namespace rtp_llm
