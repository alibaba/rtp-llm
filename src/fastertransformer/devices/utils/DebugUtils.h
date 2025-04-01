#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

namespace fastertransformer {

void printBufferData(const Buffer& buffer, const std::string& hint, DeviceBase* device = nullptr, bool force_print = false, bool show_stats_only = false);

void printTorchTensorData(const torch::Tensor& tensor, const std::string& hint, DeviceBase* device = nullptr, bool force_print = false, bool show_stats_only = false);

void saveBufferDataToTorch(const Buffer& buffer, DeviceBase* device, const std::string& fileName);

void saveTorchDataTofile(const torch::Tensor& tensor, const std::string& fileName);

torch::Tensor loadTensorFromFile(const std::string& fileName);

template<typename TensorAccessor>
std::pair<double, double> calculateTensorSum(TensorAccessor&& accessor, size_t dim) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double value = accessor(i).template item<double>();
        sum1 += value;
        sum2 += value * value;
    }
    return {sum1, sum2};
}
}
