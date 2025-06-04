#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

#define printBufferData(buffer, hint)                           \
    do {                                                        \
        if (rtp_llm::Logger::getEngineLogger().isTraceMode()) { \
            printBufferData_(buffer, hint);                     \
        }                                                       \
    } while(0)

void printBufferData_(const Buffer& buffer, const std::string& hint, DeviceBase* device = nullptr, bool show_stats_only = false);

#define printTorchTensorData(buffer, hint)                      \
    do {                                                        \
        if (rtp_llm::Logger::getEngineLogger().isTraceMode()) { \
            printTorchTensorData_(buffer, hint);                \
        }                                                       \
    } while(0)

void printTorchTensorData_(const torch::Tensor& tensor, const std::string& hint, DeviceBase* device = nullptr, bool show_stats_only = false);

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
