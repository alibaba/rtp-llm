#pragma once

#include <torch/torch.h>
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void printTorchBufferSample(const std::string& hint, torch::Tensor& tensor, uint32_t n_samples);

#define printBufferData(tensor, hint)                                                                                  \
    do {                                                                                                               \
        if (rtp_llm::Logger::getEngineLogger().isTraceMode()) {                                                        \
            printTorchTensorData_(tensor, hint);                                                                       \
        }                                                                                                              \
    } while (0)

#define printBufferDataDebug(tensor, hint)                                                                             \
    do {                                                                                                               \
        if (rtp_llm::Logger::getEngineLogger().isDebugMode()) {                                                        \
            printTorchTensorData_(tensor, hint);                                                                       \
        }                                                                                                              \
    } while (0)

#define forcePrintBufferData(tensor, hint)                                                                             \
    do {                                                                                                               \
        printTorchTensorData_(tensor, hint);                                                                           \
    } while (0)

#define printTorchTensorData(tensor, hint)                                                                             \
    do {                                                                                                               \
        if (rtp_llm::Logger::getEngineLogger().isTraceMode()) {                                                        \
            printTorchTensorData_(tensor, hint);                                                                       \
        }                                                                                                              \
    } while (0)

void printTorchTensorData_(const torch::Tensor& tensor, const std::string& hint, bool show_stats_only = false);

torch::Tensor loadTensorFromFile(const std::string& fileName);

#ifdef ENABLE_DUMP_BUFFER_DATA
#undef printBufferData
#define printBufferData(tensor, hint) saveTensorData_(tensor, hint, std::string(__FILE__));
#endif

void saveTensorData_(const torch::Tensor& tensor, const std::string& fileName, const std::string& sourceFile);

void saveTorchDataTofile(const torch::Tensor& tensor, const std::string& fileName);

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
}  // namespace rtp_llm
