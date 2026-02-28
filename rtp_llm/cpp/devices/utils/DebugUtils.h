#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

void printBufferSample(const std::string& hint, Buffer& buffer, uint32_t n_samples);

void printTorchBufferSample(const std::string& hint, torch::Tensor& tensor, uint32_t n_samples);

#define printBufferData(buffer, hint)                                                                                  \
    do {                                                                                                               \
        if (rtp_llm::Logger::getEngineLogger().isTraceMode()) {                                                        \
            printBufferData_(buffer, hint);                                                                            \
        }                                                                                                              \
    } while (0)

#define printBufferDataDebug(buffer, hint)                                                                             \
    do {                                                                                                               \
        if (rtp_llm::Logger::getEngineLogger().isDebugMode()) {                                                        \
            printBufferData_(buffer, hint);                                                                            \
        }                                                                                                              \
    } while (0)

#define forcePrintBufferData(buffer, hint)                                                                             \
    do {                                                                                                               \
        printBufferData_(buffer, hint);                                                                                \
    } while (0)

void printBufferData_(const Buffer&      buffer,
                      const std::string& hint,
                      DeviceBase*        device          = nullptr,
                      bool               show_stats_only = false);

#define printTorchTensorData(buffer, hint)                                                                             \
    do {                                                                                                               \
        if (rtp_llm::Logger::getEngineLogger().isTraceMode()) {                                                        \
            printTorchTensorData_(buffer, hint);                                                                       \
        }                                                                                                              \
    } while (0)

void printTorchTensorData_(const torch::Tensor& tensor,
                           const std::string&   hint,
                           DeviceBase*          device          = nullptr,
                           bool                 show_stats_only = false);

BufferPtr loadTorchToBuffer(const std::string& fileName, DeviceBase* device);

#ifdef ENABLE_DUMP_BUFFER_DATA
#undef printBufferData
#define printBufferData(buffer, hint) saveBufferData_(buffer, nullptr, hint, std::string(__FILE__));
#endif

void saveBufferData_(const Buffer&      buffer,
                     DeviceBase*        device,
                     const std::string& fileName,
                     const std::string& sourceFile);

void saveBufferData_(Buffer& buffer, DeviceBase* device, const std::string& fileName, const std::string& sourceFile);

void saveBufferDataToTorch(const Buffer& buffer, DeviceBase* device, const std::string& fileName);

void saveTorchDataTofile(const torch::Tensor& tensor, const std::string& fileName);

torch::Tensor loadTensorFromFile(const std::string& fileName);

void dumpTensor(const Buffer& buffer, const std::string& name, int rank = 0);

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
