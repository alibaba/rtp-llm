#include <c10/core/TensorImpl.h>
#include <sstream>
#include <torch/torch.h>

#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"

namespace rtp_llm {

void printBuffer1d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level) {
    size_t dim1 = dims[0];
    RTP_LLM_LOG(log_level, "Buffer %s: shape [%d], type: %s", hint.c_str(), dim1, tensor.options().dtype().name().data());
    std::stringstream ss;
    ss << "Buffer " << hint << " : ";
    auto   print_func = [&](size_t column_start, size_t column_end) {
        for (int i = column_start; i < column_end && i < dim1; i++) {
            double value = tensor[i].item<double>();
            ss << " i = " << i << " value = " << value;
        }
    };
    const auto [sum1, sum2] = calculateTensorSum(
        [&](size_t i) -> auto { return tensor[i]; }, // 访问器
        dim1          // 当前维度长度
    );
    print_func(column_start, column_end);
    ss << " ...... ";
    print_func(std::max((size_t)0, dim1 - (column_end - column_start)), dim1);

    ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
    RTP_LLM_LOG(log_level, ss.str());
}

void printBuffer2d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level,
                        bool show_stats_only) {
    size_t dim1 = dims[0];
    size_t dim2 = dims[1];
    RTP_LLM_LOG(log_level, "Buffer %s: shape [%d %d], type: %s", hint.c_str(), dim1, dim2, tensor.options().dtype().name().data());
    size_t line_num = 0;
    for (int i = 0; i < dim1; i++) {
        std::stringstream ss;
        ss << "Buffer " << hint << " : ";
        ss << "[" << i << "]";
        if (!show_stats_only) {
            auto   print_func = [&](size_t column_start, size_t column_end) {
                for (int j = column_start; j < column_end && j < dim2; j++) {
                    double value = tensor[i][j].item<double>();
                    ss << " k = " << j << " value = " << value;
                }
            };
            print_func(column_start, column_end);
            ss << " ...... ";
            print_func(std::max((size_t)0, dim2 - (column_end - column_start)), dim2);
        }
        const auto [sum1, sum2] = calculateTensorSum(
            [&](size_t j) -> auto { return tensor[i][j]; },
            dim2
        );
        ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
        RTP_LLM_LOG(log_level, ss.str());
        line_num++;
        if (line_num > max_print_lines) {
            return;
        }
    }
}

void printBuffer3d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level,
                        bool show_stats_only) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level, "Buffer %s: shape [%d %d %d], type: %s", hint.c_str(), dim1, dim2, dim3, tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            std::stringstream ss;
            ss << "Buffer " << hint << " : ";
            ss << "[" << i << ", " << j << "]";
            if (!show_stats_only) {
                auto   print_func = [&](size_t column_start, size_t column_end) {
                    for (int k = column_start; k < column_end && k < dim3; k++) {
                        double value = tensor[i][j][k].item<double>();
                        ss << " k = " << k << " value = " << value;
                    }
                };
                print_func(column_start, column_end);
                ss << " ...... ";
                print_func(std::max((size_t)0, dim3 - (column_end - column_start)), dim3);
            }
            const auto [sum1, sum2] = calculateTensorSum(
                [&](size_t k) -> auto { return tensor[i][j][k]; },
                dim3
            );
            ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
            RTP_LLM_LOG(log_level, ss.str());
            line_num++;
            if (line_num > max_print_lines) {
                return;
            }
        }
    }
}

void printBuffer4d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t dim4     = dims[3];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level, "Buffer %s: shape [%d %d %d %d], type: %s", hint.c_str(), dim1, dim2, dim3, dim4, tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                std::stringstream ss;
                ss << "Buffer " << hint << " : ";
                ss << "[" << i << "," << j << "," << k << "]";
                auto   print_func = [&](size_t column_start, size_t column_end) {
                    for (int x = column_start; x < column_end && x < dim4; x++) {
                        double value = tensor[i][j][k][x].item<double>();
                        ss << " k = " << x << " value = " << value;
                    }
                };
                const auto [sum1, sum2] = calculateTensorSum(
                    [&](size_t x) -> auto { return tensor[i][j][k][x]; },
                    dim4
                );
                print_func(column_start, column_end);
                ss << " ...... ";
                print_func(std::max((size_t)0, dim4 - (column_end - column_start)), dim4);
                ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
                RTP_LLM_LOG(log_level, ss.str());
                line_num++;
                if (line_num > max_print_lines) {
                    return;
                }
            }
        }
    }
}

void printBuffer5d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t dim4     = dims[3];
    size_t dim5     = dims[4];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level, "Buffer %s: shape [%d %d %d %d %d], type: %s", hint.c_str(), dim1, dim2, dim3, dim4, dim5, tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int x = 0; x < dim4; x++) {
                    std::stringstream ss;
                    ss << "Buffer " << hint << " : ";
                    ss << "[" << i << "," << j << "," << k << "," << x << "]";
                    auto   print_func = [&](size_t column_start, size_t column_end) {
                        for (int y = column_start; y < column_end && y < dim5; y++) {
                            double value = tensor[i][j][k][x][y].item<double>();
                            ss << " y = " << y << " value = " << value;
                        }
                    };
                    const auto [sum1, sum2] = calculateTensorSum(
                        [&](size_t y) -> auto { return tensor[i][j][k][x][y]; },
                        dim5
                    );
                    print_func(column_start, column_end);
                    ss << " ...... ";
                    print_func(std::max((size_t)0, dim5 - (column_end - column_start)), dim5);
                    ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
                    RTP_LLM_LOG(log_level, ss.str());
                    line_num++;
                    if (line_num > max_print_lines) {
                        return;
                    }
                }
            }
        }
    }
}

void printBuffer6d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t dim4     = dims[3];
    size_t dim5     = dims[4];
    size_t dim6     = dims[5];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level, "Buffer %s: shape [%d %d %d %d %d %d], type: %s", hint.c_str(), dim1, dim2, dim3, dim4, dim5, dim6, tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int x = 0; x < dim4; x++) {
                    for (int y = 0; y < dim5; y++) {
                        std::stringstream ss;
                        ss << "Buffer " << hint << " : ";
                        ss << "[" << i << "," << j << "," << k << "," << x << "," << y << "]";
                        auto   print_func = [&](size_t column_start, size_t column_end) {
                            for (int z = column_start; z < column_end && z < dim6; z++) {
                                double value = tensor[i][j][k][x][y][z].item<double>();
                                ss << " k = " << z << " value = " << value;
                            }
                        };
                        const auto [sum1, sum2] = calculateTensorSum(
                            [&](size_t z) -> auto { return tensor[i][j][k][x][y][z]; },
                            dim6
                        );
                        print_func(column_start, column_end);
                        ss << " ...... ";
                        print_func(std::max((size_t)0, dim6 - (column_end - column_start)), dim6);
                        ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
                        RTP_LLM_LOG(log_level, ss.str());
                        line_num++;
                        if (line_num > max_print_lines) {
                            return;
                        }
                    }
                }
            }
        }
    }
}

void printBufferData_(const Buffer& buffer, const std::string& hint, DeviceBase* device, bool show_stats_only) {
    const auto log_level = alog::LOG_LEVEL_INFO;

    if (buffer.isQBuffer()) {
        const QBuffer* q_buffer = &(reinterpret_cast<const QBuffer&>(buffer));
        printBufferData_(q_buffer->kernel(), hint + "_kernel");
        printBufferData_(q_buffer->scales(), hint + "_scales");
        if (q_buffer->zeros().type()) {
            printBufferData_(q_buffer->zeros(), hint + "_zeros");
        }
        return;
    }

    if (!device) {
        device = DeviceFactory::getDefaultDevice();
    }

    auto host_buffer = device->allocateBuffer({buffer.type(), buffer.shape(), AllocationType::HOST});
    device->copy({*host_buffer, buffer});
    device->syncAndCheck();

    torch::Tensor tensor = torch::from_blob(
        host_buffer->data(),
        bufferShapeToTorchShape(buffer),
        c10::TensorOptions().device(torch::Device(torch::kCPU)).dtype(dataTypeToTorchType(buffer.type())));

    std::vector<size_t> dims            = buffer.shape();
    size_t              column_start    = 0;
    size_t              column_end      = 20;
    size_t              max_print_lines = 20;
    if (dims.size() > 6) {
        std::stringstream ss;
        ss << "Buffer " << hint << " : " << tensor;
        RTP_LLM_LOG(log_level, "%s", ss.str().c_str());
    } else if (dims.size() == 6) {
        printBuffer6d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 5) {
        printBuffer5d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 4) {
        printBuffer4d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 3) {
        printBuffer3d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level, show_stats_only);
    } else if (dims.size() == 2) {
        printBuffer2d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level, show_stats_only);
    } else if (dims.size() == 1) {
        printBuffer1d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    }
}

void printTorchTensorData_(const torch::Tensor& tensor, const std::string& hint, DeviceBase* device, bool show_stats_only) {
    auto buffer = torchTensor2Buffer(tensor);
    printBufferData_(*buffer, hint, device, show_stats_only);
}

void saveBufferDataToTorch(const Buffer& buffer, DeviceBase* device, const std::string& fileName) {
    if (!device) {
        device = DeviceFactory::getDefaultDevice();
    }

    auto host_buffer = device->allocateBuffer({buffer.type(), buffer.shape(), AllocationType::HOST});
    device->copy({*host_buffer, buffer});
    device->syncAndCheck();
    auto tensor = torch::from_blob(
        host_buffer->data(),
        bufferShapeToTorchShape(buffer),
        c10::TensorOptions().device(torch::Device(torch::kCPU)).dtype(dataTypeToTorchType(buffer.type())));
    if (tensor.dtype() == torch::kFloat8_e4m3fn) {
        tensor = tensor.to(torch::kFloat);
    }
    auto          pickled = torch::pickle_save(tensor);
    std::ofstream fout(fileName, std::ios::out | std::ios::binary);
    fout.write(pickled.data(), pickled.size());
    fout.close();
}

void saveTorchDataTofile(const torch::Tensor& tensor, const std::string& fileName) {
    auto tensor_cpu = tensor.contiguous().cpu();
    auto          pickled = torch::pickle_save(tensor_cpu);
    std::ofstream fout(fileName, std::ios::out | std::ios::binary);
    fout.write(pickled.data(), pickled.size());
    fout.close();
}

torch::Tensor loadTensorFromFile(const std::string& fileName) {
    // Open the file
    std::ifstream fin(fileName, std::ios::in | std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    // Get the file size
    fin.seekg(0, std::ios::end);
    size_t fileSize = fin.tellg();
    fin.seekg(0, std::ios::beg);

    // Read the file content into a vector
    std::vector<char> pickledData(fileSize);
    fin.read(pickledData.data(), fileSize);
    fin.close();

    // Deserialize the Tensor
    return torch::pickle_load(pickledData).toTensor();
}

}  // namespace rtp_llm
