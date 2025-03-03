#include <c10/core/TensorImpl.h>
#include <sstream>
#include <torch/torch.h>

#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/CommonDefines.h"

namespace fastertransformer {

void printBuffer1d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level) {
    size_t dim1 = dims[0];
    FT_LOG(log_level, "Buffer %s: shape [%d]", hint.c_str(), dim1);
    std::stringstream ss;
    ss << "Buffer " << hint << " : ";
    double sum1       = 0;
    double sum2       = 0;
    auto   print_func = [&](size_t column_start, size_t column_end) {
        for (int i = column_start; i < column_end && i < dim1; i++) {
            double value = tensor[i].item<double>();
            ss << " i = " << i << " value = " << value;
            sum1 += value;
            sum2 += value * value;
        }
    };
    print_func(column_start, column_end);
    ss << " ...... ";
    print_func(std::max((size_t)0, dim1 - (column_end - column_start)), dim1);
    ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
    FT_LOG(log_level, ss.str());
}

void printBuffer2d(const std::string&  hint,
                        torch::Tensor&      tensor,
                        std::vector<size_t> dims,
                        size_t              column_start,
                        size_t              column_end,
                        size_t              max_print_lines,
                        int log_level) {
    size_t dim1 = dims[0];
    size_t dim2 = dims[1];
    FT_LOG(log_level, "Buffer %s: shape [%d %d]", hint.c_str(), dim1, dim2);
    size_t line_num = 0;
    for (int i = 0; i < dim1; i++) {
        std::stringstream ss;
        ss << "Buffer " << hint << " : ";
        ss << "[" << i << "]";
        double sum1       = 0;
        double sum2       = 0;
        auto   print_func = [&](size_t column_start, size_t column_end) {
            for (int j = column_start; j < column_end && j < dim2; j++) {
                double value = tensor[i][j].item<double>();
                ss << " k = " << j << " value = " << value;
                sum1 += value;
                sum2 += value * value;
            }
        };
        print_func(column_start, column_end);
        ss << " ...... ";
        print_func(std::max((size_t)0, dim2 - (column_end - column_start)), dim2);
        ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
        FT_LOG(log_level, ss.str());
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
                        int log_level) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t line_num = 0;
    FT_LOG(log_level, "Buffer %s: shape [%d %d %d]", hint.c_str(), dim1, dim2, dim3);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            std::stringstream ss;
            ss << "Buffer " << hint << " : ";
            ss << "[" << i << ", " << j << "]";
            double sum1       = 0;
            double sum2       = 0;
            auto   print_func = [&](size_t column_start, size_t column_end) {
                for (int k = column_start; k < column_end && k < dim3; k++) {
                    double value = tensor[i][j][k].item<double>();
                    ss << " k = " << k << " value = " << value;
                    sum1 += value;
                    sum2 += value * value;
                }
            };
            print_func(column_start, column_end);
            ss << " ...... ";
            print_func(std::max((size_t)0, dim3 - (column_end - column_start)), dim3);
            ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
            FT_LOG(log_level, ss.str());
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
    FT_LOG(log_level, "Buffer %s: shape [%d %d %d %d]", hint.c_str(), dim1, dim2, dim3, dim4);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                std::stringstream ss;
                ss << "Buffer " << hint << " : ";
                ss << "[" << i << "," << j << "," << k << "]";
                double sum1       = 0;
                double sum2       = 0;
                auto   print_func = [&](size_t column_start, size_t column_end) {
                    for (int x = column_start; x < column_end && x < dim4; x++) {
                        double value = tensor[i][j][k][x].item<double>();
                        ss << " k = " << x << " value = " << value;
                        sum1 += value;
                        sum2 += value * value;
                    }
                };
                print_func(column_start, column_end);
                ss << " ...... ";
                print_func(std::max((size_t)0, dim4 - (column_end - column_start)), dim4);
                ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
                FT_LOG(log_level, ss.str());
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
    FT_LOG(log_level, "Buffer %s: shape [%d %d %d %d %d]", hint.c_str(), dim1, dim2, dim3, dim4, dim5);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int x = 0; x < dim4; x++) {
                    std::stringstream ss;
                    ss << "Buffer " << hint << " : ";
                    ss << "[" << i << "," << j << "," << k << "," << x << "]";
                    double sum1       = 0;
                    double sum2       = 0;
                    auto   print_func = [&](size_t column_start, size_t column_end) {
                        for (int y = column_start; y < column_end && y < dim5; y++) {
                            double value = tensor[i][j][k][x][y].item<double>();
                            ss << " y = " << y << " value = " << value;
                            sum1 += value;
                            sum2 += value * value;
                        }
                    };
                    print_func(column_start, column_end);
                    ss << " ...... ";
                    print_func(std::max((size_t)0, dim5 - (column_end - column_start)), dim5);
                    ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
                    FT_LOG(log_level, ss.str());
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
    FT_LOG(log_level, "Buffer %s: shape [%d %d %d %d %d %d]", hint.c_str(), dim1, dim2, dim3, dim4, dim5, dim6);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int x = 0; x < dim4; x++) {
                    for (int y = 0; y < dim5; y++) {
                        std::stringstream ss;
                        ss << "Buffer " << hint << " : ";
                        ss << "[" << i << "," << j << "," << k << "," << x << "," << y << "]";
                        double sum1       = 0;
                        double sum2       = 0;
                        auto   print_func = [&](size_t column_start, size_t column_end) {
                            for (int z = column_start; z < column_end && z < dim6; z++) {
                                double value = tensor[i][j][k][x][y][z].item<double>();
                                ss << " k = " << z << " value = " << value;
                                sum1 += value;
                                sum2 += value * value;
                            }
                        };
                        print_func(column_start, column_end);
                        ss << " ...... ";
                        print_func(std::max((size_t)0, dim6 - (column_end - column_start)), dim6);
                        ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
                        FT_LOG(log_level, ss.str());
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

void printBufferData(const Buffer& buffer, const std::string& hint, DeviceBase* device, bool force_print) {
    const auto log_level = force_print ? alog::LOG_LEVEL_INFO : alog::LOG_LEVEL_TRACE1;

    if (!force_print) {
        if (!rtp_llm::Logger::getEngineLogger().isTraceMode()) {
            return;
        }

        if (buffer.isQBuffer()) {
            FT_LOG_INFO("skip QBuffer [%s]: %s", hint.c_str(), buffer.debugString().c_str());
            return;
        }
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
        FT_LOG(log_level, "%s", ss.str().c_str());
    } else if (dims.size() == 6) {
        printBuffer6d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 5) {
        printBuffer5d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 4) {
        printBuffer4d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 3) {
        printBuffer3d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 2) {
        printBuffer2d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 1) {
        printBuffer1d(hint, tensor, dims, column_start, column_end, max_print_lines, log_level);
    }
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

}  // namespace fastertransformer
