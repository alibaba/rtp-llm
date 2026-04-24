#include <c10/core/TensorImpl.h>
#include <sstream>
#include <fstream>
#include <torch/torch.h>
#include <filesystem>
#include <regex>

#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"

namespace fs = std::filesystem;

namespace rtp_llm {

void printTorchBufferSample(const std::string& hint, torch::Tensor& tensor, uint32_t n_samples) {
    const uint32_t SEED = 10086;

    auto    flat  = tensor.to(torch::kCPU).to(torch::kF32).view({-1}).contiguous();
    int64_t total = flat.numel();

    // LCG状态变量
    uint64_t       lcg_state = SEED;
    const uint64_t LCG_A     = 1103515245ULL;
    const uint64_t LCG_C     = 12345ULL;
    const uint64_t LCG_MOD   = 0x7fffffffULL;

    std::stringstream ss;
    ss << "[BufferSample] " << hint << " (Shape: " << tensor.sizes() << ", total: " << total << "): ";

    size_t cnt = 0;
    while (cnt < n_samples) {
        // LCG生成下一个随机索引
        lcg_state    = (lcg_state * LCG_A + LCG_C) & LCG_MOD;
        uint64_t idx = lcg_state % total;

        float val = flat[idx].item<float>();
        ss << std::fixed << std::setprecision(6) << val << " ";
        ++cnt;
    }

    RTP_LLM_LOG_INFO(ss.str());
}

void printBuffer1d(const std::string&  hint,
                   torch::Tensor&      tensor,
                   std::vector<size_t> dims,
                   size_t              column_start,
                   size_t              column_end,
                   size_t              max_print_lines,
                   int                 log_level,
                   bool                show_stats_only) {
    size_t dim1 = dims[0];
    RTP_LLM_LOG(
        log_level, "Buffer %s: shape [%d], type: %s", hint.c_str(), dim1, tensor.options().dtype().name().data());
    std::stringstream ss;
    ss << "Buffer " << hint << " : ";
    auto print_func = [&](size_t column_start, size_t column_end) {
        for (int i = column_start; i < column_end && i < dim1; i++) {
            double value = tensor[i].item<double>();
            ss << " i = " << i << " value = " << value;
        }
    };
    const auto [sum1, sum2] = calculateTensorSum([&](size_t i) -> auto { return tensor[i]; },  // 访问器
                                                 dim1                                          // 当前维度长度
    );
    if (!show_stats_only) {
        print_func(column_start, column_end);
        ss << " ...... ";
        print_func(std::max((size_t)0, dim1 - (column_end - column_start)), dim1);
    }

    ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
    RTP_LLM_LOG(log_level, ss.str());
}

void printBuffer2d(const std::string&  hint,
                   torch::Tensor&      tensor,
                   std::vector<size_t> dims,
                   size_t              column_start,
                   size_t              column_end,
                   size_t              max_print_lines,
                   int                 log_level,
                   bool                show_stats_only) {
    size_t dim1 = dims[0];
    size_t dim2 = dims[1];
    RTP_LLM_LOG(log_level,
                "Buffer %s: shape [%d %d], type: %s",
                hint.c_str(),
                dim1,
                dim2,
                tensor.options().dtype().name().data());
    size_t first_lines_start = dim1 < max_print_lines ? dim1 : max_print_lines;
    size_t last_lines_start  = dim1 - max_print_lines > first_lines_start ? dim1 - max_print_lines : first_lines_start;
    for (int i = 0; i < first_lines_start; i++) {
        std::stringstream ss;
        ss << "Buffer " << hint << " : ";
        ss << "[" << i << "]";
        if (!show_stats_only) {
            auto print_func = [&](size_t column_start, size_t column_end) {
                for (int j = column_start; j < column_end && j < dim2; j++) {
                    double value = tensor[i][j].item<double>();
                    ss << " k = " << j << " value = " << value;
                }
            };
            print_func(column_start, column_end);
            ss << " ...... ";
            print_func(std::max((size_t)0, dim2 - (column_end - column_start)), dim2);
        }
        const auto [sum1, sum2] = calculateTensorSum([&](size_t j) -> auto { return tensor[i][j]; }, dim2);
        ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
        RTP_LLM_LOG(log_level, ss.str());
    }
    for (int i = last_lines_start; i < dim1; i++) {
        std::stringstream ss;
        ss << "Buffer " << hint << " : ";
        ss << "[" << i << "]";
        if (!show_stats_only) {
            auto print_func = [&](size_t column_start, size_t column_end) {
                for (int j = column_start; j < column_end && j < dim2; j++) {
                    double value = tensor[i][j].item<double>();
                    ss << " k = " << j << " value = " << value;
                }
            };
            print_func(column_start, column_end);
            ss << " ...... ";
            print_func(std::max((size_t)0, dim2 - (column_end - column_start)), dim2);
        }
        const auto [sum1, sum2] = calculateTensorSum([&](size_t j) -> auto { return tensor[i][j]; }, dim2);
        ss << " sum1 = " << sum1 << ", square sum2 = " << sum2;
        RTP_LLM_LOG(log_level, ss.str());
    }
}

void printBuffer3d(const std::string&  hint,
                   torch::Tensor&      tensor,
                   std::vector<size_t> dims,
                   size_t              column_start,
                   size_t              column_end,
                   size_t              max_print_lines,
                   int                 log_level,
                   bool                show_stats_only) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level,
                "Buffer %s: shape [%d %d %d], type: %s",
                hint.c_str(),
                dim1,
                dim2,
                dim3,
                tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            std::stringstream ss;
            ss << "Buffer " << hint << " : ";
            ss << "[" << i << ", " << j << "]";
            if (!show_stats_only) {
                auto print_func = [&](size_t column_start, size_t column_end) {
                    for (int k = column_start; k < column_end && k < dim3; k++) {
                        double value = tensor[i][j][k].item<double>();
                        ss << " k = " << k << " value = " << value;
                    }
                };
                print_func(column_start, column_end);
                ss << " ...... ";
                print_func(std::max((size_t)0, dim3 - (column_end - column_start)), dim3);
            }
            const auto [sum1, sum2] = calculateTensorSum([&](size_t k) -> auto { return tensor[i][j][k]; }, dim3);
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
                   int                 log_level) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t dim4     = dims[3];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level,
                "Buffer %s: shape [%d %d %d %d], type: %s",
                hint.c_str(),
                dim1,
                dim2,
                dim3,
                dim4,
                tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                std::stringstream ss;
                ss << "Buffer " << hint << " : ";
                ss << "[" << i << "," << j << "," << k << "]";
                auto print_func = [&](size_t column_start, size_t column_end) {
                    for (int x = column_start; x < column_end && x < dim4; x++) {
                        double value = tensor[i][j][k][x].item<double>();
                        ss << " k = " << x << " value = " << value;
                    }
                };
                const auto [sum1, sum2] =
                    calculateTensorSum([&](size_t x) -> auto { return tensor[i][j][k][x]; }, dim4);
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
                   int                 log_level) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t dim4     = dims[3];
    size_t dim5     = dims[4];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level,
                "Buffer %s: shape [%d %d %d %d %d], type: %s",
                hint.c_str(),
                dim1,
                dim2,
                dim3,
                dim4,
                dim5,
                tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int x = 0; x < dim4; x++) {
                    std::stringstream ss;
                    ss << "Buffer " << hint << " : ";
                    ss << "[" << i << "," << j << "," << k << "," << x << "]";
                    auto print_func = [&](size_t column_start, size_t column_end) {
                        for (int y = column_start; y < column_end && y < dim5; y++) {
                            double value = tensor[i][j][k][x][y].item<double>();
                            ss << " y = " << y << " value = " << value;
                        }
                    };
                    const auto [sum1, sum2] =
                        calculateTensorSum([&](size_t y) -> auto { return tensor[i][j][k][x][y]; }, dim5);
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
                   int                 log_level) {
    size_t dim1     = dims[0];
    size_t dim2     = dims[1];
    size_t dim3     = dims[2];
    size_t dim4     = dims[3];
    size_t dim5     = dims[4];
    size_t dim6     = dims[5];
    size_t line_num = 0;
    RTP_LLM_LOG(log_level,
                "Buffer %s: shape [%d %d %d %d %d %d], type: %s",
                hint.c_str(),
                dim1,
                dim2,
                dim3,
                dim4,
                dim5,
                dim6,
                tensor.options().dtype().name().data());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int x = 0; x < dim4; x++) {
                    for (int y = 0; y < dim5; y++) {
                        std::stringstream ss;
                        ss << "Buffer " << hint << " : ";
                        ss << "[" << i << "," << j << "," << k << "," << x << "," << y << "]";
                        auto print_func = [&](size_t column_start, size_t column_end) {
                            for (int z = column_start; z < column_end && z < dim6; z++) {
                                double value = tensor[i][j][k][x][y][z].item<double>();
                                ss << " k = " << z << " value = " << value;
                            }
                        };
                        const auto [sum1, sum2] =
                            calculateTensorSum([&](size_t z) -> auto { return tensor[i][j][k][x][y][z]; }, dim6);
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

void printTorchTensorData_(const torch::Tensor& tensor, const std::string& hint, bool show_stats_only) {
    cudaCheckLastError();

    const auto log_level = alog::LOG_LEVEL_INFO;
    auto       cpu_t     = tensor.cpu();

    std::vector<size_t> dims;
    for (int64_t i = 0; i < cpu_t.dim(); i++) {
        dims.push_back(cpu_t.size(i));
    }
    size_t column_start    = 0;
    size_t column_end      = 20;
    size_t max_print_lines = 20;
    if (dims.size() > 6) {
        std::stringstream ss;
        ss << "Tensor " << hint << " : " << cpu_t;
        RTP_LLM_LOG(log_level, "%s", ss.str().c_str());
    } else if (dims.size() == 6) {
        printBuffer6d(hint, cpu_t, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 5) {
        printBuffer5d(hint, cpu_t, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 4) {
        printBuffer4d(hint, cpu_t, dims, column_start, column_end, max_print_lines, log_level);
    } else if (dims.size() == 3) {
        printBuffer3d(hint, cpu_t, dims, column_start, column_end, max_print_lines, log_level, show_stats_only);
    } else if (dims.size() == 2) {
        printBuffer2d(hint, cpu_t, dims, column_start, column_end, max_print_lines, log_level, show_stats_only);
    } else if (dims.size() == 1) {
        printBuffer1d(hint, cpu_t, dims, column_start, column_end, max_print_lines, log_level, show_stats_only);
    }
}

void saveTorchDataTofile(const torch::Tensor& tensor, const std::string& fileName) {
    auto          tensor_cpu = tensor.contiguous().cpu();
    auto          pickled    = torch::pickle_save(tensor_cpu);
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

void saveTensorData_(const torch::Tensor& tensor, const std::string& fileName, const std::string& sourceFile) {
#if (defined(USING_ROCM) || defined(USE_CUDA))
    const auto log_level = alog::LOG_LEVEL_INFO;

    auto shouldFilter = [&]() {
        static const std::vector<std::string> filters = {"cuda_impl", "rocm_impl"};
        for (const auto& keyword : filters) {
            if (sourceFile.find(keyword) != std::string::npos)
                return true;
        }
        return false;
    };

    if (shouldFilter()) {
        RTP_LLM_LOG(log_level, "filter source file : %s", sourceFile.c_str());
        return;
    }

    auto prepareOutputPath = [&]() -> std::string {
        namespace fs = std::filesystem;

        std::string folderName = fs::exists("/dev/nvidia0") ? "H20" : "308X";
        fs::path    folderPath = fs::current_path() / folderName;

        if (!fs::exists(folderPath)) {
            fs::create_directory(folderPath);
        }

        int count = std::distance(fs::directory_iterator(folderPath), fs::directory_iterator{});

        std::string cleanedFileName = std::regex_replace(fileName, std::regex(R"([\\/:*?"<>|'\s])"), "_") + "_count"
                                      + std::to_string(count) + ".pt";

        return (folderPath / cleanedFileName).string();
    };

    std::string outPath = prepareOutputPath();

    RTP_LLM_LOG(log_level, "saving tensor to file: %s", outPath.c_str());

    std::ofstream ofs(outPath, std::ios::out | std::ios::binary);
    if (ofs.is_open()) {
        auto cpu_tensor = tensor.cpu();
        if (cpu_tensor.dtype() == torch::kFloat8_e4m3fn || cpu_tensor.dtype() == torch::kFloat8_e4m3fnuz) {
            cpu_tensor = cpu_tensor.view(torch::kChar);
        }
        auto pickled = torch::pickle_save(cpu_tensor);
        ofs.write(pickled.data(), pickled.size());
        ofs.close();
    } else {
        throw std::runtime_error("Failed to open file: " + outPath);
    }
#endif
}

}  // namespace rtp_llm
