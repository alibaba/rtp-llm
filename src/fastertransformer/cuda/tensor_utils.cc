#include "src/fastertransformer/cuda/tensor_utils.h"

namespace fastertransformer {

DataType typeFromNumpyDesc(std::string type) {
    static const std::unordered_map<std::string, DataType> type_map{{"?", TYPE_BOOL},
                                                                    {"b", TYPE_BYTES},
                                                                    {"u1", TYPE_UINT8},
                                                                    {"u2", TYPE_UINT16},
                                                                    {"u4", TYPE_UINT32},
                                                                    {"u8", TYPE_UINT64},
                                                                    {"i1", TYPE_INT8},
                                                                    {"i2", TYPE_INT16},
                                                                    {"i4", TYPE_INT32},
                                                                    {"i8", TYPE_INT64},
                                                                    {"f2", TYPE_FP16},
                                                                    {"f4", TYPE_FP32},
                                                                    {"f8", TYPE_FP64}};
    return type_map.at(type);
}

void parseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data) {
    const char magic[]                   = "\x93"
                                           "NUMPY";
    char       magic_test[sizeof(magic)] = "\0";

    size_t n_elems = fread((void*)magic_test, sizeof(char), sizeof(magic) - 1, f_ptr);
    if (n_elems != sizeof(magic) - 1 || std::string(magic) != std::string(magic_test)) {
        throw std::runtime_error("Could read magic token in NPY file");
    }

    uint8_t npy_major = 0;
    uint8_t npy_minor = 0;
    n_elems           = fread((void*)&npy_major, sizeof(uint8_t), 1, f_ptr);
    n_elems += fread((void*)&npy_minor, sizeof(uint8_t), 1, f_ptr);

    if (npy_major == 1) {
        uint16_t header_len_u16 = 0;
        n_elems                 = fread((void*)&header_len_u16, sizeof(uint16_t), 1, f_ptr);
        header_len              = header_len_u16;
    } else if (npy_major == 2) {
        uint32_t header_len_u32 = 0;
        n_elems                 = fread((void*)&header_len_u32, sizeof(uint32_t), 1, f_ptr);
        header_len              = header_len_u32;
    } else {
        throw std::runtime_error("Unsupported npy version: " + std::to_string(npy_major));
    }

    start_data = 8 + 2 * npy_major + header_len;
}

int parseNpyHeader(FILE*& f_ptr, uint32_t header_len, DataType& type, std::vector<size_t>& shape) {
    char*  header_c = (char*)malloc(header_len * sizeof(char));
    size_t n_elems  = fread((void*)header_c, sizeof(char), header_len, f_ptr);
    if (n_elems != header_len) {
        free(header_c);
        return -1;
    }
    std::string header(header_c, header_len);
    free(header_c);

    size_t start, end;
    start = header.find("'descr'") + 7;
    start = header.find("'", start);
    end   = header.find("'", start + 1);
    type  = typeFromNumpyDesc(header.substr(start + 1, end - start - 1));

    start = header.find("'fortran_order'") + 15;
    start = header.find(":", start);
    end   = header.find(",", start + 1);
    if (header.substr(start + 1, end - start - 1).find("False") == std::string::npos) {
        throw std::runtime_error("Unsupported value for fortran_order while reading npy file");
    }

    start = header.find("'shape'") + 7;
    start = header.find("(", start);
    end   = header.find(")", start + 1);

    std::istringstream shape_stream(header.substr(start + 1, end - start - 1));
    std::string        token;

    shape.clear();
    while (std::getline(shape_stream, token, ',')) {
        if (token.find_first_not_of(' ') == std::string::npos) {
            break;
        }
        shape.push_back(std::stoul(token));
    }

    return 0;
}


void saveNpy(const Tensor& tensor, const std::string& filename) {
    // Save tensor to NPY 1.0 format (see https://numpy.org/neps/nep-0001-npy-format.html)
    void*  cpu_data     = (void*)tensor.data();
    bool   is_data_temp = false;
    size_t tensor_size  = tensor.size();
    if (tensor.where() == MemoryType::MEMORY_GPU) {
        cpu_data     = malloc(tensor_size * Tensor::getTypeSize(tensor.type()));
        is_data_temp = true;
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_data, tensor.data(), tensor_size * Tensor::getTypeSize(tensor.type()), cudaMemcpyDeviceToHost);
    }

    const char    magic[]   = "\x93"
                              "NUMPY";
    const uint8_t npy_major = 1;
    const uint8_t npy_minor = 0;

    std::stringstream header_stream;
    header_stream << "{'descr': '" << tensor.getNumpyTypeDesc(tensor.type()) << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        header_stream << tensor.shape()[i];
        if (i + 1 < tensor.shape().size() || tensor.shape().size() == 1) {
            header_stream << ", ";
        }
    }
    header_stream << ")}";
    int base_length = 6 + 4 + header_stream.str().size();
    int pad_length  = 16 * ((base_length + 1 + 15) / 16);  // Take ceiling of base_length + 1 (for '\n' ending)
    for (int i = 0; i < pad_length - base_length; ++i) {
        header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
    }
    std::string    header     = header_stream.str();
    const uint16_t header_len = header.size();

    FILE* f_ptr = fopen(filename.c_str(), "wb");
    FT_CHECK_WITH_INFO(f_ptr != nullptr, fmtstr("Unable to open %s for writing.\n", filename.c_str()));

    fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
    fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
    fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
    fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
    fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
    fwrite(cpu_data, Tensor::getTypeSize(tensor.type()), tensor_size, f_ptr);

    fclose(f_ptr);

    if (is_data_temp) {
        free(cpu_data);
    }
}

Tensor loadNpy(const std::string& npy_file, const MemoryType where) {
        DataType            type;
    std::vector<size_t> shape;

    FILE* f_ptr = fopen(npy_file.c_str(), "rb");
    if (f_ptr == nullptr) {
        throw std::runtime_error("Could not open file " + npy_file);
    }
    uint32_t header_len, start_data;
    parseNpyIntro(f_ptr, header_len, start_data);
    parseNpyHeader(f_ptr, header_len, type, shape);

    auto tensor = Tensor(where, type, shape, (void*)nullptr);

    const size_t size     = tensor.size();
    void*        data_cpu = (where == MEMORY_CPU) ? const_cast<void*>(tensor.data()) : malloc(tensor.sizeBytes());
    void*        data     = data_cpu;

    size_t n_elems = fread(data_cpu, Tensor::getTypeSize(type), size, f_ptr);
    FT_CHECK_WITH_INFO(n_elems == size, "reading tensor failed");

    if (where == MEMORY_GPU) {
        cudaMemcpy(const_cast<void*>(tensor.data()), data_cpu, tensor.sizeBytes(), cudaMemcpyHostToDevice);
        free(data_cpu);
    }

    fclose(f_ptr);
    return std::move(tensor);
}

void saveNpy(TensorMap& tensor_map, const std::string& base_folder) {
    mode_t mode_0755 = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
    int    ret       = mkdir(base_folder.c_str(), mode_0755);
    FT_CHECK_WITH_INFO(ret == 0 || errno == EEXIST, fmtstr("Could not create folder %s.\n", base_folder.c_str()));

    for (const auto& item : tensor_map) {
        saveNpy(item.second, base_folder + "/" + item.second.whereToString() + "-" + item.first + ".npy");
    }
}

TensorMap fromNpyFolder(const std::string& base_folder) {
    DIR* dir_p = opendir(base_folder.c_str());
    FT_CHECK_WITH_INFO(dir_p != nullptr, fmtstr("Could not open folder %s. ", base_folder.c_str()));
    struct dirent* dp;

    TensorMap ret_tensor;
    while ((dp = readdir(dir_p)) != nullptr) {
        std::string filename(dp->d_name);
        size_t      len = filename.length();
        if (len < 4 || filename.compare(len - 4, 4, ".npy")) {
            continue;
        }

        size_t pos = filename.find('-');
        FT_CHECK_WITH_INFO(pos != std::string::npos, fmtstr("Invalid filename: %s\n", filename.c_str()));

        MemoryType where;
        if (filename.compare(0, pos, "GPU") == 0) {
            where = MEMORY_GPU;
        } else if (filename.compare(0, pos, "CPU") == 0) {
            where = MEMORY_CPU;
        } else if (filename.compare(0, pos, "CPU_PINNED") == 0) {
            where = MEMORY_CPU_PINNED;
        } else {
            FT_CHECK_WITH_INFO(false, fmtstr("Invalid filename: %s\n", filename.c_str()));
        }
        std::string key = filename.substr(pos + 1, len - pos - 5);

        ret_tensor.insert({key, loadNpy(base_folder + "/" + filename, where)});
    }

    closedir(dir_p);

    return ret_tensor;
}

}
