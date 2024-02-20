#pragma once

#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/core/Tensor.h"

namespace fastertransformer {

void saveNpy(const Tensor& tensor, const std::string& filename);
Tensor loadNpy(const std::string& npy_file, const MemoryType where);
void saveNpy(TensorMap& tensor_map, const std::string& base_folder);
TensorMap fromNpyFolder(const std::string& base_folder);

} // namespace fastertransformer

