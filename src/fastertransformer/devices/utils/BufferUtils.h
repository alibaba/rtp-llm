#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template <typename T>
Buffer vector2Buffer(const std::vector<T>& vec) {
    const auto& shape = {vec.size()};
    const auto& dtype = getTensorType<T>();
    const auto& memory_type = MemoryType::MEMORY_CPU;
    return Buffer(memory_type, dtype, shape, vec.data());
};

} // namespace fastertransformer
