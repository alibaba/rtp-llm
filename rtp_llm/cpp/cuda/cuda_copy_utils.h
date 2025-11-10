#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include <cuda_runtime.h>

namespace rtp_llm {

class CudaCopyUtils {
public:
    static bool gather(std::vector<BufferPtr> src, BufferPtr dst, cudaStream_t stream);
    static bool scatter(BufferPtr src, std::vector<BufferPtr> dst, cudaStream_t stream);

private:
    size_t getDataTypeSize(DataType type) {
        size_t getElementCount(const std::vector<size_t>& shape);
        int    calculateBlockNum(size_t bytes_per_src, size_t num_srcs);
    };
}