#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include <cuda_runtime.h>
#include "rtp_llm/cpp/cuda/cuda_copy_utils.h"
#include "rtp_llm/cpp/kernels/sm_utils/sm_copy_kernel.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool CudaCopyUtils::gather(std::vector<BufferPtr> src, BufferPtr dst, cudaStream_t stream) {
    // 1. check param
    if (src.empty()) {
        // TODO(qisa.cb) need report error？
        RTP_LLM_LOG_ERROR("call gather with empty src");
        return false;
    }
    if (!dst || !dst->data_) {
        RTP_LLM_LOG_ERROR("call gather with empty dst");
        return false;
    }
    // 2. check src buffers
    const auto& first_src = src[0];
    if (!first_src || !first_src->data_) {
        RTP_LLM_LOG_ERROR("call gather with empty first src");
        return false;
    }
    // check all src buffers with same (where / type / shape)
    for (size_t i = 1; i < src.size(); ++i) {
        if (!src[i] || !src[i]->data_) {
            RTP_LLM_LOG_ERROR("src buffer [%zu] is null", i);
            return false;
        }
        if (src[i]->type_ != first_src->type_) {
            RTP_LLM_LOG_ERROR("src buffer [%zu] data type is not same as first src", i);
            return false;
        }
        if (src[i]->where_ != first_src->where_) {
            RTP_LLM_LOG_ERROR("src buffer [%zu] memory space is not same as first src", i);
        }
        if (src[i]->shape_ != first_src->shape_) {
            RTP_LLM_LOG_ERROR("src buffer [%zu] shape is not same as first src", i);
            return false;
        }
    }

    // 3. calc buffer size
    size_t element_size      = getDataTypeSize(first_src->type_);
    size_t src_element_count = getElementCount(first_src->shape_);
    size_t src_bytes         = src_element_count * element_size;

    // 4. check dst buffer size
    size_t dst_element_count = getElementCount(dst->shape_);
    size_t dst_bytes         = dst_element_count * element_size;

    if (dst_bytes != src_bytes * src.size()) {
        RTP_LLM_LOG_ERROR("Destination buffer size mismatch. Expected: %zu bytes but got %zu bytes. ",
                          src_bytes * src.size(),
                          dst_bytes);
        return false;
    }

    // 5. generate src ptrs
    std::vector<const void*> src_ptrs(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        src_ptrs[i] = static_cast<const void*>(src[i]->data_);
    }

    // 6. call kernel , support ppu & cuda
    launch_gather_copy(src_ptrs.data(),                           // const void** src_ptrs
                       0,                                         // size_t offset
                       src_bytes,                                 // size_t size
                       dst->data(),                               // void* dst
                       static_cast<int>(src_ptrs.size()),         // int num_srcs
                       calculateBlockNum(src_bytes, src.size()),  // int block_num
                       stream                                     // cudaStream_t stream
    );
}

bool CudaCopyUtils::scatter(BufferPtr src, std::vector<BufferPtr> dst, cudaStream_t stream) {
    // 1. check param
    if (dst.empty()) {
        RTP_LLM_LOG_ERROR("call scatter with empty dst");
        return false;
    }
    if (!src || !src->data_) {
        RTP_LLM_LOG_ERROR("call scatter with empty src");
        return false;
    }

    // 2. check dst buffers
    const auto& first_dst = dst[0];
    if (!first_dst || !first_dst->data_) {
        RTP_LLM_LOG_ERROR("call scatter with empty first dst");
        return false;
    }

    // check all dst buffers with same (where / type / shape)
    for (size_t i = 1; i < dst.size(); ++i) {
        if (!dst[i] || !dst[i]->data_) {
            RTP_LLM_LOG_ERROR("dst buffer [%zu] is null", i);
            return false;
        }
        if (dst[i]->type_ != first_dst->type_) {
            RTP_LLM_LOG_ERROR("dst buffer [%zu] data type is not same as first dst", i);
            return false;
        }
        if (dst[i]->where_ != first_dst->where_) {
            RTP_LLM_LOG_ERROR("dst buffer [%zu] memory space is not same as first dst", i);
            return false;
        }
        if (dst[i]->shape_ != first_dst->shape_) {
            RTP_LLM_LOG_ERROR("dst buffer [%zu] shape is not same as first dst", i);
            return false;
        }
    }

    // 3. calc buffer size
    size_t element_size      = getDataTypeSize(first_dst->type_);
    size_t dst_element_count = getElementCount(first_dst->shape_);
    size_t dst_bytes         = dst_element_count * element_size;

    // 4. check src buffer size
    size_t src_element_count = getElementCount(src->shape_);
    size_t src_bytes         = src_element_count * element_size;

    if (src_bytes != dst_bytes * dst.size()) {
        RTP_LLM_LOG_ERROR(
            "Source buffer size mismatch. Expected: %zu bytes but got %zu bytes.", dst_bytes * dst.size(), src_bytes);
        return false;
    }

    // 5. generate dst ptrs
    std::vector<void*> dst_ptrs(dst.size());
    for (size_t i = 0; i < dst.size(); ++i) {
        dst_ptrs[i] = static_cast<void*>(dst[i]->data_);
    }

    // 6. call kernel , support ppu & cuda
    launch_scatter_copy(src->data(),                               // const void* src
                        0,                                         // size_t offset
                        dst_bytes,                                 // size_t size
                        dst_ptrs.data(),                           // void** dst_ptrs
                        static_cast<int>(dst_ptrs.size()),         // int num_dsts
                        calculateBlockNum(dst_bytes, dst.size()),  // int block_num
                        stream                                     // cudaStream_t stream
    );

    return true;
}

size_t CudaCopyUtils::getDataTypeSize(DataType type) {
    switch (type) {
        case DataType::FLOAT32:
            return sizeof(float);
        case DataType::FLOAT16:
            return sizeof(half);
        case DataType::INT32:
            return sizeof(int32_t);
        case DataType::INT64:
            return sizeof(int64_t);
        case DataType::INT8:
            return sizeof(int8_t);
        case DataType::UINT8:
            return sizeof(uint8_t);
        // TODO(qisa.cb) more type...
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}
size_t CudaCopyUtils::getElementCount(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }
    return count;
}

// TODO(qisa.cb) by param?
int CudaCopyUtils::calculateBlockNum(size_t bytes_per_src, size_t num_srcs) {
    constexpr size_t bytes_per_block = 256;
    size_t           total_bytes     = bytes_per_src * num_srcs;
    size_t           num_blocks      = (total_bytes + bytes_per_block - 1) / bytes_per_block;
    constexpr size_t max_blocks      = 65535;
    return static_cast<int>(std::min(num_blocks, max_blocks));
}

}  // namespace rtp_llm