#pragma once

#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/models/GptModel.h"

namespace ft = fastertransformer;

namespace rtp_llm {

inline void tpSync(GptModelInputs &inputs, ft::NcclParam tensor_para, ft::NcclParam pipeline_para, ft::DeviceBase* device) {
    // TODO: use device nccl not cuda
    cudaStream_t stream = dynamic_cast<ft::CudaDevice*>(device)->stream();
    if (tensor_para.world_size_ <= 1) {
        return;
    }
    const size_t shape_hints_size = 6;
    auto shape_hints = device->allocateBuffer({ft::DataType::TYPE_INT32, {shape_hints_size}, ft::AllocationType::HOST});
    auto shape_hints_ptr = shape_hints->data<int32_t>();
    shape_hints_ptr[0] = inputs.combo_tokens.get() ? inputs.combo_tokens->size() : 0; // combo_token_size
    shape_hints_ptr[1] = inputs.combo_tokens.get() ? inputs.input_lengths->size() : 0; // total_batch_size
    shape_hints_ptr[2] = inputs.combo_tokens.get() ? inputs.sequence_lengths->size() : 0; // generate_batch_size
    shape_hints_ptr[3] = inputs.combo_tokens.get() ? inputs.kv_cache_blocks->shape()[0] : 0; // layer_num
    shape_hints_ptr[4] = inputs.combo_tokens.get() ? inputs.kv_cache_blocks->shape()[3] : 0; // block_size
    shape_hints_ptr[5] = inputs.kv_cache_scales.get() != nullptr; // use_block_scale
    ft::ftNcclBroadCast<char>((char*)shape_hints_ptr, shape_hints->sizeBytes(), 0, tensor_para, stream);
    ft::ftNcclStreamSynchronize(tensor_para, pipeline_para, stream, false);
    cudaStreamSynchronize(stream);
    sync_check_cuda_error();
    if (tensor_para.rank_) {
        inputs.combo_tokens = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[0]}, ft::AllocationType::HOST});
        inputs.input_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[1]}, ft::AllocationType::HOST});
        inputs.sequence_lengths = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[2]}, ft::AllocationType::HOST});
        inputs.kv_cache_blocks = device->allocateBuffer({ft::DataType::TYPE_UINT64, {(size_t)shape_hints_ptr[3], (size_t)shape_hints_ptr[1], 2, (size_t)shape_hints_ptr[4]}, ft::AllocationType::HOST});
        if (shape_hints_ptr[5]) {
            inputs.kv_cache_scales = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[3], (size_t)shape_hints_ptr[0], 2, (size_t)shape_hints_ptr[4]}, ft::AllocationType::HOST});
        }
    }
    std::vector<ft::Buffer *> buffers;
    buffers.emplace_back(inputs.combo_tokens.get());
    buffers.emplace_back(inputs.input_lengths.get());
    buffers.emplace_back(inputs.sequence_lengths.get());
    buffers.emplace_back(inputs.kv_cache_blocks.get());
    if (shape_hints_ptr[5]) {
        buffers.emplace_back(inputs.kv_cache_scales.get());
    }
    ft::ftNcclGroupStart();
    for (auto buffer : buffers) {
        ft::ftNcclBroadCast<char>((char *)buffer->data(), buffer->sizeBytes(), 0, tensor_para, stream);
    }
    ft::ftNcclGroupEnd();
    ft::ftNcclStreamSynchronize(tensor_para, pipeline_para, stream);
    sync_check_cuda_error();
    cudaStreamSynchronize(stream);
}
 
}  // namespace rtp_llm
