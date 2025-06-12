#pragma once

#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "3rdparty/xqa/mha.h"

namespace rtp_llm {

/**
 * @brief 
 * 
 * @param input_type bf16
 * @param output_type bf16
 * @param kv_cache_type fp8e4m3
 * @param group_size 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
 * @param head_dim 64, 128, 256
 * @param page_size 16, 32, 64, 128
 * @return true 
 * @return false 
 */
bool supportXqa(DataType input_type,
                DataType output_type,
                DataType kv_cache_type,
                size_t group_size,
                size_t head_dim,
                size_t page_size);

/**
 * @brief run xqa for decoding attention
 * 
 * @param input q [batch_size, beam_width, head_num, head_dim] or 
 *              qkv [batch_size, beam_width, head_num + kv_head_num * 2, head_dim]
 * @param output [batch_size, beam_width, head_num, head_dim]
 * @param head_num 
 * @param kv_head_num 
 * @param head_dim 
 * @param batch_size 
 * @param max_seq_len max kv seq len
 * @param page_size 
 * @param kv_cache_pool head ptr [block_nums, kv_head_num, page_size, head_dim]
 * @param kv_cache_page_list [batch_size, beam_width, 2, max_pages_per_seq]
 * @param sequence_lengths kv seq len
 * @param device 
 * @param rope_dim 
 * @param rope_theta 
 * @param max_q_len max q seqlen
 * @param q_cu_seqlens accumulate q seqlen 
 * @param max_position_embeddings 
 * @param q_scale 
 * @param max_batch_size for semaphores
 * @param beam_width 
 */
void runXqa(void* input,
            void* output,
            size_t head_num,
            size_t kv_head_num,
            size_t head_dim,
            size_t batch_size,
            size_t max_seq_len,
            size_t page_size,
            void* kv_cache_pool,
            int32_t* kv_cache_page_list,
            uint32_t* sequence_lengths,
            CudaDevice *device,
            size_t      max_q_len,
            void*       q_cu_seqlens,
            int rope_dim = 128,
            int rope_theta = 1000000,
            int max_position_embeddings = 128000,
            float q_scale = 1.f,
            size_t max_batch_size = 1024,
            uint32_t beam_width = beamWidth);

/**
 * @brief 
 * 
 * @tparam rope_dim 
 * @param device 
 * @param rope_theta 
 * @param max_position_embeddings 
 * @return BufferPtr 
 */
template <int rope_dim>
BufferPtr genRopeCosSin(CudaDevice* device, int rope_theta, int max_position_embeddings) {
    auto inv_freq = 1.0 / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(max_position_embeddings, torch::kInt64).to(torch::kFloat32);
    auto freqs = torch::outer(t, inv_freq);
    auto cos = freqs.cos().to(torch::kFloat32);
    auto sin = freqs.sin().to(torch::kFloat32);
    auto emb = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();

    BufferPtr rope_cos_sin = device->allocateBuffer({DataType::TYPE_UINT8,
                                                    {max_position_embeddings * sizeof(Vec<float, rope_dim>)},
                                                    AllocationType::DEVICE},
                                                    {"rope_cos_sin"});
    auto rope_cos_sin_ptr = reinterpret_cast<Vec<float, rope_dim>*>(rope_cos_sin->data());
    for (size_t i = 0; i < max_position_embeddings; ++i) {
        check_cuda_value(cudaMemcpyAsync(&(rope_cos_sin_ptr[i].data[0]),
                                           reinterpret_cast<char*>(emb.data_ptr()) + i * sizeof(float) * rope_dim,
                                           sizeof(float) * rope_dim,
                                           cudaMemcpyHostToDevice,
                                           device->getStream()));
    }
    check_cuda_value(cudaStreamSynchronize(device->getStream()));

    return rope_cos_sin;
}

}
