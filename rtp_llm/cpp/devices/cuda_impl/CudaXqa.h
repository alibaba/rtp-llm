#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#include "3rdparty/xqa/mha.h"

namespace rtp_llm {

/**
 * @brief 
 * 
 * @param input_type bf16
 * @param output_type bf16
 * @param kv_cache_type fp8e4m3
 * @param group_size 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
 * @param head_dim 128
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
 * @param decode_batch_size 
 * @param decode_max_seq_len max kv seq len
 * @param page_size 
 * @param kv_cache_pool head ptr [block_nums, kv_head_num, page_size, head_dim]
 * @param kv_cache_page_list [batch_size, beam_width, 2, max_pages_per_seq]
 * @param sequence_lengths kv seq len
 * @param device 
 * @param rope_theta 
 * @param max_position_embeddings 
 * @param q_scale 
 * @param max_decode_batch_size for semaphores
 * @param beam_width 
 */
void runXqa(void* input,
            void* output,
            size_t head_num,
            size_t kv_head_num,
            size_t head_dim,
            size_t decode_batch_size,
            size_t decode_max_seq_len,
            size_t page_size,
            void* kv_cache_pool,
            int32_t* kv_cache_page_list,
            uint32_t* sequence_lengths,
            CudaDevice *device,
            int rope_theta = 1000000,
            int max_position_embeddings = 40960,
            float q_scale = 1.f,
            size_t max_decode_batch_size = 1024,
            uint32_t beam_width = beamWidth);

}
