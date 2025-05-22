#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#include "3rdparty/xqa/mha.h"

namespace rtp_llm {

/**
 * @brief run xqa for decoding attention
 * 
 * @param input q [batch_size, beam_width, head_num, head_dim] or 
 *              qkv [batch_size, beam_width, head_num + kv_head_num * 2, head_dim]
 * @param output [batch_size, beam_width, head_num, head_dim]
 * @param head_num 
 * @param kv_head_num 
 * @param decode_batch_size 
 * @param decode_max_seq_len max kv seq len
 * @param tokens_per_block 
 * @param kv_cache_pool head ptr [block_nums, kv_head_num, tokens_per_block, head_dim]
 * @param kv_cache_page_list [batch_size, beam_width, 2, max_pages_per_seq]
 * @param sequence_lengths kv seq len
 * @param device 
 * @param rope_theta 
 * @param rope_dim 
 * @param max_position_embeddings 
 * @param q_scale 
 * @param max_decode_batch_size for semaphores
 * @param beam_width 
 */
void runXqa(void* input,
            void* output,
            size_t head_num,
            size_t kv_head_num,
            size_t decode_batch_size,
            size_t decode_max_seq_len,
            size_t tokens_per_block,
            void* kv_cache_pool,
            int32_t* kv_cache_page_list,
            uint32_t* sequence_lengths,
            CudaDevice *device,
            int rope_theta = 1000000,
            int rope_dim = validElemsPerHead,
            int max_position_embeddings = 40960,
            float q_scale = 1.f,
            size_t max_decode_batch_size = 1024,
            uint32_t beam_width = beamWidth);

}
