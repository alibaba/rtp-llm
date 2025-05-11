#pragma once

#include "maga_transformer/cpp/devices/DeviceBase.h"
#include "maga_transformer/cpp/cuda/cuda_utils.h"
#include "3rdparty/xqa/mha.h"

namespace rtp_llm {

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
            float q_scale = 1.f,
            size_t max_decode_batch_size = 256,
            uint32_t beam_width = beamWidth);

}
