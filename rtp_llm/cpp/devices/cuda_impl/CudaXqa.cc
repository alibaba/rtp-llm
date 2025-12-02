#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#include "3rdparty/xqa/mha.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/math_utils.h"
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

static const cudaDeviceProp& getDeviceProp() {
    static cudaDeviceProp prop = []() {
        int device_id;
        check_cuda_value(cudaGetDevice(&device_id));
        cudaDeviceProp device_prop;
        check_cuda_value(cudaGetDeviceProperties(&device_prop, device_id));
        return device_prop;
    }();
    return prop;
}

static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

torch::Tensor getKVCacheScale() {
    float scale          = 1.f;
    auto  kv_cache_scale = torch::tensor({scale}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return kv_cache_scale;
}

torch::Tensor getSemaphores(size_t kv_head_num, size_t group_size, size_t max_q_len, size_t max_batch_size) {
    size_t nb_blocks_per_grp = std::max(ceil_div<size_t>(max_q_len * group_size, M_TILESIZE),
                                        ceil_div<size_t>(max_q_len, M_TILESIZE / group_size));
    size_t sem_size          = kv_head_num * nb_blocks_per_grp * max_batch_size;
    size_t real_sem_size     = round_up<size_t>(sem_size, 2) + 2 + sem_size + 2;

    auto semaphores =
        torch::zeros({static_cast<int64_t>(real_sem_size)}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
    return semaphores;
}

void* getScratch(size_t group_size, uint32_t beam_width) {
    size_t               scratch_size = (256u << 20) * 4;
    static torch::Tensor scratch =
        torch::zeros({static_cast<int64_t>(scratch_size)}, torch::dtype(torch::kUInt8).device(torch::kCUDA));

    auto real_scratch = reinterpret_cast<void*>(
        round_up<uintptr_t>(reinterpret_cast<uintptr_t>(scratch.data_ptr()), ioHeadBytes * group_size * beam_width));

    return real_scratch;
}

torch::Tensor getSpecQMask(size_t max_q_len, size_t max_batch_size) {
    const size_t          num_bits_per_packed_mask   = sizeof(uint32_t) * 8;
    const size_t          num_packed_masks_per_token = ceil_div<size_t>(max_q_len, num_bits_per_packed_mask);
    std::vector<bool>     host_mask(max_q_len * max_q_len);
    std::vector<uint32_t> host_packed_mask(max_batch_size * max_q_len * num_packed_masks_per_token);
    for (uint32_t i = 0; i < max_batch_size; ++i) {
        for (uint32_t token_idx = 0; token_idx < max_q_len; ++token_idx) {
            for (uint32_t kv_pos_idx = 0; kv_pos_idx < max_q_len; ++kv_pos_idx) {
                host_mask[token_idx * max_q_len + kv_pos_idx] = (token_idx >= kv_pos_idx);
            }

            // Pack boolean masks into bits.
            for (uint32_t mask_idx = 0; mask_idx < num_packed_masks_per_token; ++mask_idx) {
                uint32_t packed_mask = 0u;
                for (uint32_t pos_idx = 0; pos_idx < num_bits_per_packed_mask; ++pos_idx) {
                    uint32_t mask_pos_idx = mask_idx * num_bits_per_packed_mask + pos_idx;
                    uint32_t mask_flag    = 0u;
                    if (mask_pos_idx < max_q_len) {
                        mask_flag = host_mask[token_idx * max_q_len + mask_pos_idx];
                    }

                    packed_mask |= mask_flag << pos_idx;
                }
                host_packed_mask[(i * max_q_len + token_idx) * num_packed_masks_per_token + mask_idx] = packed_mask;
            }
        }
    }

    auto q_mask = torch::from_blob(host_packed_mask.data(),
                                   {static_cast<int64_t>(host_packed_mask.size())},
                                   torch::dtype(torch::kUInt32))
                      .to(torch::kCUDA);
    check_cuda_value(cudaStreamSynchronize(getCurrentStream()));

    return q_mask;
}

static std::once_flag xqa_info_flag;

bool supportXqa(DataType input_type,
                DataType output_type,
                DataType kv_cache_type,
                size_t   group_size,
                size_t   head_dim,
                size_t   page_size) {
    bool support = (input_type == DataType::TYPE_BF16 || input_type == DataType::TYPE_FP16)
                   && (output_type == DataType::TYPE_BF16 || output_type == DataType::TYPE_FP16
                       || output_type == DataType::TYPE_FP8_E4M3)
                   && (kv_cache_type == DataType::TYPE_BF16 || kv_cache_type == DataType::TYPE_FP16
                       || kv_cache_type == DataType::TYPE_FP8_E4M3)
                   && (group_size <= 16) && (head_dim == 64 || head_dim == 128 || head_dim == 256)
                   && (page_size == 16 || page_size == 32 || page_size == 64 || page_size == 128);
    if (!support) {
        std::call_once(xqa_info_flag, [&]() {
            RTP_LLM_LOG_WARNING(
                "xqa not supported, in_type:%d out_type:%d kv_cache_type:%d, group_size:%d head_dim:%d page_size:%d",
                int(input_type),
                int(output_type),
                int(kv_cache_type),
                int(group_size),
                int(head_dim),
                int(page_size));
        });
    }

    return support;
}

void runXqa(void*     input,
            bool      is_input_bf16,
            void*     output,
            size_t    head_num,
            size_t    kv_head_num,
            size_t    head_dim,
            size_t    batch_size,
            size_t    max_blocks_per_seq,
            size_t    max_seq_len,
            size_t    page_size,
            void*     kv_cache_pool,
            int32_t*  kv_cache_page_list,
            bool      is_kv_cache_fp8,
            uint32_t* sequence_lengths,
            float*    rcp_out_scale,
            size_t    max_q_len,
            void*     q_cu_seqlens,
            size_t    max_batch_size,
            float     q_scale,
            uint32_t  beam_width) {
    if (!input || !output || !head_num || !kv_head_num || (head_num / kv_head_num > 16)
        || (head_dim != 64 && head_dim != 128 && head_dim != 256) || !batch_size || batch_size > max_batch_size
        || !max_blocks_per_seq || !max_seq_len
        || (page_size != 16 && page_size != 32 && page_size != 64 && page_size != 128) || !kv_cache_pool
        || !kv_cache_page_list || !sequence_lengths) {
        RTP_LLM_LOG_ERROR(
            "xqa params error: input = %p, is_input_bf16 = %d, output = %p, head_num = %zu, kv_head_num = %zu, "
            "head_dim = %zu, batch_size = %zu, max_blocks_per_seq = %zu, max_seq_len = %zu, page_size = %zu, "
            "kv_cache_pool = %p, kv_cache_page_list = %p, is_kv_cache_fp8 = %d, sequence_lengths = %p, "
            "rcp_out_scale = %p, max_q_len = %d, q_cu_seqlens = %p, max_batch_size = %zu, q_scale = %f, beam_width = %zu",
            input,
            is_input_bf16,
            output,
            head_num,
            kv_head_num,
            head_dim,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
            page_size,
            kv_cache_pool,
            kv_cache_page_list,
            is_kv_cache_fp8,
            sequence_lengths,
            rcp_out_scale,
            max_q_len,
            q_cu_seqlens,
            max_batch_size,
            q_scale,
            beam_width);

        return;
    }

    bool is_spec = (max_q_len > 0 && q_cu_seqlens);

    size_t group_size = static_cast<size_t>(head_num / kv_head_num);

    size_t max_seq_len_round = round_up<size_t>(max_seq_len, page_size);

    static torch::Tensor kv_cache_scale = getKVCacheScale();

    static torch::Tensor semaphores = getSemaphores(kv_head_num, group_size, max_q_len, max_batch_size);

    static void* scratch = getScratch(group_size, beam_width);

    static torch::Tensor q_mask = getSpecQMask(max_q_len, max_batch_size);

    static auto device_prop = getDeviceProp();

    SpecDecParams spec_params{
        static_cast<uint32_t>(max_q_len), reinterpret_cast<uint32_t*>(q_cu_seqlens), q_mask.data_ptr<uint32_t>()};

    run_xqa_sm90(static_cast<uint32_t>(head_dim),
                 static_cast<uint32_t>(page_size),
                 static_cast<uint32_t>(group_size),
                 is_input_bf16,
                 is_kv_cache_fp8,
                 device_prop,
                 static_cast<uint32_t>(kv_head_num),
                 q_scale,
                 output,
                 kv_cache_pool,
                 reinterpret_cast<KVCachePageIndex*>(kv_cache_page_list),
                 static_cast<uint32_t>(max_blocks_per_seq),
                 static_cast<uint32_t>(max_seq_len_round),
                 sequence_lengths,
                 static_cast<uint32_t>(batch_size),
                 kv_cache_scale.data_ptr<float>(),
                 semaphores.data_ptr<uint32_t>(),
                 scratch,
                 getCurrentStream(),
                 input,
                 rcp_out_scale,
                 is_spec ? &spec_params : nullptr);

    check_cuda_error();
}

}  // namespace rtp_llm
