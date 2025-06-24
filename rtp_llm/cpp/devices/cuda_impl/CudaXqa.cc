#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

BufferPtr genNormalCosSin(CudaDevice* device, int rope_dim, int rope_theta, int max_position_embeddings) {
    auto inv_freq = 1.0 / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(max_position_embeddings, torch::kInt64).to(torch::kFloat32);
    auto freqs = torch::outer(t, inv_freq);
    auto cos = freqs.cos().to(torch::kFloat32);
    auto sin = freqs.sin().to(torch::kFloat32);
    auto cos_sin = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();

    BufferPtr cos_sin_d = device->allocateBuffer({DataType::TYPE_FP32,
                                                 {static_cast<size_t>(rope_dim * max_position_embeddings)},
                                                 AllocationType::DEVICE},
                                                 {"cos_sin_d"});
    check_cuda_value(cudaMemcpyAsync(cos_sin_d->data(),
                                     cos_sin.data_ptr(),
                                     rope_dim * max_position_embeddings * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     device->getStream()));
    check_cuda_value(cudaStreamSynchronize(device->getStream()));

    return cos_sin_d;
}

BufferPtr getRopeCosSin(CudaDevice* device, RopeStyle rope_style, int rope_dim, int rope_theta, int max_position_embeddings) {
    BufferPtr cos_sin = nullptr;

    switch (rope_style) {
        case RopeStyle::No:
            break;

        case RopeStyle::Base:
            cos_sin = genNormalCosSin(device, rope_dim, rope_theta, max_position_embeddings);
            break;

        default:
            RTP_LLM_LOG_ERROR("unsupported rope_style = ", rope_style);
            break;
    }

    return cos_sin;
}

BufferPtr getKVCacheScale(CudaDevice* device) {
    float scale = 1.;
    BufferPtr kv_cache_scale = device->allocateBuffer({DataType::TYPE_FP32, {1}, AllocationType::DEVICE}, {"kv_cache_scale"});
    check_cuda_value(cudaMemcpyAsync(kv_cache_scale->data(), &scale, sizeof(float), cudaMemcpyHostToDevice, device->getStream()));
    check_cuda_value(cudaStreamSynchronize(device->getStream()));

    return kv_cache_scale;
}

BufferPtr getSemaphores(CudaDevice* device, size_t kv_head_num, size_t max_batch_size, size_t max_q_len = 0, size_t group_size = 0, bool spec_dec = false) {
    size_t real_sem_size = 0;
    if (spec_dec) {
        size_t nb_blocks_per_grp = std::max(div_up(max_q_len * group_size, M_TILESIZE), div_up(max_q_len, M_TILESIZE / group_size));
        real_sem_size = kv_head_num * nb_blocks_per_grp * max_batch_size;
    } else {
        size_t sem_size = kv_head_num * max_batch_size;
        real_sem_size = round_up<size_t>(sem_size, 2) + 2 + sem_size + 2;
    }
    BufferPtr semaphores = device->allocateBuffer({DataType::TYPE_UINT32, {real_sem_size}, AllocationType::DEVICE}, {"semaphores"});
    device->bufMemset(*semaphores, 0);

    return semaphores;
}

void* getScratch(CudaDevice* device, size_t group_size, uint32_t beam_width) {
    size_t scratch_size = (256u << 20) * 4;
    static BufferPtr scratch = device->allocateBuffer({DataType::TYPE_BYTES, {scratch_size}, AllocationType::DEVICE}, {"scratch"});
    device->bufMemset(*scratch, 0);
    auto real_scratch = 
        reinterpret_cast<void*>(round_up<uintptr_t>(reinterpret_cast<uintptr_t>(scratch->data()), ioHeadBytes * group_size * beam_width));

    return real_scratch;
}

bool supportXqa(DataType input_type,
                DataType output_type,
                DataType kv_cache_type,
                size_t group_size,
                size_t head_dim,
                size_t page_size) {
    return (input_type == DataType::TYPE_BF16) && (output_type == DataType::TYPE_BF16) && (kv_cache_type == DataType::TYPE_FP8_E4M3) &&
           (group_size <= 16) && (head_dim == 64 || head_dim == 128 || head_dim == 256) &&
           (page_size == 16 || page_size == 32 || page_size == 64 || page_size == 128);
}

void runXqa(void*       input,
            void*       output,
            size_t      head_num,
            size_t      kv_head_num,
            size_t      head_dim,
            size_t      batch_size,
            size_t      max_seq_len,
            size_t      page_size,
            void*       kv_cache_pool,
            int32_t*    kv_cache_page_list,
            uint32_t*   sequence_lengths,
            CudaDevice* device,
            size_t      max_q_len,
            void*       q_cu_seqlens,
            float       q_scale,
            size_t      max_batch_size,
            uint32_t    beam_width) {
    if (!input || !output || !head_num || !kv_head_num || (head_num / kv_head_num > 16) || (head_dim != 64 && head_dim != 128 &&
        head_dim != 256) || !batch_size || batch_size > max_batch_size || !max_seq_len || (page_size != 16 && page_size != 32 &&
        page_size != 64 && page_size != 128) || !kv_cache_pool || !kv_cache_page_list || !sequence_lengths || !device ||
        (max_q_len > 0 && !q_cu_seqlens)) {
        RTP_LLM_LOG_ERROR(
            "xqa params error: input = %p, output = %p, head_num = %zu, kv_head_num = %zu, head_dim = %zu, batch_size = %zu, "
            "max_seq_len = %zu, page_size = %zu, kv_cache_pool = %p, kv_cache_page_list = %p, sequence_lengths = %p, device = %p, "
            "q_scale = %f, max_batch_size = %zu, beam_width = %zu, max_q_len = %d, q_cu_seqlens = %p",
            input,
            output,
            head_num,
            kv_head_num,
            head_dim,
            batch_size,
            max_seq_len,
            page_size,
            kv_cache_pool,
            kv_cache_page_list,
            sequence_lengths,
            device,
            q_scale,
            max_batch_size,
            beam_width,
            max_q_len,
            q_cu_seqlens);
        return;
    }

    size_t group_size = static_cast<uint32_t>(head_num / kv_head_num);

    bool is_context_attn = q_cu_seqlens != nullptr;
    // For decoding XQA attention, the max_seq_len needs to be incremented by one additionally.
    max_seq_len = round_up(max_seq_len, page_size);

    static BufferPtr kv_cache_scale = getKVCacheScale(device);

    static BufferPtr semaphores = getSemaphores(device, kv_head_num, max_batch_size, max_q_len, group_size, is_context_attn);

    static void* scratch = getScratch(device, group_size, beam_width);

    SpecDecParams specDecParams{(uint32_t)max_q_len, reinterpret_cast<uint32_t*>(q_cu_seqlens), nullptr};

    run_xqa_sm90(static_cast<uint32_t>(head_dim),
                 static_cast<uint32_t>(page_size),
                 static_cast<uint32_t>(group_size),
                 device->getDeviceProp(),
                 static_cast<uint32_t>(kv_head_num),
                 q_scale,
                 output,
                 kv_cache_pool,
                 reinterpret_cast<KVCachePageIndex*>(kv_cache_page_list),
                 static_cast<uint32_t>(max_seq_len),
                 sequence_lengths,
                 static_cast<uint32_t>(batch_size),
                 kv_cache_scale->data<float>(),
                 semaphores->data<uint32_t>(),
                 scratch,
                 device->getStream(),
                 input,
                 is_context_attn ? &specDecParams : nullptr);

    check_cuda_error();
}

}
