#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

BufferPtr getKVCacheScale(CudaDevice *device) {
    float scale = 1.;
    BufferPtr kv_cache_scale = device->allocateBuffer({DataType::TYPE_FP32, {1}, AllocationType::DEVICE}, {"kv_cache_scale"});
    check_cuda_value(cudaMemcpyAsync(kv_cache_scale->data(), &scale, sizeof(float), cudaMemcpyHostToDevice, device->getStream()));
    check_cuda_value(cudaStreamSynchronize(device->getStream()));

    return kv_cache_scale;
}

BufferPtr getSemaphores(CudaDevice *device, size_t kv_head_num, size_t max_decode_batch_size) {
    size_t sem_size = kv_head_num * max_decode_batch_size;
    size_t real_sem_size = round_up<size_t>(sem_size, 2) + 2 + sem_size + 2;
    BufferPtr semaphores = device->allocateBuffer({DataType::TYPE_UINT32, {real_sem_size}, AllocationType::DEVICE}, {"semaphores"});
    device->bufMemset(*semaphores, 0);

    return semaphores;
}

void* getScratch(CudaDevice *device, size_t group_size, uint32_t beam_width) {
    size_t scratch_size = (256u << 20) * 4;
    static BufferPtr scratch = device->allocateBuffer({DataType::TYPE_BYTES, {scratch_size}, AllocationType::DEVICE}, {"scratch"});
    device->bufMemset(*scratch, 0);
    auto real_scratch = 
        reinterpret_cast<void*>(round_up<uintptr_t>(reinterpret_cast<uintptr_t>(scratch->data()), ioHeadBytes * group_size * beam_width));

    return real_scratch;
}

BufferPtr getRopeCosSin(CudaDevice *device, int rope_theta, int rope_dim, int max_position_embeddings) {
    auto inv_freq = 1.0 / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t = torch::arange(max_position_embeddings, torch::kInt64).to(torch::kFloat32);
    auto freqs = torch::outer(t, inv_freq);
    auto cos = freqs.cos().to(torch::kFloat32);
    auto sin = freqs.sin().to(torch::kFloat32);
    auto emb = torch::stack({cos, sin}, 0).permute({1, 2, 0}).reshape({cos.size(0), -1}).contiguous();

    BufferPtr rope_cos_sin = device->allocateBuffer({DataType::TYPE_UINT8,
                                                    {max_position_embeddings * sizeof(Vec<float, validElemsPerHead>)},
                                                    AllocationType::DEVICE},
                                                    {"rope_cos_sin"});
    auto rope_cos_sin_ptr = reinterpret_cast<Vec<float, validElemsPerHead>*>(rope_cos_sin->data());
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

bool supportXqa(DataType input_type,
                DataType output_type,
                DataType kv_cache_type,
                size_t group_size,
                size_t head_dim,
                size_t page_size) {
    return (input_type == DataType::TYPE_BF16) && (output_type == DataType::TYPE_BF16) && (kv_cache_type == DataType::TYPE_FP8_E4M3) &&
           (group_size <= 16) && (head_dim == HEAD_ELEMS) && (page_size == 16 || page_size == 32 || page_size == 64 || page_size == 128);
}

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
            int rope_theta,
            int max_position_embeddings,
            float q_scale,
            size_t max_decode_batch_size,
            uint32_t beam_width) {
    if (!input || !output || !head_num || !kv_head_num || (head_num / kv_head_num > 16) || head_dim != HEAD_ELEMS ||
        !decode_batch_size || decode_batch_size > max_decode_batch_size || !decode_max_seq_len ||
        (page_size != 16 && page_size != 32 && page_size != 64 && page_size != 128) ||  !kv_cache_pool || !kv_cache_page_list ||
        !sequence_lengths || !device || rope_theta <= 0 || max_position_embeddings <= 0 || decode_max_seq_len >= max_position_embeddings) {
        RTP_LLM_LOG_ERROR("xqa params error: input = %p, output = %p, head_num = %zu, kv_head_num = %zu, head_dim = %zu, "
            "decode_batch_size = %zu, decode_max_seq_len = %zu, page_size = %zu, kv_cache_pool = %p, kv_cache_page_list = %p, "
            "sequence_lengths = %p, device = %p, rope_theta = %d, max_position_embeddings = %d, q_scale = %f, max_decode_batch_size = %zu, "
            "beam_width = %zu", input, output, head_num, kv_head_num, head_dim, decode_batch_size, decode_max_seq_len, page_size, kv_cache_pool,
            kv_cache_page_list, sequence_lengths, device, rope_theta, max_position_embeddings, q_scale, max_decode_batch_size, beam_width);
        return;
    }

    size_t group_size = static_cast<uint32_t>(head_num / kv_head_num);

    size_t max_seq_len = round_up(decode_max_seq_len + 1, page_size);

    static BufferPtr kv_cache_scale = getKVCacheScale(device);

    static BufferPtr semaphores = getSemaphores(device, kv_head_num, max_decode_batch_size);

    static void* scratch = getScratch(device, group_size, beam_width);

    static BufferPtr rope_cos_sin = getRopeCosSin(device, rope_theta, head_dim, max_position_embeddings);

    run_xqa_sm90(static_cast<uint32_t>(page_size), 
                 static_cast<uint32_t>(group_size),
                 device->getDeviceProp(),
                 static_cast<uint32_t>(kv_head_num),
                 q_scale,
                 reinterpret_cast<OutputHead*>(output),
                 reinterpret_cast<InputHead*>(input),
                 reinterpret_cast<Vec<float, validElemsPerHead>*>(rope_cos_sin->data()),
                 reinterpret_cast<GMemCacheHead*>(kv_cache_pool),
                 reinterpret_cast<KVCachePageIndex*>(kv_cache_page_list),
                 static_cast<uint32_t>(max_seq_len),
                 sequence_lengths,
                 static_cast<uint32_t>(decode_batch_size),
                 kv_cache_scale->data<float>(),
                 semaphores->data<uint32_t>(),
                 scratch,
                 device->getStream());

    check_cuda_error();
}

}
