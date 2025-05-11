
#include <iostream>
#include <numeric>
#include "maga_transformer/cpp/devices/OpData.h"
#include "maga_transformer/cpp/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/devices/CommonDefines.h"
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/cuda/Dispatch.h"
#include "maga_transformer/cpp/utils/compiler_config.h"
#include "maga_transformer/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "maga_transformer/cpp/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/devices/cuda_impl/CudaXqa.h"

using namespace std;
using namespace rtp_llm;

namespace rtp_llm {

BufferPtr getKVCacheScale(CudaDevice *device) {
    float scale = 1.;
    BufferPtr kv_cache_scale = device->allocateBuffer({DataType::TYPE_FP32, {1}, AllocationType::DEVICE}, {"kv_cache_scale"});
    check_cuda_error(cudaMemcpyAsync(kv_cache_scale->data(), &scale, sizeof(float), cudaMemcpyHostToDevice, device->getStream()));

    return kv_cache_scale;
}

BufferPtr getSemaphores(CudaDevice *device, size_t kv_head_num, size_t max_decode_batch_size) {
    size_t sem_size = kv_head_num * max_decode_batch_size;
    size_t real_sem_size = round_up<size_t>(sem_size, 2) + 2 + sem_size + 2;
    BufferPtr semaphores = device->allocateBuffer({DataType::TYPE_UINT32, {real_sem_size}, AllocationType::DEVICE}, {"semaphores"});
    device->bufMemset(*semaphores, 0);

    return semaphores;
}

void* getScratch(CudaDevice *device, size_t head_num, size_t kv_head_num, uint32_t beam_width) {
    size_t scratch_size = (256u << 20);
    static BufferPtr scratch = device->allocateBuffer({DataType::TYPE_BYTES, {scratch_size}, AllocationType::DEVICE}, {"scratch"});
    device->bufMemset(*scratch, 0);
    size_t group_size = head_num / kv_head_num;
    auto real_scratch = 
        reinterpret_cast<void*>(round_up<uintptr_t>(reinterpret_cast<uintptr_t>(scratch->data()), ioHeadBytes * group_size * beam_width));

    return real_scratch;
}

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
            float q_scale,
            size_t max_decode_batch_size,
            uint32_t beam_width)
{
    if (!input || !output || !head_num || !kv_head_num || !decode_batch_size || decode_batch_size > max_decode_batch_size ||
        !decode_max_seq_len || !tokens_per_block || !kv_cache_pool || !kv_cache_page_list || !sequence_lengths || !device) {
        RTP_LLM_LOG_ERROR("xqa params error: input = %p, output = %p, head_num = %zu, kv_head_num = %zu, decode_batch_size = %zu, "
            "decode_max_seq_len = %zu, tokens_per_block = %zu, kv_cache_pool = %p, kv_cache_page_list = %p, sequence_lengths = %p, "
            "device = %p, q_scale = %f, max_decode_batch_size = %zu, beam_width = %zu", input, output, head_num, kv_head_num, 
            decode_batch_size, decode_max_seq_len, tokens_per_block, kv_cache_pool, kv_cache_page_list, sequence_lengths, device, 
            q_scale, max_decode_batch_size, beam_width);
        return;
    }

    RTP_LLM_LOG_INFO("xqa params: input = %p, output = %p, head_num = %zu, kv_head_num = %zu, decode_batch_size = %zu, "
        "decode_max_seq_len = %zu, tokens_per_block = %zu, kv_cache_pool = %p, kv_cache_page_list = %p, sequence_lengths = %p, "
        "device = %p, q_scale = %f, max_decode_batch_size = %zu, beam_width = %zu", input, output, head_num, kv_head_num, 
        decode_batch_size, decode_max_seq_len, tokens_per_block, kv_cache_pool, kv_cache_page_list, sequence_lengths, device, 
        q_scale, max_decode_batch_size, beam_width);

    size_t max_seq_len = round_up(decode_max_seq_len, tokens_per_block);

    static BufferPtr kv_cache_scale = getKVCacheScale(device);

    BufferPtr semaphores = getSemaphores(device, kv_head_num, decode_batch_size);

    static void* scratch = getScratch(device, head_num, kv_head_num, beam_width);

    launchHopperF8MHA(device->getDeviceProp(),
                      static_cast<uint32_t>(kv_head_num),
                      q_scale,
                      reinterpret_cast<OutputHead*>(output),
                      reinterpret_cast<InputHead*>(input),
                      reinterpret_cast<GMemCacheHead*>(kv_cache_pool),
                      reinterpret_cast<KVCachePageIndex*>(kv_cache_page_list),
                      static_cast<uint32_t>(max_seq_len),
                      sequence_lengths,
                      static_cast<uint32_t>(decode_batch_size),
                      kv_cache_scale->data<float>(),
                      semaphores->data<uint32_t>(),
                      scratch,
                      device->getStream());

    sync_check_cuda_error();
}

}
