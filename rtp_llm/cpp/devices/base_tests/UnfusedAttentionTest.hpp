#pragma once

#include <torch/torch.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"

#ifdef USING_CUDA12
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#endif

using namespace std;
using namespace rtp_llm;

inline torch::Tensor create_position_ids(const std::vector<int>& input_lengths) {
    std::vector<torch::Tensor> tensors;
    for (int i = 0; i < input_lengths.size(); ++i) {
        auto position_ids = torch::arange(input_lengths[i], torch::kInt32);
        tensors.push_back(position_ids);
    }
    return torch::concat(tensors, 0);
}

inline torch::Tensor do_rotary_emb(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin) {
    auto cos_cache = cos.unsqueeze(-2).to(x.dtype());
    auto sin_cache = sin.unsqueeze(-2).to(x.dtype());
    auto xx        = torch::chunk(x, 2, -1);
    auto o1        = xx[0] * cos_cache - xx[1] * sin_cache;
    auto o2        = xx[1] * cos_cache + xx[0] * sin_cache;
    return torch::cat({o1, o2}, -1);
}

inline std::tuple<torch::Tensor, torch::Tensor> apply_rotary_emb(const torch::Tensor& q,
                                                                 const torch::Tensor& k,
                                                                 const torch::Tensor& positions,
                                                                 int                  cache_len,
                                                                 int                  rope_dim,
                                                                 int                  rope_theta) {
    auto inv_freq =
        1.0 / torch::pow(rope_theta, torch::arange(0, rope_dim, 2, torch::kInt64).to(torch::kFloat32) / rope_dim);
    auto t          = torch::arange(cache_len, torch::kInt64).to(torch::kFloat32);
    auto freqs      = torch::outer(t, inv_freq);
    auto cos        = freqs.cos().to(torch::kFloat32);
    auto sin        = freqs.sin().to(torch::kFloat32);
    auto rope_cache = torch::cat({cos, sin}, -1);

    auto pos      = positions.flatten();
    auto rope_pos = rope_cache.index_select(0, pos).reshape({q.size(0), q.size(1), -1});

    auto cache = torch::chunk(rope_pos, 2, -1);

    auto q_rope = do_rotary_emb(q, cache[0], cache[1]);
    auto k_rope = do_rotary_emb(k, cache[0], cache[1]);

    return std::make_tuple(q_rope, k_rope);
}

struct AddFusedQKVBiasTransposeImpl: torch::nn::Module {
    AddFusedQKVBiasTransposeImpl() {}

    std::vector<torch::Tensor> forward(const torch::Tensor& query_states,
                                       const torch::Tensor& key_states,
                                       const torch::Tensor& value_states,
                                       const torch::Tensor& query_bias,
                                       const torch::Tensor& key_bias,
                                       const torch::Tensor& value_bias,
                                       const torch::Tensor& positions,
                                       int                  cache_len,
                                       const int            rope_dim   = 128,
                                       const int            rope_theta = 1000000) {
        auto q = query_states + query_bias;
        auto k = key_states + key_bias;
        auto v = value_states + value_bias;

        std::tie(q, k) = apply_rotary_emb(q, k, positions, cache_len, rope_dim, rope_theta);

        return {q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)};
    }
};
TORCH_MODULE(AddFusedQKVBiasTranspose);

class UnfusedAttentionTest: public DeviceTestBase {
public:
#ifdef USING_CUDA12
    void addFusedQKVBiasTransposeTest(size_t batch_size,
                                      size_t seq_len,
                                      size_t num_heads,
                                      size_t num_key_value_heads,
                                      size_t head_dim,
                                      size_t tokens_per_block,
                                      bool   is_perf = false);

    void decodeAddFusedQKVBiasTransposeTest(size_t batch_size,
                                            size_t seq_len,
                                            size_t kv_seq_len,
                                            size_t num_heads,
                                            size_t num_key_value_heads,
                                            size_t head_dim,
                                            size_t tokens_per_block,
                                            bool   is_perf = false);
#endif
};

#ifdef USING_CUDA12
void UnfusedAttentionTest::addFusedQKVBiasTransposeTest(size_t batch_size,
                                                        size_t seq_len,
                                                        size_t num_heads,
                                                        size_t num_key_value_heads,
                                                        size_t head_dim,
                                                        size_t tokens_per_block,
                                                        bool   is_perf) {
    AddFusedQKVBiasTranspose fused = AddFusedQKVBiasTranspose();
    fused.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = fused.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    CudaDevice* device = dynamic_cast<CudaDevice*>(device_);

    auto tensor_options      = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto bf16_tensor_options = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
    auto int_tensor_options  = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    auto query_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, bf16_tensor_options);

    auto key_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto value_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto qkv_states_host =
        torch::cat({query_states_host.view({(int)batch_size * (int)seq_len, (int)num_heads, (int)head_dim}),
                    key_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim}),
                    value_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim})},
                   1);

    auto query_bias_host = torch::zeros({(int)num_heads, (int)head_dim}, bf16_tensor_options);

    auto key_bias_host = torch::zeros({(int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto value_bias_host = torch::rand({(int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto qkv_bias_host = torch::cat({query_bias_host, key_bias_host, value_bias_host}, 0);

    size_t padding_seq_len = ((seq_len + tokens_per_block - 1) / tokens_per_block + 1) * tokens_per_block;
    padding_seq_len        = (seq_len == 0) ? 2 * tokens_per_block : padding_seq_len;
    auto kvcache_pad       = torch::zeros(
        {1, (int)batch_size, 2, (int)padding_seq_len, (int)num_key_value_heads * (int)head_dim}, bf16_tensor_options);

    std::vector<int> input_lengths(batch_size, seq_len);
    std::vector<int> sequence_lengths(batch_size, 0);
    std::vector<int> cu_seqlens(batch_size + 1);
    for (int i = 0; i < batch_size + 1; ++i) {
        cu_seqlens[i] = seq_len * i;
    }

    auto input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto sequence_lengths_host =
        torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto cu_seqlens_host = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size + 1}, int_tensor_options);

    auto token_num           = batch_size * seq_len;
    auto padding_offset_host = torch::zeros({(int)token_num}, int_tensor_options);
    auto position_ids_host   = create_position_ids(input_lengths);
    auto attention_mask_host = torch::zeros({(int)batch_size, (int)seq_len, (int)seq_len}, tensor_options);

    auto qkv_states_device       = createDeviceBuffer<__nv_bfloat16>(qkv_states_host);
    auto qkv_bias_device         = createDeviceBuffer<__nv_bfloat16>(qkv_bias_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto cu_seqlens_device       = createDeviceBuffer<int>(cu_seqlens_host);
    auto padding_offset_device   = createDeviceBuffer<int>(padding_offset_host);
    auto position_ids_device     = createDeviceBuffer<int>(position_ids_host);
    auto attention_mask_device   = createDeviceBuffer<__nv_bfloat16>(attention_mask_host);

    int  rope_dim                = static_cast<int>(head_dim);
    int  rope_theta              = 1000000;
    int  max_position_embeddings = 10240;
    auto rope_config = RopeConfig({RopeStyle::Base, rope_dim, rope_theta, 1., 0., 0., max_position_embeddings});

    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num  = 2 * batch_size * ((seq_len + tokens_per_block - 1) / tokens_per_block + 1) + 1;
    auto cache_conf = makeMhaCacheConfig(
        1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokens_per_block, DataType::TYPE_BF16);
    cache_manager_             = nullptr;
    auto kv_cache_block_id     = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad);
    auto kv_cache_buffer       = cache_manager_->kvCacheBuffer();
    auto common_inputs         = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    auto layer_kv_cache_buffer = kv_cache_buffer.kv_blocks->index(0);
    common_inputs.kv_cache =
        KvCacheInfo({(int)kv_cache_buffer.kv_blocks->shape()[0], kv_cache_block_id, layer_kv_cache_buffer, nullptr});
    common_inputs.cu_seqlens                = cu_seqlens_device;
    common_inputs.padding_offset            = padding_offset_device;
    common_inputs.position_ids              = position_ids_device;
    common_inputs.attention_mask            = attention_mask_device;
    common_inputs.context_batch_size        = batch_size;
    common_inputs.context_max_seq_len       = seq_len;
    common_inputs.decoder_batch_size        = 0;
    common_inputs.decoder_max_seq_len       = 0;
    common_inputs.max_prefix_length         = 0;

    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = std::make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, qkv_bias_device));

    AttentionConfigs attention_config;
    attention_config.head_num         = num_heads;
    attention_config.kv_head_num      = num_key_value_heads;
    attention_config.size_per_head    = head_dim;
    attention_config.rope_config      = rope_config;
    attention_config.tokens_per_block = tokens_per_block;

    auto qkv_output = device->allocateBuffer({qkv_states_device->type(), {token_num, num_heads, head_dim}});

    AttentionModuleParams params = {
        0, *qkv_states_device, *qkv_output, common_inputs, attention_weight, attention_config};

    auto attn = device->prepareTrtAttn(
        attention_config, kv_cache_buffer.kv_blocks, params.common.kv_cache->kv_cache_block_id, batch_size);
    auto trt_attn = reinterpret_cast<TRTAttn*>(attn.get());
    TRTAttn::setKvCache(trt_attn->kv_block_array, *common_inputs.kv_cache);

    PrefixPromptBatchWeightsParam prefix_prompt_param;
    prefix_prompt_param.kv_block_array = trt_attn->kv_block_array;

    auto q_no_transpose_output = device->allocateBuffer(
        {params.input.type(), {token_num, num_heads, head_dim}, AllocationType::DEVICE}, {"q_no_transpose_output"});

    auto q_output = device->allocateBuffer(
        {params.input.type(), {batch_size, num_heads, seq_len, head_dim}, AllocationType::DEVICE}, {"q_output"});

    auto k_output = device->allocateBuffer(
        {params.input.type(), {batch_size, num_key_value_heads, seq_len, head_dim}, AllocationType::DEVICE},
        {"k_output"});

    auto v_output = device->allocateBuffer(
        {params.input.type(), {batch_size, num_key_value_heads, seq_len, head_dim}, AllocationType::DEVICE},
        {"v_output"});

    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;

    auto qkv_buf_fp8 =
        cache_conf.cache_specs[0]->dtype == DataType::TYPE_FP8_E4M3 ?
            device->allocateBuffer({DataType::TYPE_FP8_E4M3,
                                    {batch_size, (num_heads + num_key_value_heads * 2), seq_len_with_prefix, head_dim},
                                    AllocationType::DEVICE},
                                   {"qkv_fp8_output"}) :
            nullptr;
    if (qkv_buf_fp8) {
        device->bufMemset(*qkv_buf_fp8, 0);
    }

    float* scale_out_ptr        = nullptr;
    int    int8_mode            = 0;
    bool   use_paged_fmha       = false;
    bool   store_qkv            = false;
    bool   store_q_no_transpose = false;

    auto rope_cache = getRopeCacheOnce(rope_config, max_position_embeddings);

    if (is_perf) {
        bool store_q     = true;
        bool store_kv    = false;
        bool store_cache = true;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // warm up
        for (int i = 0; i < 3; ++i) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                params.input.type(),
                invokeAddFusedQKVBiasTranspose,
                q_no_transpose_output->data(),
                q_output->data(),
                k_output->data(),
                v_output->data(),
                &prefix_prompt_param,
                params.input.data(),
                qkv_buf_fp8 ? qkv_buf_fp8->data() : nullptr,
                params.common.position_ids->data<int>(),
                nullptr,  // params.weights.qkv_weight->bias->data()
                params.common.padding_offset->data<int>(),
                params.common.cu_seqlens->data<int>(),
                rope_cache.used,
                checkRopeCache(rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() : nullptr,
                batch_size,
                seq_len,
                token_num,
                num_heads,
                num_key_value_heads,
                head_dim,
                params.configs.rope_config,
                params.configs.use_logn_attn,
                scale_out_ptr,
                int8_mode,
                use_paged_fmha,
                store_qkv,
                store_q_no_transpose,
                store_q,
                store_kv,
                store_cache,
                device->getStream());
        }

        device->syncAndCheck();

        // perf
        int iters = 100;
        cudaEventRecord(start, device->getStream());

        for (int i = 0; i < iters; ++i) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                params.input.type(),
                invokeAddFusedQKVBiasTranspose,
                q_no_transpose_output->data(),
                q_output->data(),
                k_output->data(),
                v_output->data(),
                &prefix_prompt_param,
                params.input.data(),
                qkv_buf_fp8 ? qkv_buf_fp8->data() : nullptr,
                params.common.position_ids->data<int>(),
                nullptr,  // params.weights.qkv_weight->bias->data()
                params.common.padding_offset->data<int>(),
                params.common.cu_seqlens->data<int>(),
                rope_cache.used,
                checkRopeCache(rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() : nullptr,
                batch_size,
                seq_len,
                token_num,
                num_heads,
                num_key_value_heads,
                head_dim,
                params.configs.rope_config,
                params.configs.use_logn_attn,
                scale_out_ptr,
                int8_mode,
                use_paged_fmha,
                store_qkv,
                store_q_no_transpose,
                store_q,
                store_kv,
                store_cache,
                device->getStream());
        }

        cudaEventRecord(stop, device->getStream());
        cudaEventSynchronize(stop);
        float total_time = 0.f;
        cudaEventElapsedTime(&total_time, start, stop);

        float time = total_time / iters;
        std::cout << "addFusedQKVBiasTransposeTest perf time: " << time << ", batch_size: " << batch_size
                  << ", seq_q: " << seq_len << ", head_q: " << num_heads << ", head_kv: " << num_key_value_heads
                  << ", head_dim: " << head_dim << ", tokens_per_block: " << tokens_per_block << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        bool store_q     = true;
        bool store_kv    = true;
        bool store_cache = false;

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(),
                                         invokeAddFusedQKVBiasTranspose,
                                         q_no_transpose_output->data(),
                                         q_output->data(),
                                         k_output->data(),
                                         v_output->data(),
                                         &prefix_prompt_param,
                                         params.input.data(),
                                         qkv_buf_fp8 ? qkv_buf_fp8->data() : nullptr,
                                         params.common.position_ids->data<int>(),
                                         params.weights.qkv_weight->bias->data(),
                                         params.common.padding_offset->data<int>(),
                                         params.common.cu_seqlens->data<int>(),
                                         rope_cache.used,
                                         checkRopeCache(rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() :
                                                                                   nullptr,
                                         batch_size,
                                         seq_len,
                                         token_num,
                                         num_heads,
                                         num_key_value_heads,
                                         head_dim,
                                         params.configs.rope_config,
                                         params.configs.use_logn_attn,
                                         scale_out_ptr,
                                         int8_mode,
                                         use_paged_fmha,
                                         store_qkv,
                                         store_q_no_transpose,
                                         store_q,
                                         store_kv,
                                         store_cache,
                                         device->getStream());

        device->syncAndCheck();

        auto ref = fused->forward(query_states_host,
                                  key_states_host,
                                  value_states_host,
                                  query_bias_host,
                                  key_bias_host,
                                  value_bias_host,
                                  position_ids_host,
                                  seq_len,
                                  rope_dim,
                                  rope_theta);

        auto result_q = bufferToTensor(*q_output);
        auto result_k = bufferToTensor(*k_output);
        auto result_v = bufferToTensor(*v_output);

        assertTensorClose(result_q, ref[0].to(result_q.dtype()), 1e-5, 1e-2);
        assertTensorClose(result_k, ref[1].to(result_k.dtype()), 1e-5, 1e-2);
        assertTensorClose(result_v, ref[2].to(result_v.dtype()), 1e-5, 1e-5);
    }
}

void UnfusedAttentionTest::decodeAddFusedQKVBiasTransposeTest(size_t batch_size,
                                                              size_t seq_len,
                                                              size_t kv_seq_len,
                                                              size_t num_heads,
                                                              size_t num_key_value_heads,
                                                              size_t head_dim,
                                                              size_t tokens_per_block,
                                                              bool   is_perf) {
    AddFusedQKVBiasTranspose fused = AddFusedQKVBiasTranspose();
    fused.ptr()->to(torch::Device(torch::kCPU));
    auto               state_dict = fused.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    CudaDevice* device = dynamic_cast<CudaDevice*>(device_);

    auto tensor_options      = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto bf16_tensor_options = torch::TensorOptions(torch::kBFloat16).device(torch::Device(torch::kCPU));
    auto int_tensor_options  = torch::TensorOptions(torch::kInt).device(torch::Device(torch::kCPU));

    auto query_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_heads, (int)head_dim}, bf16_tensor_options);

    auto key_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto value_states_host =
        torch::rand({(int)batch_size, (int)seq_len, (int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto qkv_states_host =
        torch::cat({query_states_host.view({(int)batch_size * (int)seq_len, (int)num_heads, (int)head_dim}),
                    key_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim}),
                    value_states_host.view({(int)batch_size * (int)seq_len, (int)num_key_value_heads, (int)head_dim})},
                   1);

    auto query_bias_host = torch::zeros({(int)num_heads, (int)head_dim}, bf16_tensor_options);

    auto key_bias_host = torch::zeros({(int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto value_bias_host = torch::rand({(int)num_key_value_heads, (int)head_dim}, bf16_tensor_options);

    auto qkv_bias_host = torch::cat({query_bias_host, key_bias_host, value_bias_host}, 0);

    size_t padding_seq_len = ((kv_seq_len + tokens_per_block - 1) / tokens_per_block + 1) * tokens_per_block;
    padding_seq_len        = (kv_seq_len == 0) ? 2 * tokens_per_block : padding_seq_len;
    auto kvcache_pad       = torch::zeros(
        {1, (int)batch_size, 2, (int)padding_seq_len, (int)num_key_value_heads * (int)head_dim}, bf16_tensor_options);

    std::vector<int> input_lengths(batch_size, kv_seq_len);
    std::vector<int> sequence_lengths(batch_size, kv_seq_len);
    std::vector<int> cu_seqlens(batch_size + 1);
    for (int i = 0; i < batch_size + 1; ++i) {
        cu_seqlens[i] = seq_len * i;
    }
    std::vector<int> positions(batch_size, kv_seq_len);

    auto input_lengths_host = torch::from_blob((void*)input_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto sequence_lengths_host =
        torch::from_blob((void*)sequence_lengths.data(), {(int)batch_size}, int_tensor_options);
    auto cu_seqlens_host = torch::from_blob((void*)cu_seqlens.data(), {(int)batch_size + 1}, int_tensor_options);

    auto token_num         = batch_size * seq_len;
    auto position_ids_host = torch::from_blob((void*)positions.data(), {(int)batch_size}, int_tensor_options);
    auto attention_mask_host =
        torch::zeros({(int)batch_size, (int)seq_len, (int)kv_seq_len + (int)seq_len}, tensor_options);

    auto qkv_states_device       = createDeviceBuffer<__nv_bfloat16>(qkv_states_host);
    auto qkv_bias_device         = createDeviceBuffer<__nv_bfloat16>(qkv_bias_host);
    auto input_lengths_device    = createDeviceBuffer<int>(input_lengths_host);
    auto sequence_lengths_device = createDeviceBuffer<int>(sequence_lengths_host);
    auto cu_seqlens_device       = createDeviceBuffer<int>(cu_seqlens_host);
    auto position_ids_device     = createDeviceBuffer<int>(position_ids_host);
    auto attention_mask_device   = createDeviceBuffer<__nv_bfloat16>(attention_mask_host);

    int  rope_dim                = static_cast<int>(head_dim);
    int  rope_theta              = 1000000;
    int  max_position_embeddings = 10240;
    auto rope_config = RopeConfig({RopeStyle::Base, rope_dim, rope_theta, 1., 0., 0., max_position_embeddings});

    // cache manager need one block for preserve and every seq need one block for preserve.
    auto block_num  = 2 * batch_size * ((kv_seq_len + seq_len + tokens_per_block - 1) / tokens_per_block + 1) + 1;
    auto cache_conf = makeMhaCacheConfig(
        1, (uint)block_num, (uint)num_key_value_heads, (uint)head_dim, (uint)tokens_per_block, DataType::TYPE_BF16);
    cache_manager_             = nullptr;
    auto kv_cache_block_id     = allocateKVBlocks(cache_conf, input_lengths, kvcache_pad);
    auto kv_cache_buffer       = cache_manager_->kvCacheBuffer();
    auto common_inputs         = AttentionCommonInputs({input_lengths_device, sequence_lengths_device});
    auto layer_kv_cache_buffer = kv_cache_buffer.kv_blocks->index(0);
    common_inputs.kv_cache =
        KvCacheInfo({(int)kv_cache_buffer.kv_blocks->shape()[0], kv_cache_block_id, layer_kv_cache_buffer, nullptr});
    common_inputs.cu_seqlens          = cu_seqlens_device;
    common_inputs.position_ids        = position_ids_device;
    common_inputs.attention_mask      = attention_mask_device;
    common_inputs.context_batch_size  = 0;
    common_inputs.context_max_seq_len = 0;
    common_inputs.decoder_batch_size  = batch_size;
    common_inputs.decoder_max_seq_len = kv_seq_len;
    common_inputs.max_prefix_length   = 0;

    auto buffer_nullptr         = BufferPtr(nullptr);
    auto attention_weight       = AttentionLayerWeights();
    attention_weight.qkv_weight = std::make_shared<const DenseWeights>(DenseWeights(buffer_nullptr, qkv_bias_device));

    AttentionConfigs attention_config;
    attention_config.head_num         = num_heads;
    attention_config.kv_head_num      = num_key_value_heads;
    attention_config.size_per_head    = head_dim;
    attention_config.rope_config      = rope_config;
    attention_config.tokens_per_block = tokens_per_block;

    auto qkv_output = device->allocateBuffer({qkv_states_device->type(), {token_num, num_heads, head_dim}});

    AttentionModuleParams params = {
        0, *qkv_states_device, *qkv_output, common_inputs, attention_weight, attention_config};

    auto attn = device->prepareTrtAttn(
        attention_config, kv_cache_buffer.kv_blocks, params.common.kv_cache->kv_cache_block_id, batch_size);
    auto trt_attn = reinterpret_cast<TRTAttn*>(attn.get());
    TRTAttn::setKvCache(trt_attn->kv_block_array, *common_inputs.kv_cache);

    auto q_output = device->allocateBuffer(
        {params.input.type(), {batch_size, num_heads, seq_len, head_dim}, AllocationType::DEVICE}, {"q_output"});

    auto k_output = device->allocateBuffer(
        {params.input.type(), {batch_size, num_key_value_heads, seq_len, head_dim}, AllocationType::DEVICE},
        {"k_output"});

    auto v_output = device->allocateBuffer(
        {params.input.type(), {batch_size, num_key_value_heads, seq_len, head_dim}, AllocationType::DEVICE},
        {"v_output"});

    auto rope_cache = getRopeCacheOnce(rope_config, max_position_embeddings);

    if (is_perf) {
        bool store_q     = true;
        bool store_kv    = false;
        bool store_cache = true;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // warm up
        for (int i = 0; i < 3; ++i) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                params.input.type(),
                invokeDecodeAddFusedQKVBiasTranspose,
                q_output->data(),
                k_output->data(),
                v_output->data(),
                trt_attn->kv_block_array,
                params.input.data(),
                params.common.position_ids->data<int>(),
                nullptr,  // params.weights.qkv_weight->bias->data()
                rope_cache.used,
                checkRopeCache(rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() : nullptr,
                batch_size,
                num_heads,
                num_key_value_heads,
                head_dim,
                params.configs.rope_config,
                params.configs.use_logn_attn,
                store_q,
                store_kv,
                store_cache,
                device->getStream());
        }

        device->syncAndCheck();

        // perf
        int iters = 100;
        cudaEventRecord(start, device->getStream());

        for (int i = 0; i < iters; ++i) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                params.input.type(),
                invokeDecodeAddFusedQKVBiasTranspose,
                q_output->data(),
                k_output->data(),
                v_output->data(),
                trt_attn->kv_block_array,
                params.input.data(),
                params.common.position_ids->data<int>(),
                nullptr,  // params.weights.qkv_weight->bias->data()
                rope_cache.used,
                checkRopeCache(rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() : nullptr,
                batch_size,
                num_heads,
                num_key_value_heads,
                head_dim,
                params.configs.rope_config,
                params.configs.use_logn_attn,
                store_q,
                store_kv,
                store_cache,
                device->getStream());
        }

        cudaEventRecord(stop, device->getStream());
        cudaEventSynchronize(stop);
        float total_time = 0.f;
        cudaEventElapsedTime(&total_time, start, stop);

        float time = total_time / iters;
        std::cout << "decodeAddFusedQKVBiasTransposeTest perf time: " << time << ", batch_size: " << batch_size
                  << ", seq_q: " << seq_len << ", kv_seq_len: " << kv_seq_len << ", head_q: " << num_heads
                  << ", head_kv: " << num_key_value_heads << ", head_dim: " << head_dim
                  << ", tokens_per_block: " << tokens_per_block << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        bool store_q     = true;
        bool store_kv    = true;
        bool store_cache = false;

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.input.type(),
                                         invokeDecodeAddFusedQKVBiasTranspose,
                                         q_output->data(),
                                         k_output->data(),
                                         v_output->data(),
                                         trt_attn->kv_block_array,
                                         params.input.data(),
                                         params.common.position_ids->data<int>(),
                                         params.weights.qkv_weight->bias->data(),
                                         rope_cache.used,
                                         checkRopeCache(rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() :
                                                                                   nullptr,
                                         batch_size,
                                         num_heads,
                                         num_key_value_heads,
                                         head_dim,
                                         params.configs.rope_config,
                                         params.configs.use_logn_attn,
                                         store_q,
                                         store_kv,
                                         store_cache,
                                         device->getStream());

        device->syncAndCheck();

        auto ref = fused->forward(query_states_host,
                                  key_states_host,
                                  value_states_host,
                                  query_bias_host,
                                  key_bias_host,
                                  value_bias_host,
                                  position_ids_host,
                                  kv_seq_len + seq_len,
                                  rope_dim,
                                  rope_theta);

        auto result_q = bufferToTensor(*q_output);
        auto result_k = bufferToTensor(*k_output);
        auto result_v = bufferToTensor(*v_output);

        assertTensorClose(result_q, ref[0].to(result_q.dtype()), 1e-5, 1e-2);
        assertTensorClose(result_k, ref[1].to(result_k.dtype()), 1e-5, 1e-2);
        assertTensorClose(result_v, ref[2].to(result_v.dtype()), 1e-5, 1e-5);
    }
}
#endif
