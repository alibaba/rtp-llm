#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/devices/cuda_impl/Dispatch.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"

#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"


using namespace std;

namespace fastertransformer {

AttentionModuleOutput CudaDevice::contextAttention(const AttentionModuleParams& params) {
    auto datatype       = params.input.type();
    auto token_num      = params.input.shape()[0];
    auto batch_size     = params.common.context_batch_size;
    auto seq_len        = params.common.context_max_seq_len;
    auto head_num       = params.configs.head_num;
    auto kv_head_num    = params.configs.kv_head_num;
    auto size_per_head  = params.configs.size_per_head;

    auto q_output = allocateBuffer({params.input.type(),
                                    {batch_size, head_num, seq_len, size_per_head},
                                    AllocationType::DEVICE},
                                    {});

    auto k_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len, size_per_head},
                                    AllocationType::DEVICE},
                                    {});

    auto v_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len, size_per_head},
                                    AllocationType::DEVICE},
                                    {});

    PrefixPromptBatchWeightsParam prefix_prompt_param;

    // int8
    float*  scale_out_ptr    = nullptr;
    int     int8_mode        = 0;

    // logn attention
    bool        use_logn_attention  = false;
    const int   logn_seq_len        = 0;

    // rope
    const auto rope_embedding_dim              = params.configs.rope_config.embedding_dim;
    const auto rope_embedding_style            = (int) params.configs.rope_config.embedding_style;
    const auto rope_embedding_base             = params.configs.rope_config.embedding_base;
    const auto rope_dynamic_embedding_scale    = params.configs.rope_config.dynamic_embedding_scale;
    const auto rope_dynamic_embedding_max_pos  = params.configs.rope_config.dynamic_embedding_max_pos;
    const auto rope_position_embeddings_scale  = params.configs.rope_config.position_embeddings_scale;
    const auto rope_base_scale                 = params.configs.rope_config.base_scale;

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype, invokeAddFusedQKVBiasTranspose,
        q_output->data(),
        k_output->data(),
        v_output->data(),
        &prefix_prompt_param,
        params.input.data(),
        params.common.position_ids.has_value() ? params.common.position_ids.value().get().data<int>() : nullptr,
        params.weights.qkv_weight->bias ? params.weights.qkv_weight->bias->data() : nullptr,
        params.common.padding_offset->data<int>(),
        params.common.cu_seqlens->data<int>(),
        batch_size,
        seq_len,
        token_num,
        head_num,
        kv_head_num,
        size_per_head,
        rope_embedding_dim,
        rope_embedding_style,
        rope_embedding_base,
        rope_dynamic_embedding_scale,
        rope_dynamic_embedding_max_pos,
        rope_position_embeddings_scale,
        rope_base_scale,
        logn_seq_len,
        use_logn_attention,
        scale_out_ptr,
        int8_mode,
        stream_
    );

    printBufferData(*q_output, "q_output: ");
    printBufferData(*k_output, "k_output: ");
    printBufferData(*v_output, "v_output: ");

    // TODO(lidongjin): Only support float32 gemm output.
    auto qk_output = gemm({*q_output,
                            *k_output,
                            std::nullopt,
                            DataType::TYPE_FP32,
                            TransposeOperation::NONE,
                            TransposeOperation::TRANSPOSE});
    printBufferData(*qk_output, "qk_output: ");

    float scale = (1.0f / sqrtf(size_per_head * 1.0f));

    // TODO(lidongjin): Only support float32(in)\float16(output).
    auto softmax_type = qk_output->type();
    RUNTIME_ASSERT_OP_ARG(
        params.common.attention_mask.has_value(),
        "attention_mask must be provided for default context attention implementation");
    auto softmax_qk_output = softmax({std::move(qk_output),
                                      params.common.attention_mask.value().get(),
                                      scale,
                                      DataType::TYPE_FP16});
    printBufferData(*softmax_qk_output, "softmax_qk_output: ");

    auto qkv_output = gemm({*softmax_qk_output, *v_output});
    printBufferData(*qkv_output, "qkv_output: ");

    auto &qkv_transpose_output = params.output;

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype, invokeTransposeQKV,
        qkv_transpose_output.data(),
        qkv_output->data(),
        batch_size,
        seq_len,
        head_num,
        size_per_head,
        nullptr,
        0,
        stream_);
}

struct SelfAttentionArgs {

};

template<typename T>
void selfAttentionwrapper(const AttentionModuleParams params,
                          cudaStream_t stream)
{
    size_t token_num            = params.input.shape()[0];
    size_t batch_size           = params.common.decoder_batch_size;
    size_t step                 = params.common.decoder_max_seq_len + 1;
    size_t local_head_num       = params.configs.head_num;
    size_t local_head_num_kv    = params.configs.kv_head_num;
    size_t size_per_head        = params.configs.size_per_head;
    const auto& output = params.output;

    const T* qkv_buf_ptr = params.input.data<T>();
    T* qkv_buf_2_ = output.data<T>();

    const T* bias_ptr = (params.weights.qkv_weight->bias == nullptr) ?
                         nullptr :
                         params.weights.qkv_weight->bias->data<T>();

    // TODO(lidongjin) support relative attention
    const T* relative_attention_bias_ptr = nullptr;

    // rope
    int rotary_embedding_dim = params.configs.rope_config.embedding_dim;
    int rotary_embedding_style = (int)params.configs.rope_config.embedding_style;
    int rotary_embedding_base  = params.configs.rope_config.embedding_base;
    float dynamic_embedding_scale = params.configs.rope_config.dynamic_embedding_scale;
    int dynamic_embedding_max_pos = params.configs.rope_config.dynamic_embedding_max_pos;
    int position_embeddings_scale = params.configs.rope_config.position_embeddings_scale;
    int base_scale = params.configs.rope_config.base_scale;

    // logn attention
    int logn_seq_len    = 0;
    bool use_logn_attn  = false;

    // prefix prompt

    int* prefix_prompt_lengths = nullptr;
    if (params.common.prefix_prompt_lengths.has_value()) {
        prefix_prompt_lengths = params.common.prefix_prompt_lengths.value().get().data<int>();
    }

    int max_prefix_prompt_length = 0;
    bool count_prefix_length = false;

    const auto* input_lengths = params.common.input_lengths.data<int>();
    const auto* sequence_lengths = params.common.sequence_lengths.data<int>();

    float q_scaling = 1.f;
    int relative_attention_bias_stride = 0;
    const T* linear_bias_slopes = nullptr;
    const bool* masked_tokens = nullptr;

    // TODO(lidongjin) support int8
    const float* query_weight_scale_out = nullptr;
    const float* attention_output_weight_scale_out = nullptr;
    int int8_mode = 0;
    // TODO(lidongjin) support multi block
    bool multi_block_mode = false;
    int max_seq_len_tile = 0;
    T* partial_out = nullptr;
    float* partial_sum = nullptr;
    float* partial_max = nullptr;
    int* block_counter = nullptr;

    if (!params.common.kv_cache_blocks.has_value()) {
        throw std::runtime_error("kv cache block pointers can not be null");
    }
    const auto max_blocks_per_seq = params.common.kv_cache_blocks.value().get().shape()[1];
    KVBlockArray kv_block_array(batch_size, max_blocks_per_seq, params.configs.tokens_per_block, 0);
    kv_block_array.data = reinterpret_cast<int64_t*>(
        params.common.kv_cache_blocks.value().get().data());

    fusedQKV_masked_attention_dispatch<T, KVBlockArray>(
        qkv_buf_ptr,
        bias_ptr,
        relative_attention_bias_ptr,
        nullptr, // cache_indir
        qkv_buf_2_,
        nullptr, // finished
        sequence_lengths,
        batch_size,
        1, // beam_width
        local_head_num,
        local_head_num_kv,
        size_per_head,
        rotary_embedding_dim,
        rotary_embedding_style,
        rotary_embedding_base,
        logn_seq_len,
        use_logn_attn,
        dynamic_embedding_scale,
        dynamic_embedding_max_pos,
        position_embeddings_scale,
        base_scale,
        step,
        prefix_prompt_lengths,
        max_prefix_prompt_length,
        count_prefix_length,
        input_lengths,
        step,
        q_scaling,
        relative_attention_bias_stride,
        linear_bias_slopes,
        masked_tokens,
        query_weight_scale_out,
        attention_output_weight_scale_out,
        int8_mode,
        multi_block_mode,
        max_seq_len_tile,
        partial_out,
        partial_sum,
        partial_max,
        block_counter,
        kv_block_array,
        stream);

    sync_check_cuda_error();
}

AttentionModuleOutput CudaDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    auto datatype = params.input.type();
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype, selfAttentionwrapper, params, stream_);
}

} // namespace fastertransformer
