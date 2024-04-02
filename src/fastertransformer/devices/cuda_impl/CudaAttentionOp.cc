#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"

#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "3rdparty/flash_attention2/flash.h"


using namespace std;

namespace fastertransformer {

template<typename T>
AttentionModuleOutput contextAttentionwrapper(const AttentionModuleParams& params, CudaDevice* self) {
    
    //  qkv input shape [token_num, head_num + 2 * kv_head_num, size_per_head]
    T* qkv_input_ptr = params.input.data<T>();

    auto token_num      = params.input.shape()[0];
    auto batch_size     = params.configs.batch_size;
    auto seq_len        = params.configs.seq_len;
    auto head_num       = params.configs.head_num;
    auto kv_head_num    = params.configs.kv_head_num;
    auto size_per_head  = params.configs.size_per_head;


    auto q_output = self->allocateBuffer({params.input.type(),
                                         {batch_size, head_num, seq_len, size_per_head},
                                         AllocationType::DEVICE},
                                         {});

    auto k_output = self->allocateBuffer({params.input.type(),
                                         {batch_size, kv_head_num, seq_len, size_per_head},
                                         AllocationType::DEVICE},
                                         {});


    auto v_output = self->allocateBuffer({params.input.type(),
                                         {batch_size, kv_head_num, seq_len, size_per_head},
                                         AllocationType::DEVICE},
                                         {});

    T* q_output_ptr  = q_output->data<T>();
    T* k_output_ptr  = k_output->data<T>();
    T* v_output_ptr  = v_output->data<T>();

    const int* position_ids     = params.common.position_ids.value().get().data<int>();
    const T*   bias_ptr         = params.weights.qkv_weight->bias->data<T>();
    const int* padding_offset   = params.common.padding_offset.value().get().data<int>();
    const int* cu_seqlens_ptr   = params.common.cu_seqlens.value().get().data<int>();

    PrefixPromptBatchWeightsParam<T> param;

    // int8
    float*  scale_out_ptr    = nullptr;
    int     int8_mode        = 0;

    // logn attention
    bool        use_logn_attention  = false;
    const int   logn_seq_len        = 0;

    // rope 
    const int rope_embedding_dim              = params.configs.rope_config.embedding_dim;
    const int rope_embedding_style            = (int) params.configs.rope_config.embedding_style;
    const int rope_embedding_base             = params.configs.rope_config.embedding_base;
    float rope_dynamic_embedding_scale        = params.configs.rope_config.dynamic_embedding_scale;
    const int rope_dynamic_embedding_max_pos  = params.configs.rope_config.dynamic_embedding_max_pos;
    const int rope_position_embeddings_scale  = params.configs.rope_config.position_embeddings_scale;
    const int rope_base_scale                 = params.configs.rope_config.base_scale;


    invokeAddFusedQKVBiasTranspose(q_output_ptr,
                                   k_output_ptr,
                                   v_output_ptr,
                                   param,  // prefix prompt
                                   qkv_input_ptr,
                                   position_ids,
                                   bias_ptr,
                                   padding_offset,
                                   cu_seqlens_ptr,
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
                                   self->getStream());
    sync_check_cuda_error();

    // TODO(lidongjin): Only support float32 gemm output.
    auto qk_output = self->gemm({*q_output,
                                 *k_output,
                                 std::nullopt,
                                 DataType::TYPE_FP32,
                                 TransposeOperation::NONE,
                                 TransposeOperation::TRANSPOSE});

    float scale = (1.0f / sqrtf(size_per_head* 1.0f));

    // TODO(lidongjin): Only support float32(in)\float16(output).
    auto softmax_qk_output = self->softmax({*qk_output,
                                            params.common.attention_mask.value().get(),
                                            scale,
                                            DataType::TYPE_FP16});

    auto qkv_output = self->gemm({*softmax_qk_output, *v_output});

    T* qkv_ptr = qkv_output->data<T>();

    auto qkv_transpose_output = self->allocateBuffer({params.input.type(),
                                                     {batch_size, seq_len, head_num, size_per_head},
                                                     AllocationType::DEVICE},
                                                     {});

    T* qkv_transpose_ptr = qkv_transpose_output->data<T>();

    invokeTransposeQKV(qkv_transpose_ptr,
                       qkv_ptr,
                       batch_size,
                       seq_len,
                       head_num,
                       size_per_head,
                       nullptr,
                       0,
                       self->getStream());

    sync_check_cuda_error();

    return AttentionModuleOutput({std::move(qkv_transpose_output)});
}


/// @brief   Context Attention ops
/// @details
AttentionModuleOutput CudaDevice::contextAttention(const AttentionModuleParams& params) {
    if (params.input.type() == DataType::TYPE_FP16) {
        return contextAttentionwrapper<half>(params, this);
    } else if (params.input.type() == DataType::TYPE_FP32) {
        return contextAttentionwrapper<float>(params, this);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    
}


} // namespace fastertransformer

