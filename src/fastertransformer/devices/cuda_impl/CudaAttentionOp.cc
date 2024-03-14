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


/// @brief   Context Attention ops
/// @details 
void CudaDevice::contextAttention(const ContextAttentionParams& params) {
    
    params.check();

    // prepare params

    half* qkv_input_ptr = reinterpret_cast<half*>(params.qkv_input.data());
    half* q_output_ptr  = reinterpret_cast<half*>(params.q_output.data());
    half* k_output_ptr  = reinterpret_cast<half*>(params.k_output.data());
    half* v_output_ptr  = reinterpret_cast<half*>(params.v_output.data());

    const int* position_ids   = reinterpret_cast<const int*>(params.position_ids.data());
    const half* bias_ptr      = reinterpret_cast<const half*>(params.bias.data());
    const int* padding_offset = reinterpret_cast<const int*>(params.padding_offset.data());;
    const int* cu_seqlens_ptr = reinterpret_cast<const int*>(params.cu_seqlens.data());

    PrefixPromptBatchWeightsParam<half> param;

    const int context_batch_size  = params.q_output.shape()[0];
    const int context_seq_len     = params.q_output.shape()[2];
    const int context_h_token_num = params.qkv_input.shape()[0];
    const int local_head_num      = params.q_output.shape()[1];
    const int local_head_num_kv   = params.k_output.shape()[1];
    const int size_per_head       = params.qkv_input.shape()[2];


    std::cout << "batch size is " << context_batch_size << "\n" \
              << "seq length is " << context_seq_len    << "\n" \
              << "context_h_token_num is " << context_h_token_num    << "\n" \
              << "local_head_num is " << local_head_num    << "\n" \
              << "local_head_num_kv is " << local_head_num_kv    << "\n" \
              << "size_per_head is " << size_per_head    << "\n";


    float* scale_out_ptr = nullptr;
    int int8_mode = 0;

    bool use_logn_attention = false;
    const int logn_seq_len = 0;


    // rope 

    const int rope_embedding_dim              = params.rope_config.embedding_dim;
    const int rope_embedding_style            = (int) params.rope_config.embedding_style;
    const int rope_embedding_base             = params.rope_config.embedding_base;
    float rope_dynamic_embedding_scale        = params.rope_config.dynamic_embedding_scale;
    const int rope_dynamic_embedding_max_pos  = params.rope_config.dynamic_embedding_max_pos;
    const int rope_position_embeddings_scale  = params.rope_config.position_embeddings_scale;
    const int rope_base_scale                 = params.rope_config.base_scale;

    invokeAddFusedQKVBiasTranspose(q_output_ptr,
                                   k_output_ptr,
                                   v_output_ptr,
                                   param,  // prefix prompt
                                   qkv_input_ptr,
                                   position_ids,
                                   bias_ptr,
                                   padding_offset,
                                   cu_seqlens_ptr,
                                   context_batch_size,
                                   context_seq_len,
                                   context_h_token_num,
                                   local_head_num,
                                   local_head_num_kv,
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
                                   stream_);

    sync_check_cuda_error();

    gemm({TransposeOperation::NONE,
          TransposeOperation::TRANSPOSE,
          params.q_output,
          params.k_output,
          params.qk_output});

    float scale = (1.0f / sqrtf(size_per_head* 1.0f));

    softmax({params.qk_output, params.softmax_qk_output, params.attention_mask, scale});

    gemm({params.softmax_qk_output, params.v_output, params.qkv_output});

    half* qkv_ptr = reinterpret_cast<half*>(params.qkv_output.data());
    half* qkv_transpose_ptr = reinterpret_cast<half*>(params.qkv_transpose_output.data());
    invokeTransposeQKV(qkv_transpose_ptr,
                       qkv_ptr,
                       context_batch_size,
                       context_seq_len,
                       local_head_num,
                       size_per_head,
                       nullptr,
                       0,
                       stream_);
    sync_check_cuda_error();

}


} // namespace fastertransformer

