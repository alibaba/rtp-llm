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
#include "3rdparty/flash_attention2/flash.h"
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"


using namespace std;

namespace fastertransformer {

tensorrt_llm::kernels::Data_type trtDtypeConvert(DataType dtype)
{
    switch (dtype) {
        case DataType::TYPE_FP16: return tensorrt_llm::kernels::DATA_TYPE_FP16;
#ifdef ENABLE_BF16
        case DataType::TYPE_BF16: return tensorrt_llm::kernels::DATA_TYPE_BF16;
#endif
        default: throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

}

void trtFmha(const AttentionModuleParams& params,
             tensorrt_llm::kernels::FusedMHARunnerV2* mFMHARunner,
             cudaStream_t stream)
{
    auto datatype       = params.input.type();
    auto token_num      = params.input.shape()[0];
    auto batch_size     = params.common.context_batch_size;
    auto seq_len        = params.common.context_max_seq_len;

    auto head_num       = params.configs.head_num;
    auto kv_head_num    = params.configs.kv_head_num;
    auto size_per_head  = params.configs.size_per_head;
    float q_scaling     = params.configs.q_scaling;


    FT_LOG_INFO("use TRT fmha");
    bool mFMHAForceFP32Acc  = false;
    bool mRemovePadding     = false;
    bool is_causal_         = (params.configs.mask_type == AttentionMaskType::causalMask);
    mFMHARunner->setup_flags(mFMHAForceFP32Acc,
                                mRemovePadding,
                                is_causal_,
                                kv_head_num);
    bool is_alibi = false;
    bool is_alibi_with_sacle = false;
    mFMHARunner->setup(batch_size,
                    seq_len,
                    seq_len,
                    token_num,
                    is_alibi,
                    is_alibi_with_sacle,
                    1,
                    0);
    mFMHARunner->run(params.input.data(),
                    params.common.cu_seqlens->data(),
                    params.output.data(), stream);

    sync_check_cuda_error();




}

void OpenSourceFMHA(const AttentionModuleParams& params,
                    void* softmax_lse_,
                    cudaStream_t stream)
{
    auto datatype       = params.input.type();
    auto token_num      = params.input.shape()[0];
    auto batch_size     = params.common.context_batch_size;
    auto seq_len        = params.common.context_max_seq_len;

    auto head_num       = params.configs.head_num;
    auto kv_head_num    = params.configs.kv_head_num;
    auto size_per_head  = params.configs.size_per_head;
    float q_scaling     = params.configs.q_scaling;

    auto      round_multiple    = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(size_per_head, 32);
    const int seqlen_rounded    = round_multiple(seq_len, 128);
    Flash_fwd_params flash_fwd_params_;
    memset(&flash_fwd_params_, 0, sizeof(flash_fwd_params_));
    flash_fwd_params_.is_bf16 = (datatype == DataType::TYPE_BF16);

    const int hidden_units          = head_num * size_per_head;
    const int hidden_units_kv       = kv_head_num * size_per_head;
    flash_fwd_params_.q_ptr = params.input.data();
    flash_fwd_params_.k_ptr = params.input.dataWithOffset(hidden_units);
    flash_fwd_params_.v_ptr = params.input.dataWithOffset(hidden_units + hidden_units_kv);


    flash_fwd_params_.q_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.k_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.v_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.q_head_stride = size_per_head;
    flash_fwd_params_.k_head_stride = size_per_head;
    flash_fwd_params_.v_head_stride = size_per_head;
    flash_fwd_params_.o_ptr         = params.output.data();
    flash_fwd_params_.o_row_stride  = hidden_units;
    flash_fwd_params_.o_head_stride = size_per_head;

    if (params.common.cu_seqlens == nullptr) {
        flash_fwd_params_.q_batch_stride = seq_len * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.k_batch_stride = seq_len * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.v_batch_stride = seq_len * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.o_batch_stride = seq_len * hidden_units;
    }

    flash_fwd_params_.cu_seqlens_q = params.common.cu_seqlens.get()->data<int>();
    flash_fwd_params_.cu_seqlens_k = params.common.cu_seqlens.get()->data<int>();

    // P = softmax(QK^T)
    flash_fwd_params_.p_ptr = nullptr;

    // Softmax sum
    flash_fwd_params_.softmax_lse_ptr = softmax_lse_;

    // Set the dimensions.
    flash_fwd_params_.b                = batch_size;
    flash_fwd_params_.h                = head_num;
    flash_fwd_params_.h_k              = kv_head_num;
    flash_fwd_params_.h_h_k_ratio      = head_num / kv_head_num;
    flash_fwd_params_.seqlen_q         = seq_len;
    flash_fwd_params_.seqlen_k         = seq_len;
    flash_fwd_params_.seqlen_q_rounded = seqlen_rounded;
    flash_fwd_params_.seqlen_k_rounded = seqlen_rounded;
    flash_fwd_params_.d                = size_per_head;
    flash_fwd_params_.d_rounded        = head_size_rounded;

    // Set the different scale values.
    float softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f));
    flash_fwd_params_.scale_softmax      = softmax_scale;
    flash_fwd_params_.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    float p_dropout             = 0.0f;
    flash_fwd_params_.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    flash_fwd_params_.p_dropout_in_uint8_t     = uint8_t(std::floor(flash_fwd_params_.p_dropout * 255.0));
    flash_fwd_params_.rp_dropout               = 1.f / flash_fwd_params_.p_dropout;
    flash_fwd_params_.scale_softmax_rp_dropout = flash_fwd_params_.rp_dropout * flash_fwd_params_.scale_softmax;

    flash_fwd_params_.is_causal = (params.configs.mask_type == AttentionMaskType::causalMask);;
    flash_fwd_params_.is_alibi  = false;
    if (params.common.linear_bias_slopes) {
        flash_fwd_params_.is_alibi           = true;
        flash_fwd_params_.linear_bias_slopes = params.common.linear_bias_slopes.get()->data();
    }
    flash_fwd_params_.is_seqlens_k_cumulative = true;

    run_mha_fwd(flash_fwd_params_, stream);
    sync_check_cuda_error();

}


template <typename T>
void writeContextKvCache(
        const AttentionModuleParams& params,
        const Buffer& k, const Buffer& v,
        cudaStream_t stream)
{
    const auto& kv_blocks = params.common.kv_cache_blocks.value().get();
    const auto batch_size = params.common.context_batch_size;
    RUNTIME_ASSERT_OP_ARG(
        kv_blocks.shape()[0] == batch_size,
        "context attention kv blocks batch size expected [%d] but got shape [%s]",
        batch_size, kv_blocks.debugString().c_str());
    const auto max_blocks_per_batch = kv_blocks.shape()[2];
    KVBlockArray kv_block_array(
        batch_size, max_blocks_per_batch, params.configs.tokens_per_block, 0);
    KvCacheDataType cache_type = KvCacheDataType::BASE;
    kv_block_array.data        = (int64_t *)kv_blocks.data();

    invokeTranspose4dBatchMajor<T>(
        k.data<T>(),
        v.data<T>(),
        kv_block_array,
        batch_size,
        params.common.context_max_seq_len,  // max input length + prefix prompt length
        params.configs.size_per_head,
        params.configs.kv_head_num,
        cache_type,
        nullptr,  // kvScaleOrigQuant
        params.common.input_lengths.dataWithOffset<int32_t>(params.common.decoder_batch_size),
        0, // d_prefix_prompt_lengths,
        stream);
}

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
                                    {"q_output"});

    auto k_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len, size_per_head},
                                    AllocationType::DEVICE},
                                    {"k_output"});

    auto v_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len, size_per_head},
                                    AllocationType::DEVICE},
                                    {"v_output"});

    PrefixPromptBatchWeightsParam prefix_prompt_param;

    // int8
    float*  scale_out_ptr    = nullptr;
    int     int8_mode        = 0;

    // logn attention
    bool        use_logn_attn = params.configs.rope_config.use_logn_attn;
    const int   logn_seq_len  = params.configs.rope_config.logn_seq_len;
    // rope
    const auto rope_embedding_dim              = params.configs.rope_config.embedding_dim;
    const auto rope_embedding_style            = (int) params.configs.rope_config.embedding_style;
    const auto rope_embedding_base             = params.configs.rope_config.embedding_base;
    const auto rope_rotary_embedding_scale     = params.configs.rope_config.rotary_embedding_scale;
    const auto rope_dynamic_embedding_max_pos  = params.configs.rope_config.dynamic_embedding_max_pos;
    const auto rope_base_scale                 = params.configs.rope_config.base_scale;

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype, invokeAddFusedQKVBiasTranspose,
        q_output->data(),
        k_output->data(),
        v_output->data(),
        &prefix_prompt_param,
        params.input.data(),
        params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
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
        rope_rotary_embedding_scale,
        rope_dynamic_embedding_max_pos,
        rope_base_scale,
        logn_seq_len,
        use_logn_attn,
        scale_out_ptr,
        int8_mode,
        stream_
    );

    if (params.common.kv_cache_blocks) {
        ARG_CASTED_FUNC_CALL(half, writeContextKvCache,
            std::cref(params),
            std::cref(*k_output),
            std::cref(*v_output),
            stream_);
    }
#ifdef USE_OLD_TRT_FMHA
    auto mFMHARunnerV1 = new FusedMHARunnerFP16v2(head_num, size_per_head, get_sm(), params.configs.q_scaling);

    bool use_trtv1_fmha_ = use_trtv1_fmha &&
                           mFMHARunnerV1->fmha_supported(
                            (params.configs.mask_type == AttentionMaskType::causalMask)) &&
                           head_num == kv_head_num &&
                           (datatype == DataType::TYPE_FP16);
#endif
    auto mFMHARunner = new tensorrt_llm::kernels::FusedMHARunnerV2(
            trtDtypeConvert(datatype), head_num, size_per_head, params.configs.q_scaling);

    bool use_trtv2_fmha_ = use_trtv2_fmha &&
                           (params.configs.mask_type == AttentionMaskType::causalMask ||
                            params.configs.mask_type == AttentionMaskType::noMask) &&
                            mFMHARunner->fmha_supported();

    bool use_openSource_fmha_ = use_openSource_fmha &&
                                (params.configs.mask_type == AttentionMaskType::causalMask ||
                                params.configs.mask_type == AttentionMaskType::noMask) &&
                                (head_num % kv_head_num == 0) &&
                                ((size_per_head == 64) || (size_per_head == 96) || (size_per_head == 128));

    if (use_trtv2_fmha_) {
        trtFmha(params, mFMHARunner, stream_);

    }
    else if (use_openSource_fmha_) {
        auto softmax_lse_ = allocateBuffer({DataType::TYPE_FP32,
                                            {batch_size, head_num, seq_len},
                                            AllocationType::DEVICE},
                                            {"softmax_lse"});
        OpenSourceFMHA(params,
                       softmax_lse_->data(),
                       stream_);
    }
#ifdef USE_OLD_TRT_FMHA
    else if (use_trtv1_fmha_) {
        if (params.configs.mask_type == AttentionMaskType::causalMask) {
            mFMHARunnerV1->setup_causal_masked_fmha(seq_len, batch_size);
            mFMHARunnerV1->run_causal_masked_fmha(params.input.data(),
                                                  params.common.cu_seqlens.get()->data(),
                                                  params.output.data(),
                                                  true,
                                                  stream_);
        }
        else {
            auto qkv_buf_temp   = allocateBuffer({DataType::TYPE_FP16,
                                                 {token_num, head_num + 2 * kv_head_num, size_per_head},
                                                 AllocationType::DEVICE},
                                                 {"qkv_buf_temp"});
            invokeTransposeAxis12(qkv_buf_temp->data<half>(),
                                  params.input.data<half>(),
                                  token_num,
                                  3,
                                  head_num,
                                  size_per_head,
                                  stream_);
            auto max_length  = mFMHARunnerV1->getSFromMaxSeqLen(seq_len);
            mFMHARunnerV1->setup(max_length, batch_size);
            mFMHARunnerV1->run(qkv_buf_temp->data<half>(),
                               nullptr,
                               params.common.cu_seqlens->data<int>(),
                               nullptr,
                               params.output.data<half>(),
                               stream_);
        }
    }
#endif
    else {
        // TODO(lidongjin): Only support float32 gemm output.
        auto qk_output = gemm({*q_output,
                               *k_output,
                               std::nullopt,
                               nullptr,
                               DataType::TYPE_FP32,
                               TransposeOperation::NONE,
                               TransposeOperation::TRANSPOSE});
        printBufferData(*qk_output, "qk_output: ");

        float scale = (1.0f / sqrtf(size_per_head * 1.0f));

        // TODO(lidongjin): Only support float32(in)\float16(output).
        auto softmax_type = qk_output->type();
        RUNTIME_ASSERT_OP_ARG(
            params.common.attention_mask,
            "attention_mask must be provided for default context attention implementation");
        auto softmax_qk_output = softmax({std::move(qk_output),
                                        *params.common.attention_mask,
                                        scale,
                                        DataType::TYPE_FP16});
        printBufferData(*softmax_qk_output, "softmax_qk_output: ");

        auto qkv_output = gemm({*softmax_qk_output, *v_output});

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
    float rotary_embedding_base  = params.configs.rope_config.embedding_base;
    float rotary_embedding_scale = params.configs.rope_config.rotary_embedding_scale;
    int dynamic_embedding_max_pos = params.configs.rope_config.dynamic_embedding_max_pos;
    int base_scale = params.configs.rope_config.base_scale;

    // logn attention
    bool        use_logn_attn = params.configs.rope_config.use_logn_attn;
    const int   logn_seq_len  = params.configs.rope_config.logn_seq_len;

    // prefix prompt

    int* prefix_prompt_lengths = nullptr;
    if (params.common.prefix_prompt_lengths) {
        prefix_prompt_lengths = params.common.prefix_prompt_lengths->data<int>();
    }

    int max_prefix_prompt_length = 0;
    bool count_prefix_length = false;

    const auto* input_lengths = params.common.input_lengths.data<int>();
    const auto* sequence_lengths = params.common.sequence_lengths.data<int>();

    float q_scaling = params.configs.q_scaling;
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
    const auto max_blocks_per_seq = params.common.kv_cache_blocks.value().get().shape()[2];
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
        rotary_embedding_scale,
        dynamic_embedding_max_pos,
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
