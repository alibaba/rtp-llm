#include "cufmha.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/utils/compiler_config.h"

namespace fastertransformer{

tensorrt_llm::kernels::Data_type trtDtypeConvert(DataType dtype)
{
    switch (dtype) {
        case DataType::TYPE_FP16: return tensorrt_llm::kernels::DATA_TYPE_FP16;
#ifdef ENABLE_BF16
        case DataType::TYPE_BF16: return tensorrt_llm::kernels::DATA_TYPE_BF16;
#endif
        default: throw std::runtime_error("not support dtype");
    }

}



bool cufmha::trtV1FmhaSupport() {
#ifdef USE_OLD_TRT_FMHA
    trtv1_fmha_runner_.reset(
    new FusedMHARunnerFP16v2(head_num_, size_per_head_, get_sm(), q_scaling_));

    return trtv1_fmha_runner_->fmha_supported(mtype_ == AttentionMaskType::causalMask) &&
        (head_num_ == kv_head_num_) && (dtype_ == DataType::TYPE_FP16);
#else
    return false;
#endif
}

void cufmha::runTrtV1Fmha(void* input,
                          void* cu_seqlens,
                          void* output,
                          void* qkv_buf_temp,
                          size_t batch_size, 
                          size_t seq_len,
                          size_t token_num)
{
#ifdef USE_OLD_TRT_FMHA
    if (mtype_ == AttentionMaskType::causalMask) {
        trtv1_fmha_runner_->setup_causal_masked_fmha(seq_len, batch_size);
        trtv1_fmha_runner_->run_causal_masked_fmha(input,
                                                cu_seqlens,
                                                output,
                                                true,
                                                stream_);
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(dtype_, invokeTransposeAxis12,
                                        qkv_buf_temp,
                                        input,
                                        token_num,
                                        3,
                                        head_num_,
                                        size_per_head_,
                                        stream_);

        auto max_length  = trtv1_fmha_runner_->getSFromMaxSeqLen(seq_len);
        trtv1_fmha_runner_->setup(max_length, batch_size);
        trtv1_fmha_runner_->run(qkv_buf_temp,
                                nullptr,
                                cu_seqlens,
                                nullptr,
                                output,
                                stream_);
    }
#else
    return;
#endif

}


bool cufmha::trtV2FmhaSupport() {
    trtv2_fmha_runner_.reset(
        new tensorrt_llm::kernels::FusedMHARunnerV2(
            trtDtypeConvert(dtype_), head_num_, size_per_head_, q_scaling_));
    
    return trtv2_fmha_runner_->fmha_supported() &&
           (mtype_ == AttentionMaskType::causalMask ||
            mtype_ == AttentionMaskType::noMask);
}

bool cufmha::openSourceFmhaSupport()
{
    return (head_num_ % kv_head_num_ == 0) &&
           (mtype_ == AttentionMaskType::causalMask ||
            mtype_ == AttentionMaskType::noMask) &&
           ((size_per_head_ == 64) || (size_per_head_ == 96) || (size_per_head_ == 128));
}

void cufmha::runTrtV2Fmha(void* input,
                          void* cu_seqlens,
                          void* output,
                          size_t batch_size,
                          size_t seq_len,
                          size_t token_num,
                          bool mFMHAForceFP32Acc,
                          bool mRemovePadding,
                          bool is_alibi,
                          bool is_alibi_with_sacle) {

    trtv2_fmha_runner_->setup_flags(mFMHAForceFP32Acc,
                                    mRemovePadding,
                                    (mtype_ == AttentionMaskType::causalMask),
                                    kv_head_num_);
    
    trtv2_fmha_runner_->setup(batch_size,
                              seq_len,
                              seq_len,
                              token_num,
                              is_alibi,
                              is_alibi_with_sacle,
                              1,
                              0);
    trtv2_fmha_runner_->run(input,
                            cu_seqlens,
                            output, 
                            stream_);

    sync_check_cuda_error();
}

void cufmha::runOpenSourceFmha(void* q,
                               void* k,
                               void* v,
                               void* output,
                               int* cu_seqlens,
                               void* softmax_lse_,
                               size_t token_num,
                               size_t batch_size,
                               size_t seq_len,
                               void* linear_bias_slopes)
{
    auto      round_multiple    = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(size_per_head_, 32);
    const int seqlen_rounded    = round_multiple(seq_len, 128);
    Flash_fwd_params flash_fwd_params_;
    memset(&flash_fwd_params_, 0, sizeof(flash_fwd_params_));
    flash_fwd_params_.is_bf16 = (dtype_ == DataType::TYPE_BF16);
    const int hidden_units = head_num_ * size_per_head_;
    const int hidden_units_kv = kv_head_num_ * size_per_head_;
    flash_fwd_params_.q_ptr = q;
    flash_fwd_params_.k_ptr = k;
    flash_fwd_params_.v_ptr = v;

    flash_fwd_params_.q_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.k_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.v_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.q_head_stride = size_per_head_;
    flash_fwd_params_.k_head_stride = size_per_head_;
    flash_fwd_params_.v_head_stride = size_per_head_;
    flash_fwd_params_.o_ptr         = output;
    flash_fwd_params_.o_row_stride  = hidden_units;
    flash_fwd_params_.o_head_stride = size_per_head_;

    if (cu_seqlens == nullptr) {
        flash_fwd_params_.q_batch_stride = seq_len * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.k_batch_stride = seq_len * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.v_batch_stride = seq_len * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.o_batch_stride = seq_len * hidden_units;
    }

    flash_fwd_params_.cu_seqlens_q = cu_seqlens;
    flash_fwd_params_.cu_seqlens_k = cu_seqlens;

    // P = softmax(QK^T)
    flash_fwd_params_.p_ptr = nullptr;

    // Softmax sum
    flash_fwd_params_.softmax_lse_ptr = softmax_lse_;

    // Set the dimensions.
    flash_fwd_params_.b                = batch_size;
    flash_fwd_params_.h                = head_num_;
    flash_fwd_params_.h_k              = kv_head_num_;
    flash_fwd_params_.h_h_k_ratio      = head_num_ / kv_head_num_;
    flash_fwd_params_.seqlen_q         = seq_len;
    flash_fwd_params_.seqlen_k         = seq_len;
    flash_fwd_params_.seqlen_q_rounded = seqlen_rounded;
    flash_fwd_params_.seqlen_k_rounded = seqlen_rounded;
    flash_fwd_params_.d                = size_per_head_;
    flash_fwd_params_.d_rounded        = head_size_rounded;

    // Set the different scale values.
    float softmax_scale = (1.0f / sqrtf(size_per_head_ * 1.0f));
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

    flash_fwd_params_.is_causal = (mtype_ == AttentionMaskType::causalMask);
    flash_fwd_params_.is_alibi  = false;
    if (linear_bias_slopes) {
        flash_fwd_params_.is_alibi           = true;
        flash_fwd_params_.linear_bias_slopes = linear_bias_slopes;
    }
    flash_fwd_params_.is_seqlens_k_cumulative = true;

    run_mha_fwd(flash_fwd_params_, stream_);
    sync_check_cuda_error();
}

}