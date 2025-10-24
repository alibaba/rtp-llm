#include "cufmha.h"
#include "fmha_profiling_interface.h"
#include "rtp_llm/cpp/kernels/tensor_ops_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"

using namespace tensorrt_llm::kernels;

namespace rtp_llm {

tensorrt_llm::kernels::Data_type trtDtypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_FP16:
            return tensorrt_llm::kernels::DATA_TYPE_FP16;
        case DataType::TYPE_BF16:
            return tensorrt_llm::kernels::DATA_TYPE_BF16;
#ifdef ENABLE_FP8
        case DataType::TYPE_FP8_E4M3:
            return tensorrt_llm::kernels::DATA_TYPE_E4M3;
#endif
        default:
            throw std::runtime_error("cufmha not support dtype: " + std::to_string(static_cast<int>(dtype)));
    }
}

cufmha::cufmha(DataType          dtype,
               AttentionMaskType mtype,
               size_t            head_num,
               size_t            kv_head_num,
               size_t            size_per_head,
               size_t            seq_size_per_block,
               float             q_scaling,
               bool              use_linear_bias_slopes,
               bool              can_use_trtv1_fmha,
               bool              can_use_trtv2_fmha,
               bool              can_use_trtv2_fmha_paged,
               bool              can_use_open_source_fmha,
               bool              can_use_open_source_fmha_paged,
               bool              is_s_padded,
               cudaStream_t      stream) {

    dtype_         = dtype;
    mtype_         = mtype;
    head_num_      = head_num;
    kv_head_num_   = kv_head_num;
    size_per_head_ = size_per_head;
    is_s_padded_   = is_s_padded;
    // size_per_head_v_ equals to size_per_head in general.
    size_per_head_v_           = size_per_head;
    seq_size_per_block_        = seq_size_per_block;
    q_scaling_                 = q_scaling;
    use_linear_bias_slopes_    = use_linear_bias_slopes;
    support_trt_v1_fmha_       = can_use_trtv1_fmha && initTrtV1FmhaAndCheckSupport();
    support_trt_v2_fhma_       = can_use_trtv2_fmha && initTrtV2FmhaAndCheckSupport();
    support_trt_v2_paged_fmha_ = can_use_trtv2_fmha_paged && initTrtV2FmhaPagedAndCheckSupport();
    // sm 90 use open source has bug currently
    support_open_source_fmha_ = (can_use_open_source_fmha || can_use_open_source_fmha_paged)
                                && initOpenSourceFmhaAndCheckSupport() && get_sm() < 90;
    stream_ = stream;
}

cudaStream_t cufmha::getStream() {
    auto run_stream = stream_;
    if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
        run_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    }
    return run_stream;
}

bool cufmha::checkSignature(DataType          dtype,
                            AttentionMaskType mtype,
                            size_t            head_num,
                            size_t            kv_head_num,
                            size_t            size_per_head,
                            float             q_scaling,
                            bool              use_linear_bias_slopes) {
    return dtype == dtype_ && mtype == mtype_ && head_num == head_num_ && kv_head_num == kv_head_num_
           && size_per_head == size_per_head_ && q_scaling == q_scaling_
           && use_linear_bias_slopes == use_linear_bias_slopes_;
}

bool cufmha::initTrtV1FmhaAndCheckSupport() {
#ifdef USE_OLD_TRT_FMHA
    trtv1_fmha_runner_.reset(new FusedMHARunnerFP16v2(head_num_, size_per_head_, get_sm(), q_scaling_));

    return trtv1_fmha_runner_->fmha_supported(mtype_ == AttentionMaskType::causalMask) && (head_num_ == kv_head_num_)
           && (dtype_ == DataType::TYPE_FP16);
#else
    return false;
#endif
}

void cufmha::runTrtV1Fmha(void*  input,
                          void*  cu_seqlens,
                          void*  output,
                          void*  qkv_buf_temp,
                          size_t batch_size,
                          size_t seq_len,
                          size_t token_num) {
#ifdef USE_OLD_TRT_FMHA
    auto run_stream = getStream();
    if (mtype_ == AttentionMaskType::causalMask) {
        trtv1_fmha_runner_->setup_causal_masked_fmha(seq_len, batch_size);
        trtv1_fmha_runner_->run_causal_masked_fmha(input, cu_seqlens, output, true, run_stream);
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            dtype_, invokeTransposeAxis12, qkv_buf_temp, input, token_num, 3, head_num_, size_per_head_, run_stream);

        auto max_length = trtv1_fmha_runner_->getSFromMaxSeqLen(seq_len);
        trtv1_fmha_runner_->setup(max_length, batch_size);
        trtv1_fmha_runner_->run(qkv_buf_temp, nullptr, cu_seqlens, nullptr, output, run_stream);
    }
#else
    return;
#endif
}

MHARunnerFixedParams cufmha::createMHARunnerFixedParams(bool paged, bool isSPadded) {
    MHARunnerFixedParams fixedParams;
    fixedParams.dataType     = trtDtypeConvert(dtype_);
    fixedParams.dataTypeKv   = trtDtypeConvert(dtype_);
    fixedParams.dataTypeOut  = trtDtypeConvert(dtype_);
    fixedParams.forceFp32Acc = false;
    fixedParams.attentionMaskType =
        mtype_ == AttentionMaskType::causalMask ? ContextAttentionMaskType::CAUSAL : ContextAttentionMaskType::PADDING;
    fixedParams.attentionInputLayout      = paged ? AttentionInputLayout::Q_PAGED_KV : AttentionInputLayout::PACKED_QKV;
    fixedParams.isSPadded                 = isSPadded;
    fixedParams.numQHeads                 = head_num_;
    fixedParams.numKvHeads                = kv_head_num_;
    fixedParams.numTokensPerBlock         = seq_size_per_block_;
    fixedParams.headSize                  = size_per_head_;
    fixedParams.headSizeV                 = size_per_head_v_;
    fixedParams.qScaling                  = q_scaling_;
    fixedParams.attnLogitSoftcappingScale = 0.f;
    fixedParams.hasAlibi                  = use_linear_bias_slopes_;
    fixedParams.scaleAlibi                = false;
    fixedParams.saveSoftmax               = false;
    return fixedParams;
}

MHARunnerParams cufmha::createMHARunnerParams(void*        input,
                                              void*        cu_seqlens,
                                              void*        cu_kv_seqlens,
                                              void*        output,
                                              uint32_t*    tile_counter_ptr,
                                              float*       attention_output_orig_quant_scale,
                                              size_t       batch_size,
                                              size_t       max_input_length,
                                              size_t       max_kv_length,
                                              size_t       total_q_seq_len,
                                              size_t       total_kv_seq_len,
                                              KVBlockArray kv_block_array,
                                              void*        custom_mask) {
    MHARunnerParams runnerParams;
    runnerParams.b = batch_size;
    // no in use ?
    runnerParams.numGroupedHeads      = head_num_;
    runnerParams.qSeqLen              = max_input_length;
    runnerParams.kvSeqLen             = max_kv_length;
    runnerParams.slidingWindowSize    = total_kv_seq_len;  // TODO: support sliding window
    runnerParams.chunkedAttentionSize = INT_MAX;           // TODO: support chunked attention
    runnerParams.totalQSeqLen         = total_q_seq_len;
    runnerParams.totalKvSeqLen        = total_kv_seq_len;
    runnerParams.qkvPtr               = input;
    // for page attention, input is Q
    runnerParams.qPtr              = input;
    runnerParams.kvPtr             = nullptr;
    runnerParams.pagedKvCache      = kv_block_array;
    runnerParams.outputPtr         = output;
    runnerParams.outputSfPtr       = nullptr;
    runnerParams.softmaxStatsPtr   = nullptr;
    runnerParams.attentionSinksPtr = nullptr;
    runnerParams.packedMaskPtr     = nullptr;
    runnerParams.cuQSeqLenPtr      = cu_seqlens;
    runnerParams.cuKvSeqLenPtr     = cu_kv_seqlens;
    runnerParams.cuMaskRowsPtr     = nullptr;
    runnerParams.tileCounterPtr    = tile_counter_ptr;

    // TODO: fix scaleBmm1Ptr and scaleBmm2Ptr
    runnerParams.scaleBmm1Ptr = nullptr;
    runnerParams.scaleBmm2Ptr = attention_output_orig_quant_scale;
    runnerParams.oSfScalePtr  = attention_output_orig_quant_scale;
    runnerParams.stream       = getStream();
    // for sage attention
    runnerParams.qScalePtr  = nullptr;
    runnerParams.kScalePtr  = nullptr;
    runnerParams.vScalePtr  = nullptr;
    runnerParams.qMaxNBlock = 0;
    runnerParams.kMaxNBlock = 0;
    runnerParams.vMaxNBlock = 0;
    return runnerParams;
}

// for cuda graph batch prefill test
void cufmha::setIsPadded(bool is_s_padded) {
    is_s_padded_                      = is_s_padded;
    MHARunnerFixedParams fixedParams1 = createMHARunnerFixedParams(false, is_s_padded_);
    trtv2_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2(fixedParams1));
    MHARunnerFixedParams fixedParams2 = createMHARunnerFixedParams(true, is_s_padded_);
    trtv2_paged_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2(fixedParams2));
}

bool cufmha::initTrtV2FmhaAndCheckSupport() {
    if (get_sm() == tensorrt_llm::kernels::kSM_70) {
        trtv2_sm70_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2Sm70(
            trtDtypeConvert(dtype_), head_num_, size_per_head_, q_scaling_));
        return trtv2_sm70_fmha_runner_->fmha_supported()
               && (mtype_ == AttentionMaskType::causalMask || mtype_ == AttentionMaskType::noMask)
               && !(mtype_ == AttentionMaskType::noMask && use_linear_bias_slopes_);
    }
    if (get_sm() < tensorrt_llm::kernels::kSM_80) {
        RTP_LLM_LOG_INFO("cuda sm %d < 80, not support trt v2 fmha", get_sm());
        return false;
    }
    MHARunnerFixedParams fixedParams = createMHARunnerFixedParams(false, is_s_padded_);
    trtv2_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2(fixedParams));

    return trtv2_fmha_runner_->isFmhaSupported()
           && (mtype_ == AttentionMaskType::causalMask || mtype_ == AttentionMaskType::noMask)
           && !(mtype_ == AttentionMaskType::noMask && use_linear_bias_slopes_);
}

bool cufmha::initTrtV2FmhaPagedAndCheckSupport() {
    if (get_sm() < tensorrt_llm::kernels::kSM_80) {
        RTP_LLM_LOG_INFO("cuda sm %d < 80, not support trt paged fmha", get_sm());
        return false;
    }
    MHARunnerFixedParams fixedParams = createMHARunnerFixedParams(true, is_s_padded_);
    trtv2_paged_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2(fixedParams));
    return trtv2_paged_fmha_runner_->isFmhaSupported()
           && (mtype_ == AttentionMaskType::causalMask || mtype_ == AttentionMaskType::noMask)
           && !(mtype_ == AttentionMaskType::noMask && use_linear_bias_slopes_);
}

bool cufmha::initOpenSourceFmhaAndCheckSupport() {
    return (kv_head_num_ != 0 && head_num_ % kv_head_num_ == 0)
           && (mtype_ == AttentionMaskType::causalMask || mtype_ == AttentionMaskType::noMask)
           && ((size_per_head_ == 64) || (size_per_head_ == 96) || (size_per_head_ == 128) || (size_per_head_ == 192));
}

void cufmha::runTrtV2FmhaPaged(void*        input,
                               void*        cu_q_seqlens,
                               void*        cu_kv_seqlens,
                               void*        output,
                               uint32_t*    tile_counter_ptr,
                               float*       attention_output_orig_quant_scale,
                               size_t       batch_size,
                               size_t       max_input_seq_len,
                               size_t       max_past_kv_len,
                               size_t       token_num,
                               size_t       token_num_kv,
                               KVBlockArray kv_block_array,
                               void*        custom_mask) {
    auto runnerParams = createMHARunnerParams(input,
                                              cu_q_seqlens,
                                              cu_kv_seqlens,
                                              output,
                                              tile_counter_ptr,
                                              attention_output_orig_quant_scale,
                                              batch_size,
                                              max_input_seq_len,
                                              max_past_kv_len,
                                              token_num,
                                              token_num_kv,
                                              kv_block_array,
                                              custom_mask);
    trtv2_paged_fmha_runner_->run(runnerParams);
    check_cuda_error();
}

void cufmha::runTrtV2Fmha(void*        input,
                          void*        cu_seqlens,
                          void*        output,
                          uint32_t*    tile_counter_ptr,
                          float*       attention_output_orig_quant_scale,
                          size_t       batch_size,
                          size_t       max_seq_len,
                          size_t       token_num,
                          KVBlockArray kv_block_array,
                          void*        custom_mask) {
    if (trtv2_fmha_runner_) {
        auto runnerParams = createMHARunnerParams(input,
                                                  cu_seqlens,
                                                  cu_seqlens,
                                                  output,
                                                  tile_counter_ptr,
                                                  attention_output_orig_quant_scale,
                                                  batch_size,
                                                  max_seq_len,
                                                  max_seq_len,
                                                  token_num,
                                                  token_num,
                                                  kv_block_array,
                                                  custom_mask);
        trtv2_fmha_runner_->run(runnerParams);
    } else {
        trtv2_sm70_fmha_runner_->setup_flags(false, false, (mtype_ == AttentionMaskType::causalMask), kv_head_num_);

        trtv2_sm70_fmha_runner_->setup(
            batch_size, max_seq_len, max_seq_len, token_num, use_linear_bias_slopes_, false, 1, 0);
        trtv2_sm70_fmha_runner_->run(input, cu_seqlens, output, stream_);
    }
    check_cuda_error();
}

void cufmha::runOpenSourceFmhaPaged(void*            q,
                                    void*            k,
                                    void*            v,
                                    void*            output,
                                    int*             cu_seqlens,
                                    int*             cu_kv_seqlens,
                                    int*             block_table,
                                    size_t           batch_size,
                                    size_t           block_table_batch_stride,
                                    size_t           seq_size_per_block,
                                    size_t           seq_len,
                                    void*            workspace,
                                    DeviceInitParams device_params,
                                    float*           linear_bias_slopes,
                                    float            softmax_extra_scale) {
    RTP_LLM_CHECK_WITH_INFO(head_num_ % kv_head_num_ == 0,
                            "Number of heads in key/value must divide number of heads in query");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block % 256 == 0,
                            "open source fmha paged seq_size_per_block must be divided by 256");
    const auto seq_len_round = roundMultiple(seq_len, 32);

    RTP_LLM_CHECK_WITH_INFO(block_table, "open source paged must have block_table");
    Flash_fwd_params flash_fwd_params = genFlashFwdParams(q,
                                                          k,
                                                          v,
                                                          output,
                                                          cu_seqlens,
                                                          cu_kv_seqlens,
                                                          workspace,
                                                          batch_size,
                                                          seq_len,
                                                          seq_size_per_block * block_table_batch_stride,
                                                          linear_bias_slopes,
                                                          softmax_extra_scale);
    const int        hidden_units_kv  = kv_head_num_ * size_per_head_;

    // Set the pointers and strides.
    // All stride are in elements, not bytes.
    flash_fwd_params.k_row_stride   = size_per_head_;
    flash_fwd_params.v_row_stride   = size_per_head_;
    flash_fwd_params.k_head_stride  = seq_size_per_block * size_per_head_;
    flash_fwd_params.v_head_stride  = seq_size_per_block * size_per_head_;
    flash_fwd_params.k_batch_stride = seq_size_per_block * (hidden_units_kv) * 2;
    flash_fwd_params.v_batch_stride = seq_size_per_block * (hidden_units_kv) * 2;

    flash_fwd_params.block_table              = block_table;
    flash_fwd_params.block_table_batch_stride = block_table_batch_stride;
    flash_fwd_params.page_block_size          = seq_size_per_block;

    flash_fwd_params.num_splits = getNumSplits(batch_size, seq_len, seq_size_per_block * block_table_batch_stride);
    auto run_stream             = getStream();
    RTP_LLM_CHECK_WITH_INFO(flash_fwd_params.num_splits <= 128, "open source not support split head 128");
    if (flash_fwd_params.num_splits > 1) {
        flash_fwd_params.softmax_lseaccum_ptr =
            (char*)workspace + sizeof(float) * batch_size * head_num_ * seq_len_round;
        flash_fwd_params.oaccum_ptr =
            (char*)(flash_fwd_params.softmax_lseaccum_ptr)
            + sizeof(float) * flash_fwd_params.num_splits * batch_size * head_num_ * seq_len_round;
    }
    // export FMHA_SHOW_PARAMS=1
    rtp_llm::fmha::FmhaProfParam fmha_prof_params;
    if (rtp_llm::fmha::ProfilingInterface::Instance(device_params.fmha_config).get_op_info()) {
        fmha_prof_params.set_flash_attn_params(true /*dir*/,
                                               flash_fwd_params.is_bf16 /*data_type*/,
                                               flash_fwd_params.is_causal /*custom_mask*/,
                                               flash_fwd_params.b /*batch_size*/,
                                               flash_fwd_params.h /*num_heads*/,
                                               flash_fwd_params.h_k /*num_heads_k*/,
                                               flash_fwd_params.d /*head_dim*/,
                                               flash_fwd_params.d /*head_dim_value*/,
                                               flash_fwd_params.seqlen_q /*seqlen_q*/,
                                               flash_fwd_params.seqlen_k /*seqlen_k*/,
                                               flash_fwd_params.p_dropout /*dropout*/,
                                               flash_fwd_params.scale_softmax /*scale*/,
                                               flash_fwd_params.window_size_left,  /*window_size_left*/
                                               flash_fwd_params.window_size_right, /*window_size_right*/
                                               true /*is_fixed_seqs**/,
                                               flash_fwd_params.alibi_slopes_ptr != nullptr /*alibi*/
        );
    }
    rtp_llm::fmha::ProfilingInterface::Instance(device_params.fmha_config).instrument(true, fmha_prof_params);
    try {
        run_mha_fwd(flash_fwd_params, run_stream, block_table);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("run opensource paged flash attention failed, err: %s", e.what());
        throw;
    }
    rtp_llm::fmha::ProfilingInterface::Instance(device_params.fmha_config).instrument(false, fmha_prof_params);
    check_cuda_error();
}

void cufmha::runOpenSourceFmha(void*            q,
                               void*            k,
                               void*            v,
                               void*            output,
                               int*             cu_seqlens,
                               size_t           batch_size,
                               size_t           seq_len,
                               void*            workspace,
                               DeviceInitParams device_params,
                               float*           linear_bias_slopes,
                               float            softmax_extra_scale) {
    auto             run_stream       = getStream();
    Flash_fwd_params flash_fwd_params = genFlashFwdParams(q,
                                                          k,
                                                          v,
                                                          output,
                                                          cu_seqlens,
                                                          cu_seqlens,
                                                          workspace,
                                                          batch_size,
                                                          seq_len,
                                                          seq_len,
                                                          linear_bias_slopes,
                                                          softmax_extra_scale);
    // export FMHA_SHOW_PARAMS=1
    rtp_llm::fmha::FmhaProfParam fmha_prof_params;
    if (rtp_llm::fmha::ProfilingInterface::Instance(device_params.fmha_config).get_op_info()) {
        fmha_prof_params.set_flash_attn_params(true /*dir*/,
                                               flash_fwd_params.is_bf16 /*data_type*/,
                                               flash_fwd_params.is_causal /*custom_mask*/,
                                               flash_fwd_params.b /*batch_size*/,
                                               flash_fwd_params.h /*num_heads*/,
                                               flash_fwd_params.h_k /*num_heads_k*/,
                                               flash_fwd_params.d /*head_dim*/,
                                               flash_fwd_params.d /*head_dim_value*/,
                                               flash_fwd_params.seqlen_q /*seqlen_q*/,
                                               flash_fwd_params.seqlen_k /*seqlen_k*/,
                                               flash_fwd_params.p_dropout /*dropout*/,
                                               flash_fwd_params.scale_softmax /*scale*/,
                                               flash_fwd_params.window_size_left,  /*window_size_left*/
                                               flash_fwd_params.window_size_right, /*window_size_right*/
                                               true /*is_fixed_seqs**/,
                                               flash_fwd_params.alibi_slopes_ptr != nullptr /*alibi*/
        );
    }
    rtp_llm::fmha::ProfilingInterface::Instance(device_params.fmha_config).instrument(true, fmha_prof_params);
    try {
        run_mha_fwd(flash_fwd_params, run_stream, false);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("run opensource flash attention failed, err: %s", e.what());
        throw;
    }
    rtp_llm::fmha::ProfilingInterface::Instance(device_params.fmha_config).instrument(false, fmha_prof_params);
    check_cuda_error();
}

Flash_fwd_params cufmha::genFlashFwdParams(void*  q,
                                           void*  k,
                                           void*  v,
                                           void*  output,
                                           int*   cu_seqlens,
                                           int*   cu_kv_seqlens,
                                           void*  softmax_lse,
                                           size_t batch_size,
                                           size_t seq_len_q,
                                           size_t seq_len_kv,
                                           float* linear_bias_slopes,
                                           float  softmax_extra_scale) const {
    const int        head_size_rounded = roundMultiple(size_per_head_, 32);
    const int        seqlen_q_rounded  = roundMultiple(seq_len_q, 128);
    const int        seqlen_kv_rounded = roundMultiple(seq_len_kv, 128);
    Flash_fwd_params flash_fwd_params;
    memset(&flash_fwd_params, 0, sizeof(flash_fwd_params));
    flash_fwd_params.is_bf16  = (dtype_ == DataType::TYPE_BF16);
    const int hidden_units    = head_num_ * size_per_head_;
    const int hidden_units_kv = kv_head_num_ * size_per_head_;
    flash_fwd_params.q_ptr    = q;
    flash_fwd_params.k_ptr    = k;
    flash_fwd_params.v_ptr    = v;

    flash_fwd_params.q_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params.k_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params.v_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params.q_head_stride = size_per_head_;
    flash_fwd_params.k_head_stride = size_per_head_;
    flash_fwd_params.v_head_stride = size_per_head_;
    flash_fwd_params.o_ptr         = output;
    flash_fwd_params.o_row_stride  = hidden_units;
    flash_fwd_params.o_head_stride = size_per_head_;

    flash_fwd_params.q_batch_stride = seq_len_q * (hidden_units + 2 * hidden_units_kv);
    flash_fwd_params.k_batch_stride = seq_len_kv * (hidden_units + 2 * hidden_units_kv);
    flash_fwd_params.v_batch_stride = seq_len_kv * (hidden_units + 2 * hidden_units_kv);
    flash_fwd_params.o_batch_stride = seq_len_q * hidden_units;

    flash_fwd_params.cu_seqlens_q = cu_seqlens;
    flash_fwd_params.cu_seqlens_k = cu_kv_seqlens;

    // P = softmax(QK^T)
    flash_fwd_params.p_ptr = nullptr;

    // Softmax sum
    flash_fwd_params.softmax_lse_ptr = softmax_lse;

    // Set the dimensions.
    flash_fwd_params.b                = batch_size;
    flash_fwd_params.h                = head_num_;
    flash_fwd_params.h_k              = kv_head_num_;
    flash_fwd_params.h_h_k_ratio      = head_num_ / kv_head_num_;
    flash_fwd_params.seqlen_q         = seq_len_q;
    flash_fwd_params.seqlen_k         = seq_len_kv;
    flash_fwd_params.seqlen_q_rounded = seqlen_q_rounded;
    flash_fwd_params.seqlen_k_rounded = seqlen_kv_rounded;
    flash_fwd_params.d                = size_per_head_;
    flash_fwd_params.d_rounded        = head_size_rounded;

    // Set the different scale values.
    float softmax_scale                 = (1.0f / sqrtf(size_per_head_ * 1.0f)) * softmax_extra_scale;
    flash_fwd_params.scale_softmax      = softmax_scale;
    flash_fwd_params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    float p_dropout            = 0.0f;
    flash_fwd_params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    flash_fwd_params.p_dropout_in_uint8_t     = uint8_t(std::floor(flash_fwd_params.p_dropout * 255.0));
    flash_fwd_params.rp_dropout               = 1.f / flash_fwd_params.p_dropout;
    flash_fwd_params.scale_softmax_rp_dropout = flash_fwd_params.rp_dropout * flash_fwd_params.scale_softmax;

    flash_fwd_params.is_causal = (mtype_ == AttentionMaskType::causalMask);
    if (linear_bias_slopes) {
        flash_fwd_params.alibi_slopes_ptr = linear_bias_slopes;
    }
    flash_fwd_params.is_seqlens_k_cumulative = true;
    return flash_fwd_params;
}

size_t cufmha::getOpenSourceWorkSpaceSize(size_t batch_size, size_t seqlen_q, size_t seqlen_k, bool paged) {
    size_t       total_size     = 0;
    const size_t seqlen_q_round = roundMultiple(seqlen_q, 32);
    total_size += sizeof(float) * batch_size * head_num_ * seqlen_q_round;  // softmax_lse
    if (paged) {
        const size_t num_splits = getNumSplits(batch_size, seqlen_q, seqlen_k);
        if (num_splits > 1) {
            total_size += sizeof(float) * num_splits * batch_size * head_num_ * seqlen_q_round;  // softmax_lseaccum
            total_size += sizeof(float) * num_splits * batch_size * head_num_ * roundMultiple(size_per_head_, 32)
                          * seqlen_q_round;  // oaccum
        }
    }
    return total_size;
}

int cufmha::getNumSplits(size_t batch_size, size_t seqlen_q, size_t seqlen_k) const {
    if (seqlen_q > 1) {
        return 0;
    }
    int       multi_processor_count = getMultiProcessorCount();
    const int block_n               = size_per_head_ <= 64 ? 256 : (size_per_head_ <= 128 ? 128 : 64);
    const int num_n_blocks          = (seqlen_k + block_n - 1) / block_n;
    const int num_m_blocks          = (seqlen_q + 64 - 1) / 64;
    return num_splits_heuristic(batch_size * head_num_ * num_m_blocks, multi_processor_count, num_n_blocks, 128);
}

}  // namespace rtp_llm
