#include "rtp_llm/cpp/cuda/cufmha/TrtV2FmhaRunner.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

namespace {

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
            throw std::runtime_error("TrtV2FmhaRunner not support dtype: " + std::to_string(static_cast<int>(dtype)));
    }
}

}  // namespace

TrtV2FmhaRunner::TrtV2FmhaRunner(const TrtV2FmhaRunnerConfig& config,
                                 DataType                     attn_dtype,
                                 bool                         is_s_padded,
                                 cudaStream_t                 stream):
    config_(config),
    attn_dtype_(attn_dtype),
    is_s_padded_(is_s_padded),
    q_scaling_(config.q_scaling / config.softmax_extra_scale),
    stream_(stream) {

    // 初始化 TRT V2 FMHA
    support_trt_v2_fmha_       = initTrtV2FmhaAndCheckSupport();
    support_trt_v2_paged_fmha_ = initTrtV2FmhaPagedAndCheckSupport();
}

bool TrtV2FmhaRunner::initTrtV2FmhaAndCheckSupport() {
    if (get_sm() == tensorrt_llm::kernels::kSM_70) {
        trtv2_sm70_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2Sm70(
            trtDtypeConvert(attn_dtype_), config_.head_num, config_.size_per_head, q_scaling_));
        return trtv2_sm70_fmha_runner_->fmha_supported();
    }

    if (get_sm() < tensorrt_llm::kernels::kSM_80) {
        RTP_LLM_LOG_DEBUG("cuda sm %d < 80, not support trt v2 fmha", get_sm());
        return false;
    }

    auto fixedParams = createMHARunnerFixedParams(false);
    trtv2_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2(fixedParams));

    return trtv2_fmha_runner_->isFmhaSupported();
}

bool TrtV2FmhaRunner::initTrtV2FmhaPagedAndCheckSupport() {
    if (get_sm() < tensorrt_llm::kernels::kSM_80) {
        RTP_LLM_LOG_DEBUG("cuda sm %d < 80, not support trt v2 paged fmha", get_sm());
        return false;
    }

    auto fixedParams = createMHARunnerFixedParams(true);
    trtv2_paged_fmha_runner_.reset(new tensorrt_llm::kernels::FusedMHARunnerV2(fixedParams));

    return trtv2_paged_fmha_runner_->isFmhaSupported();
}

tensorrt_llm::kernels::MHARunnerFixedParams TrtV2FmhaRunner::createMHARunnerFixedParams(bool paged) {
    tensorrt_llm::kernels::MHARunnerFixedParams fixedParams;
    fixedParams.dataType             = trtDtypeConvert(attn_dtype_);
    fixedParams.dataTypeKv           = trtDtypeConvert(attn_dtype_);
    fixedParams.dataTypeOut          = trtDtypeConvert(attn_dtype_);
    fixedParams.forceFp32Acc         = false;
    fixedParams.attentionMaskType    = config_.is_causal ? tensorrt_llm::kernels::ContextAttentionMaskType::CAUSAL :
                                                           tensorrt_llm::kernels::ContextAttentionMaskType::PADDING;
    fixedParams.attentionInputLayout = paged ? tensorrt_llm::kernels::AttentionInputLayout::Q_PAGED_KV :
                                               tensorrt_llm::kernels::AttentionInputLayout::PACKED_QKV;
    fixedParams.isSPadded            = is_s_padded_;
    fixedParams.numQHeads            = config_.head_num;
    fixedParams.numKvHeads           = config_.kv_head_num;
    fixedParams.numTokensPerBlock    = config_.tokens_per_block;
    fixedParams.headSize             = config_.size_per_head;
    fixedParams.headSizeV            = config_.size_per_head;
    fixedParams.qScaling             = q_scaling_;
    fixedParams.attnLogitSoftcappingScale = 0.f;
    fixedParams.hasAlibi                  = false;
    fixedParams.scaleAlibi                = false;
    fixedParams.saveSoftmax               = false;
    return fixedParams;
}

tensorrt_llm::kernels::MHARunnerParams TrtV2FmhaRunner::createMHARunnerParams(void*     input,
                                                                              void*     cu_seqlens,
                                                                              void*     cu_kv_seqlens,
                                                                              void*     output,
                                                                              uint32_t* tile_counter_ptr,
                                                                              float* attention_output_orig_quant_scale,
                                                                              size_t batch_size,
                                                                              size_t max_input_length,
                                                                              size_t max_kv_length,
                                                                              size_t total_q_seq_len,
                                                                              size_t total_kv_seq_len,
                                                                              KVBlockArray kv_block_array,
                                                                              void*        custom_mask) {
    tensorrt_llm::kernels::MHARunnerParams runnerParams;

    auto run_stream = stream_;
    if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
        run_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    }

    runnerParams.b                    = batch_size;
    runnerParams.numGroupedHeads      = config_.head_num;
    runnerParams.qSeqLen              = max_input_length;
    runnerParams.kvSeqLen             = max_kv_length;
    runnerParams.slidingWindowSize    = total_kv_seq_len;
    runnerParams.chunkedAttentionSize = INT_MAX;
    runnerParams.totalQSeqLen         = total_q_seq_len;
    runnerParams.totalKvSeqLen        = total_kv_seq_len;
    runnerParams.qkvPtr               = input;
    runnerParams.qPtr                 = input;
    runnerParams.kvPtr                = nullptr;
    runnerParams.pagedKvCache         = kv_block_array;
    runnerParams.outputPtr            = output;
    runnerParams.outputSfPtr          = nullptr;
    runnerParams.softmaxStatsPtr      = nullptr;
    runnerParams.attentionSinksPtr    = nullptr;
    runnerParams.packedMaskPtr        = custom_mask;
    runnerParams.cuQSeqLenPtr         = reinterpret_cast<int*>(cu_seqlens);
    runnerParams.cuKvSeqLenPtr        = reinterpret_cast<int*>(cu_kv_seqlens);
    runnerParams.cuMaskRowsPtr        = nullptr;
    runnerParams.tileCounterPtr       = tile_counter_ptr;
    runnerParams.scaleBmm1Ptr         = nullptr;
    runnerParams.scaleBmm2Ptr         = attention_output_orig_quant_scale;
    runnerParams.oSfScalePtr          = attention_output_orig_quant_scale;
    runnerParams.stream               = run_stream;
    runnerParams.qScalePtr            = nullptr;
    runnerParams.kScalePtr            = nullptr;
    runnerParams.vScalePtr            = nullptr;
    runnerParams.qMaxNBlock           = 0;
    runnerParams.kMaxNBlock           = 0;
    runnerParams.vMaxNBlock           = 0;
    return runnerParams;
}

void TrtV2FmhaRunner::runTrtV2Fmha(void*        input,
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
    } else if (trtv2_sm70_fmha_runner_) {
        trtv2_sm70_fmha_runner_->setup_flags(false, false, config_.is_causal, config_.kv_head_num);
        trtv2_sm70_fmha_runner_->setup(batch_size, max_seq_len, max_seq_len, token_num, false, false, 1, 0);
        trtv2_sm70_fmha_runner_->run(input, cu_seqlens, output, stream_);
    }
    check_cuda_error();
}

void TrtV2FmhaRunner::runTrtV2FmhaPaged(void*        input,
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

std::shared_ptr<TRTAttn> prepareTrtAttnParams(const AttentionConfigs& configs,
                                              const BufferPtr&        kv_cache_block_id,
                                              int                     batch_size,
                                              bool                    use_fp8_fmha,
                                              cudaStream_t            stream,
                                              bool                    enable_paged_trt_v2) {
    if (!kv_cache_block_id || 0 == batch_size) {
        return nullptr;
    }

    auto trt_attn = std::make_shared<TRTAttn>();

    int             ele_size   = 2;
    KvCacheDataType cache_type = KvCacheDataType::BASE;
#ifdef ENABLE_FP8
    if (use_fp8_fmha) {
        cache_type = KvCacheDataType::FP8;
        ele_size   = 1;
    } else
#endif
        if (configs.kv_cache_dtype == KvCacheDataType::INT8) {
        cache_type = KvCacheDataType::INT8;
        ele_size   = 1;
    } else if (configs.kv_cache_dtype == KvCacheDataType::FP8) {
        cache_type = KvCacheDataType::FP8;
        ele_size   = 1;
    }

    RTP_LLM_CHECK_WITH_INFO(kv_cache_block_id->shape()[0] == batch_size,
                            "context attention kv blocks batch size expected [%d] but buffer size is [%d]",
                            (int)batch_size,
                            (int)kv_cache_block_id->shape()[0]);

    const size_t max_blocks_per_batch = kv_cache_block_id->shape()[1];

    // Create torch::Tensor for kv_cache_offset
    trt_attn->kv_cache_offset = torch::empty({int64_t(batch_size), 1, 2, int64_t(max_blocks_per_batch)},
                                             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    trt_attn->kv_block_array                     = KVBlockArray(batch_size,
                                            max_blocks_per_batch,
                                            configs.tokens_per_block,
                                            configs.kv_head_num * configs.size_per_head * ele_size,
                                            0,
                                            0,
                                            nullptr,
                                            nullptr,
                                            (rtp_llm::KVCacheIndex*)trt_attn->kv_cache_offset.data_ptr<int>());
    trt_attn->kv_block_array.cache_type          = cache_type;
    trt_attn->kv_block_array.mScaleBytesPerBlock = configs.tokens_per_block * configs.kv_head_num * sizeof(float);

    invokeConvertOffsetToBlockArrayData(trt_attn->kv_cache_offset.data_ptr<int>(),
                                        kv_cache_block_id->data<int>(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        stream);

    if (is_sm90() && enable_paged_trt_v2) {
        trt_attn->kv_cache_offset_h                        = trt_attn->kv_cache_offset.to(torch::kCPU);
        trt_attn->kv_block_array.pagedKVBlockOffsetsOnHost = trt_attn->kv_cache_offset_h.data_ptr();
    }

    check_cuda_error();
    return trt_attn;
}

}  // namespace rtp_llm
