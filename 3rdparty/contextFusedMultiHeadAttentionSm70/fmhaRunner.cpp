/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fmhaRunner.h"
#include "fused_multihead_attention_v2.h"
#include "fused_multihead_attention_common.h"
#include "tmaDescriptor.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{

union __half2_uint32_t_union
{
    half2 fp162;
    uint32_t u32;
};

union __float_uint32_t_union
{
    float fp32;
    uint32_t u32;
};

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_FP32 || dtype == DATA_TYPE_BF16)
    {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<const uint32_t&>(inorm);
    }
    else
    {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

class FusedMHARunnerV2Sm70::mhaImpl
{
public:
    mhaImpl(const Data_type data_type, const int numHeads, const int headSize, const float qScaling, int sm_)
        : sm(sm_)
        , mDataType(data_type)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mQScaling(qScaling)
    {
        RTP_LLM_CHECK_WITH_INFO(
            (sm == kSM_70), "Unsupported architecture");
        RTP_LLM_CHECK_WITH_INFO((mDataType == DATA_TYPE_FP16 || mDataType == DATA_TYPE_BF16), "Unsupported data type");

        xmmaKernel = getXMMAKernelsV2(mDataType, sm);

        mParams.clear();

        // get device attributes
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&mLaunchParams.multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
        cudaDeviceGetAttribute(&mLaunchParams.device_l2_cache_size, cudaDevAttrL2CacheSize, device_id);
    }

    ~mhaImpl() {}

    // Shared setup function.
    template <typename Params>
    void setup_params(Params& params, const int b, const int s_q, const int s_kv, const int sliding_window_size,
        const int total_seqlen, const bool has_alibi, const bool scale_alibi, const int tp_size, const int tp_rank)
    {

        const float inv_sqrt_scale = (1.f / (sqrtf(mHeadSize) * mQScaling));
        // Note that we apply scales and bias in the order of
        const float scale_bmm1 = scale_alibi ? 1.0f : inv_sqrt_scale;
        const float scale_softmax = 1.f; // Seems to be only required for int8
        const float scale_bmm2 = 1.f;

        Data_type scale_type = mLaunchParams.force_fp32_acc ? DATA_TYPE_FP32 : mDataType;
        // Use exp2f optimization for warp-specialized ws kernels on Hopper.
        if (mLaunchParams.useBase2ExpTrick)
        {
            // The kernel adopts the log2f optimziation.
            constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
            set_alpha(params.scale_bmm1, scale_bmm1 * float(kLog2e), DATA_TYPE_FP32);
        }
        else
        {
            set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        }
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = b;
        params.h = mNumHeads;
        params.s = s_q;
        params.d = mHeadSize;
        params.sliding_window_size = sliding_window_size;

        params.o_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);

        // Total sequence length needed by TMA descriptor
        // it should be actual total seq length if non-padded input is given.
        mTotalSeqLen = total_seqlen;
    }

    // Support packed QKV.
    void setup(const int b, const int s, const int sliding_window_size, const int total_seqlen, const bool has_alibi,
        const bool scale_alibi, const int tp_size, const int tp_rank)
    {

        // Determine launch parameters.
        // Hopper: fallback to original fmha_v2 when head_size <= 64 and seq_len <= 256
        mLaunchParams.set_default_kernel_selection_params();

        // Next power of 2 head size.
        RTP_LLM_CHECK_WITH_INFO(mHeadSize > 0, "Head size should be greater than 0.");
        mLaunchParams.padded_d = (mHeadSize & (mHeadSize - 1)) == 0 ? mHeadSize : pow(2, int(log2(mHeadSize)) + 1);

        const bool isSm70 = (sm == kSM_70);
        if (isSm70)
        {
            mLaunchParams.flash_attention = true;
            mLaunchParams.force_unroll = true;          // need more profile
            mLaunchParams.useKernelWithoutAlibi = true; // Volta do not support alibi
        }

        // Use specialized ws kernels on Hopper for cases without alibi.
        if (mLaunchParams.warp_specialization && !has_alibi)
        {
            // Use specialized ws kernels for cases without alibi.
            mLaunchParams.useKernelWithoutAlibi = true;
            // Enable exp2f optimization (which helps improve performance).
            //    - note that this is not compatible with alibi bias due to the accuracy issues.
            //    - only hopper warp-specialized kernels have this optimization.
            mLaunchParams.useBase2ExpTrick = true;
        }

        // Sliding_window_causal mask.
        if (s > sliding_window_size && mLaunchParams.attention_mask_type == ContextAttentionMaskTypeSm70::CAUSAL)
        {
            RTP_LLM_CHECK_WITH_INFO(!isSm70, "Sliding window attention is not supported for FMHA on Volta");
            mLaunchParams.attention_mask_type = ContextAttentionMaskTypeSm70::SLIDING_WINDOW_CAUSAL;
        }

        // Set kernel parameters.
        setup_params(mParams, b, s, s, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
        mParams.qkv_stride_in_bytes = (mNumHeads + 2 * mParams.h_kv) * mHeadSize * sizeof(half);
    }


    // NOTE: assume that heads_interleaved = false (b, s, 3, h, d), and sequences are padded/non-padded
    // TMA descriptors are used as grid_constant parameters (remove MemCpyH2D operations)
    void set_tma_descriptors()
    {
        // split D into multiple groups in order to match the TMA swizzle mode (128B)
        const uint32_t d_in_bytes = mLaunchParams.padded_d * sizeof(uint16_t);
        const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

        // separate q, k, and v tma descriptors
        Multiple_tma_descriptor<4> qkv_tma_descriptor;

        // tensor size
        uint32_t tensor_size_qkv[4];
        if (mParams.h_kv < mParams.h)
        {
            // if multi-query or grouped-query
            tensor_size_qkv[2] = 1;
            tensor_size_qkv[1] = (mParams.h + 2 * mParams.h_kv);
            tensor_size_qkv[0] = mParams.d; // mParams.d;
        }
        else
        {
            tensor_size_qkv[2] = 3;
            tensor_size_qkv[1] = mParams.h;
            tensor_size_qkv[0] = mParams.d; // mParams.d;
        }

        // box size for k and v
        uint32_t box_size[4];
        // Update this on device?
        box_size[2] = 1;
        box_size[1] = 1;
        box_size[0] = mLaunchParams.padded_d / d_groups;

        // stride size in bytes. Assumes least significant dim is 1 (?)
        uint64_t tensor_stride_qkv[3];
        tensor_stride_qkv[0] = tensor_size_qkv[0] * sizeof(uint16_t);     // d
        tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0]; // d*h
        tensor_stride_qkv[2] = tensor_size_qkv[2] * tensor_stride_qkv[1]; // d*h*3

        // traversal stride
        uint32_t traversal_stride_qkv[4] = {1, 1, 1, 1};

        // OOB fill zeros
        uint32_t oob_fill = 0;

        // FP32 to TF32 conversion disabled
        uint32_t fp32_to_tf32 = 0;

        // gmma descriptor mode
        const uint32_t d_bytes_per_group = (mLaunchParams.padded_d * sizeof(uint16_t)) / d_groups;
        const cudaTmaDescSwizzle swizzle_mode = (d_bytes_per_group > 64
                ? cudaTmaDescSwizzle::SWIZZLE_128B
                : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

        uint32_t q_step = 0, kv_step = 0;
        for (unsigned int i = 0u; i < sizeof(sTmaMetaInfo) / sizeof(sTmaMetaInfo[0]); ++i)
        {
            if (sTmaMetaInfo[i].mD == mLaunchParams.padded_d)
            {
                q_step = sTmaMetaInfo[i].mQStep;
                kv_step = sTmaMetaInfo[i].mKVStep;
                break;
            }
        }

        // QKV [TOTAL, 3, h, d]
        // NOTE: we may need to use actual seqlen to set oob_value
        const char* qkv_ptr = reinterpret_cast<const char*>(mParams.qkv_ptr);
        tensor_size_qkv[3] = mTotalSeqLen;

        // Q: STEP_Q
        box_size[3] = q_step;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &mParams.tma_desc_q);

        // K/V: STEP_KV
        box_size[3] = kv_step;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &mParams.tma_desc_k);
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &mParams.tma_desc_v);
    }

    void setup_flags(const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask, const int num_kv_heads)
    {
        // BF16 FMHA only accumulates on FP32
        mLaunchParams.force_fp32_acc = mDataType == DATA_TYPE_BF16 || force_fp32_acc;
        mLaunchParams.attention_mask_type
            = causal_mask ? ContextAttentionMaskTypeSm70::CAUSAL : ContextAttentionMaskTypeSm70::PADDING;

        // Contiguous Cache.
        mParams.h_kv = num_kv_heads;
        mParams.is_s_padded = is_s_padded;
    }

    bool fmha_supported()
    {
        return MHARunnerSm70::fmha_supported(mHeadSize, sm);
    }

    void run(const void* qkvPtr, const void* cuSeqlenPtr, void* outputPtr, cudaStream_t stream)
    {
        mParams.qkv_ptr = qkvPtr;
        mParams.o_ptr = outputPtr;
        mParams.cu_seqlens = reinterpret_cast<const int*>(cuSeqlenPtr);

        xmmaKernel->run(mParams, mLaunchParams, stream);
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

    int getSFromMaxSeqLen(const int max_seq_len)
    {
        int S = 1024;

        if (max_seq_len <= 64)
        {
            S = 64;
        }
        else if (max_seq_len <= 128)
        {
            S = 128;
        }
        else if (max_seq_len <= 256)
        {
            S = 256;
        }
        else if (max_seq_len <= 384)
        {
            S = 384;
        }
        else if (max_seq_len <= 512)
        {
            S = 512;
        }
        // for bert and vit, use flash attention when s >= 512
        else if (max_seq_len > 512)
        {
            S = max_seq_len;
        }

        return S;
    }

private:
    Fused_multihead_attention_params_v2Sm70 mParams;
    Launch_paramsSm70 mLaunchParams;
    int sm;
    const FusedMultiHeadAttentionXMMAKernelV2Sm70* xmmaKernel;
    bool use_flash_attention = false;
    const Data_type mDataType;
    const int mNumHeads;
    const int mHeadSize;
    const float mQScaling;
    int mTotalSeqLen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

FusedMHARunnerV2Sm70::FusedMHARunnerV2Sm70(
    const Data_type data_type, const int numHeads, const int headSize, const float qScaling)
    : pimpl(new mhaImpl(data_type, numHeads, headSize, qScaling, rtp_llm::get_sm()))
{
}

FusedMHARunnerV2Sm70::~FusedMHARunnerV2Sm70() = default;

void FusedMHARunnerV2Sm70::setup(const int b, const int s, const int sliding_window_size, const int total_seqlen,
    const bool has_alibi, const bool scale_alibi, const int tp_size, const int tp_rank)
{
    pimpl->setup(b, s, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
}

bool FusedMHARunnerV2Sm70::fmha_supported()
{
    return pimpl->fmha_supported();
}

void FusedMHARunnerV2Sm70::setup_flags(
    const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask, const int num_kv_heads)
{
    pimpl->setup_flags(force_fp32_acc, is_s_padded, causal_mask, num_kv_heads);
}

void FusedMHARunnerV2Sm70::run(const void* qkvPtr, const void* cuSeqlenPtr, void* outputPtr, cudaStream_t stream)
{
    pimpl->run(qkvPtr, cuSeqlenPtr, outputPtr, stream);
}


bool FusedMHARunnerV2Sm70::isValid(int s) const
{
    return pimpl->isValid(s);
}

// static function to check if fmha is supported when building plugins
bool MHARunnerSm70::fmha_supported(const int headSize, const int sm)
{
    if (sm == kSM_70)
    {
        return (headSize == 32 || headSize == 40 || headSize == 64 || headSize == 80 || headSize == 128
            || headSize == 160 || headSize == 256);
    }

    return false;
}

} // namespace kernels
} // namespace tensorrt_llm
