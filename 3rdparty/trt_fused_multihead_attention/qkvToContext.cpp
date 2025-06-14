/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef USE_OLD_TRT_FMHA

#include "qkvToContext.h"
#include "3rdparty/trt_fused_multihead_attention/fused_multihead_attention_common.h"
#include "3rdparty/trt_fused_multihead_attention/fused_multihead_attention_v2.h"

#include <cassert>
#include <cstring>

namespace rtp_llm {

union __half2_uint32_t_union {
    half2    fp162;
    uint32_t u32;
};
union __float_uint32_t_union {
    float    fp32;
    uint32_t u32;
};

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16) {
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha      = temp.u32;
    }
    else if (dtype == DATA_TYPE_FP32 || dtype == DATA_TYPE_BF16) {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha     = temp.u32;
    }
    else if (dtype == DATA_TYPE_INT32) {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha         = reinterpret_cast<const uint32_t&>(inorm);
    }
    else {
        assert(false);
    }
}

class FusedMHARunnerFP16v2::mhaImpl {
public:
    mhaImpl(FusedMHARunnerFP16v2* interface):
        interface(interface), sm(interface->mSm), xmmaKernel(getXMMAKernelsV2(DATA_TYPE_FP16, sm))
    {
        assert((sm == kSM_70 || sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 ||
            sm == kSM_89) && "Unsupported architecture");
        params.clear();
    }

    ~mhaImpl() {}

    size_t getPackedMaskSizeInBytes() const
    {
        // check that we initialized
        assert(xmmas_m > 0);
        assert(threads_per_cta > 0);
        assert(interface->mB > 0);
        return interface->mB * xmmas_m * threads_per_cta * sizeof(uint32_t);
    }

    void setup(const int S, const int B)
    {
        // TODO these implementation details might be better centralized into the XMMA code, since they are needed in
        // several places (also outside of this plugin)
        size_t warps_m = 2, warps_n = 2, warps_k = 1;
        if (sm == 70) {
            if (S == 64 || S == 96) {
                warps_m = 2;
                warps_n = 2;
            }
            else if (S == 128) {
                warps_m = 1;
                warps_n = 4;
            }
            else if (S == 256 || S == 384) {
                warps_m = 1;
                warps_n = 8;
            }
            else {
                // S >= 512, flash attention
                warps_m = 4;
                warps_n = 1;
            }
        }
        else {
            if (S == 32 || S == 64 || S == 96 || S == 128) {
                warps_m = 2;
                warps_n = 2;
            }
            else if (S == 192 || S == 256) {
                warps_m = 1;
                warps_n = 4;
            }
            else if (S == 384 || S == 512) {
                warps_m = 1;
                warps_n = 8;
            }
            else {
                // S >= 512, flash attention
                warps_m = 4;
                warps_n = 1;
            }
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

        const float scale_bmm1    = interface->mRsqrtHeadSize;
        const float scale_softmax = 1.f;  // Seems to be only required for int8
        const float scale_bmm2    = 1.f;

        Data_type scale_type = DATA_TYPE_FP16;
        set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;
        // TODO:
        // For now we set window_num = 0, to avoid using fused multi-head window-attention kernel
        params.window_num = 0;

        // mLdQKV = 3 * B * mNumHeads * mHeadSize;
        // mLdOut = B * mNumHeads * mHeadSize;

        params.qkv_stride_in_bytes         = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
        params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
        params.o_stride_in_bytes           = interface->mNumHeads * interface->mHeadSize * sizeof(half);

        // for bert and vit, use flash attention when s >= 512
        use_flash_attention = (params.s >= 512);
        params.force_unroll = use_flash_attention;
    }

    void setup_causal_masked_fmha(const int S, const int B)
    {
        const float scale_bmm1 = interface->mRsqrtHeadSize;
        const float scale_softmax = 1.f; // Seems to be only required for int8
        const float scale_bmm2 = 1.f;

        Data_type scale_type = DATA_TYPE_FP16;
        set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = B;
        params.h = interface->mNumHeads;
        params.s = S;
        params.d = interface->mHeadSize;

        // mLdQKV = 3 * B * mNumHeads * mHeadSize;
        // mLdOut = B * mNumHeads * mHeadSize;

        params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
        params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(half);

        // fallback to original fmha_v2 when head_size <= 64 and seq_len <- 128
        use_flash_attention = true;
        if (params.d <= 64 && params.s <= 128) {
            use_flash_attention = false;
            // get max sequence length
            if (params.s > 64) {
                params.s = 128;
            }
            else {
                params.s = 64;
            }
        }

        // set flags
        params.force_unroll = use_flash_attention;
    }

    bool fmha_supported(bool causal_mask)
    {
        if (causal_mask) {
            return (sm == kSM_70 || sm == kSM_72 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 || sm == kSM_89) &&
                (interface->mHeadSize == 32 || interface->mHeadSize == 40 || interface->mHeadSize == 64 ||
                interface->mHeadSize == 80 || interface->mHeadSize == 128 || interface->mHeadSize == 144
                || interface->mHeadSize == 160 || interface->mHeadSize == 256);
        }
        else {
            return (sm == kSM_75 || sm == kSM_80 || sm == kSM_86) &&
                (interface->mHeadSize == 32 || interface->mHeadSize == 64);
        }
    }

    void setup(const int S, const int B, const int window_num)
    {
        // TODO these implementation details might be better centralized into the XMMA code, since they are needed in
        // several places (also outside of this plugin)
        size_t warps_m = 2, warps_n = 2, warps_k = 1;
        if (S == 64 || S == 128) {
            warps_m = 2;
            warps_n = 2;
        }
        else if (S == 256) {
            warps_m = 1;
            warps_n = 4;
        }
        else if (S == 384) {
            warps_m = 1;
            warps_n = 8;
        }
        else {
            assert(false && "Unsupporte seqlen");
        }
        // The number of threads per CTA.
        threads_per_cta = warps_m * warps_n * warps_k * 32;
        // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
        xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
        // The number of xmmas in the N dimension.
        xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

        const float scale_bmm1    = interface->mRsqrtHeadSize;
        const float scale_softmax = 1.f;  // Seems to be only required for int8
        const float scale_bmm2    = 1.f;

        Data_type scale_type = DATA_TYPE_FP16;
        set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b          = B;
        params.h          = interface->mNumHeads;
        params.s          = S;
        params.d          = interface->mHeadSize;
        params.window_num = window_num;

        // mLdQKV = 3 * B * mNumHeads * mHeadSize;
        // mLdOut = B * mNumHeads * mHeadSize;

        params.qkv_stride_in_bytes         = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
        params.packed_mask_stride_in_bytes = S * sizeof(half);
        params.o_stride_in_bytes           = interface->mNumHeads * interface->mHeadSize * sizeof(half);
    }

    void run(const void*  qkvPtr,
             const void*  maskPtr,
             const void*  cuSeqlenPtr,
             void*        output,
             void*        workspace,
             cudaStream_t stream)
    {
        params.qkv_ptr = const_cast<void*>(qkvPtr);

        params.packed_mask_ptr = const_cast<void*>(maskPtr);

        params.o_ptr = output;

        params.cu_seqlens = static_cast<int*>(const_cast<void*>(cuSeqlenPtr));
        xmmaKernel->run(params, stream, use_flash_attention);
        check_cuda_value(cudaPeekAtLastError());
    }

    void run(const void*  qkvPtr,
             const void*  maskPtr,
             const void*  relative_position_bias,
             const int    actual_seqlen,
             void*        output,
             void*        workspace,
             cudaStream_t stream)
    {
        params.qkv_ptr = const_cast<void*>(qkvPtr);

        params.packed_mask_ptr = const_cast<void*>(maskPtr);

        params.packed_relative_position_bias_ptr = const_cast<void*>(relative_position_bias);

        params.o_ptr = output;

        params.actual_seqlen = actual_seqlen;

        params.cu_seqlens = nullptr;
        xmmaKernel->run(params, stream);
        check_cuda_value(cudaPeekAtLastError());
    }

    void run_causal_masked_fmha(const void* qkvPtr, const void* cuSeqlenPtr, void* output, bool causal_mask, cudaStream_t stream)
    {
        params.qkv_ptr = const_cast<void*>(qkvPtr);

        params.o_ptr = output;

        params.cu_seqlens = static_cast<int*>(const_cast<void*>(cuSeqlenPtr));

        xmmaKernel->run(params, stream, use_flash_attention, causal_mask);
    }

    bool isValid(int s, const bool withRelativePositionBias) const
    {
        return xmmaKernel->isValid(s) || (s >= 512 && !withRelativePositionBias);
    }

    int getSFromMaxSeqLen(const int max_seq_len, const bool withRelativePositionBias)
    {
        int S = 1024;
        if (withRelativePositionBias) {
            if (max_seq_len <= 64) {
                S = 64;
            }
            else if (max_seq_len <= 128) {
                S = 128;
            }
            else if (max_seq_len <= 256) {
                S = 256;
            }
        }
        else {
            if (max_seq_len <= 32) {
                if (sm == 70)
                    S = 64;
                else
                    S = 32;
            }
            else if (max_seq_len <= 64) {
                S = 64;
            }
            else if (max_seq_len <= 96) {
                S = 96;
            }
            else if (max_seq_len <= 128) {
                S = 128;
            }
            else if (max_seq_len <= 192) {
                if (sm == 70)
                    S = 256;
                else
                    S = 192;
            }
            else if (max_seq_len <= 256) {
                S = 256;
            }
            else if (max_seq_len <= 384) {
                S = 384;
            }
            else if (max_seq_len <= 512) {
                S = 512;
            }
            // for bert and vit, use flash attention when s >= 512
            else if (max_seq_len > 512) {
                S = max_seq_len;
            }
        }
        return S;
    }

private:
    FusedMHARunnerFP16v2*                      interface;
    Fused_multihead_attention_params_v2        params;
    int                                        sm;
    const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
    size_t                                     xmmas_m;
    size_t                                     xmmas_n;
    size_t                                     threads_per_cta;
    bool                                       use_flash_attention = false;
};

FusedMHARunnerFP16v2::FusedMHARunnerFP16v2(const int numHeads, const int headSize, const int sm, const float q_scaling):
    MHARunner(numHeads, headSize, 2, q_scaling), mSm(sm), pimpl(new mhaImpl(this))
{
}

void FusedMHARunnerFP16v2::setup(const int S, const int B)
{
    MHARunner::setup(S, B);
    pimpl->setup(S, B);
}

void FusedMHARunnerFP16v2::setup_causal_masked_fmha(const int S, const int B)
{
    MHARunner::setup_causal_masked_fmha(S, B);
    pimpl->setup_causal_masked_fmha(S, B);
}

bool FusedMHARunnerFP16v2::fmha_supported(bool causal_mask)
{
    MHARunner::fmha_supported(causal_mask);
    return pimpl->fmha_supported(causal_mask);
}

void FusedMHARunnerFP16v2::setup(const int S, const int B, const int window_num)
{
    MHARunner::setup(S, B, window_num);
    pimpl->setup(S, B, window_num);
}

size_t FusedMHARunnerFP16v2::getWorkspaceSize() const
{
    return 0;
}

void FusedMHARunnerFP16v2::run(const void* input, const void* mask, void* workspace, void* output, cudaStream_t stream)
{
    assert(false && "not implemented");
}

void FusedMHARunnerFP16v2::run(
    const void* input, const void* mask, const void* seqlen, void* workspace, void* output, cudaStream_t stream)
{
    pimpl->run(input, mask, seqlen, output, workspace, stream);
}

void FusedMHARunnerFP16v2::run(const void*  input,
                               const void*  mask,
                               const void*  relative_position_bias,
                               const int    actual_seqlen,
                               void*        workspace,
                               void*        output,
                               cudaStream_t stream)
{
    pimpl->run(input, mask, relative_position_bias, actual_seqlen, output, workspace, stream);
}

void FusedMHARunnerFP16v2::run_causal_masked_fmha(const void* qkvPtr, const void* cuSeqlenPtr, void* output, bool causal_mask, cudaStream_t stream)
{
    pimpl->run_causal_masked_fmha(qkvPtr, cuSeqlenPtr, output, causal_mask, stream);
}

FusedMHARunnerFP16v2::~FusedMHARunnerFP16v2() {
    if (pimpl != nullptr) {
        delete pimpl;
    }
}

bool FusedMHARunnerFP16v2::isValid(int s, const bool withRelativePositionBias) const
{
    return pimpl->isValid(s, withRelativePositionBias);
}

void FusedMHARunnerFP16v2::setScaleList(const float scaleQkv, const float dqProbs, const float scaleCtx) {}

int FusedMHARunnerFP16v2::getSFromMaxSeqLen(const int max_seq_len, const bool withRelativePositionBias)
{
    return pimpl->getSFromMaxSeqLen(max_seq_len, withRelativePositionBias);
}

}  // namespace rtp_llm

#endif
