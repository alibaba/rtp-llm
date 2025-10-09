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

#pragma once
#ifdef USE_OLD_TRT_FMHA
#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace rtp_llm
{

class MHARunner
{
public:
    MHARunner(const int numHeads, const int headSize, const int wordSize, const float q_scaling)
        : mS(0)
        , mB(0)
        , mOmatSize(0)
        , mNumMats(0)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mWordSize(wordSize)
        , mLdQKV(0)
        , mStrideQKV(0)
        , mLdOut(0)
        , mStrideOut(0)
        , mRsqrtHeadSize(1.f / (sqrtf(headSize) * q_scaling))
    {
    }

    virtual ~MHARunner() = default;

    virtual void setup(const int S, const int B)
    {
        assert(S);
        assert(B);
        mB = B;
        mS = S;

        mLdQKV = 3 * B * mNumHeads * mHeadSize;
        mStrideQKV = 3 * mHeadSize;

        mLdOut = B * mNumHeads * mHeadSize;
        mStrideOut = mHeadSize;
        mOmatSize = S * S;
        mNumMats = B * mNumHeads;
    }

    virtual void setup_causal_masked_fmha(const int S, const int B)
    {
        setup(S, B);
    };

    virtual bool fmha_supported(bool causal_mask) { return false; };

    virtual void setup(const int S, const int B, const int window_num)
    {
        setup(S, B);
    }
    virtual void setup_flags(const bool interleaved, const bool ignore_b1opt, const bool force_unroll, const bool use_int8_scale_max, const bool use_tma) {return;};

    virtual void run(const void* input, const void* mask, void* workspace, void* output, cudaStream_t stream) = 0;
    virtual void run(const void* input, const void* mask, const void* seqlen, void* workspace, void* output, cudaStream_t stream) = 0;
    virtual void run(const void* input, const void* mask, const void* relative_position_bias, const int actual_seqlen, void* workspace, void* output, cudaStream_t stream) = 0;
    virtual void run_causal_masked_fmha(const void* input, const void* cu_seqlens, void* output, bool causal_mask, cudaStream_t stream)
    {
        // unimplemented
        ;
    }

    virtual void setScaleList(const float scaleQkv, const float dqProbs, const float scaleCtx) = 0;

    virtual size_t getWorkspaceSize() const = 0;

    virtual bool isValid(int s, const bool withRelativePositionBias) const = 0;

    virtual int getSFromMaxSeqLen(const int max_seq_len, const bool withRelativePositionBias = false) = 0;
protected:

    int mS;
    int mB;
    int mOmatSize;
    int mNumMats;
    int mNumHeads;
    int mHeadSize;
    int mWordSize;
    int mLdQKV;
    int mStrideQKV;
    int mLdOut;
    int mStrideOut;

    float mRsqrtHeadSize;
};

class FusedMHARunnerFP16v2 : public MHARunner
{
public:
    FusedMHARunnerFP16v2(const int numHeads, const int headSize, const int sm, const float q_scaling);
    ~FusedMHARunnerFP16v2(); // for pimpl

    virtual void setup(const int S, const int B) override;
    virtual void setup_causal_masked_fmha(const int S, const int B) override;
    virtual void setup(const int S, const int B, const int window_num) override;

    virtual bool fmha_supported(bool causal_mask) override;

    void run(const void* input, const void* mask, void* workspace, void* output, cudaStream_t stream);
    void run(const void* input, const void* mask, const void* seqlen, void* workspace, void* output, cudaStream_t stream) override;
    void run_causal_masked_fmha(const void* input, const void* cu_seqlens, void* output, bool causal_mask, cudaStream_t stream) override;
    void run(const void* input, const void* mask, const void* relative_position_bias, const int actual_seqlen, void* workspace, void* output, cudaStream_t stream) override;

    void setScaleList(const float scaleQkv, const float dqProbs, const float scaleCtx) override; 

    size_t getWorkspaceSize() const override;

    bool isValid(int s, const bool withRelativePositionBias) const override;

    int getSFromMaxSeqLen(const int max_seq_len, const bool withRelativePositionBias = false) override;

private:
    int mSm;
    class mhaImpl;
    mhaImpl* pimpl;
};

} // namespace rtp_llm

#endif