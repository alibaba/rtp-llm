/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <assert.h>
#include <cmath>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_index.h"

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

#if USING_ROCM
template<KvCacheDataType CType>
struct ElementSizeInBytes;

template<>
struct ElementSizeInBytes<KvCacheDataType::BASE> {
    static constexpr int value = 2;
};
template<>
struct ElementSizeInBytes<KvCacheDataType::INT8> {
    static constexpr int value = 1;
};
template<>
struct ElementSizeInBytes<KvCacheDataType::FP8> {
    static constexpr int value = 1;
};
#endif

// Internal for K and V cache indexing
enum class KVIdxType : int32_t {
    K_IDX = 0,
    V_IDX = 1
};

// Struct operates on paged kv cache providing
// only the fields necessary for context FMHA
struct KVBlockArrayForContextFMHA {
    using DataType = KVCacheIndex const;

    // Current number of sequences
    int32_t mMaxSeqs;
    // Max number of blocks per sequence
    int32_t mMaxBlocksPerSeq;
    // Number of tokens. It must be power of 2.
    int32_t mTokensPerBlock;
    // Exponent of number of tokens with base 2.
    // E.g. for mTokensPerBlock 64, mTokensPerBlockLog2 equals to 6
    int32_t mTokensPerBlockLog2;
    // Table maps logical block idx to the data pointer of k/v cache block pool
    // Shape [B, W, 2, M], where 2 is table for K and V,
    // B is current number of sequences
    // W is beam width
    // M is Max number of blocks per sequence

    // Size of KV cache blocks in bytes (H*D*T*sizeof(DataType))
    int32_t mBytesPerBlock;
    // Pointer to beginning of pool.
    void* mPrimaryPoolPtr;
    // Pointer to block offsets.
    DataType* data;

    KVBlockArrayForContextFMHA() = default;

    KVBlockArrayForContextFMHA(int32_t   batchSize,
                               int32_t   maxBlocksPerSeq,
                               int32_t   tokensPerBlock,
                               int32_t   bytesPerToken,
                               void*     primaryPoolPtr,
                               DataType* data):
        mMaxSeqs(batchSize),
        mMaxBlocksPerSeq(maxBlocksPerSeq),
        mTokensPerBlock(tokensPerBlock),
        mBytesPerBlock{tokensPerBlock * bytesPerToken},
        mPrimaryPoolPtr{primaryPoolPtr},
        data{data} {
        float const tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
        RTP_LLM_CHECK_WITH_INFO(ceil(tokensPerBlockSeqLog2) == floor(tokensPerBlockSeqLog2),
                                "tokensPerBlock must be power of 2");
        // NOTE: pointer offset arithmetic offset is performed on int32_t (see this.getRowPtr).
        // If needed, we could do it on uint32_t or even uint64_t, but that might have performance implications
        RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(mMaxSeqs - 1) * mMaxBlocksPerSeq * 2 + maxBlocksPerSeq
                                    <= std::numeric_limits<int32_t>::max(),
                                "kv cache is too large for gpt_attention_plugin");
        mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
    }
};

// Struct operates on paged kv cache providing
// functions for accessing blocks of in K and V caches
// and elements inside these blocks
struct KVBlockArray: public KVBlockArrayForContextFMHA {
#if USING_ROCM
    friend class OffsetIndexedKVBlockArray;
#endif

    // Pointer to beginning of pool.
    void* mSecondaryPoolPtr;
    // Maximum kv cache length per sequence
    int32_t mMaxAttentionWindow;
    // Number of sink tokens.
    int32_t mSinkTokens;
    // Cyclic cache length.
    int32_t mCyclicCacheLen;
    // Bubble length.
    int32_t mBubbleLen;
    // Enable one more block to save the kv tokens
    bool            mEnableOneMoreBlock;
    KvCacheDataType cache_type          = KvCacheDataType::BASE;
    void*           scale               = nullptr;
    int             mScaleBytesPerBlock = 1;
    void*           pagedKVBlockOffsetsOnHost;

    KVBlockArray() = default;

    KVBlockArray(int32_t   batchSize,
                 int32_t   maxBlocksPerSeq,
                 int32_t   tokensPerBlock,
                 int32_t   bytesPerToken,
                 int32_t   maxAttentionWindow,
                 int32_t   sinkTokenLen,
                 void*     primaryPoolPtr,
                 void*     secondaryPoolPtr,
                 DataType* data):
        KVBlockArrayForContextFMHA(batchSize, maxBlocksPerSeq, tokensPerBlock, bytesPerToken, primaryPoolPtr, data),
        mSecondaryPoolPtr{secondaryPoolPtr},
        mMaxAttentionWindow(maxAttentionWindow),
        mSinkTokens(sinkTokenLen) {
        auto sinkTokensInLastBlock = mSinkTokens % mTokensPerBlock;
        mBubbleLen                 = sinkTokensInLastBlock == 0 ? 0 : mTokensPerBlock - sinkTokensInLastBlock;
        mEnableOneMoreBlock        = (maxBlocksPerSeq - 1) * tokensPerBlock >= mMaxAttentionWindow + mBubbleLen;
        mCyclicCacheLen            = (mEnableOneMoreBlock) ? mMaxAttentionWindow + mTokensPerBlock - mSinkTokens :
                                                             mMaxAttentionWindow - mSinkTokens;
    }

    KVBlockArrayForContextFMHA copyKVBlockArrayForContextFMHA() const {
        return KVBlockArrayForContextFMHA{
            mMaxSeqs, mMaxBlocksPerSeq, mTokensPerBlock, mBytesPerBlock / mTokensPerBlock, mPrimaryPoolPtr, data};
    }

    __host__ __device__ inline bool isSinkToken(int32_t tokenIdx) const {
        return tokenIdx < mSinkTokens;
    }

    __host__ __device__ inline int32_t getKVTokenIdx(int32_t tokenIdx) const {
        if (!isSinkToken(tokenIdx)) {
            // Apply cyclic kv cache if tokenIdx >= max_attention_window_size.
            return mSinkTokens + mBubbleLen + (tokenIdx - mSinkTokens) % mCyclicCacheLen;
        }
        return tokenIdx;
    }

    __host__ __device__ inline DataType const* getRowPtr(KVIdxType kvIdx, int32_t seqIdx) const {
        // Returns pointer to array of offsets to K or V cache for one specific sequence seqIdx.
        // seqIdx is in range [0; B]
        return data + (seqIdx * mMaxBlocksPerSeq * 2 + static_cast<int32_t>(kvIdx) * mMaxBlocksPerSeq);
    }

    __host__ __device__ inline void* getBlockPtr(DataType const* offsets, int32_t tokenIdx) const {
        auto const offset = offsets[tokenIdx >> mTokensPerBlockLog2];
        return reinterpret_cast<void*>(reinterpret_cast<char*>(getPoolPtr(offset))
                                       + offset.get() * static_cast<uint64_t>(mBytesPerBlock));
    }

    __host__ __device__ inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx) const {
        return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
    }

    __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx) const {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
    }

    __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx) const {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
    }

    __host__ __device__ inline int32_t getLocalIdx(int32_t globalIdx) const {
        return globalIdx & ((1 << mTokensPerBlockLog2) - 1);
    }

    __host__ __device__ inline int32_t
    getKVLocalIdx(int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const {
        // For K or V, the hidden dimension per head is *not* decomposed. The layout of each block of K or V is:
        // [numHeads, tokensPerBlock, hiddenSizePerHead].
        // This member function computes the corresponding linear index.
        // NOTE: we have remapped K layout as the same of V.
        return headIdx * mTokensPerBlock * dimsPerHead + getLocalIdx(globalTokenIdx) * dimsPerHead + channelIdx;
    }

#if USING_ROCM
    template<KvCacheDataType CType>
    __host__ __device__ inline int32_t
    getKLocalIdx(int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const {
        constexpr int element_size = ElementSizeInBytes<CType>::value;
        static_assert(16 % element_size == 0, "kv cache element size must divide 16");
        constexpr int vectorize_size = 16 / element_size;

        assert(dimsPerHead % vectorize_size == 0);
        // shape: [numHeads, dimsPerHead/vs, mTokensPerBlock, vs]
        // stride: [dimsPerHead*mTokensPerBlock, mTokensPerBlock*vs, vs, 1]
        return headIdx * dimsPerHead * mTokensPerBlock + channelIdx / vectorize_size * mTokensPerBlock * vectorize_size
               + getLocalIdx(globalTokenIdx) * vectorize_size + channelIdx % vectorize_size;
    }

    template<KvCacheDataType CType>
    __host__ __device__ inline int32_t
    getVLocalIdx(int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const {
        constexpr int element_size = ElementSizeInBytes<CType>::value;
        static_assert(16 % element_size == 0, "kv cache element size must divide 16");
        constexpr int vectorize_size = 16 / element_size;

        assert(mTokensPerBlock % vectorize_size == 0);
        int32_t localTokenIdx = getLocalIdx(globalTokenIdx);
        // shape:  [numHeads,                    mTokensPerBlock/vs, dimsPerHead, vs]
        // stride: [mTokensPerBlock*dimsPerHead, dimsPerHead*vs,     vs,          1]
        return headIdx * dimsPerHead * mTokensPerBlock + localTokenIdx / vectorize_size * dimsPerHead * vectorize_size
               + channelIdx * vectorize_size + localTokenIdx % vectorize_size;
    }

    __host__ __device__ inline int32_t
    getVLocalIdx(int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const {
        // shape: [numHeads, dimsPerHead, mTokensPerBlock]
        // stride: [dimsPerHead*mTokensPerBlock, mTokensPerBlock, 1]
        return headIdx * dimsPerHead * mTokensPerBlock + channelIdx * mTokensPerBlock + getLocalIdx(globalTokenIdx);
    }

#endif

    __host__ __device__ inline void* getScaleBlockPtr(DataType const* offsets, int32_t tokenIdx) const {
        auto const offset = offsets[tokenIdx >> mTokensPerBlockLog2];
        return reinterpret_cast<void*>(reinterpret_cast<char*>(scale)
                                       + offset.get() * static_cast<uint64_t>(mScaleBytesPerBlock));
    }

    __host__ __device__ inline void* getScalePtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx) {
        return getScaleBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
    }

    __host__ __device__ inline void* getKScalePtr(int32_t seqIdx, int32_t tokenIdx) {
        return getScalePtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
    }

    __host__ __device__ inline void* getVScalePtr(int32_t seqIdx, int32_t tokenIdx) {
        return getScalePtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
    }

    __host__ __device__ inline int32_t getKVScaleLocalIdx(int32_t globalTokenIdx, int32_t headIdx) {
        // For K or V, the hidden dimension per head is *not* decomposed. The layout of each block of K or V is:
        // [numHeads, tokensPerBlock, hiddenSizePerHead].
        // This member function computes the corresponding linear index.
        // NOTE: we have remapped K layout as the same of V.
        return headIdx * mTokensPerBlock + getLocalIdx(globalTokenIdx);
    }

private:
    __host__ __device__ void* getPoolPtr(DataType offset) const {
        return offset.isPrimary() ? mPrimaryPoolPtr : mSecondaryPoolPtr;
    }
};

#if USING_ROCM
// Use raw offset (kv_block_offset) to index KV blocks (skip
// invokeConvertOffsetToBlockArrayData)
struct OffsetIndexedKVBlockArray: public KVBlockArray {
    OffsetIndexedKVBlockArray() = default;
    OffsetIndexedKVBlockArray(KVBlockArray& base, DataType* raw_block_table): KVBlockArray(base) {
        cache_type = base.cache_type;
        data       = raw_block_table;
    }
    __host__ __device__ inline DataType const* getRowPtr(KVIdxType kvIdx, int32_t seqIdx) const {
        return data + seqIdx * mMaxBlocksPerSeq;
    }
    __host__ __device__ inline void* getBlockPtr(DataType const* offsets, int32_t tokenIdx, KVIdxType kvIdx) const {
        auto const offset = offsets[tokenIdx >> mTokensPerBlockLog2];
        return reinterpret_cast<void*>(reinterpret_cast<char*>(getPoolPtr(offset))
                                       + (offset.get() * 2 + static_cast<int32_t>(kvIdx))
                                             * static_cast<uint64_t>(mBytesPerBlock));
    }
    __host__ __device__ inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx) const {
        return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx, kvIdx);
    }
    __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx) const {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
    }
    __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx) const {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
    }
};
#endif

// MLA KVBlockArray

}  // namespace rtp_llm
