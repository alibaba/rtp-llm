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

#if USING_ROCM
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

#include "beamSearchKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, int PBM, bool IS_V2>
void beamSearchKernelLauncher(
    T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

#define CASE_K(PBM)                                                                                                    \
    {                                                                                                                  \
        beamSearchKernelLauncher<T, PBM, IS_V2>(logProbs, bias, workspace, bh, stream);                                \
        break;                                                                                                         \
    }

template <typename T, bool IS_V2>
void invokeTopkBeamSearch(T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream)
{
    int maxBeamWidth = max(bh.nBeamWidthIn, bh.nBeamWidthOut);
    int const nPadBeamWidth{padToNextPowerOfTwo(maxBeamWidth)};

    // case X means X/2 < max_beam_width <= X
    if constexpr (IS_V2)
    {
        switch (nPadBeamWidth)
        {
        case 1:
        case 2:
        case 4: CASE_K(4)
        case 8: CASE_K(8)
        case 16: CASE_K(16)
#ifndef FAST_BUILD // Skip max beam width > 16
        case 32: CASE_K(32)
        case 64: CASE_K(64)
        case 128: CASE_K(128)
        case 256: CASE_K(256)
        case 512: CASE_K(512)
        case 1024: CASE_K(1024)
        case 2048: CASE_K(2048)
        case 4096: CASE_K(4096)
#endif // FAST_BUILD
        }
    }
    else // V1, only use kernels of `beam_width <= kMaxBeamWidthForV1`
    {
        switch (nPadBeamWidth)
        {
        case 1:
        case 2:
        case 4: CASE_K(4)
        case 8: CASE_K(8)
        }
    }
}

#undef CASE_K

template void invokeTopkBeamSearch<float, false>(
    float const* logProbs, float const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template void invokeTopkBeamSearch<float, true>(
    float const* logProbs, float const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template void invokeTopkBeamSearch<half, false>(
    half const* logProbs, half const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template void invokeTopkBeamSearch<half, true>(
    half const* logProbs, half const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

__global__ void updateCacheIndirectionKernel(
    int* tgtCI, int const* srcCI, BeamHypotheses bh, int const nMaxAttentionWindow, int const nSinkTokenLength)
{
    // Update cache indirections which steps are between `bh.inputLength[x]` to `sequenceLengths[x]`
    int const step = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const nBMIn{bh.nBeamWidthIn};
    size_t const nBMOut{bh.nBeamWidthOut};
    size_t const nMSL{bh.nMaxSeqLen};
    int const indexBatch = blockIdx.y;
    int const batchSlot = bh.batchSlots ? bh.batchSlots[indexBatch] : indexBatch;
    int const tgtIndexBeam = blockIdx.z;
    int const tgtIndexBatchBeam = batchSlot * nBMOut + tgtIndexBeam;
    int const lastStep{bh.sequenceLengthsOut[tgtIndexBatchBeam] - 1}; // minus 1 since it is updated in stage 3 kernel

    // Return early when at least one of the conditions is true:
    // 1. `step` is out of the bound
    // 2. `step` is inside of input part (since context KV Cache is shared)
    // 3. `step` is outside of attention widow
    if (step >= nMSL || step < bh.sequenceLengthsOut[tgtIndexBatchBeam] || step < (nMSL - nMaxAttentionWindow))
    {
        return;
    }

    // Keep all past tokens by parentIdsPtr
    int const srcIndexBeam = bh.parentIdsPtr[batchSlot * nBMOut + tgtIndexBeam];
    // Return early when the source beam isfinished
    if (bh.finished && bh.finished[tgtIndexBatchBeam].isFinished())
    {
        return;
    }

    int const stepCirc = (step >= nSinkTokenLength)
        ? nSinkTokenLength + (step - nSinkTokenLength) % (nMaxAttentionWindow - nSinkTokenLength)
        : step;
    // Consider cyclic kv cache for the indir tables
    uint32_t const tgtOffset = batchSlot * nBMOut * nMaxAttentionWindow + tgtIndexBeam * nMaxAttentionWindow + stepCirc;
    uint32_t const srcOffset = batchSlot * nBMIn * nMaxAttentionWindow + srcIndexBeam * nMaxAttentionWindow + stepCirc;
    tgtCI[tgtOffset] = (step == lastStep) ? tgtIndexBeam : srcCI[srcOffset];
}

void invokeUpdateCacheIndirection(int* tgtCI, int const* srcCI, BeamHypotheses& bh,
    runtime::SizeType32 const maxAttentionWindow, runtime::SizeType32 sinkTokenLength, cudaStream_t stream)
{
    dim3 const grid(common::roundUp(bh.nMaxSeqLen, 32), bh.nBatchSize, bh.nBeamWidthOut);
    updateCacheIndirectionKernel<<<grid, 32, 0, stream>>>(tgtCI, srcCI, bh, maxAttentionWindow, sinkTokenLength);
    check_cuda_error();
}

// See beamSearchKernels.h comment: host wrappers below keep the kernel launch
// in the same TU as the kernel definition so nvcc 13's local-linkage
// instantiations can be resolved.
namespace {
template <typename T>
__global__ void addCumLogProbsKernel(
    T* __restrict pStage1LogProbs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBMIn, size_t const nBMOut)
{
    int const bid = blockIdx.x; // Index of request in batch
    runtime::SizeType32 const slot = batchSlots ? batchSlots[bid] : bid;
    float const diversityRate{diversityRates == nullptr ? kBeamSearchDiversity : diversityRates[slot]};
    T* pLocalLogProbs = pStage1LogProbs + bid * nBMIn * nBMOut;

    for (int i = threadIdx.x; i < nBMIn * nBMOut; i += blockDim.x)
    {
        int const iBMIn = i / nBMOut;
        if (finished && finished[slot * nBMIn + iBMIn].isFinished())
        {
            // TODO(known-broken): i is a candidate-slot index, endIds[slot] is a vocab
            // token id — this compares across index spaces (pre-existing). Unreachable
            // in production while `finished` stays unwired.
            pLocalLogProbs[i] += endIds && (i == endIds[slot]) ? T(1.0f) : T(0.0f);
        }
        else
        {
            pLocalLogProbs[i] += cumLogProbs[slot * nBMIn + iBMIn] + diversityRate * iBMIn;
        }
    }
    return;
}

} // namespace

template <typename T>
void launchAddCumLogProbs(
    T* pStage1LogProbs, float const* cumLogProbs, FinishedState const* finished,
    int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t nBS, size_t nBMIn, size_t nBMOut,
    int nThread, cudaStream_t stream)
{
    addCumLogProbsKernel<T><<<nBS, nThread, 0, stream>>>(
        pStage1LogProbs, cumLogProbs, finished, endIds, diversityRates, batchSlots, nBS, nBMIn, nBMOut);
    check_cuda_error();
}

template void launchAddCumLogProbs<float>(
    float*, float const*, FinishedState const*, int const*, float const*,
    runtime::SizeType32 const*, size_t, size_t, size_t, int, cudaStream_t);

template void launchAddCumLogProbs<half>(
    half*, float const*, FinishedState const*, int const*, float const*,
    runtime::SizeType32 const*, size_t, size_t, size_t, int, cudaStream_t);

__global__ void gatherId(int const* __restrict pStage1Id, int* __restrict pStage2Id, size_t const nBS,
    size_t const nBMIn, size_t const nBMOut, size_t const nV)
{
    // Use topK output `pStage1Id` and `pStage2Id` to get the index of a new token in `logProbs` for each beam.
    // Layouts: pStage1Id is [nBS, nBMIn, nBMOut] flat (per-beam candidates, index-ordered);
    // pStage2Id is [nBS, nBMOut] flat, holding the flat stage-1 indices that survived the
    // merged top-k.
    //
    // For output slot j of batch a = blockIdx.x:
    //   stage2Id = pStage2Id[a * nBMOut + j]        flat candidate index into stage-1
    //   b = stage2Id / nBMOut                       beam the candidate came from
    //   d = stage2Id % nBMOut                       slot inside that beam's candidates
    //   c = a * nBMIn + b                           row in pStage1Id
    //   e = pStage1Id[c * nBMOut + d]               token id within the vocab
    //   f = b * nV                                  padding for previous tokens
    //   output: pStage2Id[a * nBMOut + j] = e + f   final index in logProbs
    int const a = blockIdx.x; // Index of request in batch
    for (int j = threadIdx.x; j < nBMOut; j += blockDim.x)
    {
        int const index = a * nBMOut + j;
        int const stage2Id = pStage2Id[index];
        int const b = stage2Id / nBMOut;
        int const c = a * nBMIn + b;
        int const d = stage2Id % nBMOut;
        int const e = pStage1Id[c * nBMOut + d];
        int const f = b * nV;
        pStage2Id[index] = e + f;
    }
    return;
}

__global__ void populateTokenIds(int* tokenIdsOut, int const* tokenIdsIn, int const* sequenceLengthsOut, int const* parentIdsPtr, int const* outputIdsPtr, 
                                 size_t const batchSize, size_t const maxSeqLen, size_t const beamWidthOut, size_t const beamWidthIn) {
    int const totalBeamNumOut = batchSize * beamWidthOut;
    for (int beamIdxOut = blockIdx.x; beamIdxOut < totalBeamNumOut; beamIdxOut += gridDim.x) {
        int beamIdxIn = parentIdsPtr[beamIdxOut];
        int newTokenId = outputIdsPtr[beamIdxOut];
        int newTokenPos = sequenceLengthsOut[beamIdxOut] - 1;

        int const* curTokenIdsIn = tokenIdsIn + (beamIdxOut / beamWidthOut * beamWidthIn + beamIdxIn) * maxSeqLen;
        // TODO(zhangjianning.zjn): only copy idx from 0 to seqlen to reduce copy size
        for (int i = threadIdx.x; i < maxSeqLen; i += blockDim.x) {
            int tokenId = i == newTokenPos ? newTokenId : curTokenIdsIn[i];
            tokenIdsOut[beamIdxOut * maxSeqLen + i] = tokenId;
        }
    }
}

__global__ void populateNewTokenIds(int* tokenIdsOut, int const* sequenceLengthsOut, int const* outputIdsPtr, 
                                    size_t const batchSize, size_t const maxSeqLen, size_t const beamWidthOut) {
    int const totalBeamNumOut = batchSize * beamWidthOut;
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const stride = gridDim.x * blockDim.x;
    for (int beamIdxOut = tid; beamIdxOut < totalBeamNumOut; beamIdxOut += stride) {
        int newTokenId = outputIdsPtr[beamIdxOut];
        int newTokenPos = sequenceLengthsOut[beamIdxOut] - 1;
        tokenIdsOut[beamIdxOut * maxSeqLen + newTokenPos] = newTokenId;
    }
}

void invokePopulateTokenIds(int* tokenIdsOut, int const* tokenIdsIn, int const* sequenceLengthsOut, int const* parentIdsPtr, int const* outputIdsPtr, 
                            size_t const batchSize, size_t const maxSeqLen, size_t const beamWidthOut, size_t const beamWidthIn, 
                            cudaStream_t stream) {
    if (tokenIdsOut == nullptr || batchSize == 0 || maxSeqLen == 0 || beamWidthIn == 0 || beamWidthOut == 0) {
        return;
    }

    constexpr size_t maxBlockSize = 1024;

    int smCount = getMultiProcessorCount();

    if (tokenIdsIn == tokenIdsOut) {
        int blockSize = maxBlockSize;
        int blockNum = 0;
        check_cuda_value(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockNum, populateNewTokenIds, blockSize, 0));
        populateNewTokenIds<<<smCount * blockNum, blockSize, 0, stream>>>(tokenIdsOut, sequenceLengthsOut, outputIdsPtr, batchSize, maxSeqLen, beamWidthOut);
        check_cuda_error();
    } else {
        int blockSize = min(maxSeqLen, maxBlockSize);
        int blockNum = 0;
        check_cuda_value(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockNum, populateTokenIds, blockSize, 0));
        populateTokenIds<<<smCount * blockNum, blockSize, 0, stream>>>(tokenIdsOut, tokenIdsIn, sequenceLengthsOut, parentIdsPtr, outputIdsPtr, batchSize, maxSeqLen, beamWidthOut, beamWidthIn);
        check_cuda_error();
    }
}

void BeamHypotheses::print()
{
#if BEAM_SEARCH_DEBUG
    cudaDeviceSynchronize();
    printf("================ print BeamHypotheses start\n");

    PRINT(this->bReturnNormedScore);
    PRINT(this->bVBWS);
    PRINT(this->nMaxBatchSize);
    PRINT(this->nBatchSize);
    PRINT(this->nBeamWidth);
    PRINT(this->nBeamWidthIn);
    PRINT(this->nBeamWidthOut);
    PRINT(this->nMaxSeqLen);
    PRINT(this->nVocabSize);
    PRINT(this->nVPart);
    PRINT(this->nByteMaxSharedMemoryPerBlock);
    PRINT(this->nByteSharedMemoryStage1);
    PRINT(this->nByteSharedMemoryStage3);
    size_t const mbs = this->nMaxBatchSize;
    size_t const nbs = this->nBatchSize;
    size_t const nbm = this->nBeamWidth;
    size_t const nbmo = this->nBeamWidthOut;
    size_t const msl = this->nMaxSeqLen;

    PH2(this->diversityRates, nbs);
    PH2(this->lengthPenalties, nbs);
    PH2(this->earlyStoppings, nbs);
    PH3(this->beamWidthArraysHost, nbs * kMaxBeamWidthArrayLength, kMaxBeamWidthArrayLength);
    PH2(this->nBeamWidthInHost, nbs);
    PH2(this->nBeamWidthOutHost, nbs);

    PH2(this->inputLengths, nbs * nbm);
    PH2(this->endIds, nbs);
    PH2(this->batchSlots, nbs);

    PH3(this->outputIds, nbs * nbm * msl, msl);
    PH3(this->logProbs, nbs * nbm * msl, msl);
    PH3(this->sequenceLengths, nbs * nbm, nbm);
    PH3(this->cumLogProbs, nbs * nbm, nbm);

    PH3(this->outputIdsCBA, mbs * nbmo * 2 * msl, msl);
    PH3(this->logProbsCBA, mbs * nbmo * 2 * msl, msl);
    PH3(this->sequenceLengthsCBA, mbs * nbmo * 2, nbmo * 2);
    PH3(this->cumLogProbsCBA, mbs * nbmo * 2, nbmo * 2);
    PH3(this->normedScoresCBA, mbs * nbmo * 2, nbmo * 2);
    PH2(this->numBeamsCBA, mbs);
    PH2(this->minNormedScoresCBA, mbs);

    // PH2(this->batchDones, nbs);
    uint8_t* finished = reinterpret_cast<uint8_t*>(this->finished);
    PH2(finished, nbs * nbm);

    std::vector<runtime::SizeType32> batchSlots(nbs, 0);
    cudaMemcpy(batchSlots.data(), this->batchSlots, sizeof(runtime::SizeType32) * nbs, cudaMemcpyDeviceToHost);

    std::vector<int*> outputIdsPtr(nbs, 0);
    cudaMemcpy(outputIdsPtr.data(), this->outputIdsPtr, sizeof(int*) * nbs, cudaMemcpyDeviceToHost);

    std::vector<int*> parentIdsPtr(nbs, 0);
    cudaMemcpy(parentIdsPtr.data(), this->parentIdsPtr, sizeof(int*) * nbs, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < nbs; ++i)
    {
        int slot = batchSlots[i];
        printf("slot=%d\n", slot);
        printf("outputIdsPtr[slot]=%p\n", outputIdsPtr[slot]);
        PH3(outputIdsPtr[slot], nbm * msl, msl);
    }
    for (int i = 0; i < nbs; ++i)
    {
        int slot = batchSlots[i];
        printf("slot=%d\n", slot);
        printf("parentIdsPtr[slot]=%p\n", parentIdsPtr[slot]);
        PH3(parentIdsPtr[slot], nbm * msl, msl);
    }

    // May not available in some context
    // PH3(this->outputIdsUnfinish, nbs * nbm * msl, msl);
    // PH3(this->parentIdsUnfinish, nbs * nbm * msl, msl);

    printf("================ print BeamHypotheses stop\n");
#endif
}

// template <typename T>
// void printLogProbs(T const* x, int const nBS, int const nBMIn, int const nBM, int const nV)
// {
//     for (int bs = 0; bs < nBS; ++bs)
//     {
//         T const* ptrBatch = x + bs * nBM * nV;
//         printArrayInfo(ptrBatch, nBMIn * nV, std::string("Request ") + std::to_string(bs));
//         for (int bm = 0; bm < nBMIn; ++bm)
//         {
//             T const* ptrBeam = ptrBatch + bm * nV;
//             printArrayInfo(ptrBeam, nV, std::string("Beam ") + std::to_string(bm), true);
//         }
//     }
// }

// template void printLogProbs<float>(float const* x, int const nBS, int const nBMIn, int const nBM, int const nV);
// template void printLogProbs<half>(half const* x, int const nBS, int const nBMIn, int const nBM, int const nV);

} // namespace kernels
} // namespace tensorrt_llm
