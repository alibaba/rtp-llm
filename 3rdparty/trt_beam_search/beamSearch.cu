/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "rtp_llm/cpp/utils/Logger.h"
#include "beamSearch.h"
#include "beamSearchKernels.h"
#include "beamSearchKernelsTemplate.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm {

#define GET_INFO_STAGE1(paddedBeamWidth)                                                                               \
    {                                                                                                                  \
        int constexpr nBlock = (paddedBeamWidth < 16) ? ((paddedBeamWidth < 8) ? kThreadForSmallBeamWidth : 128) : 64; \
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                 \
            &nMaxActiveBlock, beamStage1Kernel<T, 2 * paddedBeamWidth, nBlock>, nBlock, 0));                           \
        TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage1Kernel<T, 2 * paddedBeamWidth, nBlock>));               \
        break;                                                                                                         \
    }

#define GET_INFO_STAGE2(paddedBeamWidth)                                                                               \
    {                                                                                                                  \
        if (nByteDynamicSharedMemoryStage2 > nByteMaxSharedMemoryPerBlock)                                             \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 128, false>));           \
        }                                                                                                              \
        else if (nVPart <= 32)                                                                                         \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 32, true>));             \
        }                                                                                                              \
        else if (nVPart <= 64)                                                                                         \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 64, true>));             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, paddedBeamWidth, 128, true>));            \
        }                                                                                                              \
        break;                                                                                                         \
    }

#define GET_INFO_STAGE3(paddedBeamWidth, isV2)                                                                         \
    {                                                                                                                  \
        int constexpr nThreadStage3 = (paddedBeamWidth + 31) / 32 * 32;                                                \
        TLLM_CUDA_CHECK(                                                                                               \
            cudaFuncGetAttributes(&attr, beamStage3Kernel<T, paddedBeamWidth, nThreadStage3, true, isV2>));            \
        break;                                                                                                         \
    }

template<typename T>
BeamSearchConfig configureBeamSearch(runtime::SizeType32 batchSize,
                                     runtime::SizeType32 beamWidthIn,
                                     runtime::SizeType32 beamWidthOut,
                                     runtime::SizeType32 vocabSize) {

    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    BeamSearchConfig config;

    config.mVBWS = beamWidthIn != beamWidthOut;

    SizeType32 const paddedBeamWidthIn{padToNextPowerOfTwo(beamWidthIn)};
    SizeType32 const paddedBeamWidthOut{padToNextPowerOfTwo(beamWidthOut)};
    cudaFuncAttributes attr;

    // Find device information to determine `nVPart`.
    int const nByteMaxSharedMemoryPerSM = getMaxSharedMemoryPerSM();
    int const nByteMaxSharedMemoryPerBlock = getMaxSharedMemoryPerBlockOptin();
    int const nByteReservedSharedMemoryPerBlock = nByteMaxSharedMemoryPerSM - nByteMaxSharedMemoryPerBlock;
    config.mByteMaxSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock;

    if (beamWidthIn == beamWidthOut && beamWidthIn <= kMaxBeamWidthForV1)
    {
        // V1 workflow for small beam width and non-VBWS
        // Stage 1
        int nMaxActiveBlock = -1;
        switch (paddedBeamWidthIn)
        {
        case 1: GET_INFO_STAGE1(1);
        case 2: GET_INFO_STAGE1(2);
        case 4: GET_INFO_STAGE1(4);
        case 8: GET_INFO_STAGE1(8);
        default: break;
        }
        int nByteStaticSharedMemory = attr.sharedSizeBytes;
        int nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        // Find the maximum of `nBlock` (maximum of `nVPart`, minimum of `nByteDynamicSharedMemoryStage1`), s.t.
        // `nVPart <= kMaxVPartStage1 && nByteDynamicSharedMemoryStage1 * nVPart >= sizeof(T) * vocabSize`
        TLLM_CHECK_WITH_INFO(nByteMaxDynamicSharedMemoryPerBlock * kMaxVPartStage1 >= sizeof(T) * vocabSize,
            "vocab_size is too large for Beam search.");
        int nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        int nBlock = nMaxActiveBlock;
        int nVPart = kMaxVPartStage1 + 1;
        for (; nBlock > 0 && nVPart > kMaxVPartStage1; --nBlock)
        {
            int nByteDynamicSharedMemoryStage1 = nByteMaxSharedMemoryPerSM / nBlock - nByteExtralSharedMemory;
            nByteDynamicSharedMemoryStage1 -= nByteDynamicSharedMemoryStage1 % sizeof(T);
            nVPart = ceilDiv(sizeof(T) * vocabSize, nByteDynamicSharedMemoryStage1);
        }
        TLLM_CHECK_WITH_INFO(nBlock >= 0, "No enough active blocks for Beam Search stage 1 kernel.");

        int const nByteDynamicSharedMemoryStage1 = sizeof(T) * ceilDiv(vocabSize, nVPart);
        config.mVPart = nVPart;
        config.mByteSharedMemoryStage1 = nByteDynamicSharedMemoryStage1; // Only dynamic shared memory

        // Stage 2
        TLLM_CHECK_WITH_INFO(batchSize * beamWidthIn * paddedBeamWidthIn < (1 << 21),
            "batch_size or beam_width is too large for Beam search");
        size_t const nByteDynamicSharedMemoryStage2 = common::roundUp(
            sizeof(float) * nVPart * (paddedBeamWidthIn * 4) + sizeof(cub::KeyValuePair<int, T>) * paddedBeamWidthIn * 2,
            4);
        switch (paddedBeamWidthIn)
        {
        case 1: GET_INFO_STAGE2(1);
        case 2: GET_INFO_STAGE2(2);
        case 4: GET_INFO_STAGE2(4);
        case 8: GET_INFO_STAGE2(8);
        default: break;
        }
        nByteStaticSharedMemory = attr.sharedSizeBytes;
        nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        bool const bUseGlobalMemoryStage2 = (nByteDynamicSharedMemoryStage2 > nByteMaxDynamicSharedMemoryPerBlock);

        // Stage 3
        // Keep top 2K candidates in case of k candidates finishes in one iteration
        size_t const nByteDynamicSharedMemoryStage3
            = common::roundUp(sizeof(T) * paddedBeamWidthIn * paddedBeamWidthIn * 2, 4);
        switch (paddedBeamWidthIn)
        {
        case 1: GET_INFO_STAGE3(1, false);
        case 2: GET_INFO_STAGE3(2, false);
        case 4: GET_INFO_STAGE3(4, false);
        case 8: GET_INFO_STAGE3(8, false);
        }
        nByteStaticSharedMemory = attr.sharedSizeBytes;
        nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        bool const bUseGlobalMemoryStage3 = (nByteDynamicSharedMemoryStage3 > nByteMaxDynamicSharedMemoryPerBlock);
        config.mByteSharedMemoryStage3 = nByteStaticSharedMemory; // Only static shared memory

        // Compute workspace size, see `beamSearchKernelsTemplate.h` for detailed information
        // |<----- Workspace ----->|
        // |<- A ->|<- B ->|<- C ->|
        //         |<---- D ---->|
        // A for data exchange between stage 2 and 3
        // B for data exchange between stage 1 and 2, can be reuse for stage 3
        // C for stage 2 if `bUseGlobalMemoryStage2 == true`, can be reuse for stage 3
        // D for stage 3 if `bUseGlobalMemoryStage3 == true`
        size_t const nByteA = common::roundUp(sizeof(T) * batchSize * paddedBeamWidthIn * paddedBeamWidthOut * 4, 4);
        size_t const nByteB
            = common::roundUp(sizeof(T) * batchSize * paddedBeamWidthIn * kMaxVPartStage1 * paddedBeamWidthOut * 4, 4);
        size_t const nByteC = (bUseGlobalMemoryStage2) ? nByteDynamicSharedMemoryStage2 : 0;
        size_t const nByteD = (bUseGlobalMemoryStage3) ? nByteDynamicSharedMemoryStage3 : 0;
        config.mWorkspaceSize = nByteA + std::max(nByteB + nByteC, nByteD);
    }
    else // V2 workflow for large beam width or VBWS
    {
        config.mV2 = true;
        switch (max(paddedBeamWidthIn, paddedBeamWidthOut))
        {
        case 1: GET_INFO_STAGE3(1, true);
        case 2: GET_INFO_STAGE3(2, true);
        case 4: GET_INFO_STAGE3(4, true);
        case 8: GET_INFO_STAGE3(8, true);
        case 16: GET_INFO_STAGE3(16, true);
        case 32: GET_INFO_STAGE3(32, true);
        case 64: GET_INFO_STAGE3(64, true);
        case 128: GET_INFO_STAGE3(128, true);
        case 256: GET_INFO_STAGE3(256, true);
        case 512: GET_INFO_STAGE3(512, true);
        case 1024: GET_INFO_STAGE3(1024, true);
        case 2048: GET_INFO_STAGE3(2048, true);
        case 4096: GET_INFO_STAGE3(4096, true);
        }
        config.mByteSharedMemoryStage3 = attr.sharedSizeBytes; // Only static shared memory

        // Compute shared memory size for stage 3
        // Compute workspace size, see `beamSearchKernelsTemplate.h` for detailed information
        // |<----------------------------------------- Workspace ------------------------------------------>|
        // |<- Stage2Ids ->|<- Stage2LogProbs ->|<- Stage1Ids ->|<- Stage1LogProbs ->|<---- Stage1TopK ---->|
        //                                                                           |<- stage2TopK ->|
        //                                      |<------------------ Stage3 ------------------>|
        size_t const nByteStage1LogProbs = roundUp(sizeof(T) * batchSize * beamWidthIn * beamWidthOut * 2, 4);
        size_t const nByteStage1Ids = roundUp(sizeof(int) * batchSize * beamWidthIn * beamWidthOut * 2, 4);
        size_t const nByteStage2LogProbs = roundUp(sizeof(T) * batchSize * beamWidthOut * 2, 4);
        size_t const nByteStage2Ids = roundUp(sizeof(int) * batchSize * beamWidthOut * 2, 4);
        size_t const nByteStage1TopK
            = invokeComputeTopkLastDimWorkspaceSize<T>(batchSize * beamWidthIn, vocabSize, beamWidthOut * 2, true);
        size_t const nByteStage2TopK = invokeComputeTopkLastDimWorkspaceSize<T>(
            batchSize, beamWidthIn * beamWidthOut * 2, beamWidthOut * 2, true);
        size_t const nByteStage3 = sizeof(T) * beamWidthIn * beamWidthOut * 2;
        config.mWorkspaceSize = nByteStage2LogProbs + nByteStage2Ids
            + max(nByteStage1LogProbs + nByteStage1Ids + max(nByteStage1TopK, nByteStage2TopK), nByteStage3);
    }

    RTP_LLM_LOG_DEBUG("configureBeamSearch: "
                      "config.mVPart = %zu, "
                      "config.mByteMaxSharedMemoryPerBlock = %zu, "
                      "config.mByteSharedMemoryStage1 = %zu, "
                      "config.mByteSharedMemoryStage3 = %zu, "
                      "config.mWorkspaceSize = %zu, "
                      "config.mVBWS = %s, "
                      "config.mV2 = %s",
                      config.mVPart,
                      config.mByteMaxSharedMemoryPerBlock,
                      config.mByteSharedMemoryStage1,
                      config.mByteSharedMemoryStage3,
                      config.mWorkspaceSize,
                      config.mVBWS ? "true" : "false",
                      config.mV2 ? "true" : "false");

    return config;
}

template BeamSearchConfig configureBeamSearch<float>(runtime::SizeType32 batchSize,
                                                     runtime::SizeType32 beamWidthIn,
                                                     runtime::SizeType32 beamWidthOut,
                                                     runtime::SizeType32 vocabSize);
template BeamSearchConfig configureBeamSearch<half>(runtime::SizeType32 batchSize,
                                                    runtime::SizeType32 beamWidthIn,
                                                    runtime::SizeType32 beamWidthOut,
                                                    runtime::SizeType32 vocabSize);

}