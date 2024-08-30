/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#pragma once

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off


extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_80_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_128_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_cu_cubin[];


extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_80_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_128_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_cu_cubin_len;

static const struct FusedMultiHeadAttentionKernelMetaInfoV2Sm70
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    bool mInterleaved;
    bool mFlashAttention;
    bool mFP32Accumulation;
    int mAttentionMaskType;
    bool mAlibiSupported;
    bool mTiled;
} sMhaKernelMetaInfosV2Sm70[] = {
{ DATA_TYPE_FP16, 0, 32, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm70_kernel", 12288, 128, 0, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 32, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm70_kernel_nl", 12288, 128, 64, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 40, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm70_kernel", 24576, 128, 0, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 40, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm70_kernel_nl", 24576, 128, 64, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 64, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm70_kernel", 24576, 128, 0, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 64, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm70_kernel_nl", 24576, 128, 64, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 80, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_80_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_80_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_80_sm70_kernel", 49152, 128, 0, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 80, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_80_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_80_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_80_sm70_kernel_nl", 49152, 128, 64, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 128, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_128_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_128_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_128_sm70_kernel", 49152, 128, 0, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 128, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_128_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_128_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_128_sm70_kernel_nl", 49152, 128, 64, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 160, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm70_kernel", 98304, 128, 0, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 160, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm70_kernel_nl", 98304, 128, 64, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 256, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm70_kernel", 98304, 128, 0, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 256, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm70_kernel_nl", 98304, 128, 64, false, true, false, 0, false, false },
{ DATA_TYPE_FP16, 0, 32, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_kernel", 12288, 128, 0, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 32, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_Causal_S_32_sm70_kernel_nl", 12288, 128, 64, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 40, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_kernel", 24576, 128, 0, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 40, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_Causal_S_40_sm70_kernel_nl", 24576, 128, 64, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 64, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_kernel", 24576, 128, 0, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 64, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_Causal_S_64_sm70_kernel_nl", 24576, 128, 64, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 80, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_kernel", 49152, 128, 0, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 80, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_80_sm70_kernel_nl", 49152, 128, 64, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 128, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_kernel", 49152, 128, 0, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 128, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_128_sm70_kernel_nl", 49152, 128, 64, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 160, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_kernel", 98304, 128, 0, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 160, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_160_sm70_kernel_nl", 98304, 128, 64, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 256, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_kernel", 98304, 128, 0, false, true, false, 1, false, false },
{ DATA_TYPE_FP16, 0, 256, kSM_70,  cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_Causal_S_256_sm70_kernel_nl", 98304, 128, 64, false, true, false, 1, false, false }
};

// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
