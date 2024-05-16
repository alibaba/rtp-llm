// Copyright (c) 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include "dtype.h"

namespace xft {

void invokeLayerLLaMA(DataType dt, ActivationType at, NormType nt, int batchSize, int inputSeqLen, int attHeadDim,
        int attHeadNum, int kvHeadNum, int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step,
        int hiddenSize, int intermediateSize, void *output, int outputStride, const void *input, int inputStride,
        const float *ln1Gamma, const float *ln1Beta, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *ln2Gamma, const float *ln2Beta,
        const void *gateWeight, const void *upWeight, const void *downWeight, const float *queryBias = nullptr,
        const float *keyBias = nullptr, const float *valueBias = nullptr, const float *attnOutBias = nullptr);

} // namespace xft