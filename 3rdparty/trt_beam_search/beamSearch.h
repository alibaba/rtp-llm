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

#pragma once

#include "common.h"

namespace tensorrt_llm {

struct BeamSearchConfig {
    size_t mVPart;
    size_t mByteMaxSharedMemoryPerBlock;
    size_t mByteSharedMemoryStage1;
    size_t mByteSharedMemoryStage3;
    size_t mWorkspaceSize;
    bool   mVBWS;
    bool   mV2;
};

template<typename T>
BeamSearchConfig configureBeamSearch(runtime::SizeType32 batchSize,
                                     runtime::SizeType32 beamWidthIn,
                                     runtime::SizeType32 beamWidthOut,
                                     runtime::SizeType32 vocabSize);

}