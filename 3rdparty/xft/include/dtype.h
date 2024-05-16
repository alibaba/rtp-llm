// Copyright (c) 2023 Intel Corporation
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

namespace xft {
enum DataType {
    fp32 = 0,
    bf16,
    fp16,
    int8,
    w8a8,
    int4,
    nf4,
    bf16_fp16,
    bf16_int8,
    bf16_w8a8,
    bf16_int4,
    bf16_nf4,
    w8a8_int8,
    w8a8_int4,
    w8a8_nf4,
    unknown,
};

enum DeviceKind {
    iCPU = 0,
    iGPU,
};

enum NormType {
    RMS = 0,
    LN,
};

enum ActivationType {
    RELU = 0,
    GELU,
    SWIGLU,
    SILU,
};

} // namespace xft
