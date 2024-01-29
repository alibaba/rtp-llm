/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

namespace ft_nvtx {

std::string getScope();
void        addScope(std::string name);
void        setScope(std::string name);
void        resetScope();
void        setDeviceDomain(int deviceId);
int         getDeviceDomain();
void        resetDeviceDomain();
bool        isEnableNvtx();

void ftNvtxRangePush(std::string name, cudaStream_t stream);
void ftNvtxRangePop();
}  // namespace ft_nvtx

#define PUSH_RANGE(stream, name)                                                                                       \
    { ft_nvtx::ftNvtxRangePush(name, stream); }

#define POP_RANGE                                                                                                      \
    { ft_nvtx::ftNvtxRangePop(); }
