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

#include <iostream>
#include <vector>
#include "nvtx_utils.h"
#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif
#include "rtp_llm/cpp/th_op/ConfigModules.h"

namespace ft_nvtx {

static std::vector<rtp_llm::KernelProfiler*> profilers;
static std::string                                     scope;
static int                                             domain = 0;

std::string getScope() {
    return scope;
}
void addScope(std::string name) {
    scope = scope + name + "/";
    return;
}
void setScope(std::string name) {
    scope = name + "/";
    return;
}
void resetScope() {
    scope = "";
    return;
}
void setDeviceDomain(int deviceId) {
    domain = deviceId;
    return;
}
void resetDeviceDomain() {
    domain = 0;
    return;
}
int getDeviceDomain() {
    return domain;
}

}  // namespace ft_nvtx
