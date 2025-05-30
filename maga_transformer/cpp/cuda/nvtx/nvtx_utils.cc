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

namespace ft_nvtx {

static bool                                            has_read_nvtx_env = false;
static bool                                            is_enable_ft_nvtx = false;
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

bool isEnableNvtx() {
    if (!has_read_nvtx_env) {
        static char* ft_nvtx_env_char = std::getenv("FT_NVTX");
        is_enable_ft_nvtx = (ft_nvtx_env_char != nullptr && std::string(ft_nvtx_env_char) == "ON") ? true : false;
        has_read_nvtx_env = true;
    }
    return is_enable_ft_nvtx;
}

void ftNvtxRangePush(std::string name, cudaStream_t stream) {
    std::string                        kernel_name = (getScope() + name).c_str();
    if (NvtxResource::Instance().getCounter().shouldReport()) {
        rtp_llm::KernelProfiler* profiler =
            new rtp_llm::KernelProfiler(stream, kernel_name, NvtxResource::Instance().getMetricReporter());
        profiler->start();
        profilers.push_back(profiler);
    }

#ifdef USE_NVTX
    if (isEnableNvtx()) {
        nvtxRangePushA((getScope() + name).c_str());
    }
#endif
}

void ftNvtxRangePop() {
    if (NvtxResource::Instance().getCounter().shouldReport()) {
        rtp_llm::KernelProfiler* profiler = profilers.back();
        profiler->stop();
        profilers.pop_back();
        delete profiler;
    }
#ifdef USE_NVTX
    if (isEnableNvtx()) {
        nvtxRangePop();
    }
#endif
}

}  // namespace ft_nvtx
