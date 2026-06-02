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

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cstddef>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

template<typename T>
void check(T result, const char* const file, int const line);
#define check_cuda_value(val) rtp_llm::check((val), __FILE__, __LINE__)

void syncAndCheckInDebug(const char* const file, int const line);
#define check_cuda_error() rtp_llm::syncAndCheckInDebug(__FILE__, __LINE__)

int  get_sm();
bool is_sm70();
bool is_sm8x();
bool is_sm90();
bool is_sm100();

float                      timing_function(const std::function<void(cudaStream_t)>& operation,
                                           int64_t                                  timing_iterations,
                                           cudaStream_t                             stream);
int                        getDevice();
int                        getDeviceCount();
int                        currentDeviceId();
void                       priorityRange(int* low_priority, int* high_priority, int device_id = -1);
std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm);
std::string                getDriverVersion();
int                        getCudaVersion();
bool                       checkAllNVLinks(std::vector<size_t> device_ids);
bool                       checkOnSameNumaNodes(std::vector<size_t> device_ids);
int                        getVisibleDeviceNum();
bool                       checkP2PAvailable(const std::vector<size_t>& tp_ranks, size_t rank);
int                        getMultiProcessorCount(int device_id = -1);
int                        getMaxSharedMemoryPerMultiprocessor(int device_id = -1);
int                        getMaxSharedMemoryPerBlockOptin(int device_id = -1);
int                        getMaxThreadsPerMultiprocessor(int device_id = -1);
int                        getMaxBlocksPerMultiprocessor(int device_id = -1);
int                        getComputeCapabilityMajor(int device_id = -1);
int                        getComputeCapabilityMinor(int device_id = -1);
std::pair<int, int>        getComputeCapability(int device_id = -1);

struct CaptureCheck {
    static bool in_cuda_graph_capture;
};

}  // namespace rtp_llm
