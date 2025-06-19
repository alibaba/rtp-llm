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
#include "rtp_llm/cpp/cuda/nvtx/kernel_profiler.h"

namespace ft_nvtx {

class ReportCounter{

public:
    void setReportStep(int step) {
        step_ = step;
    }
    void increment() {
        if (step_ <= 0) {
            return;
        }
        counter_++;
        if (counter_ == step_) {
            counter_ = 0;
        }
    }
    bool shouldReport() {
        return step_ > 0 && counter_ == 0;
    }

protected:
    int step_    = -1;
    int counter_ = -1;
};

class NvtxResource {
private:
    ReportCounter counter_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    NvtxResource(): counter_(ReportCounter()), metrics_reporter_(nullptr) {} 
    NvtxResource(const NvtxResource&) = delete;
    NvtxResource(const NvtxResource&&) = delete;
    NvtxResource& operator=(const NvtxResource&) = delete;
public:
    ReportCounter& getCounter() {
        return counter_;
    }

    void setMetricReporter(kmonitor::MetricsReporterPtr& ptr) {
        metrics_reporter_ = ptr;
    }

    kmonitor::MetricsReporterPtr& getMetricReporter() {
        return metrics_reporter_;
    }

    static NvtxResource& Instance() {
        static NvtxResource instance;
        return instance;
    }
};


std::string getScope();
void        addScope(std::string name);
void        setScope(std::string name);
void        resetScope();
void        setDeviceDomain(int deviceId);
int         getDeviceDomain();
void        resetDeviceDomain();
}  // namespace ft_nvtx
