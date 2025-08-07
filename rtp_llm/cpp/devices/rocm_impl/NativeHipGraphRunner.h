#pragma once

#include "rtp_llm/cpp/devices/NativeGraphRunnerBase.h"
#include <ATen/hip/HIPGraph.h>

namespace rtp_llm {

class ROCmDevice;

class HipGraphExecutor: public ExecutorBase {
public:
    HipGraphExecutor(ROCmDevice* device, std::shared_ptr<at::hip::HIPStream> stream):
        ExecutorBase(), device_(device), graph_(std::make_shared<at::cuda::CUDAGraph>()), capture_stream_(stream) {}
    void replay() override {
        graph_->replay();
    }
    void captureBegin() override;
    void captureEnd() override;

private:
    ROCmDevice*                          device_         = nullptr;
    std::shared_ptr<at::cuda::CUDAGraph> graph_          = nullptr;
    std::shared_ptr<at::hip::HIPStream>  capture_stream_ = nullptr;
    std::shared_ptr<at::hip::HIPStream>  origin_stream_  = nullptr;
};

template<typename Input, typename Output>
class NativeHipGraphRunner: public NativeGraphRunnerBase<Input, Output> {
public:
    NativeHipGraphRunner(DeviceBase* device):
        NativeGraphRunnerBase<Input, Output>(device),
        capture_stream_(std::make_shared<at::hip::HIPStream>(at::hip::getStreamFromPool(true))) {}
    std::shared_ptr<ExecutorBase> makeExecutor() override {
        return std::make_shared<HipGraphExecutor>(dynamic_cast<ROCmDevice*>(this->device_), capture_stream_);
    }

private:
    std::shared_ptr<at::hip::HIPStream> capture_stream_ = nullptr;
};

#define INSTANTIATE_HIPGRAPH_RUNNER(I, O) template class NativeHipGraphRunner<I, O>;

};  // namespace rtp_llm