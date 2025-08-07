#include "rtp_llm/cpp/devices/rocm_impl/NativeHipGraphRunner.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

namespace rtp_llm {

void HipGraphExecutor::captureBegin() {
    origin_stream_ = std::make_shared<at::hip::HIPStream>(at::hip::getCurrentHIPStream(at::hip::current_device()));
    device_->setStream(capture_stream_->stream());
    at::hip::setCurrentHIPStream(*capture_stream_);
    device_->syncAndCheck();
    graph_->capture_begin();
}
void HipGraphExecutor::captureEnd() {
    graph_->capture_end();
    device_->syncAndCheck();
    device_->setStream(origin_stream_->stream());
    at::hip::setCurrentHIPStream(*origin_stream_);
    device_->registerARGraphBuffers();
}

INSTANTIATE_HIPGRAPH_RUNNER(GptModelInputs, GptModelOutputs)

};  // namespace rtp_llm