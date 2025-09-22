#pragma once

#include "rtp_llm/cpp/core/Event.h"
#include <torch/all.h>
#ifdef USING_ROCM
#include <ATen/hip/HIPContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#endif


namespace rtp_llm {

struct TorchEvent: public DeviceEvent {
#ifdef USING_ROCM
    TorchEvent(const torch::Stream& stream = c10::hip::getCurrentHIPStream()) {
        event = std::make_shared<torch::Event>(torch::kHIP);
        event->record(stream);
    };
#else
    TorchEvent(const torch::Stream& stream = c10::cuda::getCurrentCUDAStream()) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    };
#endif
    ~TorchEvent() override = default;

    void synchronize() const override {
        throw std::runtime_error("TorchEvent::synchronize() is not implemented.");
    }

    bool checkReadiness() const override {
        throw std::runtime_error("TorchEvent::checkReadiness() is not implemented.");
    }

    std::shared_ptr<torch::Event> event;
};

}  // namespace rtp_llm
