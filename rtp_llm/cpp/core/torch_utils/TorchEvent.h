#pragma once

#include "rtp_llm/cpp/core/Event.h"
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

struct TorchEvent: public AsyncEvent {
    TorchEvent(const torch::Stream& stream = c10::cuda::getCurrentCUDAStream()) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    };

    ~TorchEvent() override = default;

    void synchronize() const override {
        event->synchronize();
    }

    bool checkReadiness() const override {
        return event->query();
    }

    std::shared_ptr<torch::Event> event;
};

}  // namespace rtp_llm
