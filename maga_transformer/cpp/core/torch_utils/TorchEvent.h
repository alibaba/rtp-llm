#pragma once

#include "maga_transformer/cpp/core/Event.h"
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

struct TorchEvent : public DeviceEvent {
    TorchEvent(const torch::Stream& stream = c10::cuda::getCurrentCUDAStream()) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    };

    ~TorchEvent() override = default;

    void synchronize() const override {
        event->synchronize();
    }

    std::shared_ptr<torch::Event> event;
};

}
