#pragma once

#include "rtp_llm/cpp/utils/Logger.h"

#if USING_ROCM
#include <ATen/hip/HIPGraph.h>

class GraphStreamLife {
public:
    GraphStreamLife(at::hip::HIPStream capture_stream):
        origin_stream_(at::hip::getCurrentHIPStream(at::hip::current_device())) {
        at::hip::setCurrentHIPStream(capture_stream);
        RTP_LLM_LOG_INFO("Set HIP Stream: capture_stream -> %d, origin_stream -> %d",
                         capture_stream.stream(),
                         origin_stream_.stream());
    }
    ~GraphStreamLife() {
        at::hip::setCurrentHIPStream(origin_stream_);
    }

    GraphStreamLife(const GraphStreamLife&)            = delete;
    GraphStreamLife& operator=(const GraphStreamLife&) = delete;
    GraphStreamLife(GraphStreamLife&&)                 = delete;
    GraphStreamLife& operator=(GraphStreamLife&&)      = delete;

private:
    at::hip::HIPStream origin_stream_;
};

#else
#include <ATen/cuda/CUDAGraph.h>

class GraphStreamLife {
public:
    GraphStreamLife(at::cuda::CUDAStream capture_stream):
        origin_stream_(at::cuda::getCurrentCUDAStream(at::cuda::current_device())) {
        at::cuda::setCurrentCUDAStream(capture_stream);
        RTP_LLM_LOG_INFO("Set Cuda Stream: capture_stream -> %d, origin_stream -> %d",
                         capture_stream.stream(),
                         origin_stream_.stream());
    }
    ~GraphStreamLife() {
        at::cuda::setCurrentCUDAStream(origin_stream_);
    }

    GraphStreamLife(const GraphStreamLife&)            = delete;
    GraphStreamLife& operator=(const GraphStreamLife&) = delete;
    GraphStreamLife(GraphStreamLife&&)                 = delete;
    GraphStreamLife& operator=(GraphStreamLife&&)      = delete;

private:
    at::cuda::CUDAStream origin_stream_;
};

#endif
