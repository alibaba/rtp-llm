#pragma once

#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/layers/DynamicDecodeLayer.h"


namespace ft = fastertransformer;

namespace rtp_llm {

class FakeSampler {
public:
    FakeSampler(const ft::GptInitParameter& gpt_init_parameter);
    SamplerOutput forward(SamplerInputs& inputs);
    void          allocateBuffer(size_t total_batch_size);
    void          freeBuffer();

private:
    ft::CudaDevice*               device_;
    ft::DynamicDecodeLayer<half>* dynamic_decode_layer_;
    cudaDeviceProp                prop_;
    cudaStream_t                  stream_;
    size_t                        vocab_size_;
    int*                          top_k_;
    int*                          end_id_;
    ft::IAllocator*               allocator_;
};

}  // namespace rtp_llm
