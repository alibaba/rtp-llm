
#pragma once

#include "absl/status/statusor.h"
#include "src/fastertransformer/utils/DenseWeight.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"

namespace rtp_llm {

template<typename T>
class LinearSoftmaxHandlerImpl: public IHandlerImpl {
public:
    LinearSoftmaxHandlerImpl(const ft::GptInitParameter& params);
    ~LinearSoftmaxHandlerImpl();
    void loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) override;
    void allocateBuffer(size_t batch_size);
    void freeBuffer();
    th::Tensor forward(th::Tensor hidden_states, th::Tensor input_lengths) override;
private:
    ft::CudaDevice*      device_;
    ft::IAllocator*      allocator_;
    ft::cublasMMWrapper* cublas_wrapper_;
    ft::DenseWeight<T>   linear_weight;
    T*                   linear_buffer;
    cudaStream_t         stream_;
    bool                 is_initalized_        = false;
    ft::BufferPtr        transposed_weight_    = nullptr;
    T*                   sliced_hidden_buffer_ = nullptr;
    int*                 input_lengths_gpu_buf = nullptr;
    int*                 cu_seqlens_           = nullptr;
};

class LinearSoftmaxHandler: public HandlerBase {
public:
    LinearSoftmaxHandler(const ft::GptInitParameter& params);
};

} // namespace rtp_llm