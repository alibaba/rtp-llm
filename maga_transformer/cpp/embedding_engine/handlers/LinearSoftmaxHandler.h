
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
    std::vector<std::string> tensorInfo();
    absl::Status loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors);
    absl::StatusOr<std::unique_ptr<GptModelOutputs>> forward(const ModelRequest& model_input, const GptModelOutputs& model_output) const;    
private:
    ft::CudaDevice*      device_;
    ft::IAllocator*      allocator_;
    ft::cublasMMWrapper* cublas_wrapper_;
    ft::DenseWeight<T>   linear_weight;
    T*                   linear_buffer;
    cudaStream_t         stream_;
    bool                 is_initalized_;
    ft::BufferPtr        transposed_weight_;
};

class LinearSoftmaxHandler: public HandlerBase {
public:
    LinearSoftmaxHandler(const ft::GptInitParameter& params);    
};

} // namespace rtp_llm