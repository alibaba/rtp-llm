#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "maga_transformer/cpp/embedding_engine/handlers/LinearSoftmaxHandler.h"

using namespace fastertransformer;
namespace rtp_llm {

template<typename T>
LinearSoftmaxHandlerImpl<T>::LinearSoftmaxHandlerImpl(const GptInitParameter& params): IHandlerImpl(params), is_initalized_(false) {
    ft::DeviceFactory::initDevices(ft::DeviceFactory::getDefaultGlobalDeviceParams());
    device_ = dynamic_cast<CudaDevice*>(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda));
    allocator_      = device_->getAllocator();
    cublas_wrapper_ = device_->cublasMMWrapperPtr();
    stream_         = device_->stream();
}

template<typename T>
std::vector<std::string> LinearSoftmaxHandlerImpl<T>::tensorInfo() {
    return {"w_out.weight", "w_out.bias"};
}

template<typename T>
absl::Status LinearSoftmaxHandlerImpl<T>::loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) {
    // weight
    auto weight_it = tensors.find("w_out.weight");
    if (weight_it == tensors.end()) {
        return absl::InternalError("can't find w_out.weight");
    } else {                
        transposed_weight_ = device_->transpose({*(weight_it->second)});// weight_it->second->data<T>();a
        linear_weight.kernel = transposed_weight_->data<T>();
    }
    // bias
    auto bias_it = tensors.find("w_out.bias");
    if (bias_it == tensors.end()) {
        return absl::InternalError("can't find w_out.bias");
    }
    else {
        linear_weight.bias = bias_it->second->data<T>();
    }    
    is_initalized_ = true;
    return absl::OkStatus();
}

template<typename T>
absl::StatusOr<std::unique_ptr<GptModelOutputs>> LinearSoftmaxHandlerImpl<T>::forward(const ModelRequest& model_input, const GptModelOutputs& model_output) const {
    if (!is_initalized_) {
        return absl::InternalError("mainse handler not initalized!");
    }
    const size_t length = model_input.input_lengths->shape()[0];
    const size_t hidden_units = params_.head_num_ * params_.size_per_head_;
    std::unique_ptr<GptModelOutputs> mainse_output = std::make_unique<GptModelOutputs>();
    mainse_output->hidden_states = device_->allocateBuffer({ft::getTensorType<T>(), {length, 2}, AllocationType::DEVICE}, {});
    float     alpha            = 1.0f;
    float     beta             = 0.0f;
    const cudaDataType_t gemm_data_type = ft::getCudaDataType<T>();
    // gemm
    print_bsd(-1, "origin", model_output.hidden_states->data<T>(), 1, length, hidden_units);
    print_bsd(-1, "kernel", linear_weight.kernel, 1, 2, hidden_units);
    print_bsd(-1, "out", mainse_output->hidden_states->data<T>(), 1, length, 2);
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          2,  // n
                          length, // m
                          hidden_units,  // k
                          &alpha,
                          linear_weight.kernel,
                          gemm_data_type,
                          2,               // k
                          model_output.hidden_states->data<T>(),
                          gemm_data_type,
                          hidden_units,  // k
                          &beta,
                          mainse_output->hidden_states->data<T>(),
                          gemm_data_type,
                          2, /* n */
                          CUDA_R_32F,
                          cublasGemmAlgo_t(-1));
    // bias softmax
    invokeAddBiasSoftMax(mainse_output->hidden_states->data<T>(), linear_weight.bias, nullptr, nullptr, length, 2, 2, stream_);
    print_bsd(-1, "mainse_output", mainse_output->hidden_states->data<T>(), 1, 1, 2);
    return mainse_output;
}

LinearSoftmaxHandler::LinearSoftmaxHandler(const GptInitParameter& params): HandlerBase(params) {
    //@miji FIXME
    DataType data_type = DataType::TYPE_FP16;
    switch (data_type) {
        case DataType::TYPE_FP32:
            handler_impl_ = std::make_unique<LinearSoftmaxHandlerImpl<float>>(params);
            break;
        case DataType::TYPE_FP16:
            handler_impl_ = std::make_unique<LinearSoftmaxHandlerImpl<half>>(params);
            break;
        case DataType::TYPE_BF16:            
            // bfloat16 add_bias_softmax not implemented
            throw std::runtime_error("not support bfloat16");                            
            break;
        default:
            throw std::runtime_error("Wrong tensor type::" + std::to_string(data_type));
    }
}

template class LinearSoftmaxHandlerImpl<float>;
template class LinearSoftmaxHandlerImpl<half>;

} // namespace rtp_llmπ