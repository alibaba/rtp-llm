#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
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
LinearSoftmaxHandlerImpl<T>::~LinearSoftmaxHandlerImpl(){
    freeBuffer();
}

template<typename T>
void LinearSoftmaxHandlerImpl<T>::allocateBuffer(size_t batch_size) {
    const size_t hidden_units = params_.head_num_ * params_.size_per_head_;
    sliced_hidden_buffer_ =
        reinterpret_cast<T*>(allocator_->reMalloc(sliced_hidden_buffer_, sizeof(T) * batch_size * hidden_units));
    cu_seqlens_     = reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1)));
    input_lengths_gpu_buf = reinterpret_cast<int*>(allocator_->reMalloc(input_lengths_gpu_buf, sizeof(int) * batch_size));
}

template<typename T>
void LinearSoftmaxHandlerImpl<T>::freeBuffer() {
    allocator_->free((void**)sliced_hidden_buffer_);
    allocator_->free((void**)cu_seqlens_);
    allocator_->free((void**)input_lengths_gpu_buf);
}

template<typename T>
void LinearSoftmaxHandlerImpl<T>::loadTensor(std::unordered_map<std::string, ft::ConstBufferPtr>& tensors) {
    // weight
    auto weight_it = tensors.find("w_out.weight");
    if (weight_it == tensors.end()) {
        throw std::runtime_error("can't find w_out.weight");
    } else {
        transposed_weight_ = device_->transpose({*(weight_it->second)});// weight_it->second->data<T>();a
        linear_weight.kernel = transposed_weight_->data<T>();
    }
    // bias
    auto bias_it = tensors.find("w_out.bias");
    if (bias_it == tensors.end()) {
        throw std::runtime_error("can't find w_out.bias");
    }
    else {
        linear_weight.bias = bias_it->second->data<T>();
    }
    is_initalized_ = true;
}

template<typename T>
th::Tensor LinearSoftmaxHandlerImpl<T>::forward(th::Tensor hidden_states, th::Tensor input_lengths) {
    const int* input_lengths_cpu_buf = input_lengths.data_ptr<int>();
    const size_t hidden_units = params_.head_num_ * params_.size_per_head_;
    const size_t length = input_lengths.size(0);
    const size_t max_context_seq_length = *std::max_element(input_lengths_cpu_buf, input_lengths_cpu_buf + (int)length);
    if (!is_initalized_) {
        throw std::runtime_error("mainse handler not initalized!");
    }

    allocateBuffer(length);
    cudaMemcpyAsync(input_lengths_gpu_buf, input_lengths_cpu_buf, sizeof(int) * length, cudaMemcpyHostToDevice, stream_);

    invokeLookupHiddenStateOfFirstToken(
        sliced_hidden_buffer_,
        (T*)hidden_states.data_ptr(),
        input_lengths_gpu_buf,
        length,
        hidden_units,
        stream_);

    th::Tensor decoder_output =
        torch::zeros({(int64_t)length, (int64_t)2},
                     torch::dtype(at::ScalarType::Half).device(torch::kCUDA).requires_grad(false));
    T* decoder_output_buf = (T*)decoder_output.data_ptr();

    float     alpha            = 1.0f;
    float     beta             = 0.0f;
    const cudaDataType_t gemm_data_type = ft::getCudaDataType<T>();
    // gemm
    print_bsd(-1, "origin", sliced_hidden_buffer_, 1, length, hidden_units);

    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          2,  // n
                          length, // m
                          hidden_units,  // k
                          &alpha,
                          linear_weight.kernel,
                          gemm_data_type,
                          2,               // k
                          sliced_hidden_buffer_,
                          gemm_data_type,
                          hidden_units,  // k
                          &beta,
                          decoder_output_buf,
                          gemm_data_type,
                          2, /* n */
                          CUDA_R_32F,
                          cublasGemmAlgo_t(-1));
    // bias softmax
    invokeAddBiasSoftMax(decoder_output_buf, linear_weight.bias, nullptr, nullptr, length, 2, 2, stream_);
    print_bsd(-1, "mainse_output", decoder_output_buf, 1, 1, 2, 0, 2);
    return decoder_output.cpu();
}

LinearSoftmaxHandler::LinearSoftmaxHandler(const GptInitParameter& params): HandlerBase(params) {
    //@miji FIXME
    DataType data_type = DataType::TYPE_FP16;
    switch (data_type) {
        case DataType::TYPE_FP32:
            throw std::runtime_error("not support fp32");
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

template class LinearSoftmaxHandlerImpl<half>;

} // namespace rtp_llmπ