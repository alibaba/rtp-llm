#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/embedding_engine/handlers/LinearSoftmaxHandler.h"
#include <cstdlib>
#include <stdlib.h>

using namespace fastertransformer;

namespace rtp_llm {

LinearSoftmaxHandlerImpl::LinearSoftmaxHandlerImpl(const GptInitParameter& params): IHandlerImpl(params), is_initalized_(false) {
    DeviceFactory::initDevices(params);
    device_ = DeviceFactory::getDefaultDevice();
}

LinearSoftmaxHandlerImpl::~LinearSoftmaxHandlerImpl(){}

void LinearSoftmaxHandlerImpl::loadTensor(std::unordered_map<std::string, ConstBufferPtr>& tensors) {
    // weight
    auto weight_it = tensors.find("w_out.weight");
    if (weight_it == tensors.end()) {
        throw std::runtime_error("can't find w_out.weight");
    } else {
        weight_ = device_->transpose({*(weight_it->second)});
    }
    // bias
    auto bias_it = tensors.find("w_out.bias");
    if (bias_it == tensors.end()) {
        throw std::runtime_error("can't find w_out.bias");
    } else {
        bias_ = bias_it->second;
    }
    is_initalized_ = true;
}

void getStateIndexes(int32_t*       select_indexes,
                     const int32_t* sequence_length,
                     const int32_t  batch_size)
{
    int32_t        total_seq_len        = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        select_indexes[i] = total_seq_len;
        total_seq_len += sequence_length[i];
    }
}

th::Tensor LinearSoftmaxHandlerImpl::forward(th::Tensor hidden_states, th::Tensor input_lengths) {
    if (!is_initalized_) {
        throw std::runtime_error("mainse handler not initalized!");
    }

    const size_t batch_size = input_lengths.size(0);

    auto indexes_cpu = device_->allocateBuffer(
        {DataType::TYPE_INT32, {batch_size}, AllocationType::HOST});
    getStateIndexes(indexes_cpu->data<int32_t>(), input_lengths.data_ptr<int32_t>(), batch_size);
    auto input_buf = torchTensor2Buffer(hidden_states);
    auto indexes_buf = device_->clone({*indexes_cpu});
    auto sliced_hidden_buffer = device_->select({*input_buf, *indexes_buf});

    printBufferData(*input_buf, "input_buf");
    printBufferData(*sliced_hidden_buffer, "sliced_hidden_buffer");

    // out with fp32
    auto gemm_output = device_->gemm({*sliced_hidden_buffer, *weight_, std::nullopt, nullptr, DataType::TYPE_FP32});
    auto decoder_output = device_->softmax({gemm_output, std::nullopt, *bias_});
    auto output_cpu = device_->clone({*decoder_output, AllocationType::HOST});
    return Buffer2torchTensor(output_cpu);
}

LinearSoftmaxHandler::LinearSoftmaxHandler(const GptInitParameter& params): HandlerBase(params) {
    handler_impl_ = std::make_unique<LinearSoftmaxHandlerImpl>(params);
}

} // namespace rtp_llm
