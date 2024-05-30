#include "maga_transformer/cpp/deprecated/FakeSampler.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

namespace ft = fastertransformer;
using namespace fastertransformer;

namespace rtp_llm {

FakeSampler::FakeSampler(const ft::GptInitParameter& gpt_init_parameter) {
    device_               = dynamic_cast<CudaDevice*>(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda));
    allocator_            = device_->getAllocator();
    stream_               = device_->stream();
    vocab_size_           = gpt_init_parameter.vocab_size_;
    dynamic_decode_layer_ = new ft::DynamicDecodeLayer<half>(gpt_init_parameter.vocab_size_,
                                                             gpt_init_parameter.vocab_size_,
                                                             0,
                                                             device_->stream(),
                                                             device_->cublasMMWrapperPtr(),
                                                             allocator_,
                                                             false,
                                                             &prop_);
}

void FakeSampler::allocateBuffer(size_t total_batch_size) {
    top_k_  = (int*)allocator_->reMalloc(top_k_, sizeof(int));
    end_id_ = (int*)allocator_->reMalloc(end_id_, sizeof(int) * total_batch_size);
}

void FakeSampler::freeBuffer() {
    allocator_->free((void**)top_k_);
    allocator_->free((void**)end_id_);
}

SamplerOutput FakeSampler::forward(SamplerInputs& inputs) {
    allocateBuffer(inputs.batch_size);
    int              step             = inputs.step;
    int              max_input_length = step;
    int              _top_k           = 1;
    std::vector<int> _end_id;
    ft::Tensor       top_k(ft::MEMORY_CPU, ft::DataType::TYPE_UINT32, {1}, &_top_k);
    ft::Tensor       end_id(ft::MEMORY_CPU, ft::DataType::TYPE_INT32, {inputs.batch_size}, _end_id.data());
    ft::Tensor       output_token_ids(
        ft::MEMORY_GPU, ft::DataType::TYPE_INT32, inputs.token_ids->shape(), inputs.token_ids->data());

    ft::Tensor logits(
        ft::MEMORY_GPU, ft::DataType::TYPE_FP16, {inputs.batch_size, 1, vocab_size_}, inputs.logits->data());

    // cudaMemcpyAsync(
    //     top_k.getPtr<int>(), inputs.top_k->data(), sizeof(int), cudaMemcpyHostToDevice, stream_);

    ft::TensorMap runtime_args({{"runtime_top_k", top_k}});
    dynamic_decode_layer_->setup(inputs.batch_size, 1, &runtime_args);

    int           ite              = 0;
    int           local_batch_size = inputs.batch_size;
    ft::TensorMap input_tensors{
        {"logits", logits},
        {"step", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &step)},
        {"max_input_length", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &max_input_length)},
        {"ite", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_UINT32, {1}, &ite)},
        {"local_batch_size", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &local_batch_size)},
        {"end_id", end_id}};
    ft::TensorMap output_tensors{{"output_ids", output_token_ids}};
    dynamic_decode_layer_->forward(&output_tensors, &input_tensors);
    SamplerOutput output;
    output.token_ids = std::move(inputs.token_ids);
    return output;
}

}  // namespace rtp_llm
