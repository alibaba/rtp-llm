#include "rtp_llm/cpp/models/test/CudaGraphDecodePaddingOp.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphDecodePaddingOp::init(py::object py_instance) {
    cuda_graph_runner = createCudaGraphRunner(std::move(py_instance));
    // initializeResource();
    cuda_graph_runner->init_capture();
}

int CudaGraphDecodePaddingOp::get_current_real_graph_size() {
    return cuda_graph_runner->get_current_real_graph_bs();
}

CudaGraphRunnerPtr CudaGraphDecodePaddingOp::createCudaGraphRunner(py::object py_instance) {
    DeviceInitParams params;
    DeviceBase*      device                     = rtp_llm::DeviceFactory::getDefaultDevice();
    params.hw_kernel_config.enable_cuda_graph   = true;
    params.concurrency_config.concurrency_limit = 8;
    params.hw_kernel_config.enable_cuda_graph   = true;
    params.hw_kernel_config.disable_padding     = false;
    params.hidden_size                          = 896;
    params.max_seq_len                          = 64;
    params.tokens_per_block                     = 64;
    // int  layer_num                              = 24;
    // int  block_num                              = 26037;
    auto runner_ptr = std::make_shared<CudaGraphRunner>(params, std::move(py_instance), 663676, device, true);
    return runner_ptr;
}

PyModelInputs CudaGraphDecodePaddingOp::build_inputs(int64_t batch_size,
                                                     int64_t max_seq_len,
                                                     int64_t num_tokens_per_bs,
                                                     int64_t seq_size_per_block) {
    PyModelInputs inputs;
    inputs.attention_inputs.is_prefill = false;
    // int  hidden_size                   = 896;
    // int  layer_num                     = 24;
    // int  block_num                     = 26037;
    int  max_num_token = batch_size * num_tokens_per_bs;
    auto options2      = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    auto options3      = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token}, options3);
    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    inputs.attention_inputs.prefix_lengths = torch::empty(0);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::ones({int(batch_size)}, options2);
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(batch_size)}, options2);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device =
        torch::zeros({int(batch_size), ((max_seq_len + seq_size_per_block - 1) / seq_size_per_block)}, options3);
    inputs.attention_inputs.kv_cache_block_id_host =
        torch::zeros({int(batch_size), ((max_seq_len + seq_size_per_block - 1) / seq_size_per_block)}, options2);
    inputs.attention_inputs.is_prefill      = false;
    inputs.attention_inputs.dtype           = caffe2::TypeMeta::Make<c10::Half>();
    inputs.attention_inputs.kv_block_offset = 663676;
    std::cout << "kv_cache_block_id_device addr: "
              << reinterpret_cast<int64_t>(inputs.attention_inputs.kv_cache_block_id_device.data_ptr()) << std::endl;
    // max_bs = 8
    size_t    cu_len = batch_size + 1;
    BufferPtr cu_seqlens_buf =
        cuda_graph_runner->device_->allocateBuffer({DataType::TYPE_INT32, {cu_len}, AllocationType::HOST});
    inputs.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    return inputs;
}

PYBIND11_MODULE(libtest_cuda_graph_ops, m) {
    py::class_<cuda_graph::CudaGraphDecodePaddingOp>(m, "CudaGraphDecodePaddingOp")
        .def(py::init<>())
        .def("init", &CudaGraphDecodePaddingOp::init)
        .def("forward", &cuda_graph::CudaGraphDecodePaddingOp::forward)
        .def("get_current_real_graph_size", &cuda_graph::CudaGraphDecodePaddingOp::get_current_real_graph_size)
        .def("build_inputs", &cuda_graph::CudaGraphDecodePaddingOp::build_inputs);
}

}  // namespace cuda_graph
