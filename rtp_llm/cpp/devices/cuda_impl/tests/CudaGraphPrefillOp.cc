#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphPrefillOp.h"
namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphPrefillOp::init(py::object py_instance) {
    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance));
    // initializeResource();
    // model warm up
    auto inputs = buildInputs(2, 64, 64, 64, true);
    cuda_graph_runner_->normalForward(inputs);
    setCufmhaPadded(true);
    cuda_graph_runner_->setQKVDim(4608);
    cuda_graph_runner_->setMaxPrefillCudaGraphLen(960);
    cuda_graph_runner_->initCapture();
}

int CudaGraphPrefillOp::getCurrentRealGraphSize() {
    return cuda_graph_runner_->getCurrentRealGraphBs();
}

void CudaGraphPrefillOp::setCufmhaPadded(bool is_s_padded) {
    DeviceBase* device = rtp_llm::DeviceFactory::getDefaultDevice();
    RTP_LLM_CHECK_WITH_INFO(device != nullptr, "device can't be nullptr");
    CudaDevice* cuda_device = dynamic_cast<CudaDevice*>(device);
    cuda_device->setIsPadded(is_s_padded);
}

CudaGraphRunnerPtr CudaGraphPrefillOp::createCudaGraphRunner(py::object py_instance) {
    DeviceInitParams params;
    DeviceBase*      device                              = rtp_llm::DeviceFactory::getDefaultDevice();
    params.hw_kernel_config.enable_cuda_graph            = true;
    params.fifo_scheduler_config.max_context_batch_size  = 128;
    params.hw_kernel_config.enable_cuda_graph_debug_mode = false;
    params.hidden_size                                   = 3584;
    params.max_seq_len                                   = 64;
    params.tokens_per_block                              = 64;
    params.hw_kernel_config.enable_cuda_graph_debug_mode = true;
    // int  layer_num                              = 24;
    // int  block_num                              = 26037;
    auto               runner_ptr            = device->getDeviceGraphRunner(params, std::move(py_instance), 0, true);
    CudaGraphRunnerPtr cuda_graph_runner_ptr = dynamic_cast<CudaGraphRunner*>(runner_ptr);
    cuda_graph_runner_ptr->setModelDataType(torch::scalarTypeToTypeMeta(torch::kBFloat16));
    return cuda_graph_runner_ptr;
}

PyModelInputs CudaGraphPrefillOp::buildInputs(int64_t batch_size,
                                              int64_t max_seq_len,
                                              int64_t num_tokens_per_bs,
                                              int64_t seq_size_per_block,
                                              bool    use_max_padded_mode) {
    PyModelInputs inputs;
    inputs.attention_inputs.is_prefill = true;
    // int  hidden_size                   = 896;
    // int  layer_num                     = 24;
    // int  block_num                     = 26037;

    // 创建新的序列长度数组：10, 20, 30, ..., 10 * batch_size
    std::vector<int32_t> input_lengths_data(batch_size);
    int32_t              total_tokens = 0;
    for (int64_t i = 0; i < batch_size; i++) {
        if (use_max_padded_mode) {
            // 当使用 max_padded_mode 时，所有序列都 padding 到 max_seq_len
            input_lengths_data[i] = max_seq_len;
        } else {
            // 否则使用递增长度：10, 20, 30, ..., 10 * batch_size
            input_lengths_data[i] = std::min(max_seq_len, 10 * (i + 1));
        }
        total_tokens += input_lengths_data[i];
    }

    auto options2 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    auto options3 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    RTP_LLM_LOG_INFO(
        "buildInputs check, batch_size: %d, max_seq_len: %d, num_tokens_per_bs: %d, seq_size_per_block: %d, total_tokens: %d, use_max_padded_mode: %s",
        batch_size,
        max_seq_len,
        num_tokens_per_bs,
        seq_size_per_block,
        total_tokens,
        use_max_padded_mode ? "true" : "false");

    // input_ids [tokens_nums] = [total_tokens]
    if (use_max_padded_mode) {
        // 当使用 max_padded_mode 时，input_ids 需要按照 padded 的方式生成
        // 每个 batch 都有 max_seq_len 个 token，但只有前面的部分是有意义的
        inputs.input_ids    = torch::ones({total_tokens}, options3);
        int32_t current_pos = 0;
        int32_t current_id  = 1;
        for (int64_t i = 0; i < batch_size; i++) {
            // 每个 batch 的前 10*(i+1) 个位置填充有意义的数据
            int32_t actual_length = std::min(max_seq_len, 10 * (i + 1));
            for (int32_t j = 0; j < actual_length; j++) {
                inputs.input_ids[current_pos + j] = (current_id++) % 10 + 1;
            }
            // 剩余位置保持为 0（padding）
            current_pos += max_seq_len;
        }
    } else {
        // 否则使用连续递增的方式
        inputs.input_ids = torch::arange(1, total_tokens + 1, options3);
        for (int i = 0; i < inputs.input_ids.size(0); i++) {
            inputs.input_ids[i] = (i + 1) % 10 + 1;
        }
    }
    RTP_LLM_LOG_INFO("build input_ids shapes: [%d]\n", inputs.input_ids.size(0));

    // input_lengths [batch_size, int32] - 使用新的序列长度
    inputs.attention_inputs.input_lengths =
        torch::from_blob(input_lengths_data.data(),
                         {int(batch_size)},
                         torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
            .clone();
    RTP_LLM_LOG_INFO("input_lengths build success\n");
    // sequence_lengths [batch_size, int32] - 与 input_lengths 相同
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.input_lengths.clone();
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    RTP_LLM_LOG_INFO("sequence_lengths build success\n");
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device =
        torch::zeros({int(batch_size), ((max_seq_len + seq_size_per_block - 1) / seq_size_per_block)}, options3);
    inputs.attention_inputs.kv_cache_block_id_host =
        torch::zeros({int(batch_size), ((max_seq_len + seq_size_per_block - 1) / seq_size_per_block)}, options2);
    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    inputs.attention_inputs.prefix_lengths = torch::zeros(int(batch_size), options2);
    inputs.attention_inputs.prefix_lengths.pin_memory();
    inputs.attention_inputs.is_prefill      = true;
    inputs.attention_inputs.dtype           = torch::kBFloat16;
    inputs.attention_inputs.kv_block_offset = 0;
    RTP_LLM_LOG_INFO("kv_cache_block_id_device build success\n");
    // 计算 cu_seqlens
    size_t    cu_len = batch_size + 1;
    BufferPtr cu_seqlens_buf =
        cuda_graph_runner_->device_->allocateBuffer({DataType::TYPE_INT32, {cu_len}, AllocationType::HOST});

    // 手动计算 cu_seqlens - 使用真实有效的长度
    int32_t* cu_seqlens_data = cu_seqlens_buf->data<int32_t>();
    int32_t  total_seq_len   = 0;
    for (int64_t i = 0; i < batch_size; i++) {
        cu_seqlens_data[i] = total_seq_len;
        if (use_max_padded_mode) {
            // 当使用 max_padded_mode 时，cu_seqlens 记录的是实际的有效长度
            // 即 10*(i+1)，而不是 padded 后的 max_seq_len
            int32_t actual_length = std::min(max_seq_len, 10 * (i + 1));
            total_seq_len += actual_length;
        } else {
            // 否则使用实际的序列长度
            total_seq_len += input_lengths_data[i];
        }
    }
    cu_seqlens_data[batch_size] = total_seq_len;
    RTP_LLM_LOG_INFO("cu_seqlens_data build success\n");
    inputs.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    if (!use_max_padded_mode) {
        calculatePaddingOffset(inputs.attention_inputs);
    }
    RTP_LLM_LOG_INFO("padding_offset build success\n");
    return inputs;
}

PYBIND11_MODULE(libtest_cuda_graph_prefill_ops, m) {
    py::class_<cuda_graph::CudaGraphPrefillOp>(m, "CudaGraphPrefillOp")
        .def(py::init<>())
        .def("init", &CudaGraphPrefillOp::init)
        .def("forward", &cuda_graph::CudaGraphPrefillOp::forward)
        .def("getCurrentRealGraphSize", &cuda_graph::CudaGraphPrefillOp::getCurrentRealGraphSize)
        .def("buildInputs", &cuda_graph::CudaGraphPrefillOp::buildInputs)
        .def("setCufmhaPadded", &cuda_graph::CudaGraphPrefillOp::setCufmhaPadded);
}

}  // namespace cuda_graph
