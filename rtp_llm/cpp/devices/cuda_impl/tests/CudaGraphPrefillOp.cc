#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphPrefillOp.h"
namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphPrefillOp::init(py::object py_instance) {
    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance));
    // initializeResource();
    // model warm up
    auto inputs = buildInputs(2, 64, 64, 64, true);
    cuda_graph_runner_->normalForward(inputs);
    cuda_graph_runner_->setMaxPrefillCudaGraphLen(960);
    cuda_graph_runner_->initCapture();
}

int CudaGraphPrefillOp::getCurrentRealGraphSize() {
    return cuda_graph_runner_->getCurrentRealGraphBs();
}

CudaGraphRunnerPtr CudaGraphPrefillOp::createCudaGraphRunner(py::object py_instance) {
    DeviceInitParams params;
    params.hw_kernel_config.enable_cuda_graph            = true;
    params.fifo_scheduler_config.max_context_batch_size  = 128;
    params.hw_kernel_config.enable_cuda_graph_debug_mode = false;
    params.hidden_size                                   = 3584;
    params.max_seq_len                                   = 64;
    params.tokens_per_block                              = 64;
    params.hw_kernel_config.enable_cuda_graph_debug_mode = true;
    params.hw_kernel_config.prefill_capture_seq_lens     = {
        6,   10,  14,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75,  77,  80,  85,  90,  95,
        100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200,
        205, 210, 215, 220, 225, 230, 235, 240, 245, 248, 250, 252, 255, 256, 260, 265, 270, 275, 280, 285, 290,
        295, 300, 305, 310, 311, 315, 317, 320, 321, 325, 330, 335, 340, 345, 350, 355, 356, 360, 365, 370, 375,
        380, 385, 390, 395, 399, 400, 405, 410, 411, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470,
        475, 480, 485, 490, 495, 500, 512, 520, 540, 560, 576, 580, 600, 620, 629, 640, 660, 673, 680, 685, 697,
        700, 703, 720, 740, 760, 780, 793, 797, 800, 820, 837, 840, 844, 856, 860, 880, 889, 900, 920, 940, 960};
    // int  layer_num                              = 24;
    // int  block_num                              = 26037;
    at::cuda::CUDAStream capture_stream    = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    caffe2::TypeMeta     dtype             = torch::scalarTypeToTypeMeta(torch::kBFloat16);
    int                  num_tokens_per_bs = params.max_seq_len;  // prefill mode
    CudaGraphRunnerPtr   cuda_graph_runner_ptr =
        new CudaGraphRunner(params, std::move(py_instance), 0, capture_stream, dtype, num_tokens_per_bs, true);
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
    inputs.attention_inputs.prefix_lengths  = torch::zeros(int(batch_size), options2);
    inputs.attention_inputs.prefix_lengths  = inputs.attention_inputs.prefix_lengths.pin_memory();
    inputs.attention_inputs.is_prefill      = true;
    inputs.attention_inputs.dtype           = torch::kBFloat16;
    inputs.attention_inputs.is_s_padded     = use_max_padded_mode;
    RTP_LLM_LOG_INFO("kv_cache_block_id_device build success\n");
    // 计算 cu_seqlens
    size_t cu_len = batch_size + 1;

    // 使用 torch 创建 cu_seqlens tensor
    torch::Tensor cu_seqlens_tensor = torch::zeros({int(cu_len)}, options2).pin_memory();
    int32_t*      cu_seqlens_data   = cu_seqlens_tensor.data_ptr<int32_t>();

    // 手动计算 cu_seqlens - 使用真实有效的长度
    int32_t total_seq_len = 0;
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
    cu_seqlens_data[batch_size]                = total_seq_len;
    cu_seqlens_without_prefix_data[batch_size] = total_seq_len_without_prefix;
    RTP_LLM_LOG_INFO("cu_seqlens_data build success\n");

    inputs.attention_inputs.cu_seqlens              = cu_seqlens_tensor;
    inputs.attention_inputs.cu_kv_seqlens           = cu_seqlens_tensor.clone().pin_memory();
    inputs.attention_inputs.context_total_kv_length = total_seq_len;
    inputs.attention_inputs.total_tokens            = total_seq_len;
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
        .def("buildInputs", &cuda_graph::CudaGraphPrefillOp::buildInputs);
}

}  // namespace cuda_graph
