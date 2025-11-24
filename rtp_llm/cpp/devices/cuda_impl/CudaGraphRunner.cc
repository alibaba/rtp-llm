#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"

using namespace torch_ext;
namespace rtp_llm {

GraphBase* CudaDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_prefill_cuda_graph_mode) {
    if (!graph_runner_) {
        graph_runner_ = new CudaGraphRunner(
            params, std::move(py_instance), kv_cache_block_offset, this, is_prefill_cuda_graph_mode);
    }
    return graph_runner_;
}

CudaGraphRunner::CudaGraphRunner(const DeviceInitParams& params,
                                 py::object              py_instance,
                                 int                     kv_cache_block_offset,
                                 DeviceBase*             device,
                                 bool                    is_prefill_cuda_graph_mode):
    GraphBase(params, std::move(py_instance), kv_cache_block_offset, device, is_prefill_cuda_graph_mode),
    capture_stream_(CudaGraphUtils::getStreamFromPool()) {}

CudaGraphRunner::~CudaGraphRunner() {
    RTP_LLM_LOG_INFO("Release CudaGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release CudaGraphRunner Successfully");
}

std::unique_ptr<void, std::function<void(void*)>> CudaGraphRunner::createStreamLife(void* capture_stream) {
    auto* stream_life = new CudaGraphStreamLife(*static_cast<at::cuda::CUDAStream*>(capture_stream), device_);
    return std::unique_ptr<void, std::function<void(void*)>>(
        stream_life, [](void* ptr) { delete static_cast<CudaGraphStreamLife*>(ptr); });
}

void CudaGraphRunner::setParamsPtr(int bs, const PyModelOutputs& outputs) {
    if (outputs.params_ptr->check_recycle()) {
        graph_instances_[bs].mem_hold_.params_ptr = ParamsBasePtr(outputs.params_ptr.get(), [&](ParamsBase* ptr) {});
    } else {
        graph_instances_[bs].mem_hold_.params_ptr = outputs.params_ptr;
    }
}

void CudaGraphRunner::capture() {
    RTP_LLM_LOG_INFO("Capture Start");
    int capture_range_size = capture_range_.size();
    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        inputs.input_ids        = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, bs * num_tokens_per_bs_);
        auto options_cpu_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        // input_lengths [batch_size, int32]
        inputs.attention_inputs.input_lengths = torch::full({int(bs)}, num_tokens_per_bs_, options_cpu_int32);
        // sequence_lengths [batch_size, int32] (decode only)
        // sequence_length should in pinned memory
        inputs.attention_inputs.sequence_lengths = torch::ones({int(bs)}, options_cpu_int32);
        inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
        // kv_cache_block_id_device [batch_size, block_num]
        inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
            {int(bs), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32);
        inputs.attention_inputs.kv_cache_block_id_host =
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, bs);
        // pinned memory
        inputs.attention_inputs.cu_seqlens =
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, bs + 1);
        inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        inputs.attention_inputs.dtype          = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
        inputs.attention_inputs.padding_offset =
            capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, bs * num_tokens_per_bs_);
        // Copy BertEmbeddingInputs from capture_mem_hold_
        inputs.bert_embedding_inputs = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
        graph_instances_[bs].mem_hold_ =
            CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, bs * num_tokens_per_bs_),
                              inputs,
                              kv_cache_block_offset_,
                              is_prefill_graph_mode_);
        captureOneBatchSize(bs);
        RTP_LLM_LOG_INFO("replay start check for %d", bs);
        replay(bs);
        cudaDeviceSynchronize();
        RTP_LLM_LOG_INFO("replay end check for %d", bs);
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture End");
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    auto options_cpu_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    inputs.attention_inputs.is_prefill = is_prefill_graph_mode_;
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32);
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32);
    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    inputs.attention_inputs.prefix_lengths         = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32);
    inputs.attention_inputs.kv_cache_block_id_host = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cpu_int32);
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32);
    inputs.attention_inputs.padding_offset = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype          = model_data_type_;
}

void CudaGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void CudaGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void CudaGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void CudaGraphRunner::setModelDataType(caffe2::TypeMeta data_type) {
    model_data_type_ = data_type;
}

void CudaGraphRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    // Initialize BertEmbeddingInputs for capture
    // combo_position_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_position_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // position_encoding: from weights
    inputs.bert_embedding_inputs.position_encoding = position_encoding_;

    // combo_tokens_type_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // token_type_embedding: from weights
    inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding_;

    // input_embedding_scalar: fixed value
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

py::object CudaGraphRunner::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

void CudaGraphRunner::initCapture() {
    if (enable_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        if (is_prefill_graph_mode_) {
            RTP_LLM_LOG_INFO("CUDA graph capture for embedding");
            // for embedding model which is prefill-only, the `input_ids` shape should be: [bs, max_seq_len_].
            // we will do mask for extra tokens in attention mechenism.
            num_tokens_per_bs_ = max_seq_len_;
        }
        // Capture
        at::cuda::CUDAGraph graph;
        capture_range_          = GraphUtils::getBatchSizesToCapture(concurrency_limit_);
        max_bs_                 = *(std::max_element(capture_range_.begin(), capture_range_.end()));
        max_num_token_          = max_bs_ * num_tokens_per_bs_;
        auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32);
        // input_lengths [batch_size, int32] (decode only)
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_prefill_graph_mode_);
        initKernelInternalMemory();
        // get real output data type
        auto py_outputs_obj     = py_forward_method_(capture_mem_hold_.py_model_inputs_);
        auto outputs            = py_outputs_obj.cast<PyModelOutputs>();
        auto options_cuda_float = torch::TensorOptions()
                                      .dtype(outputs.hidden_states.dtype().toScalarType())
                                      .device(torch::kCUDA)
                                      .requires_grad(false);
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float);
        capture_mem_hold_.setHiddenStates(output);
        capture();
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

}  // namespace rtp_llm
