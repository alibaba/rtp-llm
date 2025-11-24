#include "rtp_llm/cpp/devices/GraphBase.h"

namespace rtp_llm {

// Constructor implementations
GraphBase::GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}

GraphBase::GraphBase(const DeviceInitParams& params,
                     py::object              py_instance,
                     int                     kv_cache_block_offset,
                     DeviceBase*             device,
                     bool                    is_prefill_mode):
    enable_graph_(params.hw_kernel_config.enable_cuda_graph),
    is_prefill_graph_mode_(is_prefill_mode),
    enable_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
    concurrency_limit_(params.concurrency_config.concurrency_limit),
    hidden_size_(params.hidden_size),
    max_seq_len_(params.max_seq_len),
    seq_size_per_block_(params.tokens_per_block),
    kv_cache_block_offset_(kv_cache_block_offset),
    device_(device),
    py_instance_(std::move(py_instance)) {

    py::gil_scoped_acquire gil;
    if (!py_instance_ || py_instance_.is_none()) {
        throw std::runtime_error("GraphRunner constructor: Python instance is null or none.");
    }
    py_forward_method_     = py_instance_.attr("forward");
    py_fill_params_method_ = py_instance_.attr("fill_params");

    RTP_LLM_LOG_INFO(
        "Initialize CudaGraphRunner with parameters below: enable_graph=%d, concurrency_limit=%d, debug_mode=%d, "
        "hidden_size=%d, max_seq_len=%d, seq_size_per_block=%d, kv_cache_offset=%d, prefill_mode=%d",
        enable_graph_,
        concurrency_limit_,
        enable_graph_debug_mode_,
        hidden_size_,
        max_seq_len_,
        seq_size_per_block_,
        kv_cache_block_offset_,
        is_prefill_graph_mode_);
}

// Device-specific virtual functions
void GraphBase::initCapture() {}

void GraphBase::setPositionEncoding(torch::Tensor position_encoding) {}

void GraphBase::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {}

void GraphBase::setInputEmbeddingScalar(float input_embedding_scalar) {}

void GraphBase::setModelDataType(caffe2::TypeMeta data_type) {}

void GraphBase::replay(int bs) {}

void GraphBase::deviceSpecificSync() {}

std::unique_ptr<void, std::function<void(void*)>> GraphBase::createStreamLife(void* capture_stream) {
    // 默认实现：返回空的智能指针
    return std::unique_ptr<void, std::function<void(void*)>>(nullptr, [](void*) {});
}

void* GraphBase::getDeviceStream() {
    // 默认实现：返回空指针
    return nullptr;
}

void GraphBase::setParamsPtr(int bs, const PyModelOutputs& outputs) {
    // 默认实现：空操作
}

// Main interface methods
PyModelOutputs GraphBase::forward(PyModelInputs& inputs) {
    if (canRun(inputs)) {
        prepareInputs(inputs);
        graph_instances_[current_real_graph_bs_].graph_.replay();

        PyModelOutputs outputs;
        if (is_prefill_graph_mode_) {
            outputs.hidden_states      = capture_mem_hold_.decoder_layer_hidden_states_;
            auto    input_lengths      = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
            int32_t total_valid_tokens = 0;
            for (int i = 0; i < current_batch_size_; i++) {
                total_valid_tokens += input_lengths[i];
            }
            extractValidHiddenStates(outputs, inputs, total_valid_tokens);
        } else {
            outputs.hidden_states = capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, seq_len_sum_);
        }
        return outputs;
    } else {
        auto py_outputs_obj = normalForward(inputs);
        return py_outputs_obj.cast<PyModelOutputs>();
    }
}

bool GraphBase::canRun(PyModelInputs& inputs) {
    return enable_graph_ && (!inputs.attention_inputs.is_prefill || is_prefill_graph_mode_)
           && tryGetRealGraphBatchSize(inputs);
}

int GraphBase::getCurrentRealGraphBs() {
    return current_real_graph_bs_;
}

bool GraphBase::tryGetRealGraphBatchSize(PyModelInputs& inputs) {
    current_batch_size_    = inputs.attention_inputs.input_lengths.size(0);
    bool is_bs_supported   = (current_batch_size_ <= max_bs_);
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_batch_size_);
    current_real_graph_bs_ = *it;
    seq_len_sum_           = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    return is_bs_supported;
}

void GraphBase::extractValidHiddenStates(PyModelOutputs&      outputs,
                                         const PyModelInputs& inputs,
                                         int32_t              total_valid_tokens) {
    GraphUtils::extractValidHiddenStates(outputs,
                                         inputs,
                                         total_valid_tokens,
                                         capture_mem_hold_.decoder_layer_hidden_states_,
                                         current_batch_size_,
                                         num_tokens_per_bs_);
}

py::object GraphBase::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

void GraphBase::initKernelInternalMemory() {
    BufferPtr cu_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
                            "capture_mem_hold_ sequence_lengths is not pinned memory");
}

void GraphBase::captureOneBatchSize(int bs) {
    auto inputs = graph_instances_[bs].mem_hold_.py_model_inputs_;

    RTP_LLM_LOG_INFO("WarmUp for batch size %d start.", bs);
    py_forward_method_(inputs);
    py_forward_method_(inputs);
    RTP_LLM_LOG_INFO("WarmUp for batch size %d successfully.", bs);

    {
        auto  stream_life = createStreamLife(getDeviceStream());
        auto& graph       = graph_instances_[bs].graph_;

        if (enable_graph_debug_mode_) {
            graph.enable_debug_mode();
        }

        RTP_LLM_LOG_INFO("Capture for batch size %d begin.", bs);
        graph.capture_begin();

        auto py_outputs_obj = py_forward_method_(inputs);
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[bs].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);

        graph.capture_end();
        setParamsPtr(bs, outputs);

        if (enable_graph_debug_mode_) {
            graph.debug_dump("cuda_graph_visualization.dot");
        }
    }
}

void GraphBase::prepareInputs(PyModelInputs& inputs) {
    auto& py_model_inputs_ = graph_instances_[current_real_graph_bs_].mem_hold_.py_model_inputs_;
    py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
        inputs.attention_inputs.input_lengths;
    py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
        inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);

    if (!is_prefill_graph_mode_) {
        py_model_inputs_.input_ids.fill_(0);
        py_model_inputs_.input_ids.slice(0, 0, inputs.input_ids.size(0)) = inputs.input_ids;
        py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.sequence_lengths;
        GraphUtils::copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                                          py_model_inputs_.attention_inputs.kv_cache_block_id_device);
        graph_instances_[current_real_graph_bs_].mem_hold_.params_ptr->fillParams(
            inputs.attention_inputs.sequence_lengths,
            inputs.attention_inputs.input_lengths,
            inputs.attention_inputs.kv_cache_block_id_host,
            current_batch_size_,
            seq_size_per_block_);
    } else {
        preparePrefillInputs(inputs, py_model_inputs_);
    }
}

void GraphBase::capture() {}

void GraphBase::preparePrefillInputs(PyModelInputs& inputs, PyModelInputs& py_model_inputs_) {
    auto input_lengths_ptr  = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
    auto padding_offset_ptr = py_model_inputs_.attention_inputs.padding_offset.data_ptr<int32_t>();

    int32_t cum_offset = 0, index = 0;
    for (int32_t i = 0; i < current_batch_size_; i++) {
        index           = i * num_tokens_per_bs_;
        int32_t seq_len = input_lengths_ptr[i];
        for (int32_t j = 0; j < seq_len; j++) {
            padding_offset_ptr[index++] = cum_offset;
        }
        cum_offset += num_tokens_per_bs_ - seq_len;
    }

    py_model_inputs_.input_ids.fill_(0);
    auto lengths   = inputs.attention_inputs.input_lengths.data_ptr<int>();
    int  start_idx = 0;

    for (int i = 0; i < current_batch_size_; i++) {
        int dst_start = i * num_tokens_per_bs_;
        int dst_end   = dst_start + lengths[i];
        int src_start = start_idx;
        int src_end   = src_start + lengths[i];

        py_model_inputs_.input_ids.slice(0, dst_start, dst_end) = inputs.input_ids.slice(0, src_start, src_end);

        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            py_model_inputs_.bert_embedding_inputs.combo_position_ids.slice(0, dst_start, dst_end) =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, src_start, src_end);
            py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids.slice(0, dst_start, dst_end) =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, src_start, src_end);
        }
        start_idx += lengths[i];
    }
}

}  // namespace rtp_llm
