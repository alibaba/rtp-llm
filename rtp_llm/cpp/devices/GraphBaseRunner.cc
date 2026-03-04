#include "rtp_llm/cpp/devices/GraphBaseRunner.h"
#include "rtp_llm/cpp/devices/GraphStreamLife.h"

#include <algorithm>
#include <cstring>

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/rocm/hip_capture_check.h"
#else
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {

GraphBaseRunner::GraphBaseRunner(const DeviceInitParams& params,
                                 py::object              py_instance,
                                 c10::ScalarType         model_data_type,
                                 int                     num_tokens_per_bs,
                                 bool                    is_prefill_graph_mode):
#if USING_ROCM
    nccl_capture_ctx_(std::make_shared<HipGraphNcclCaptureContext>()),
#endif
    py_instance_(std::move(py_instance)),
    enable_graph_(params.hw_kernel_config.enable_cuda_graph),
    is_prefill_graph_mode_(is_prefill_graph_mode),
    enable_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
    num_tokens_per_bs_(num_tokens_per_bs),
    max_seq_len_(params.max_seq_len),
    seq_size_per_block_(params.tokens_per_block),
    hidden_size_(params.hidden_size),
    prefill_capture_seq_lens_(params.hw_kernel_config.prefill_capture_seq_lens),
    decode_capture_batch_sizes_(params.hw_kernel_config.decode_capture_batch_sizes),
    model_data_type_(model_data_type),
    kv_cache_layer_to_group_(params.kv_cache_layer_to_group),
    kv_cache_group_num_(params.kv_cache_group_num),
    options_device_int32_(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false)),
    options_cpu_int32_(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false)),
    options_device_float_(torch::TensorOptions().dtype(model_data_type).device(torch::kCUDA).requires_grad(false)),
#if USING_ROCM
    forward_event_(c10::DeviceType::HIP)
#else
    forward_event_(c10::DeviceType::CUDA)
#endif
{
    // All members are now fully constructed – safe to capture [this] in lambdas.
    buildDeviceOps();

    py::gil_scoped_acquire gil;
    RTP_LLM_CHECK_WITH_INFO(py_instance_ && !py_instance_.is_none(), "GraphBaseRunner constructor py_instance is null");
    py_attn_pyobj_method_ = py_instance_.attr("prepare_fmha_impl");
    py_forward_method_    = py_instance_.attr("forward");

    if (is_prefill_graph_mode_) {
        max_bs_ = params.runtime_config.fifo_scheduler_config.max_context_batch_size;
    } else {
        max_bs_ = params.concurrency_config.concurrency_limit;
    }
}

GraphBaseRunner::~GraphBaseRunner() {}

// ─── buildDeviceOps ──────────────────────────────────────────────────────────
// Called from the constructor body after all members are constructed.
// Uses [this] captures – safe because device_ops_ and nccl_capture_ctx_ exist.
void GraphBaseRunner::buildDeviceOps() {
#if USING_ROCM
    device_ops_.event_device_type = c10::DeviceType::HIP;
    device_ops_.debug_file_prefix = "hip_graph_tokens";
#else
    device_ops_.event_device_type = c10::DeviceType::CUDA;
    device_ops_.debug_file_prefix = "cuda_graph_tokens";
#endif

    device_ops_.memcpy_async = [](const torch::Tensor& src, torch::Tensor& dst, size_t size) {
        if (!src.defined() || src.numel() <= 0) {
            return;
        }
#if USING_ROCM
        hipStream_t stream = at::hip::getCurrentHIPStream().stream();
        if (src.is_cuda() && dst.is_cuda()) {
            ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToDevice, stream));
        } else if (!src.is_cuda() && !dst.is_cuda()) {
            std::memcpy(dst.data_ptr(), src.data_ptr(), size);
        } else if (src.is_cuda() && !dst.is_cuda()) {
            ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToHost, stream));
        } else {
            ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyHostToDevice, stream));
        }
#else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        if (src.is_cuda() && dst.is_cuda()) {
            check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToDevice, stream));
        } else if (!src.is_cuda() && !dst.is_cuda()) {
            std::memcpy(dst.data_ptr(), src.data_ptr(), size);
        } else if (src.is_cuda() && !dst.is_cuda()) {
            check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToHost, stream));
        } else {
            check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyHostToDevice, stream));
        }
#endif
    };

#if USING_ROCM
    device_ops_.device_synchronize         = []() { ROCM_CHECK(hipDeviceSynchronize()); };
    device_ops_.record_forward_event       = [](torch::Event& event) { event.record(at::hip::getCurrentHIPStream()); };
    device_ops_.synchronize_forward_stream = []() {
        auto stream = at::hip::getCurrentHIPStream();
        ROCM_CHECK(hipStreamSynchronize(stream.stream()));
    };
    auto capture_stream             = at::hip::getStreamFromPool(true);
    device_ops_.with_capture_stream = [capture_stream](const std::function<void()>& fn) {
        GraphStreamLife stream_life(capture_stream);
        fn();
    };
#else
    device_ops_.device_synchronize   = []() { check_cuda_value(cudaDeviceSynchronize()); };
    device_ops_.record_forward_event = [](torch::Event& event) { event.record(at::cuda::getCurrentCUDAStream()); };
    device_ops_.synchronize_forward_stream = []() {};
    auto capture_stream                    = at::cuda::getStreamFromPool(true);
    device_ops_.with_capture_stream        = [capture_stream](const std::function<void()>& fn) {
        GraphStreamLife stream_life(capture_stream);
        fn();
    };
#endif

#if USING_ROCM
    device_ops_.should_skip_decode_capture = [](py::object py_instance, bool is_prefill_mode) {
        if (is_prefill_mode) {
            return false;
        }
        py::gil_scoped_acquire gil;
        bool                   has_kv_cache = true;
        if (py::hasattr(py_instance, "kv_cache")) {
            has_kv_cache = !py_instance.attr("kv_cache").is_none();
        }
        if (!has_kv_cache) {
            RTP_LLM_LOG_WARNING("HIP graph capture is enabled but kv_cache is not available. "
                                "Skipping decode graph capture for this instance.");
        }
        return !has_kv_cache;
    };
    device_ops_.before_capture_stream = [](py::object py_instance, int key, const char* key_type) {
        (void)py_instance;
        py::gil_scoped_acquire gil;
        try {
            py::module_ torch_dist = py::module_::import("torch.distributed");
            if (torch_dist.attr("is_initialized")().cast<bool>()) {
                RTP_LLM_LOG_INFO("Executing torch.distributed.barrier() before graph capture for %s %d", key_type, key);
                torch_dist.attr("barrier")();
                RTP_LLM_LOG_INFO("torch.distributed.barrier() completed for %s %d", key_type, key);
            }
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("Failed to execute torch.distributed.barrier(): %s", e.what());
        }
    };
#else
    device_ops_.should_skip_decode_capture = [](py::object, bool) { return false; };
    device_ops_.before_capture_stream      = [](py::object, int, const char*) {};
#endif

    // enter_capture / exit_capture: set the graph-capture flag and, for ROCm,
    // also notify collective_torch so NCCL ops are captured properly.
#if USING_ROCM
    device_ops_.enter_capture = [this](py::object) {
        rocm::CaptureCheck::in_hip_graph_capture = true;
        if (nccl_capture_ctx_ && nccl_capture_ctx_->comm_handle != 0) {
            try {
                py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
                collective_torch.attr("enter_graph_capture_mode")(
                    nccl_capture_ctx_->comm_handle, nccl_capture_ctx_->world_size, nccl_capture_ctx_->rank);
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("Failed to enter graph capture mode: %s", e.what());
            }
        }
    };
    device_ops_.exit_capture = [this](py::object) {
        rocm::CaptureCheck::in_hip_graph_capture = false;
        if (nccl_capture_ctx_ && nccl_capture_ctx_->comm_handle != 0) {
            try {
                py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
                collective_torch.attr("exit_graph_capture_mode")();
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("Failed to exit graph capture mode: %s", e.what());
            }
        }
    };
#else
    device_ops_.enter_capture = [](py::object) { CaptureCheck::in_cuda_graph_capture = true; };
    device_ops_.exit_capture  = [](py::object) { CaptureCheck::in_cuda_graph_capture = false; };
#endif

    device_ops_.kv_block_cols = [](int max_seq_len, int seq_size_per_block) {
#if USING_ROCM
        return (max_seq_len + seq_size_per_block - 1) / seq_size_per_block;
#else
        return (max_seq_len + seq_size_per_block - 1) / seq_size_per_block + 1;
#endif
    };

    device_ops_.sequence_lengths_plus_one_tensor = [](int max_bs, const at::TensorOptions& opts) {
#if USING_ROCM
        return torch::full({max_bs}, 2, opts);
#else
        return torch::zeros({max_bs}, opts);
#endif
    };
}

// ─── NCCL (ROCm only) ────────────────────────────────────────────────────────
#if USING_ROCM
void GraphBaseRunner::setNcclCommHandle(void* nccl_comm, size_t rank, size_t world_size) {
    nccl_capture_ctx_->comm_handle = reinterpret_cast<int64_t>(nccl_comm);
    nccl_capture_ctx_->rank        = static_cast<int>(rank);
    nccl_capture_ctx_->world_size  = static_cast<int>(world_size);
}
#endif

// ─── Helpers ─────────────────────────────────────────────────────────────────
py::object GraphBaseRunner::normalForward(PyModelInputs& inputs) {
    auto attn_pyobj = py_attn_pyobj_method_(inputs, false);
    return py_forward_method_(inputs, attn_pyobj);
}

void GraphBaseRunner::copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
    RTP_LLM_CHECK_WITH_INFO(source_tensor.dim() == target_tensor.dim(), "source and target dim mismatch");
    for (int i = 0; i < source_tensor.dim(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(source_tensor.size(i) <= target_tensor.size(i),
                                "target dim[%d]=%d smaller than source dim=%d",
                                i,
                                (int)target_tensor.size(i),
                                (int)source_tensor.size(i));
    }
    torch::Tensor target_slice = target_tensor;
    for (int i = 0; i < source_tensor.dim(); ++i) {
        target_slice = target_slice.slice(i, 0, source_tensor.size(i));
    }
    target_slice.copy_(source_tensor);
}

void GraphBaseRunner::prepareInputs(PyModelInputs& inputs) {
    forward_event_.synchronize();
    auto& py_model_inputs_ = is_prefill_graph_mode_ ?
                                 graph_instances_[state_.current_real_graph_seq_len].mem_hold_.py_model_inputs_ :
                                 graph_instances_[state_.current_real_graph_bs].mem_hold_.py_model_inputs_;

    if (!is_prefill_graph_mode_) {
        // clear kv_cache_block_id_device, otherwise it will cause the cache block pollution
        py_model_inputs_.attention_inputs.kv_cache_block_id_device.fill_(0);

        device_ops_.memcpy_async(inputs.attention_inputs.prefix_lengths,
                                 py_model_inputs_.attention_inputs.prefix_lengths,
                                 state_.current_batch_size * sizeof(int));

        py_model_inputs_.input_ids.fill_(0);
        device_ops_.memcpy_async(inputs.input_ids, py_model_inputs_.input_ids, inputs.input_ids.size(0) * sizeof(int));
        device_ops_.memcpy_async(inputs.input_hiddens,
                                 py_model_inputs_.input_hiddens,
                                 inputs.input_hiddens.numel() * inputs.input_hiddens.element_size());
        device_ops_.memcpy_async(inputs.attention_inputs.sequence_lengths,
                                 py_model_inputs_.attention_inputs.sequence_lengths,
                                 state_.current_batch_size * sizeof(int));

        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);

        device_ops_.memcpy_async(inputs.attention_inputs.sequence_lengths_plus_1_d,
                                 py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                                 state_.current_batch_size * sizeof(int));
        device_ops_.memcpy_async(inputs.attention_inputs.decode_cu_seqlens_d,
                                 py_model_inputs_.attention_inputs.decode_cu_seqlens_d,
                                 (state_.current_batch_size + 1) * sizeof(int));
        auto attn_pyobj = graph_instances_[state_.current_real_graph_bs].mem_hold_.attn_pyobj_;
        attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs_.attention_inputs);
    } else {
        // clear kv_cache_block_id_device, otherwise it will cause the cache block pollution
        py_model_inputs_.attention_inputs.kv_cache_block_id_device.fill_(0);

        device_ops_.memcpy_async(inputs.input_ids, py_model_inputs_.input_ids, state_.current_seq_len * sizeof(int));

        device_ops_.memcpy_async(inputs.attention_inputs.padding_offset,
                                 py_model_inputs_.attention_inputs.padding_offset,
                                 state_.current_seq_len * sizeof(int));
        if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
            (*(py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params->cuda_graph_prefill_batch_size
                   .data_ptr<int>())) = state_.current_batch_size;
        }
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            device_ops_.memcpy_async(inputs.bert_embedding_inputs.combo_position_ids,
                                     py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                                     state_.current_seq_len * sizeof(int));
            device_ops_.memcpy_async(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                                     py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                                     state_.current_seq_len * sizeof(int));
        }
    }

    // Common async copies for both decode and prefill paths
    device_ops_.memcpy_async(inputs.attention_inputs.input_lengths,
                             py_model_inputs_.attention_inputs.input_lengths,
                             state_.current_batch_size * sizeof(int));

    device_ops_.memcpy_async(inputs.attention_inputs.cu_seqlens,
                             py_model_inputs_.attention_inputs.cu_seqlens,
                             (state_.current_batch_size + 1) * sizeof(int));

    device_ops_.memcpy_async(inputs.attention_inputs.cu_kv_seqlens,
                             py_model_inputs_.attention_inputs.cu_kv_seqlens,
                             (state_.current_batch_size + 1) * sizeof(int));

    // Hybrid cache: update per-group block tables (including group 0).
    if (!inputs.attention_inputs.kv_cache_block_id_device_by_group.empty()
        && !inputs.attention_inputs.kv_cache_block_id_host_by_group.empty()
        && !py_model_inputs_.attention_inputs.kv_cache_block_id_device_by_group.empty()
        && !py_model_inputs_.attention_inputs.kv_cache_block_id_host_by_group.empty()) {
        RTP_LLM_CHECK_WITH_INFO(inputs.attention_inputs.kv_cache_block_id_device_by_group.size()
                                    == py_model_inputs_.attention_inputs.kv_cache_block_id_device_by_group.size(),
                                "kv_cache_block_id_device_by_group size mismatch");
        const size_t group = inputs.attention_inputs.kv_cache_block_id_device_by_group.size();
        RTP_LLM_CHECK_WITH_INFO(inputs.attention_inputs.kv_cache_block_id_host_by_group.size() == group
                                    && py_model_inputs_.attention_inputs.kv_cache_block_id_host_by_group.size()
                                           == group,
                                "kv_cache_block_id_host_by_group size mismatch");
        for (size_t g = 0; g < group; ++g) {
            copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device_by_group[g],
                                  py_model_inputs_.attention_inputs.kv_cache_block_id_device_by_group[g]);
            copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_host_by_group[g],
                                  py_model_inputs_.attention_inputs.kv_cache_block_id_host_by_group[g]);
        }
    }

    if (inputs.attention_inputs.kv_cache_layer_to_group.defined()
        && inputs.attention_inputs.kv_cache_layer_to_group.numel() > 0) {
        device_ops_.memcpy_async(inputs.attention_inputs.kv_cache_layer_to_group,
                                 py_model_inputs_.attention_inputs.kv_cache_layer_to_group,
                                 inputs.attention_inputs.kv_cache_layer_to_group.numel() * sizeof(int32_t));
    }
}

PyModelOutputs GraphBaseRunner::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    prepareInputs(inputs);
    if (is_prefill_graph_mode_) {
        replayPrefill(state_.current_real_graph_seq_len);
        outputs.hidden_states =
            graph_instances_[state_.current_real_graph_seq_len].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state_.current_seq_len);
    } else {
        replayDecode(state_.current_real_graph_bs);
        outputs.hidden_states =
            graph_instances_[state_.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state_.seq_len_sum);
    }
    device_ops_.record_forward_event(forward_event_);
    device_ops_.synchronize_forward_stream();
    return outputs;
}

void GraphBaseRunner::tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs) {
    state_.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_seq_len);
    state_.current_real_graph_seq_len = *it;
    state_.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
}

void GraphBaseRunner::tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs) {
    int graph_bs              = inputs.attention_inputs.input_lengths.size(0);
    state_.current_batch_size = graph_bs;
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_batch_size);
    state_.current_real_graph_bs = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", state_.current_real_graph_bs);
    state_.seq_len_sum =
        inputs.attention_inputs.is_prefill ? inputs.attention_inputs.input_lengths.sum(0).item<int>() : graph_bs;
}

bool GraphBaseRunner::canRun(PyModelInputs& inputs) {
    if (!is_prefill_graph_mode_ && inputs.attention_inputs.prefix_lengths.defined()
        && inputs.attention_inputs.prefix_lengths.numel() > 0
        && inputs.attention_inputs.prefix_lengths.data_ptr<int>()[0] > 0) {
        auto input_lengths_cpu = inputs.attention_inputs.input_lengths;
        bool all_same          = true;
        for (int i = 0; i < input_lengths_cpu.size(0); i++) {
            if (input_lengths_cpu[i].item<int>() != num_tokens_per_bs_) {
                all_same = false;
                break;
            }
        }
        if (all_same && num_tokens_per_bs_ > 1) {
            tryGetRealGraphDecodeBatchSize(inputs);
            return true;
        }
    }
    if (!enable_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_graph_mode_)) {
        return false;
    }

    // Hybrid kv cache group check
    if (!inputs.attention_inputs.kv_cache_block_id_device_by_group.empty()) {
        const size_t group = inputs.attention_inputs.kv_cache_block_id_device_by_group.size();
        if (kv_cache_group_num_ <= 0) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache detected but kv_cache_group_num_ is not set, fallback to normal run.");
            return false;
        }
        if (group != static_cast<size_t>(kv_cache_group_num_)) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache group size mismatch: inputs=%zu, captured=%d, fallback to normal run.",
                                group,
                                kv_cache_group_num_);
            return false;
        }
    }

    if (is_prefill_graph_mode_) {
        tryGetRealGraphPrefillSeqLen(inputs);
        if (state_.current_seq_len > max_prefill_graph_len_) {
            return false;
        }
    } else {
        tryGetRealGraphDecodeBatchSize(inputs);
    }
    return true;
}

void GraphBaseRunner::initKernelInternalMemory() {
    auto input_lengths  = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths;
    auto prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
    if (!input_lengths.defined() || input_lengths.numel() == 0 || !prefix_lengths.defined()
        || prefix_lengths.numel() == 0) {
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens =
            torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens =
            torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
        return;
    }

    torch::Tensor cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));

    cu_seqlens.slice(0, 1, max_bs_ + 1)    = input_lengths.cumsum(0);
    cu_kv_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.add(prefix_lengths).cumsum(0);

    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens    = cu_seqlens.pin_memory();
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens = cu_kv_seqlens.pin_memory();
}

int GraphBaseRunner::getCurrentRealGraphBs() const {
    return state_.current_real_graph_bs;
}

void GraphBaseRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.input_ids = torch::zeros({max_num_token_}, options_device_int32_);
    inputs.attention_inputs.input_lengths =
        torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_).pin_memory();
    inputs.attention_inputs.sequence_lengths         = torch::ones({int(max_bs_)}, options_cpu_int32_).pin_memory();
    int kv_cols                                      = device_ops_.kv_block_cols(max_seq_len_, seq_size_per_block_);
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros({int(max_bs_), kv_cols}, options_device_int32_);
    if (num_tokens_per_bs_ > 1 && !is_prefill_graph_mode_) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ + num_tokens_per_bs_, options_cpu_int32_).pin_memory();
    } else {
        inputs.attention_inputs.prefix_lengths = torch::zeros({int(max_bs_)}, options_cpu_int32_).pin_memory();
    }
    inputs.attention_inputs.kv_cache_block_id_host =
        torch::zeros({int(max_bs_), kv_cols}, options_cpu_int32_).pin_memory();

    // Hybrid cache: kv_cache_layer_to_group tensor
    auto layer_num = kv_cache_layer_to_group_.size();
    if (layer_num > 0) {
        auto kv_cache_layer_to_group_capture =
            torch::empty({static_cast<int64_t>(layer_num)}, options_cpu_int32_).pin_memory();
        auto* dst = kv_cache_layer_to_group_capture.data_ptr<int32_t>();
        for (size_t i = 0; i < layer_num; ++i) {
            dst[i] = static_cast<int32_t>(kv_cache_layer_to_group_[i]);
        }
        inputs.attention_inputs.kv_cache_layer_to_group = kv_cache_layer_to_group_capture;
    }

    // Hybrid cache: per-group block tables
    inputs.attention_inputs.kv_cache_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_block_id_host_by_group.clear();
    if (kv_cache_group_num_ > 1) {
        inputs.attention_inputs.kv_cache_block_id_device_by_group.reserve(kv_cache_group_num_);
        inputs.attention_inputs.kv_cache_block_id_host_by_group.reserve(kv_cache_group_num_);
        for (int g = 0; g < kv_cache_group_num_; ++g) {
            inputs.attention_inputs.kv_cache_block_id_device_by_group.push_back(
                torch::zeros({int(max_bs_), kv_cols}, options_device_int32_));
            inputs.attention_inputs.kv_cache_block_id_host_by_group.push_back(
                torch::zeros({int(max_bs_), kv_cols}, options_cpu_int32_).pin_memory());
        }
    }

    inputs.attention_inputs.padding_offset =
        torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_).pin_memory();
    inputs.attention_inputs.dtype       = model_data_type_;
    inputs.attention_inputs.is_s_padded = true;
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        device_ops_.sequence_lengths_plus_one_tensor(int(max_bs_), options_device_int32_);
    inputs.attention_inputs.decode_cu_seqlens_d = torch::zeros({int(max_bs_)}, options_device_int32_);
}

void GraphBaseRunner::initCaptureAttentionInputsPost() {
    auto&         inputs                        = capture_mem_hold_.py_model_inputs_;
    torch::Tensor cuda_graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    cuda_graph_prefill_batch_size.fill_(1);
    inputs.attention_inputs.prefill_cuda_graph_copy_params =
        PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, max_seq_len_, int(max_bs_)};
}

void GraphBaseRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    (void)max_num_token;
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    inputs.bert_embedding_inputs.combo_position_ids     = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);
    inputs.bert_embedding_inputs.position_encoding      = position_encoding_;
    inputs.bert_embedding_inputs.combo_tokens_type_ids  = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);
    inputs.bert_embedding_inputs.token_type_embedding   = token_type_embedding_;
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

void GraphBaseRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void GraphBaseRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void GraphBaseRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void GraphBaseRunner::setMaxPrefillGraphLen(int max_prefill_graph_len) {
    max_prefill_graph_len_ = max_prefill_graph_len;
}

void GraphBaseRunner::initCapture() {
    if (enable_graph_) {
        if (device_ops_.should_skip_decode_capture(py_instance_, is_prefill_graph_mode_)) {
            initKernelInternalMemory();
            return;
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        if (is_prefill_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
            if (!capture_range_.empty()) {
                max_prefill_graph_len_ = *std::max_element(capture_range_.begin(), capture_range_.end());
            }
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        inputs.input_ids     = torch::zeros({max_num_token_}, options_device_int32_);
        inputs.input_hiddens = torch::zeros({max_num_token_, hidden_size_}, options_device_float_);
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, is_prefill_graph_mode_);
        initKernelInternalMemory();
        auto attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
        attn_pyobj.attr("prepare_cuda_graph")(capture_mem_hold_.py_model_inputs_.attention_inputs);
        py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
        output = torch::zeros({max_num_token_, hidden_size_}, options_device_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();

        if (is_prefill_graph_mode_) {
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.data_ptr<int>()[0] = max_num_token_;
            PyModelInputs inputs = capture_mem_hold_.py_model_inputs_;
            inputs.attention_inputs.cu_seqlens =
                capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, 2);
            inputs.attention_inputs.cu_kv_seqlens =
                capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, 2);
            inputs.attention_inputs.input_lengths =
                capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, 1);
            py_forward_method_(inputs);
            capturePrefill();
        } else {
            captureDecode();
        }
    } else {
        initKernelInternalMemory();
    }
}

void GraphBaseRunner::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void GraphBaseRunner::captureOneGraphInstance(int key, const char* key_type) {
    auto inputs     = graph_instances_[key].mem_hold_.py_model_inputs_;
    auto attn_pyobj = graph_instances_[key].mem_hold_.attn_pyobj_;
    attn_pyobj.attr("prepare_cuda_graph")(inputs.attention_inputs);
    py_forward_method_(inputs, attn_pyobj);
    py_forward_method_(inputs, attn_pyobj);

    device_ops_.device_synchronize();
    device_ops_.before_capture_stream(py_instance_, key, key_type);

    device_ops_.with_capture_stream([&]() {
        at::cuda::CUDAGraph& graph = graph_instances_[key].graph_;
        std::string          output_dot_filename;
        if (enable_graph_debug_mode_) {
            graph.enable_debug_mode();
            std::string key_type_str = std::string(key_type);
            std::replace(key_type_str.begin(), key_type_str.end(), ' ', '_');
            output_dot_filename = std::string(device_ops_.debug_file_prefix) + std::to_string(num_tokens_per_bs_) + "_"
                                  + key_type_str + "_" + std::to_string(key) + "_visualization.dot";
        }

        device_ops_.enter_capture(py_instance_);
        graph.capture_begin();
        auto py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
        graph.capture_end();
        device_ops_.exit_capture(py_instance_);

        if (enable_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename.c_str());
        }
    });
}

void GraphBaseRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    device_ops_.device_synchronize();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void GraphBaseRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    inputs.attention_inputs.is_prefill = is_prefill_graph_mode_ || num_tokens_per_bs_ > 1;
    inputs.input_ids                   = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.input_hiddens = capture_mem_hold_.py_model_inputs_.input_hiddens.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.prefix_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.sequence_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);

    // Hybrid cache: per-group block tables
    const auto& cap_attn = capture_mem_hold_.py_model_inputs_.attention_inputs;
    inputs.attention_inputs.kv_cache_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_block_id_host_by_group.clear();
    if (!cap_attn.kv_cache_block_id_device_by_group.empty() && !cap_attn.kv_cache_block_id_host_by_group.empty()) {
        const size_t group = cap_attn.kv_cache_block_id_device_by_group.size();
        inputs.attention_inputs.kv_cache_block_id_device_by_group.reserve(group);
        inputs.attention_inputs.kv_cache_block_id_host_by_group.reserve(group);
        for (size_t g = 0; g < group; ++g) {
            inputs.attention_inputs.kv_cache_block_id_device_by_group.push_back(
                cap_attn.kv_cache_block_id_device_by_group[g].slice(0, 0, batch_size));
            inputs.attention_inputs.kv_cache_block_id_host_by_group.push_back(
                cap_attn.kv_cache_block_id_host_by_group[g].slice(0, 0, batch_size));
        }
    }

    // Common direct assignments (no slice needed)
    inputs.attention_inputs.dtype = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.attention_inputs.kv_cache_layer_to_group =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_layer_to_group;
    inputs.bert_embedding_inputs        = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded = true;
}

CaptureMemoryHold GraphBaseRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             is_prefill_graph_mode_ || num_tokens_per_bs_ > 1);
}

}  // namespace rtp_llm
