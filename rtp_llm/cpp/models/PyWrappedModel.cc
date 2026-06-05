#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <vector>
#include <algorithm>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include "rtp_llm/cpp/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"
#endif

using namespace std;

namespace rtp_llm {

torch::Tensor PyWrappedModel::tensorHoldHostAndToCuda(const torch::Tensor& tensor) {
    if (tensor.device().is_cuda()) {
        return tensor;
    }

    buffer_holder_.hold_host(tensor);

    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));
    }

    // NOTE: since is_pinned() operation costs a lot cpu time, we only check it when pinned_check_remaining_ > 0.
    if (pinned_check_remaining_ > 0) {
        RTP_LLM_CHECK_WITH_INFO(tensor.is_pinned(), "tensor is not pinned, fused copy requires pinned memory");
    }

    // create tensor on cuda
    auto cuda_tensor = torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));

    d2d_copies_.add(tensor.data_ptr(), cuda_tensor.data_ptr(), tensor.nbytes());

    return cuda_tensor;
}

void PyWrappedModel::releaseBuffers() {
    if (held_attn_pyobj_.ptr()) {
        py::gil_scoped_acquire gil;
        held_attn_pyobj_ = py::object();
    }
    // TensorHolder release point (PyWrappedModel): advances model-internal
    // host staging buffers from tensorHoldHostAndToCuda()/holdInputsHostBuffers().
    buffer_holder_.release();
}

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        held_attn_pyobj_ = py::object();
        // Always release py_model_ since it's always initialized now
        py_model_.release();
        if (graph_runner_ != nullptr) {
            delete graph_runner_;
            graph_runner_ = nullptr;
        }
        RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python object instance released.");
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction: %s", e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction: %s", e.what());
    }
}

// Helper function to build PyAttentionInputs from GptModelInputs
torch_ext::PyAttentionInputs PyWrappedModel::buildPyAttentionInputs(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs");
    DevicePerfWrapper            wrapper(enable_device_perf_, "py model buildPyAttentionInputs");
    torch_ext::PyAttentionInputs py_attn_inputs;
    auto                         to_device_i32 = [this](const torch::Tensor& tensor) -> torch::Tensor {
        if (!tensor.defined()) {
            return tensor;
        }
        const auto cuda_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
        if (tensor.device().is_cuda()) {
            auto tensor_d = tensor.dtype() == torch::kInt32 ? tensor : tensor.to(cuda_i32);
            return tensor_d.is_contiguous() ? tensor_d : tensor_d.contiguous();
        }

        auto host_tensor =
            tensor.dtype() == torch::kInt32 ? tensor.contiguous() : tensor.to(torch::kInt32).contiguous();
        if (!host_tensor.is_pinned()) {
            host_tensor = host_tensor.pin_memory();
        }
        buffer_holder_.hold_host(host_tensor);
        return host_tensor.to(cuda_i32, /*non_blocking=*/true);
    };
    auto length_plus_one_device = [&to_device_i32](const torch::Tensor& tensor) -> torch::Tensor {
        if (!tensor.defined()) {
            return tensor;
        }
        auto tensor_d = to_device_i32(tensor);
        return tensor_d.numel() == 0 ? tensor_d : (tensor_d + 1).to(torch::kInt32);
    };
    auto pinned_host_i32 = [this](const torch::Tensor& tensor) -> torch::Tensor {
        if (!tensor.defined()) {
            return tensor;
        }
        torch::Tensor host_tensor = tensor;
        if (host_tensor.device().is_cuda()) {
            // TODO(async): legacy cache-store and attention params still require a CPU block table.
            host_tensor = host_tensor.cpu();
        }
        if (host_tensor.dtype() != torch::kInt32) {
            host_tensor = host_tensor.to(torch::kInt32);
        }
        host_tensor = host_tensor.contiguous().pin_memory();
        buffer_holder_.hold_host(host_tensor);
        return host_tensor;
    };

    const auto    prefix_lengths_src   = inputs.prefix_lengths;
    const auto    sequence_lengths_src = inputs.sequence_lengths;
    const auto    input_lengths_src    = inputs.input_lengths;
    torch::Tensor prefix_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor input_lengths;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs(lengths_to_device)");
        prefix_lengths   = to_device_i32(prefix_lengths_src);
        sequence_lengths = to_device_i32(sequence_lengths_src);
        input_lengths    = to_device_i32(input_lengths_src);
    }
    const auto cuda_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);

    // Keep all length tensors device-resident on the model boundary. Legacy CPU
    // consumers must opt in to an explicit .cpu() with TODO(async) at the call site.
    py_attn_inputs.prefix_lengths   = prefix_lengths;
    py_attn_inputs.sequence_lengths = sequence_lengths;
    py_attn_inputs.input_lengths    = input_lengths;

    if (inputs.kv_cache_kernel_block_id.defined() && inputs.kv_cache_kernel_block_id.dim() != 3) {
        RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs(kv_kernel_block_host)");
        py_attn_inputs.kv_cache_kernel_block_id_host = pinned_host_i32(inputs.kv_cache_kernel_block_id);
    }
    if (inputs.kv_cache_block_id.defined()) {
        RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs(kv_block_host)");
        py_attn_inputs.kv_cache_block_id_host = pinned_host_i32(inputs.kv_cache_block_id);
    }
    if (inputs.kv_cache_layer_to_group.defined()) {
        py_attn_inputs.kv_cache_layer_to_group = inputs.kv_cache_layer_to_group;
    }

    if (inputs.combo_position_ids.defined()) {
        py_attn_inputs.combo_position_ids = tensorHoldHostAndToCuda(inputs.combo_position_ids);
    }

    // Calculate cu_seqlens
    int    batch_size               = py_attn_inputs.input_lengths.size(0);
    size_t context_batch_size       = py_attn_inputs.prefix_lengths.size(0);
    size_t decode_batch_size        = py_attn_inputs.sequence_lengths.size(0);
    py_attn_inputs.dtype            = dataTypeToTorchType(description_.data_type);
    py_attn_inputs.is_prefill       = !decode_batch_size;
    py_attn_inputs.is_target_verify = inputs.is_target_verify;
    RTP_LLM_CHECK_WITH_INFO(
        context_batch_size + decode_batch_size == batch_size,
        "batch size check failed context_batch_size[%ld] decode_batch_size[%ld] total_batch_size[%ld]",
        context_batch_size,
        decode_batch_size,
        batch_size);

    // Defensive guard: PyWrappedModel currently does not support a mixed prefill+decode batch.
    // The cu_seqlens slice assignment below assumes input_lengths.cumsum spans only context streams,
    // but input_lengths actually has shape [decode + context]. When both are non-zero the sizes
    // mismatch (slice=[context_batch_size] vs cumsum=[batch_size]) and copy_ throws an opaque
    // PyTorch broadcast error. Failing here gives an actionable message and also catches any
    // future scheduler regression that lets a mixed batch reach the python model path. Schedulers
    // that talk to py_model are expected to drain decode before adding context (see
    // FIFOScheduler::evaluateRunningBatch and GatherBatchScheduler::schedule's
    // python_model_busy guard).
    RTP_LLM_CHECK_WITH_INFO(context_batch_size == 0 || decode_batch_size == 0,
                            "PyWrappedModel received a mixed prefill+decode batch which is not supported: "
                            "context_batch_size[%ld] decode_batch_size[%ld]. The scheduler must keep prefill and "
                            "decode batches separate when load_python_model is enabled.",
                            context_batch_size,
                            decode_batch_size);

    if (context_batch_size > 0) {
        RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs(context_metadata)");
        py_attn_inputs.total_tokens = inputs.combo_tokens.defined() ? static_cast<int>(inputs.combo_tokens.size(0)) : 0;
        // TODO(async): context_total_kv_length is still a legacy CPU scalar.
        // The exact value for non-zero prefix lengths is available as
        // cu_kv_seqlens[-1] on device; do not D2H here just to fill this field.
        py_attn_inputs.context_total_kv_length = py_attn_inputs.total_tokens;
        py_attn_inputs.cu_seqlens              = torch::empty({batch_size + 1}, cuda_i32);
        py_attn_inputs.cu_kv_seqlens           = torch::empty({batch_size + 1}, cuda_i32);
        py_attn_inputs.padding_offset          = torch::empty({py_attn_inputs.total_tokens}, cuda_i32);
#if USING_CUDA
        invokeBuildAttentionInputMetadata(py_attn_inputs.input_lengths,
                                          py_attn_inputs.prefix_lengths,
                                          py_attn_inputs.cu_seqlens,
                                          py_attn_inputs.cu_kv_seqlens,
                                          py_attn_inputs.padding_offset,
                                          c10::cuda::getCurrentCUDAStream().stream());
#else
        RTP_LLM_FAIL("device attention input metadata requires CUDA");
#endif
    } else {
        py_attn_inputs.total_tokens        = 0;
        py_attn_inputs.cu_seqlens          = torch::zeros({batch_size + 1}, cuda_i32);
        py_attn_inputs.cu_kv_seqlens       = torch::zeros({batch_size + 1}, cuda_i32);
        py_attn_inputs.padding_offset      = torch::empty({0}, cuda_i32);
        py_attn_inputs.decode_cu_seqlens_d = torch::arange(0, py_attn_inputs.sequence_lengths.size(0) + 1, 1, cuda_i32);
    }

    // In qwen3-next target verify mode, sequence_lengths_plus_1_d uses prefix_lengths
    {
        RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs(sequence_lengths_plus_1)");
        if (py_attn_inputs.is_target_verify && inputs.sequence_lengths_plus_1.defined()) {
            py_attn_inputs.sequence_lengths_plus_1_d = to_device_i32(inputs.sequence_lengths_plus_1);
        } else if (py_attn_inputs.is_target_verify) {
            py_attn_inputs.sequence_lengths_plus_1_d = length_plus_one_device(prefix_lengths_src);
        } else {
            py_attn_inputs.sequence_lengths_plus_1_d = length_plus_one_device(sequence_lengths_src);
        }
    }

    return py_attn_inputs;
}

static void calculatePaddingOffsetDeviceAware(torch_ext::PyAttentionInputs& py_attn_inputs) {
    if (!py_attn_inputs.input_lengths.defined() || !py_attn_inputs.input_lengths.device().is_cuda()) {
        calculatePaddingOffset(py_attn_inputs);
        return;
    }

    if (py_attn_inputs.padding_offset.defined() && py_attn_inputs.padding_offset.device().is_cuda()) {
        return;
    }

    const auto cuda_i32           = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    py_attn_inputs.padding_offset = torch::empty({py_attn_inputs.total_tokens}, cuda_i32);
    if (py_attn_inputs.total_tokens == 0) {
        return;
    }

#if USING_CUDA
    invokeBuildAttentionInputMetadata(py_attn_inputs.input_lengths,
                                      py_attn_inputs.prefix_lengths,
                                      py_attn_inputs.cu_seqlens,
                                      py_attn_inputs.cu_kv_seqlens,
                                      py_attn_inputs.padding_offset,
                                      c10::cuda::getCurrentCUDAStream().stream());
#else
    RTP_LLM_FAIL("device padding_offset requires CUDA");
#endif
}

// Helper function to setup KV cache for attention inputs
void PyWrappedModel::setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                                    const GptModelInputs&         inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.setupKVCacheForAttentionInputs");
    DevicePerfWrapper wrapper(enable_device_perf_, "py model setupKVCacheForAttentionInputs");
    if (!inputs.kv_cache_kernel_block_id.defined()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_kernel_block_id.dim() == 3, "kv_cache_kernel_block_id shape should be 3");
    // New CUDA layout: [group, batch, kernel_blocks].
    // Per-group device views are zero-copy slices; host_by_group was removed.
    const size_t group = inputs.kv_cache_kernel_block_id.size(0);

    py_attn_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    py_attn_inputs.kv_cache_kernel_block_id_device_by_group.reserve(group);

    for (size_t g = 0; g < group; ++g) {
        // Group view: [batch, kernel_blocks] on CUDA (no copy).
        py_attn_inputs.kv_cache_kernel_block_id_device_by_group.push_back(inputs.kv_cache_kernel_block_id[g]);
    }

    // Legacy 2-D device field defaults to group 0.
    py_attn_inputs.kv_cache_kernel_block_id_device = py_attn_inputs.kv_cache_kernel_block_id_device_by_group[0];

    // Gate host materialization: MHA reads device fields only, while MLA/
    // SparseMLA/ROCm/CP paths still consume the singular host block table.
    if (description_.attention_conf.use_mla) {
        torch::Tensor group0 = inputs.kv_cache_kernel_block_id[0];
        if (group0.device().is_cuda()) {
            group0 = group0.cpu();
        }
        if (group0.dtype() != torch::kInt32) {
            group0 = group0.to(torch::kInt32);
        }
        group0 = group0.contiguous().pin_memory();
        buffer_holder_.hold_host(group0);
        py_attn_inputs.kv_cache_kernel_block_id_host = group0;
    }
}

// Helper function to build BertEmbeddingInputs from GptModelInputs
torch_ext::BertEmbeddingInputs PyWrappedModel::buildBertEmbeddingInputs(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.buildBertEmbeddingInputs");
    DevicePerfWrapper              wrapper(enable_device_perf_, "py model buildBertEmbeddingInputs");
    torch_ext::BertEmbeddingInputs bert_embedding_inputs;

    // Convert combo_position_ids from Buffer to torch::Tensor
    if (inputs.combo_position_ids.defined()) {
        bert_embedding_inputs.combo_position_ids = inputs.combo_position_ids.cuda();
    }

    // Convert combo_tokens_type_ids from Buffer to torch::Tensor
    if (inputs.combo_tokens_type_ids.defined()) {
        {
            DevicePerfWrapper wrapper(enable_device_perf_, "py model combo_tokens.cuda()");
            bert_embedding_inputs.combo_tokens_type_ids = inputs.combo_tokens_type_ids.cuda();
        }
    }

    // Get position_encoding from model weights (no clone needed for weights)
    if (weights_.position_encoding) {
        DevicePerfWrapper wrapper(enable_device_perf_, "py model weights_.position_encoding->kernel");
        bert_embedding_inputs.position_encoding = weights_.position_encoding->kernel;
    }

    // Get token_type_embedding from model weights (no clone needed for weights)
    if (weights_.token_type_embedding) {
        DevicePerfWrapper wrapper(enable_device_perf_, "py model weights_.token_type_embedding->kernel");
        bert_embedding_inputs.token_type_embedding = weights_.token_type_embedding->kernel;
    }

    // Set input_embedding_scalar
    bert_embedding_inputs.input_embedding_scalar = description_.input_embedding_scalar;
    return bert_embedding_inputs;
}

// Helper function to call forwardPostLayers with common parameters
GptModelOutputs PyWrappedModel::callForwardPostLayers(torch::Tensor         hidden_states,
                                                      const GptModelInputs& inputs,
                                                      bool                  skip_final_layernorm,
                                                      size_t                num_valid_tokens) {
    RTP_LLM_PROFILE_SCOPE("py_model.callForwardPostLayers");
    size_t num_input_tokens = num_valid_tokens != -1 ? num_valid_tokens : inputs.combo_tokens.size(0);
    return forwardPostLayers(hidden_states,
                             inputs.input_lengths.size(0) != inputs.sequence_lengths.size(0),
                             inputs.need_all_logits,
                             inputs.lm_output_indexes,
                             false,
                             num_input_tokens,
                             inputs,
                             torch::Tensor(),
                             skip_final_layernorm);
}

std::optional<PyCacheStoreInputs> PyWrappedModel::prepareWriteCacheParams(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.prepareWriteCacheParams");
    std::optional<PyCacheStoreInputs> params;
    if (!inputs.warmup && inputs.pd_separation) {
        const size_t         decoder_batch_size = inputs.sequence_lengths.size(0);
        const size_t         context_batch_size = inputs.input_lengths.size(0) - decoder_batch_size;
        std::vector<int64_t> cache_keys_vec;
        if (inputs.cache_keys.defined()) {
            auto ck        = inputs.cache_keys.contiguous();
            cache_keys_vec = std::vector<int64_t>(ck.data_ptr<int64_t>(), ck.data_ptr<int64_t>() + ck.numel());
        }
        // Async-copy device length tensors to pinned host for cache store.
        // The copy is enqueued on the current CUDA stream; the event recorded
        // later in WriteCacheStoreOp guarantees completion before consumption.
        auto async_to_pinned_host = [this](const torch::Tensor& t) -> torch::Tensor {
            if (!t.defined()) {
                return t;
            }
            if (t.device().is_cpu()) {
                return t.is_pinned() ? t : t.pin_memory();
            }
            auto host = torch::empty(t.sizes(), t.options().device(torch::kCPU).pinned_memory(true));
            host.copy_(t, /*non_blocking=*/true);
            buffer_holder_.hold_host(host);
            return host;
        };
        auto input_lengths_host  = async_to_pinned_host(inputs.input_lengths);
        auto prefix_lengths_host = async_to_pinned_host(inputs.prefix_lengths);

        torch::Tensor kv_cache_layer_to_group =
            inputs.kv_cache_layer_to_group.defined() ? inputs.kv_cache_layer_to_group : torch::Tensor();
        torch::Tensor kv_cache_group_types =
            inputs.kv_cache_group_types.defined() ? inputs.kv_cache_group_types : torch::Tensor();
        PyCacheStoreInputs cache_store_inputs{context_batch_size,
                                              decoder_batch_size,
                                              inputs.request_id,
                                              inputs.request_pd_separation,
                                              kv_cache_layer_to_group,
                                              kv_cache_group_types,
                                              transVectorToString(cache_keys_vec),
                                              inputs.seq_size_per_block,
                                              inputs.kv_block_stride_bytes,
                                              inputs.kv_scale_stride_bytes,
                                              inputs.pd_separation,
                                              model_id_,
                                              inputs.decode_entrance,
                                              inputs.warmup,
                                              description_.attention_conf.use_mla
                                                  && mla_ops_type_ != rtp_llm::MlaOpsType::MHA,
                                              cache_manager_ ? cache_manager_->getCacheStore() : nullptr,
                                              cache_store_async_writer_.get()};
        params = cache_store_inputs;
    }
    return params;
}

GptModelOutputs PyWrappedModel::forwardMicroBatched(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.forwardMicroBatched");

    // d2d_copies_ accumulates across all micro-batches before one fusedCopy().
    // With planMicroBatches capped at 2, buildPyAttentionInputs + padding_offset
    // stays within MAX_FUSED_D2D_COPIES; re-check if either cap changes.
    d2d_copies_.clear();
    if (pinned_check_remaining_ > 0) {
        --pinned_check_remaining_;
    }

    {
        py::gil_scoped_acquire gil;
        if (device_props_.ffn_as_service) {
            py::object py_forward_method = py_model_.attr("forward_micro_batch");
            py::object py_outputs_obj    = py_forward_method(std::vector<PyModelInputs>{});
            return GptModelOutputs();
        }
    }

    auto micro_batch_plan  = planMicroBatches(inputs);
    auto [split_inputs, _] = splitInputsIntoMicroBatches(inputs, micro_batch_plan);
    std::vector<PyModelInputs> input_list;
    input_list.reserve(split_inputs.size());

    for (size_t i = 0; i < split_inputs.size(); ++i) {
        const auto& micro_inputs =
            split_inputs[i].kv_cache_kernel_block_id.defined() ? split_inputs[i] : split_inputs[0];
        auto py_attn_inputs        = buildPyAttentionInputs(micro_inputs);
        auto embedding_inputs      = buildPyEmbeddingInputs(micro_inputs);
        auto multimodal_inputs     = buildPyMultimodalInputs(micro_inputs);
        auto bert_embedding_inputs = buildBertEmbeddingInputs(micro_inputs);
        if (!inputs.warmup && inputs.pd_separation) {
            py_attn_inputs.cache_store_inputs = prepareWriteCacheParams(inputs);
        }
        torch::Tensor combo_position_ids = micro_inputs.combo_position_ids.defined() ?
                                               tensorHoldHostAndToCuda(micro_inputs.combo_position_ids) :
                                               torch::empty({0});
        setupKVCacheForAttentionInputs(py_attn_inputs, micro_inputs);

        calculatePaddingOffsetDeviceAware(py_attn_inputs);
        if (py_attn_inputs.padding_offset.defined() && !py_attn_inputs.padding_offset.device().is_cuda()) {
            py_attn_inputs.padding_offset = tensorHoldHostAndToCuda(py_attn_inputs.padding_offset);
        }

        torch::Tensor token_ids = micro_inputs.combo_tokens.clone().cuda();
        torch::Tensor input_hiddens =
            inputs.last_hidden_states.defined() ? inputs.last_hidden_states : torch::empty({0});
        input_list.emplace_back(PyModelInputs{token_ids,
                                              input_hiddens,
                                              combo_position_ids,
                                              embedding_inputs,
                                              multimodal_inputs,
                                              py_attn_inputs,
                                              bert_embedding_inputs});
    }

    if (!inputs.warmup && inputs.pd_separation) {
        cache_store_async_writer_->init();
    }

    fusedCopy(d2d_copies_);

    std::vector<PyModelOutputs> py_model_outputs;
    {
        py::gil_scoped_acquire gil;
        py::object             py_forward_method = py_model_.attr("forward_micro_batch");
        py::object             py_outputs_obj    = py_forward_method(input_list);
        py_model_outputs                         = py_outputs_obj.cast<std::vector<PyModelOutputs>>();
    }

    RTP_LLM_CHECK_WITH_INFO(py_model_outputs.size() == input_list.size(),
                            "py_model_outputs.size:%d != micro_batch_inputs.size:%d",
                            py_model_outputs.size(),
                            input_list.size());

    if (!inputs.warmup && inputs.pd_separation) {
        cache_store_async_writer_->waitAllDone();
    }

    // TODO: merge hidden states in one tensor
    torch::Tensor hidden_states;
    if (!micro_batch_plan.enable) {
        RTP_LLM_CHECK_WITH_INFO(py_model_outputs[0].hidden_states.size(0) == inputs.combo_tokens.size(0),
                                "py_model_outputs[0].hidden_states.size(0):%d != inputs.combo_tokens.size(0):%d",
                                py_model_outputs[0].hidden_states.size(0),
                                inputs.combo_tokens.size(0));
        hidden_states = py_model_outputs[0].hidden_states;
    } else {
        size_t total_tokens = inputs.combo_tokens.size(0);
        size_t hidden_size  = description_.attention_conf.head_num * description_.attention_conf.size_per_head;
        hidden_states =
            torch::empty({(int64_t)total_tokens, (int64_t)hidden_size},
                         torch::TensorOptions(dataTypeToTorchType(description_.data_type)).device(torch::kCUDA));
        int offset = 0;
        for (int i = 0; i < py_model_outputs.size(); i++) {
            RTP_LLM_CHECK_WITH_INFO(
                offset + py_model_outputs[i].hidden_states.size(0) <= (int)total_tokens,
                "offset + py_model_outputs[i].hidden_states.size(0):%d > inputs.combo_tokens->shape()[0]:%d",
                offset + py_model_outputs[i].hidden_states.size(0),
                total_tokens);
            auto slice_size = py_model_outputs[i].hidden_states.size(0);
            hidden_states.slice(0, offset, offset + slice_size).copy_(py_model_outputs[i].hidden_states);
            offset += slice_size;
        }
        RTP_LLM_CHECK_WITH_INFO(offset == (int)total_tokens,
                                "total out hidden size:%d != inputs.combo_tokens->shape()[0]:%d",
                                offset,
                                total_tokens);
    }

    RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");

    return callForwardPostLayers(hidden_states, inputs, false);
}

torch_ext::PyEmbeddingInputs PyWrappedModel::buildPyEmbeddingInputs(const GptModelInputs& inputs) {
    DevicePerfWrapper            wrapper(enable_device_perf_, "py model buildPyEmbeddingInputs");
    torch_ext::PyEmbeddingInputs embedding_inputs;
    if (inputs.combo_tokens_type_ids.defined()) {
        embedding_inputs.combo_tokens_type_ids = inputs.combo_tokens_type_ids.cuda();
    }
    if (inputs.text_tokens_mask.defined()) {
        embedding_inputs.text_tokens_mask = inputs.text_tokens_mask.cuda();
    }
    return embedding_inputs;
}

torch_ext::PyMultimodalInputs PyWrappedModel::buildPyMultimodalInputs(const GptModelInputs& inputs) {
    DevicePerfWrapper             wrapper(enable_device_perf_, "py model buildPyMultimodalInputs");
    torch_ext::PyMultimodalInputs multimodal_input;
    if (inputs.multimodal_features && !inputs.multimodal_features.value().empty()) {
        std::vector<torch::Tensor> multimodal_features;
        for (const auto& feature : inputs.multimodal_features.value()) {
            multimodal_features.emplace_back(feature.cuda());
        }
        multimodal_input.multimodal_features = multimodal_features;
    }
    if (inputs.mm_extra_input && !inputs.mm_extra_input.value().empty()) {
        std::vector<torch::Tensor> mm_extra_input;
        for (const auto& embed : inputs.mm_extra_input.value()) {
            mm_extra_input.emplace_back(embed.cuda());
        }
        multimodal_input.mm_extra_input = mm_extra_input;
    }
    if (inputs.mm_features_locs.defined()) {
        multimodal_input.mm_features_locs = inputs.mm_features_locs.cuda();
    }
    return multimodal_input;
}

void PyWrappedModel::prepareAttentionInputs(const GptModelInputs& inputs) {
    prepareAttentionInputs(inputs, false);
}

void PyWrappedModel::prepareAttentionInputs(const GptModelInputs& inputs, bool skip_forward_event_sync) {
    RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs");
    d2d_copies_.clear();
    if (pinned_check_remaining_ > 0) {
        --pinned_check_remaining_;
    }

    DevicePerfWrapper            wrapper(enable_device_perf_, "py model prepareAttentionInputs");
    torch_ext::PyAttentionInputs attention_inputs;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(build)");
        attention_inputs = buildPyAttentionInputs(inputs);
    }
    if (!inputs.warmup && inputs.pd_separation) {
        attention_inputs.cache_store_inputs = prepareWriteCacheParams(inputs);
        cache_store_async_writer_->init();
    }
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(setup_kv_cache)");
        setupKVCacheForAttentionInputs(attention_inputs, inputs);
    }

    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(padding_offset)");
        calculatePaddingOffsetDeviceAware(attention_inputs);
        if (attention_inputs.padding_offset.defined() && !attention_inputs.padding_offset.device().is_cuda()) {
            attention_inputs.padding_offset = tensorHoldHostAndToCuda(attention_inputs.padding_offset);
        }
    }

    attention_inputs_ = attention_inputs;
    prepared_attention_inputs_.store(true, std::memory_order_release);

    // CRITICAL ORDERING: flush queued H2D copies BEFORE graph_runner_->prepareAttentionInputs.
    // graph_runner internally launches strided D2D copies that READ from these freshly allocated
    // CUDA tensors (e.g. kv_cache_kernel_block_id_device_by_group); without flushing first, the
    // D2D copies see uninitialized device memory and pollute the capture buffer, which the
    // QKV+RoPE+KVCache kernel then dereferences as block-id pointers → WARP_ILLEGAL_ADDRESS.
    // (Pre-d318b63ea forward() did fusedCopy before graph_runner->forward; the async-prepare
    // extraction split those steps and broke the implicit ordering.)
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(fused_h2d)");
        fusedCopy(d2d_copies_);
    }

    graph_state_         = CudaGraphState();
    auto empty_tensor    = torch::Tensor();
    auto py_model_inputs = PyModelInputs({empty_tensor, empty_tensor, empty_tensor, PyEmbeddingInputs(), PyMultimodalInputs(),attention_inputs_, empty_tensor});

    if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state_)) {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(cuda_graph_prepare)");
        graph_runner_->prepareAttentionInputs(py_model_inputs, graph_state_, skip_forward_event_sync);
    }
}

void PyWrappedModel::updateKVCacheKernelBlockId(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.updateKVCacheKernelBlockId");
    if (!inputs.kv_cache_kernel_block_id.defined()) {
        return;
    }
    if (!prepared_attention_inputs_.load(std::memory_order_acquire)) {
        // No pre-built attention_inputs_ to refresh — forward() will rebuild
        // from the current `inputs` directly via prepareAttentionInputs().
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_kernel_block_id.dim() == 3,
                            "kv_cache_kernel_block_id must be 3-D (group, batch, blocks)");
    const size_t group = inputs.kv_cache_kernel_block_id.size(0);

    // Re-alias _device_by_group / _device from the freshly-gathered tensor.
    // Slices are zero-copy views, no new allocation. Same logic as
    // setupKVCacheForAttentionInputs but applied in place to attention_inputs_.
    attention_inputs_.kv_cache_kernel_block_id_device_by_group.clear();
    attention_inputs_.kv_cache_kernel_block_id_device_by_group.reserve(group);
    for (size_t g = 0; g < group; ++g) {
        attention_inputs_.kv_cache_kernel_block_id_device_by_group.push_back(inputs.kv_cache_kernel_block_id[g]);
    }
    attention_inputs_.kv_cache_kernel_block_id_device = attention_inputs_.kv_cache_kernel_block_id_device_by_group[0];

    // CUDA-graph case: refresh the captured held buffers + FlashInfer plan
    // via the focused graph_runner hook (no replay of unrelated D2D copies).
    if (enable_cuda_graph_) {
        auto empty_tensor    = torch::Tensor();
        auto py_model_inputs = PyModelInputs(
            {empty_tensor, empty_tensor, empty_tensor, PyEmbeddingInputs(), PyMultimodalInputs(), attention_inputs_, empty_tensor});
        if (graph_runner_->canRun(py_model_inputs, graph_state_)) {
            graph_runner_->updateKVCacheKernelBlockId(py_model_inputs, graph_state_);
        }
    }
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.forward");
    DevicePerfWrapper wrapper(enable_device_perf_, "py model forward");
    holdInputsHostBuffers(inputs);

    // RAII guard: ensure prepared_attention_inputs_ is always reset to false on scope exit,
    // even if forward() throws. Without this, an exception after async prepareAttentionInputs
    // would leave the flag true, causing the next forward() to use stale attention_inputs_.
    struct PreparedFlagGuard {
        std::atomic<bool>& flag;
        ~PreparedFlagGuard() {
            flag.store(false, std::memory_order_release);
        }
    } flag_guard{prepared_attention_inputs_};

    try {
        RTP_LLM_LOG_DEBUG("Calling forward method on Python object instance.");

        if (int(device_props_.enable_layer_micro_batch)) {
            return forwardMicroBatched(inputs);
        }
        PyContextParallelParams cp_params;
        if (device_props_.enable_prefill_cp) {
            context_parallel_processor_->handleInputs(const_cast<GptModelInputs&>(inputs), cp_params);
        }

        // Direct async H2D for combo_tokens (the only tensorHoldHostAndToCuda call site that
        // used to live on this code path). Bypass d2d_copies_ so forward() does not depend on
        // the fused-copy queue — that queue is now an internal detail of prepareAttentionInputs.
        // Host buffer is already kept alive by holdInputsHostBuffers() above.
        torch::Tensor token_ids;
        if (inputs.combo_tokens.device().is_cuda()) {
            token_ids = inputs.combo_tokens;
        } else {
            buffer_holder_.hold_host(inputs.combo_tokens);
            token_ids = inputs.combo_tokens.to(torch::kCUDA, /*non_blocking=*/true);
        }

        torch::Tensor input_hiddens =
            inputs.last_hidden_states.defined() ? inputs.last_hidden_states : torch::empty({0});

        torch::Tensor combo_position_ids = inputs.combo_position_ids.defined() ?
                                               tensorHoldHostAndToCuda(inputs.combo_position_ids) :
                                               torch::empty({0});

        auto embedding_inputs      = buildPyEmbeddingInputs(inputs);
        auto multimodal_inputs     = buildPyMultimodalInputs(inputs);
        auto attention_inputs      = buildPyAttentionInputs(inputs);
        auto bert_embedding_inputs = buildBertEmbeddingInputs(inputs);

        if (!prepared_attention_inputs_.load(std::memory_order_acquire)) {
            prepareAttentionInputs(inputs, /*skip_forward_event_sync=*/true);
        }

        if (device_props_.enable_prefill_cp) {
            attention_inputs_.context_parallel_info = cp_params;
        }

        // No fusedCopy here: prepareAttentionInputs (above) already flushed its queue,
        // and combo_tokens above used direct .to(non_blocking=true). Both are async on
        // the current stream and will be ordered correctly with the kernels below.

        auto           py_model_inputs = PyModelInputs({token_ids,
                                                        input_hiddens,
                                                        combo_position_ids,
                                                        embedding_inputs,
                                                        multimodal_inputs,
                                                        attention_inputs_,
                                                        bert_embedding_inputs});
        PyModelOutputs py_model_outputs;
        torch::Tensor  hidden_states;

        // Cast the Python object to PyModelOutputs and extract hidden states
        if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state_)) {
            RTP_LLM_PROFILE_SCOPE("py_model.forward(cuda_graph)");
            DevicePerfWrapper wrapper(enable_device_perf_, "cuda graph python forward");
            RTP_LLM_LOG_DEBUG(
                "[PyWrappedModel] using CUDA graph forward, is_target_verify=%d, is_prefill=%d, graph_bs=%d",
                py_model_inputs.attention_inputs.is_target_verify,
                py_model_inputs.attention_inputs.is_prefill,
                graph_state_.current_real_graph_bs);
            py_model_inputs.attention_inputs.is_s_padded = true;
            py_model_outputs                             = graph_runner_->forward(py_model_inputs, graph_state_);
            RTP_LLM_LOG_DEBUG("[PyWrappedModel] CUDA graph forward completed");
            hidden_states = py_model_outputs.hidden_states.clone();
        } else {
            py::gil_scoped_acquire gil;
            RTP_LLM_PROFILE_SCOPE("py_model.forward(normal)");
            DevicePerfWrapper wrapper(enable_device_perf_, "normal forward");
            RTP_LLM_LOG_DEBUG("[PyWrappedModel] using normal forward, is_target_verify=%d, is_prefill=%d",
                              py_model_inputs.attention_inputs.is_target_verify,
                              py_model_inputs.attention_inputs.is_prefill);
            held_attn_pyobj_      = py_model_.attr("prepare_fmha_impl")(py_model_inputs, false);
            auto py_model_forward = py_model_.attr("forward");
            auto outputs          = py_model_forward(py_model_inputs, held_attn_pyobj_);
            py_model_outputs      = outputs.cast<PyModelOutputs>();
            hidden_states         = py_model_outputs.hidden_states.clone();
        }

        if (!inputs.warmup && inputs.pd_separation) {
            cache_store_async_writer_->waitAllDone();
        }

        RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");
        // CP exit-gather + strip-padding must only run when there's an
        // actual CP-split prefill stream present.  A pure-decode batch
        // (num_prefill_stream == 0) sets empty padding_mask /
        // restore_indices in handleInputs; handleOutputs would then
        // index_select([], empty) -> hidden_states = [0, D], which makes
        // downstream Sampler::forward fail with a narrow OOB.  Detect
        // this by reusing the standard "has any context stream" test
        // already used by callForwardPostLayers.
        const bool has_context_request = inputs.input_lengths.size(0) != inputs.sequence_lengths.size(0);
        if (device_props_.enable_prefill_cp && has_context_request) {
            size_t num_valid_tokens = context_parallel_processor_->handleOutputs(hidden_states, inputs, cp_params);
            return callForwardPostLayers(hidden_states, inputs, true, num_valid_tokens);
        }
        return callForwardPostLayers(hidden_states, inputs, true);

    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("pybind11 error during forward call on Python instance: ") + e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("C++ error during forward call on Python instance: ") + e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("An unknown error occurred during forward call on Python instance.");
        throw std::runtime_error("An unknown error occurred during forward call on Python instance.");
    }
}

// --- Methods absorbed from GptModel ---

static torch::Tensor
sliceKvCacheBlockIdByBatch(const torch::Tensor& kv_cache_block_id, size_t batch_offset, size_t batch_size) {
    if (!kv_cache_block_id.defined()) {
        return torch::Tensor();
    }
    if (kv_cache_block_id.dim() == 2) {
        return kv_cache_block_id.narrow(0, batch_offset, batch_size);
    }
    if (kv_cache_block_id.dim() == 3) {
        // [group, batch, max_blocks] → narrow on dim 1
        return kv_cache_block_id.narrow(1, batch_offset, batch_size).contiguous();
    }
    return kv_cache_block_id;
}

torch::Tensor PyWrappedModel::tpSyncEmbeddingOrLogits(const torch::Tensor& input) {
    RTP_LLM_PROFILE_SCOPE("py_model.tpSyncEmbeddingOrLogits");
    const auto    tp_size = device_props_.tp_size;
    const auto    rows    = input.size(0);
    const auto    cols    = input.size(1);
    torch::Tensor all_data;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.tpSyncEmbeddingOrLogits(alloc)");
        all_data = torch::empty({rows * (int64_t)tp_size, cols}, input.options());
    }
    {
        RTP_LLM_PROFILE_SCOPE("py_model.tpSyncEmbeddingOrLogits(all_gather)");
        auto send = input.is_contiguous() ? input : input.contiguous();
        execAllGather({{all_data}, ParallelMode::TP, {send}, false});
        cudaCheckLastError();
    }
    // Transpose [tp_size, batch, hidden] -> [batch, tp_size, hidden] -> [batch, hidden * tp_size]
    {
        RTP_LLM_PROFILE_SCOPE("py_model.tpSyncEmbeddingOrLogits(transpose_concat)");
        auto transposed = all_data.reshape({(int64_t)tp_size, rows, cols})
                              .permute({1, 0, 2})
                              .contiguous()
                              .reshape({rows, cols * (int64_t)tp_size});
        cudaCheckLastError();
        return transposed;
    }
}

GptModelOutputs PyWrappedModel::forwardPostLayers(torch::Tensor         hidden,
                                                  const bool            has_context_request,
                                                  const bool            need_all_logits,
                                                  const torch::Tensor&  lm_output_indexes,
                                                  bool                  enable_sp,
                                                  size_t                token_num,
                                                  const GptModelInputs& inputs,
                                                  torch::Tensor         merged_eagle3_hidden,
                                                  bool                  skip_final_layernorm) {
    DevicePerfWrapper wrapper(enable_device_perf_, "forwardPostLayers");
    if (enable_sp && device_props_.tp_size > 1) {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(sp_all_gather)");
        auto ag_tensor =
            torch::empty({(int64_t)(hidden.size(0) * device_props_.tp_size), hidden.size(1)}, hidden.options());
        size_t m                 = ag_tensor.size(0);
        int    m_split           = device_props_.m_split;
        size_t overlap_comm_type = device_props_.overlap_comm_type;
        if (overlap_comm_type == 1 && m_split > 0) {
            size_t token_idx    = 0;
            size_t ag_token_idx = 0;
            size_t m_chunk      = m / m_split;
            if (m > 128) {
                m_chunk = (m / m_split + 127) & ~127;
            }
            while (token_idx < m) {
                const auto micro_batch_tokens    = std::min(m - token_idx, m_chunk);
                const auto ag_micro_batch_tokens = micro_batch_tokens / device_props_.tp_size;
                auto       micro_batch_recv_t    = ag_tensor.narrow(0, token_idx, micro_batch_tokens);
                auto       micro_ag_send_t       = hidden.narrow(0, ag_token_idx, ag_micro_batch_tokens);
                execAllGather({{micro_batch_recv_t}, ParallelMode::TP, {micro_ag_send_t}, false});
                token_idx += micro_batch_tokens;
                ag_token_idx += ag_micro_batch_tokens;
            }
        } else {
            execAllGather({{ag_tensor}, ParallelMode::TP, {hidden}, false});
        }

        size_t pad_mod_num = device_props_.tp_size * max((size_t)1, device_props_.m_split);
        if (token_num % pad_mod_num != 0) {
            hidden = ag_tensor.slice(0, 0, token_num).contiguous();
        } else {
            hidden = ag_tensor;
        }
    }

    if (weights_.final_layernorm && !skip_final_layernorm) {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(final_layernorm)");
        const auto& norm_w = *weights_.final_layernorm;
        const auto  eps    = description_.layernorm_eps;
        if (description_.norm_type == NormType::rmsnorm) {
            auto variance = hidden.to(torch::kFloat32).pow(2).mean(-1, /*keepdim=*/true);
            hidden        = hidden * torch::rsqrt(variance + eps);
            if (norm_w.gamma.defined()) {
                hidden = hidden * norm_w.gamma;
            }
        } else {
            auto normalized_shape = std::vector<int64_t>{hidden.size(-1)};
            auto beta             = norm_w.beta.defined() ? norm_w.beta : torch::Tensor();
            hidden                = torch::layer_norm(hidden, normalized_shape, norm_w.gamma, beta, eps);
        }
    }
    printTorchTensorData(hidden, "final_hidden");

    const auto& lm_head = weights_.lm_head;

    if (lm_head) {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(lm_head)");
        printTorchTensorData(lm_output_indexes, "lm_output_indexes");

        RTP_LLM_CHECK_WITH_INFO(lm_output_indexes.defined() && lm_output_indexes.is_cuda(),
                                "lm_output_indexes must be a CUDA tensor before forwardPostLayers, got defined=%d "
                                "device=%s",
                                static_cast<int>(lm_output_indexes.defined()),
                                lm_output_indexes.defined() ? lm_output_indexes.device().str().c_str() : "undefined");
        torch::Tensor lm_output_indexes_device;
        {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(lm_output_indexes_to_long)");
            lm_output_indexes_device = lm_output_indexes.to(torch::kLong).contiguous();
        }

        torch::Tensor last_hidden;
        if (has_context_request && !need_all_logits) {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(index_select_last_hidden)");
            last_hidden = torch::index_select(hidden, 0, lm_output_indexes_device);
        } else {
            last_hidden = hidden;
        }

        printTorchTensorData(last_hidden, "last_hidden");

        torch::Tensor logits;
        {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(lm_head_mm)");
            logits = torch::mm(last_hidden.to(lm_head->kernel.dtype()), lm_head->kernel.t()).to(torch::kFloat32);
        }
        printTorchTensorData(logits, "logits");
        if (device_props_.tp_size > 1) {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(tp_sync_logits)");
            logits = tpSyncEmbeddingOrLogits(logits);
        }
        if (check_nan_) {
            RTP_LLM_CHECK_WITH_INFO(!torch::isnan(last_hidden).any().item<bool>(), "NAN detected in last_hidden");
            RTP_LLM_CHECK_WITH_INFO(!torch::isnan(logits).any().item<bool>(), "NAN detected in logits");
        }
        torch::Tensor softmax_result_t;
        if (need_all_logits) {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(need_all_logits_index)");
            auto last_logits = torch::index_select(logits, 0, lm_output_indexes_device);
            return {last_logits, last_hidden, hidden, logits, softmax_result_t};
        }

        if (merged_eagle3_hidden.defined()) {
            hidden = merged_eagle3_hidden;
        }
        return {logits, last_hidden, hidden, torch::Tensor(), softmax_result_t};
    } else {
        return {torch::Tensor(), torch::Tensor(), hidden};
    }
}

MicroBatchPlan PyWrappedModel::planMicroBatches(const GptModelInputs& inputs) {
    if (!int(device_props_.enable_layer_micro_batch)) {
        RTP_LLM_LOG_DEBUG("micro batch disable when enable_layer_micro_batch is false");
        return {false, {}};
    }

    const auto&  input_lengths      = inputs.input_lengths;
    const auto&  sequence_lengths   = inputs.sequence_lengths;
    const size_t decoder_batch_size = sequence_lengths.size(0);
    const size_t context_batch_size = input_lengths.size(0) - decoder_batch_size;
    // TODO(async): layer micro-batch planning still needs host lengths for
    // split arithmetic. Keep the CPU mirror explicit while model inputs stay CUDA.
    const auto input_lengths_host = input_lengths.is_cuda() ? input_lengths.cpu().pin_memory() : input_lengths;
    const auto input_lengths_ptr  = input_lengths_host.data_ptr<int32_t>();

    if (decoder_batch_size + context_batch_size < 2) {
        RTP_LLM_LOG_DEBUG("micro batch disable when batch size %ld is less than 2",
                          decoder_batch_size + context_batch_size);
        return {false, {}};
    }

    if (context_batch_size && decoder_batch_size) {
        if (layer_num_ == 1) {
            size_t total_token_num = decoder_batch_size;
            for (size_t i = 0; i < context_batch_size; i++) {
                total_token_num += input_lengths_ptr[i + decoder_batch_size];
            }
            RTP_LLM_LOG_DEBUG("total_token_num %ld, decode_batch_size %ld, context_batch_size %ld",
                              total_token_num,
                              decoder_batch_size,
                              context_batch_size);
            size_t context_batch_0_size = 0;
            size_t context_batch_1_size = 0;
            size_t decode_batch_0_size  = 0;
            size_t decode_batch_1_size  = 0;
            if (total_token_num > decoder_batch_size * 2) {
                decode_batch_0_size        = decoder_batch_size;
                decode_batch_1_size        = 0;
                size_t acc_token_num       = decoder_batch_size;
                size_t context_split_point = 0;
                for (context_split_point = 0; context_split_point < context_batch_size; context_split_point++) {
                    acc_token_num += input_lengths_ptr[context_split_point + decoder_batch_size];
                    if (acc_token_num * 2 >= total_token_num) {
                        break;
                    }
                }
                context_batch_0_size = context_split_point;
                context_batch_1_size = context_batch_size - context_split_point;
            } else {
                decode_batch_0_size  = total_token_num / 2;
                decode_batch_1_size  = decoder_batch_size - total_token_num / 2;
                context_batch_0_size = 0;
                context_batch_1_size = context_batch_size;
            }
            RTP_LLM_LOG_DEBUG("split [c]%d:[d]%d in micro batch 0 and [c]%d:[d]%d in micro batch 1",
                              context_batch_0_size,
                              decode_batch_0_size,
                              context_batch_1_size,
                              decode_batch_1_size);
            return MicroBatchPlan{
                true, {{context_batch_0_size, decode_batch_0_size}, {context_batch_1_size, decode_batch_1_size}}};
        } else {
            RTP_LLM_LOG_DEBUG("split context in micro batch 0, decode in micro batch 1 disabled!");
            return {false, {}};
        }
    }

    const size_t batch_size_to_split = context_batch_size ? context_batch_size : decoder_batch_size;
    const size_t micro_batch_0_size  = (batch_size_to_split + 1) / 2;
    const size_t micro_batch_1_size  = batch_size_to_split - micro_batch_0_size;

    RTP_LLM_LOG_DEBUG("split micro batch size %ld, %ld", micro_batch_0_size, micro_batch_1_size);
    return context_batch_size ? MicroBatchPlan{true, {{micro_batch_0_size, 0}, {micro_batch_1_size, 0}}} :
                                MicroBatchPlan{true, {{0, micro_batch_0_size}, {0, micro_batch_1_size}}};
}

std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
PyWrappedModel::splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan) {
    std::vector<GptModelInputs> micro_batch_inputs;
    std::vector<TokenSliceInfo> token_slice_recipes;
    size_t                      sliced_token_idx       = 0;
    size_t                      sliced_lm_output_index = 0;
    size_t                      sliced_batch_idx       = 0;
    size_t                      decode_batch_idx       = 0;
    size_t                      prefill_batch_idx      = 0;
    // TODO(async): micro-batch token slicing still computes CPU scalar sums.
    // Convert explicitly and keep all sliced GptModelInputs device-resident.
    const auto  input_lengths_host = inputs.input_lengths.defined() && inputs.input_lengths.is_cuda() ?
                                         inputs.input_lengths.cpu().pin_memory() :
                                         inputs.input_lengths;
    const auto* input_lengths_ptr  = input_lengths_host.defined() ? input_lengths_host.data_ptr<int32_t>() : nullptr;

    if (!micro_batch_plan.enable) {
        RTP_LLM_LOG_DEBUG("micro batch disable when enable is false, use fake");
        micro_batch_inputs.push_back(inputs);

        GptModelInputs fake_inputs;
        fake_inputs.kv_cache_block_id = torch::Tensor();
        fake_inputs.combo_tokens      = inputs.combo_tokens.narrow(0, 0, 1);
        fake_inputs.input_lengths     = torch::ones({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        fake_inputs.sequence_lengths  = torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        fake_inputs.prefix_lengths    = torch::zeros({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        micro_batch_inputs.push_back(fake_inputs);
    } else {
        for (size_t i = 0; i < micro_batch_plan.batch_infos.size(); ++i) {
            const auto& p_micro_batch_size = micro_batch_plan.batch_infos[i].prefill_num;
            const auto& d_micro_batch_size = micro_batch_plan.batch_infos[i].decoder_num;
            RTP_LLM_LOG_DEBUG(
                "micro batch index %ld, prefill size %ld, decode size %ld", i, p_micro_batch_size, d_micro_batch_size);

            if (d_micro_batch_size && p_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                size_t         total_batch_size   = d_micro_batch_size + p_micro_batch_size;
                RTP_LLM_LOG_DEBUG("d and p slice from %ld %ld %ld %ld",
                                  sliced_token_idx,
                                  sliced_batch_idx,
                                  decode_batch_idx,
                                  prefill_batch_idx);
                micro_model_inputs.input_lengths = inputs.input_lengths.narrow(0, sliced_batch_idx, total_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths.narrow(0, decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, total_batch_size);
                micro_model_inputs.kv_cache_kernel_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_kernel_block_id, sliced_batch_idx, total_batch_size);
                micro_model_inputs.prefix_lengths =
                    inputs.prefix_lengths.narrow(0, prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask.defined() ?
                        inputs.attention_mask.narrow(0, sliced_batch_idx, total_batch_size) :
                        torch::Tensor();
                int32_t slice_token_num = std::accumulate(input_lengths_ptr + sliced_batch_idx + d_micro_batch_size,
                                                          input_lengths_ptr + sliced_batch_idx + total_batch_size,
                                                          0)
                                          + d_micro_batch_size;
                int32_t slice_lm_output_num = total_batch_size;
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes.narrow(0, sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens.narrow(0, sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id   = inputs.request_id.defined() ?
                                                      inputs.request_id.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                      torch::Tensor();
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation.defined() ?
                        inputs.request_pd_separation.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.cache_keys = inputs.cache_keys.defined() ?
                                                    inputs.cache_keys.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                    torch::Tensor();

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});

                micro_batch_inputs.push_back(micro_model_inputs);

                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += total_batch_size;
                prefill_batch_idx += p_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
                RTP_LLM_LOG_DEBUG(
                    "micro batch %ld sliced context and decode, batch idx %ld, token idx %ld, prefill batch idx %d, decode batch idx %d",
                    i,
                    sliced_batch_idx,
                    sliced_token_idx,
                    prefill_batch_idx,
                    decode_batch_idx);
            } else if (d_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                RTP_LLM_LOG_DEBUG("d slice from %ld %ld %ld", sliced_token_idx, sliced_batch_idx, decode_batch_idx);
                micro_model_inputs.combo_tokens  = inputs.combo_tokens.narrow(0, sliced_token_idx, d_micro_batch_size);
                micro_model_inputs.input_lengths = inputs.input_lengths.narrow(0, sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths.narrow(0, decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask.defined() ?
                        inputs.attention_mask.narrow(0, sliced_batch_idx, d_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.kv_cache_kernel_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_kernel_block_id, sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.prefix_lengths =
                    torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes.narrow(0, sliced_batch_idx, d_micro_batch_size);

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, d_micro_batch_size});

                micro_batch_inputs.push_back(micro_model_inputs);

                sliced_token_idx += d_micro_batch_size;
                sliced_batch_idx += d_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
                sliced_lm_output_index += d_micro_batch_size;
                RTP_LLM_LOG_DEBUG("micro batch %ld sliced decode, batch idx %ld, token idx %ld",
                                  i,
                                  sliced_batch_idx,
                                  sliced_token_idx);
            } else {
                GptModelInputs micro_model_inputs = inputs;
                RTP_LLM_LOG_DEBUG("p slice from %ld %ld %ld", sliced_token_idx, sliced_batch_idx, prefill_batch_idx);
                micro_model_inputs.input_lengths = inputs.input_lengths.narrow(0, sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.kv_cache_kernel_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_kernel_block_id, sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.prefix_lengths =
                    inputs.prefix_lengths.narrow(0, prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask.defined() ?
                        inputs.attention_mask.narrow(0, sliced_batch_idx, p_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.sequence_lengths =
                    torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
                int32_t slice_token_num = std::accumulate(
                    input_lengths_ptr + sliced_batch_idx, input_lengths_ptr + sliced_batch_idx + p_micro_batch_size, 0);
                int32_t slice_lm_output_num = p_micro_batch_size;
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes.narrow(0, sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens.narrow(0, sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id   = inputs.request_id.defined() ?
                                                      inputs.request_id.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                      torch::Tensor();
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation.defined() ?
                        inputs.request_pd_separation.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.cache_keys = inputs.cache_keys.defined() ?
                                                    inputs.cache_keys.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                    torch::Tensor();

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});

                micro_batch_inputs.push_back(micro_model_inputs);
                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += p_micro_batch_size;
                prefill_batch_idx += p_micro_batch_size;
                RTP_LLM_LOG_DEBUG("micro batch %ld sliced context, batch idx %ld, token idx %ld",
                                  i,
                                  sliced_batch_idx,
                                  sliced_token_idx);
            }
        }
    }
    return {micro_batch_inputs, token_slice_recipes};
}

void PyWrappedModel::holdInputsHostBuffers(const GptModelInputs& inputs) {
    buffer_holder_.hold_host(inputs.combo_tokens);
    buffer_holder_.hold_host(inputs.input_lengths);
    buffer_holder_.hold_host(inputs.sequence_lengths);
    buffer_holder_.hold_host(inputs.lm_output_indexes);
    buffer_holder_.hold_host(inputs.prefix_lengths);

    buffer_holder_.hold_host(inputs.combo_position_ids);
    buffer_holder_.hold_host(inputs.combo_tokens_type_ids);

    buffer_holder_.hold_host(inputs.last_hidden_states);

    buffer_holder_.hold_host(inputs.attention_mask);
    buffer_holder_.hold_host(inputs.kv_cache_block_id);
    buffer_holder_.hold_host(inputs.kv_cache_layer_to_group);
    buffer_holder_.hold_host(inputs.kv_cache_group_types);
    buffer_holder_.hold_host(inputs.kv_cache_update_mapping);

    if (inputs.multimodal_features.has_value()) {
        for (auto& mm_feature : inputs.multimodal_features.value()) {
            buffer_holder_.hold_host(mm_feature);
        }
    }

    buffer_holder_.hold_host(inputs.text_tokens_mask);
    buffer_holder_.hold_host(inputs.mm_features_locs);

    if (inputs.input_embeddings.has_value()) {
        for (auto& input_embedding : inputs.input_embeddings.value()) {
            buffer_holder_.hold_host(input_embedding);
        }
    }
    buffer_holder_.hold_host(inputs.input_embeddings_locs);

    buffer_holder_.hold_host(inputs.request_id);
    buffer_holder_.hold_host(inputs.request_pd_separation);
    buffer_holder_.hold_host(inputs.cache_keys);
}

}  // namespace rtp_llm
