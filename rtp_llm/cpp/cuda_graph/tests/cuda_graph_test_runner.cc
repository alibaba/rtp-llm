#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace py = pybind11;
namespace rtp_llm {

// Single wrapper for both prefill and decode tests; init_prefill / init_decode
// build role-specific GraphParams and use the common capture initializer.
// Plain pybind11 class (no torch::jit::CustomClassHolder) so the module loads without
// depending on torch's registered CustomClassHolder type.
class CudaGraphTestRunner {
public:
    static std::vector<int64_t> topologyWidths(const std::vector<uint32_t>& spans,
                                               const std::vector<uint32_t>& kernel_spans,
                                               const std::vector<int>&      types,
                                               size_t                       sequence_length,
                                               size_t                       reserved_step,
                                               size_t                       fake_count) {
        RTP_LLM_CHECK_WITH_INFO(spans.size() == kernel_spans.size() && spans.size() == types.size(),
                                "invalid test topology geometry");
        std::vector<GroupBase>   groups;
        std::vector<std::string> tags;
        for (size_t i = 0; i < spans.size(); ++i) {
            GroupBase group;
            group.tag               = std::to_string(i);
            group.spec              = std::make_shared<MHAKVCacheSpec>(group.tag, spans[i], kernel_spans[i]);
            group.policy.group_type = static_cast<CacheGroupType>(types[i]);
            tags.push_back(group.tag);
            groups.push_back(std::move(group));
        }
        const auto topology = CacheTopology::create(std::move(groups), {{0, std::move(tags)}});
        return {CudaGraphRunner::captureKernelBlockTableWidth(*topology, sequence_length, reserved_step),
                CudaGraphRunner::captureKernelBlockTableWidth(*topology, fake_count)};
    }

    void init_prefill(py::object                   py_instance,
                      int64_t                      max_context_batch_size,
                      int64_t                      max_seq_len,
                      int64_t                      kernel_block_table_width,
                      std::vector<int>             prefill_capture_seq_lens,
                      int64_t                      hidden_size,
                      std::vector<std::string>     group_tags,
                      std::optional<torch::Tensor> position_encoding,
                      std::optional<torch::Tensor> token_type_embedding) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph            = true;
        params.enable_cuda_graph_debug_mode = true;
        params.is_prefill_cuda_graph_mode   = true;
        params.max_seq_len                  = static_cast<int>(max_seq_len);

        params.num_tokens_per_bs        = static_cast<int>(max_seq_len);
        params.max_context_batch_size   = static_cast<size_t>(max_context_batch_size);
        params.hidden_size              = static_cast<size_t>(hidden_size);
        params.input_hidden_size        = static_cast<size_t>(hidden_size);
        params.model_data_type          = c10::ScalarType::BFloat16;
        params.prefill_capture_seq_lens   = std::move(prefill_capture_seq_lens);
        params.kv_cache_group_tags        = std::move(group_tags);
        params.kernel_block_table_width   = kernel_block_table_width;

        if (position_encoding.has_value()) {
            params.position_encoding = std::move(*position_encoding);
        }
        if (token_type_embedding.has_value()) {
            params.token_type_embedding = std::move(*token_type_embedding);
        }
        runner_ = CudaGraphRunner::initializeCapture(std::make_unique<CudaGraphRunner>(params, std::move(py_instance)));
    }

    void init_generation_prefill(py::object                   py_instance,
                                 int64_t                      max_requests,
                                 int64_t                      max_seq_len,
                                 int64_t                      kernel_block_table_width,
                                 std::vector<int>             prefill_capture_seq_lens,
                                 int64_t                      hidden_size,
                                 std::vector<std::string>     group_tags,
                                 int64_t                      position_id_len_factor,
                                 std::optional<torch::Tensor> position_encoding,
                                 std::optional<torch::Tensor> token_type_embedding) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph            = true;
        params.enable_cuda_graph_debug_mode = true;
        params.role                         = CudaGraphRole::GENERATION_PREFILL;
        params.max_seq_len                  = static_cast<int>(max_seq_len);

        params.num_tokens_per_bs                          = 1;
        params.max_context_batch_size                     = static_cast<size_t>(max_requests + 1);
        params.hidden_size                                = static_cast<size_t>(hidden_size);
        params.input_hidden_size                          = static_cast<size_t>(hidden_size);
        params.model_data_type                            = c10::ScalarType::BFloat16;
        params.prefill_capture_seq_lens                   = std::move(prefill_capture_seq_lens);
        params.kv_cache_group_tags                        = std::move(group_tags);
        params.generation_prefill_cuda_graph_max_requests = static_cast<int>(max_requests);
        params.generation_prefill_cuda_graph_pad_token_id = 0;
        params.position_id_len_factor                     = static_cast<int>(position_id_len_factor);
        params.kernel_block_table_width                   = kernel_block_table_width;
        if (position_encoding.has_value()) {
            params.position_encoding = std::move(*position_encoding);
        }
        if (token_type_embedding.has_value()) {
            params.token_type_embedding = std::move(*token_type_embedding);
        }
        runner_ = CudaGraphRunner::initializeCapture(std::make_unique<CudaGraphRunner>(params, std::move(py_instance)));
    }

    void init_decode(py::object               py_instance,
                     int64_t                  hidden_size,
                     int64_t                  max_seq_len,
                     int64_t                  kernel_block_table_width,
                     std::vector<int>         decode_capture_batch_sizes,
                     std::vector<std::string> group_tags,
                     bool                     is_target_verify,
                     int64_t                  num_tokens_per_bs,
                     int64_t                  position_id_len_factor) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph            = true;
        params.enable_cuda_graph_debug_mode = false;
        params.is_prefill_cuda_graph_mode   = false;
        params.max_seq_len                  = static_cast<int>(max_seq_len);

        params.input_hidden_size          = static_cast<size_t>(hidden_size);
        params.num_tokens_per_bs          = static_cast<int>(num_tokens_per_bs);
        params.hidden_size                = static_cast<size_t>(hidden_size);
        params.model_data_type            = c10::ScalarType::BFloat16;
        params.max_context_batch_size     = 128;
        params.decode_capture_batch_sizes = std::move(decode_capture_batch_sizes);
        params.kv_cache_group_tags        = std::move(group_tags);
        params.is_target_verify           = is_target_verify;
        params.position_id_len_factor     = static_cast<int>(position_id_len_factor);
        params.kernel_block_table_width   = kernel_block_table_width;

        runner_ = CudaGraphRunner::initializeCapture(std::make_unique<CudaGraphRunner>(params, std::move(py_instance)));
    }

    bool canRun(torch_ext::PyModelInputs& inputs) {
        return runner_ != nullptr && runner_->canRun(inputs, state_);
    }

    bool canPrepare(torch_ext::PyModelInputs& inputs) {
        return runner_ != nullptr && runner_->canRun(inputs, state_, CudaGraphCheckMode::PREPARE);
    }

    void clearInputIds(torch_ext::PyModelInputs& inputs) {
        inputs.input_ids = torch::Tensor();
    }

    bool prepare(torch_ext::PyModelInputs& inputs, bool skip_forward_event_sync) {
        // Match PyWrappedModel::prepareAttentionInputs exactly: token data is
        // and BERT IDs are supplied only by the later forward call.
        auto prepare_inputs                  = inputs;
        prepare_inputs.input_ids             = torch::Tensor();
        prepare_inputs.input_hiddens         = torch::Tensor();
        prepare_inputs.bert_embedding_inputs = torch_ext::BertEmbeddingInputs();
        if (runner_ == nullptr || !runner_->canRun(prepare_inputs, state_, CudaGraphCheckMode::PREPARE)) {
            return false;
        }
        prepare_inputs.attention_inputs.input_lengths_device  = prepare_inputs.attention_inputs.input_lengths.cuda();
        prepare_inputs.attention_inputs.prefix_lengths_device = prepare_inputs.attention_inputs.prefix_lengths.cuda();
        prepare_inputs.attention_inputs.combo_position_ids    = prepare_inputs.combo_position_ids;
        refreshTaggedAttentionInputs(prepare_inputs);
        runner_->prepareAttentionInputs(prepare_inputs, state_, skip_forward_event_sync);
        return true;
    }

    void makeInputLengthsScalar(torch_ext::PyModelInputs& inputs) {
        inputs.attention_inputs.input_lengths = torch::ones({}, torch::TensorOptions().dtype(torch::kInt32));
        refreshTaggedAttentionInputs(inputs);
    }

    void makeInputIdsScalar(torch_ext::PyModelInputs& inputs) {
        inputs.input_ids = torch::ones({}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }

    torch_ext::PyModelOutputs forward(torch_ext::PyModelInputs& inputs) {
        // Production PyWrappedModel creates these device mirrors. Python tests
        // cannot assign them because the bindings intentionally expose them as
        // read-only, so reproduce that input-building step in the test wrapper.
        prepareDeviceMirrors(inputs);
        return runner_->forward(inputs, state_);
    }

    void prepareAttentionInputs(torch_ext::PyModelInputs& inputs) {
        c10::InferenceMode inference_guard(true);
        prepareDeviceMirrors(inputs);
        runner_->prepareAttentionInputs(inputs, state_);
    }

    void updateBlockTables(torch_ext::PyModelInputs& inputs) {
        runner_->updateKVCacheKernelBlockId(inputs, state_);
    }

    int getCurrentRealGraphSize() {
        return runner_ != nullptr ? runner_->getCurrentRealGraphSize(state_) : 0;
    }

    bool captureSessionMayBeDirty() const {
        return runner_ != nullptr && runner_->captureSessionMayBeDirty();
    }

    std::string getGenerationPrefillStatus() const {
        return generationPrefillCudaGraphStatusString(state_.generation_prefill_status);
    }

    ~CudaGraphTestRunner() {
        reset_runner();
    }

private:
    void prepareDeviceMirrors(torch_ext::PyModelInputs& inputs) {
        inputs.attention_inputs.input_lengths_device  = inputs.attention_inputs.input_lengths.cuda();
        inputs.attention_inputs.prefix_lengths_device = inputs.attention_inputs.prefix_lengths.cuda();
        refreshTaggedAttentionInputs(inputs);
    }

    void reset_runner() {
        if (runner_ != nullptr) {
            delete runner_;
            runner_ = nullptr;
        }
    }

    CudaGraphRunner* runner_ = nullptr;
    CudaGraphState   state_{};
};

}  // namespace rtp_llm

PYBIND11_MODULE(libtest_cuda_graph_runner, m) {
    using namespace rtp_llm;
    py::register_exception<DirtyCudaGraphCaptureError>(m, "DirtyCudaGraphCaptureError", PyExc_RuntimeError);
    py::class_<CudaGraphTestRunner>(m, "CudaGraphRunner")
        .def(py::init<>())
        .def_static("topologyWidths", &CudaGraphTestRunner::topologyWidths)
        .def("init_prefill",
             &CudaGraphTestRunner::init_prefill,
             py::arg("py_instance"),
             py::arg("max_context_batch_size"),
             py::arg("max_seq_len"),
             py::arg("kernel_block_table_width"),
             py::arg("prefill_capture_seq_lens"),
             py::arg("hidden_size"),
             py::arg("group_tags")            = std::vector<std::string>{},
             py::arg("position_encoding")     = py::none(),
             py::arg("token_type_embedding")  = py::none())
        .def("init_decode",
             &CudaGraphTestRunner::init_decode,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("kernel_block_table_width"),
             py::arg("decode_capture_batch_sizes"),
             py::arg("group_tags")            = std::vector<std::string>{},
             py::arg("is_target_verify")      = false,
             py::arg("num_tokens_per_bs")     = 1,
             py::arg("position_id_len_factor") = 0)
        .def("init_generation_prefill",
             &CudaGraphTestRunner::init_generation_prefill,
             py::arg("py_instance"),
             py::arg("max_requests"),
             py::arg("max_seq_len"),
             py::arg("kernel_block_table_width"),
             py::arg("prefill_capture_seq_lens"),
             py::arg("hidden_size"),
             py::arg("group_tags")             = std::vector<std::string>{},
             py::arg("position_id_len_factor") = 0,
             py::arg("position_encoding")      = py::none(),
             py::arg("token_type_embedding")   = py::none())
        .def("updateBlockTables", &CudaGraphTestRunner::updateBlockTables)
        .def("canRun", &CudaGraphTestRunner::canRun)
        .def("canPrepare", &CudaGraphTestRunner::canPrepare)
        .def("clear_input_ids", &CudaGraphTestRunner::clearInputIds)
        .def("make_input_lengths_scalar", &CudaGraphTestRunner::makeInputLengthsScalar)
        .def("make_input_ids_scalar", &CudaGraphTestRunner::makeInputIdsScalar)
        .def("prepare", &CudaGraphTestRunner::prepare, py::arg("inputs"), py::arg("skip_forward_event_sync") = false)
        .def("forward", &CudaGraphTestRunner::forward)
        .def("prepareAttentionInputs", &CudaGraphTestRunner::prepareAttentionInputs)
        .def("getGenerationPrefillStatus", &CudaGraphTestRunner::getGenerationPrefillStatus)
        .def("getCurrentRealGraphSize", &CudaGraphTestRunner::getCurrentRealGraphSize)
        .def("captureSessionMayBeDirty", &CudaGraphTestRunner::captureSessionMayBeDirty);
}
