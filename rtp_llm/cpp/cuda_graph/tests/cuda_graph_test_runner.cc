#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace py = pybind11;
namespace rtp_llm {

// Single wrapper for both prefill and decode tests; init_prefill / init_decode
// build GraphParams and call CudaGraphRunner factory methods.
// Plain pybind11 class (no torch::jit::CustomClassHolder) so the module loads without
// depending on torch's registered CustomClassHolder type.
class CudaGraphTestRunner {
public:
    void init_prefill(py::object               py_instance,
                      int64_t                  max_context_batch_size,
                      int64_t                  max_seq_len,
                      int64_t                  tokens_per_block,
                      int64_t                  kernel_tokens_per_block,
                      std::vector<int>         prefill_capture_seq_lens,
                      int64_t                  hidden_size,
                      std::vector<std::string>                   group_tags,
                      std::map<std::string, std::pair<int, int>> group_capacities,
                      int64_t                                    sp_steps) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = true;
        params.is_prefill_cuda_graph_mode   = true;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.kernel_tokens_per_block      = static_cast<int>(kernel_tokens_per_block);
        params.num_tokens_per_bs            = static_cast<int>(max_seq_len);
        params.max_context_batch_size       = static_cast<size_t>(max_context_batch_size);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.sp_steps                     = static_cast<int>(sp_steps);
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.prefill_capture_seq_lens     = std::move(prefill_capture_seq_lens);
        bindCacheGroups(params, group_tags, max_seq_len, tokens_per_block, kernel_tokens_per_block, group_capacities);

        runner_ = CudaGraphRunner::createForPrefill(std::move(py_instance), std::move(params));
    }

    void init_decode(py::object               py_instance,
                     int64_t                  hidden_size,
                     int64_t                  max_seq_len,
                     int64_t                  tokens_per_block,
                     int64_t                  kernel_tokens_per_block,
                     std::vector<int>         decode_capture_batch_sizes,
                     std::vector<std::string> group_tags,
                     bool                     is_target_verify,
                     int64_t                                    num_tokens_per_bs,
                     std::map<std::string, std::pair<int, int>> group_capacities,
                     int64_t                                    sp_steps) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = false;
        params.is_prefill_cuda_graph_mode   = false;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.kernel_tokens_per_block      = static_cast<int>(kernel_tokens_per_block);
        params.num_tokens_per_bs            = static_cast<int>(num_tokens_per_bs);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.sp_steps                     = static_cast<int>(sp_steps);
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.max_context_batch_size       = 128;
        params.decode_capture_batch_sizes   = std::move(decode_capture_batch_sizes);
        bindCacheGroups(params, group_tags, max_seq_len, tokens_per_block, kernel_tokens_per_block, group_capacities);
        params.is_target_verify = is_target_verify;

        runner_ = CudaGraphRunner::createForDecode(std::move(py_instance), std::move(params));
    }

    bool canRun(torch_ext::PyModelInputs& inputs) {
        return runner_ != nullptr && runner_->canRun(inputs, state_);
    }

    void clearTaggedPhysicalBlockTable(torch_ext::PyModelInputs& inputs, const std::string& tag, bool device) {
        const auto it = inputs.attention_inputs_by_tag.find(tag);
        RTP_LLM_CHECK_WITH_INFO(
            it != inputs.attention_inputs_by_tag.end(), "missing tagged attention inputs for tag=%s", tag.c_str());
        if (device) {
            it->second.kv_cache_block_id_device = torch::Tensor();
        } else {
            it->second.kv_cache_block_id = torch::Tensor();
        }
    }

    torch_ext::PyModelOutputs forward(torch_ext::PyModelInputs& inputs) {
        // Production PyWrappedModel creates these device mirrors. Python tests
        // cannot assign them because the bindings intentionally expose them as
        // read-only, so reproduce that input-building step in the test wrapper.
        inputs.attention_inputs.input_lengths_device  = inputs.attention_inputs.input_lengths.cuda();
        inputs.attention_inputs.prefix_lengths_device = inputs.attention_inputs.prefix_lengths.cuda();
        refreshGroupedAttentionInputs(inputs);
        return runner_->forward(inputs, state_);
    }

    int getCurrentRealGraphSize() {
        return runner_ != nullptr ? runner_->getCurrentRealGraphBs(state_) : 0;
    }

    ~CudaGraphTestRunner() {
        reset_runner();
    }

private:
    static void bindCacheGroups(GraphParams&                                      params,
                                const std::vector<std::string>&                   group_tags,
                                int64_t                                           max_seq_len,
                                int64_t                                           physical_tokens_per_block,
                                int64_t                                           kernel_tokens_per_block,
                                const std::map<std::string, std::pair<int, int>>& group_capacities) {
        const auto default_capacity = CacheBlockTableCapacity::fromBlockSizes(
            max_seq_len, physical_tokens_per_block, kernel_tokens_per_block, params.sp_steps, "test runner");
        for (const auto& tag : group_tags) {
            const auto [it, inserted] = params.kv_cache_groups.emplace(tag, CacheGroupType::FULL);
            (void)it;
            RTP_LLM_CHECK_WITH_INFO(inserted, "duplicate CUDA graph KV cache tag=%s", tag.c_str());
            const auto capacity_it = group_capacities.find(tag);
            params.kv_cache_block_table_capacities[tag] =
                capacity_it == group_capacities.end() ?
                    default_capacity :
                    CacheBlockTableCapacity{capacity_it->second.first, capacity_it->second.second};
        }
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
    py::class_<CudaGraphTestRunner>(m, "CudaGraphRunner")
        .def(py::init<>())
        .def("init_prefill",
             &CudaGraphTestRunner::init_prefill,
             py::arg("py_instance"),
             py::arg("max_context_batch_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("prefill_capture_seq_lens"),
             py::arg("hidden_size"),
             py::arg("group_tags")       = std::vector<std::string>{},
             py::arg("group_capacities") = std::map<std::string, std::pair<int, int>>{},
             py::arg("sp_steps")         = 0)
        .def("init_decode",
             &CudaGraphTestRunner::init_decode,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("decode_capture_batch_sizes"),
             py::arg("group_tags")        = std::vector<std::string>{},
             py::arg("is_target_verify")  = false,
             py::arg("num_tokens_per_bs") = 1,
             py::arg("group_capacities")  = std::map<std::string, std::pair<int, int>>{},
             py::arg("sp_steps")          = 0)
        .def("canRun", &CudaGraphTestRunner::canRun)
        .def("clearTaggedPhysicalBlockTable",
             &CudaGraphTestRunner::clearTaggedPhysicalBlockTable,
             py::arg("inputs"),
             py::arg("tag"),
             py::arg("device"))
        .def("forward", &CudaGraphTestRunner::forward)
        .def("getCurrentRealGraphSize", &CudaGraphTestRunner::getCurrentRealGraphSize);
}
