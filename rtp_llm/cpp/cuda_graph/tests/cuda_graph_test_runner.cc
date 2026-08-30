#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace py = pybind11;
namespace rtp_llm {

// Single wrapper for both prefill and decode tests; init_prefill / init_decode
// build GraphParams and call CudaGraphRunner factory methods.
// Plain pybind11 class (no torch::jit::CustomClassHolder) so the module loads without
// depending on torch's registered CustomClassHolder type.
class CudaGraphTestRunner {
public:
    static std::shared_ptr<const CacheConfig> makeCacheConfig(int64_t                  tokens_per_block,
                                                              int64_t                  kernel_tokens_per_block,
                                                              std::vector<std::string> group_tags,
                                                              std::vector<int64_t>     group_tokens_per_block,
                                                              std::vector<int64_t>     group_kernel_tokens_per_block) {
        constexpr uint32_t           kTestBlockNum = 17;
        std::shared_ptr<CacheConfig> config;
        const bool uses_group_geometry = !group_tokens_per_block.empty() || !group_kernel_tokens_per_block.empty();
        RTP_LLM_CHECK_WITH_INFO(!uses_group_geometry || !group_tags.empty(),
                                "per-group cache geometry requires explicit group tags");
        if (group_tags.empty()) {
            group_tags.push_back("default");
        }
        if (uses_group_geometry) {
            RTP_LLM_CHECK_WITH_INFO(group_tokens_per_block.size() == group_tags.size(),
                                    "physical geometry count=%zu does not match group tag count=%zu",
                                    group_tokens_per_block.size(),
                                    group_tags.size());
            RTP_LLM_CHECK_WITH_INFO(group_kernel_tokens_per_block.size() == group_tags.size(),
                                    "kernel geometry count=%zu does not match group tag count=%zu",
                                    group_kernel_tokens_per_block.size(),
                                    group_tags.size());
        } else {
            group_tokens_per_block.assign(group_tags.size(), tokens_per_block);
            group_kernel_tokens_per_block.assign(group_tags.size(), kernel_tokens_per_block);
        }
        std::vector<CacheGroup> groups;
        CacheLayer              layer;
        for (size_t group_idx = 0; group_idx < group_tags.size(); ++group_idx) {
            const auto& tag                 = group_tags[group_idx];
            auto        spec                = std::make_shared<MHAKVCacheSpec>();
            spec->seq_size_per_block        = static_cast<uint32_t>(group_tokens_per_block[group_idx]);
            spec->kernel_seq_size_per_block = static_cast<uint32_t>(group_kernel_tokens_per_block[group_idx]);
            CacheGroup group;
            group.tag               = tag;
            group.spec              = std::move(spec);
            group.policy.group_type = CacheGroupType::FULL;
            group.block_num         = kTestBlockNum;
            groups.push_back(std::move(group));
            layer.push_back(tag);
        }
        config = std::make_shared<CacheConfig>(
            std::move(groups), std::vector<CacheLayer>{std::move(layer)}, /*main_layer_num=*/1);
        config->block_num          = kTestBlockNum;
        config->seq_size_per_block = static_cast<size_t>(tokens_per_block);
        return config;
    }

    void init_prefill(py::object               py_instance,
                      int64_t                  max_context_batch_size,
                      int64_t                  max_seq_len,
                      int64_t                  tokens_per_block,
                      int64_t                  kernel_tokens_per_block,
                      std::vector<int>         prefill_capture_seq_lens,
                      int64_t                  hidden_size,
                      std::vector<std::string> group_tags,
                      std::vector<int64_t>     group_tokens_per_block,
                      std::vector<int64_t>     group_kernel_tokens_per_block) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = true;
        params.is_prefill_cuda_graph_mode   = true;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.cache_config                 = makeCacheConfig(tokens_per_block,
                                              kernel_tokens_per_block,
                                              group_tags,
                                              group_tokens_per_block,
                                              group_kernel_tokens_per_block);
        cache_config_                       = params.cache_config;
        params.num_tokens_per_bs            = static_cast<int>(max_seq_len);
        params.max_context_batch_size       = static_cast<size_t>(max_context_batch_size);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.input_hidden_size            = static_cast<size_t>(hidden_size);
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.prefill_capture_seq_lens     = std::move(prefill_capture_seq_lens);

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
                     int64_t                  num_tokens_per_bs,
                     std::vector<int64_t>     group_tokens_per_block,
                     std::vector<int64_t>     group_kernel_tokens_per_block) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = false;
        params.is_prefill_cuda_graph_mode   = false;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.cache_config                 = makeCacheConfig(tokens_per_block,
                                              kernel_tokens_per_block,
                                              group_tags,
                                              group_tokens_per_block,
                                              group_kernel_tokens_per_block);
        cache_config_                       = params.cache_config;
        params.input_hidden_size            = static_cast<size_t>(hidden_size);
        params.num_tokens_per_bs            = static_cast<int>(num_tokens_per_bs);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.max_context_batch_size       = 128;
        params.decode_capture_batch_sizes   = std::move(decode_capture_batch_sizes);
        params.is_target_verify             = is_target_verify;

        runner_ = CudaGraphRunner::createForDecode(std::move(py_instance), std::move(params));
    }

    bool canRun(torch_ext::PyModelInputs& inputs) {
        return runner_ != nullptr && runner_->canRun(inputs, state_);
    }

    torch_ext::PyModelOutputs forward(torch_ext::PyModelInputs& inputs) {
        // Production PyWrappedModel creates these device mirrors. Python tests
        // cannot assign them because the bindings intentionally expose them as
        // read-only, so reproduce that input-building step in the test wrapper.
        inputs.attention_inputs.input_lengths_device  = inputs.attention_inputs.input_lengths.cuda();
        inputs.attention_inputs.prefix_lengths_device = inputs.attention_inputs.prefix_lengths.cuda();
        refreshGroupAttentionInputs(inputs);
        return runner_->forward(inputs, state_);
    }

    int getCurrentRealGraphSize() {
        return runner_ != nullptr ? runner_->getCurrentRealGraphBs(state_) : 0;
    }

    size_t getGroupKernelBlockRatio(const std::string& tag) const {
        RTP_LLM_CHECK_WITH_INFO(cache_config_ != nullptr, "CUDA graph test runner is not initialized");
        return cache_config_->group(tag).storedKernelBlocksPerKvBlock();
    }

    ~CudaGraphTestRunner() {
        reset_runner();
    }

private:
    void reset_runner() {
        if (runner_ != nullptr) {
            delete runner_;
            runner_ = nullptr;
        }
        cache_config_.reset();
    }

    CudaGraphRunner*                   runner_ = nullptr;
    CudaGraphState                     state_{};
    std::shared_ptr<const CacheConfig> cache_config_;
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
             py::arg("group_tags")                    = std::vector<std::string>{},
             py::arg("group_tokens_per_block")        = std::vector<int64_t>{},
             py::arg("group_kernel_tokens_per_block") = std::vector<int64_t>{})
        .def("init_decode",
             &CudaGraphTestRunner::init_decode,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("decode_capture_batch_sizes"),
             py::arg("group_tags")                    = std::vector<std::string>{},
             py::arg("is_target_verify")              = false,
             py::arg("num_tokens_per_bs")             = 1,
             py::arg("group_tokens_per_block")        = std::vector<int64_t>{},
             py::arg("group_kernel_tokens_per_block") = std::vector<int64_t>{})
        .def("canRun", &CudaGraphTestRunner::canRun)
        .def("forward", &CudaGraphTestRunner::forward)
        .def("getCurrentRealGraphSize", &CudaGraphTestRunner::getCurrentRealGraphSize)
        .def("getGroupKernelBlockRatio", &CudaGraphTestRunner::getGroupKernelBlockRatio);
}
