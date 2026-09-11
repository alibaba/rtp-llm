#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {
namespace {

struct LifecycleCalls {
    bool              has_cache                   = false;
    int               cacheless_prefill_forwards  = 0;
    int               cache_backed_graph_prepares = 0;
    int               forwards                    = 0;
    std::vector<bool> initialize_with_cache;
};

class PyWrappedModelWarmupTest: public DeviceTestBase {
protected:
    static void SetUpTestSuite() {
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
        }
        py::gil_scoped_acquire gil;
        auto                   module = py::reinterpret_steal<py::module_>(PyModule_New("_rtp_warmup_lifecycle_test"));
        py::class_<torch_ext::PyModelInitResources>(module, "ModelInitResources");
        py::class_<torch_ext::PyModelInputs>(module, "ModelInputs");
        py::class_<torch_ext::PyModelOutputs>(module, "ModelOutputs");
        py::class_<torch_ext::PyAttentionInputs>(module, "AttentionInputs");
        py::module_::import("sys").attr("modules")[module.attr("__name__")] = module;
    }

    py::object makePythonModel(const std::shared_ptr<LifecycleCalls>& calls) {
        py::gil_scoped_acquire gil;
        auto                   model         = py::module_::import("types").attr("SimpleNamespace")();
        auto                   attention     = py::module_::import("types").attr("SimpleNamespace")();
        attention.attr("prepare_cuda_graph") = py::cpp_function([](const torch_ext::PyAttentionInputs&) {});
        model.attr("initialize")        = py::cpp_function([calls](const torch_ext::PyModelInitResources& resources) {
            calls->has_cache = resources.kv_cache.has_value();
            if (calls->has_cache) {
                const auto& cache = *resources.kv_cache;
                if (cache.kv_cache_base_by_layer.empty() || !cache.kv_cache_base_by_layer[0].is_cuda()) {
                    throw std::runtime_error("lifecycle fixture requires allocated CUDA KV storage");
                }
            }
            calls->initialize_with_cache.push_back(calls->has_cache);
            return true;
        });
        model.attr("prepare_fmha_impl") = py::cpp_function(
            [calls, attention](const torch_ext::PyModelInputs& inputs, bool graph) {
                if (graph && !calls->has_cache) {
                    throw std::runtime_error("graph preparation ran before KV allocation");
                }
                if (!inputs.attention_inputs.is_prefill && !calls->has_cache) {
                    throw std::runtime_error("decode preparation ran without KV storage");
                }
                if (graph) {
                    ++calls->cache_backed_graph_prepares;
                }
                return attention;
            },
            py::arg("inputs"),
            py::arg("graph") = false);
        model.attr("forward") = py::cpp_function(
            [calls](const torch_ext::PyModelInputs& inputs, py::object) {
                if (!calls->has_cache) {
                    if (!inputs.attention_inputs.is_prefill) {
                        throw std::runtime_error("decode forward ran without KV storage");
                    }
                    ++calls->cacheless_prefill_forwards;
                }
                ++calls->forwards;
                // Deterministic GPU arithmetic stands in for model layers;
                // PyWrappedModel, KV allocation and CUDA capture/replay are real.
                auto hidden = inputs.input_ids.to(torch::kBFloat16)
                                  .unsqueeze(1)
                                  .expand({inputs.input_ids.numel(), 128})
                                  .contiguous()
                                  .add(1);
                return torch_ext::PyModelOutputs(hidden);
            },
            py::arg("inputs"),
            py::arg("fmha_impl") = py::none());
        return model;
    }

    GptModelInitParams cachelessParams(bool memory_warmup) {
        ModelConfig model;
        model.hidden_size                  = 128;
        model.vocab_size                   = 100;
        model.num_layers                   = 2;
        model.attn_config.head_num         = 2;
        model.attn_config.kv_head_num      = 2;
        model.attn_config.size_per_head    = 64;
        model.attn_config.tokens_per_block = 128;
        model.data_type                    = DataType::TYPE_BF16;
        const auto description = Executor::genModelDescription(model, ParallelismConfig{}, EPLBConfig{}, MoeConfig{});
        GptModelInitParams params({Weights{}, description, std::nullopt});
        params.prefill_memory_warmup                           = memory_warmup;
        params.hw_kernel_config.enable_cuda_graph              = true;
        params.hw_kernel_config.decode_capture_batch_sizes     = {1, 2};
        params.max_seq_len                                     = 32;
        params.hidden_size                                     = 128;
        params.tokens_per_block                                = 128;
        params.kernel_tokens_per_block                         = 128;
        params.concurrency_config.concurrency_limit            = 2;
        params.device_resource_config.enable_layer_micro_batch = 0;
        return params;
    }
};

TEST_F(PyWrappedModelWarmupTest, EnginePrefillMemoryWarmupThenRealDecodeCaptureAndReplay) {
    py::gil_scoped_acquire gil;
    ModelConfig            model;
    RuntimeConfig          runtime;
    KVCacheConfig          kv;
    auto                   params                     = createEngineInitParams(CustomConfig{}, model, runtime, kv);
    params.model_config_.data_type                    = DataType::TYPE_BF16;
    params.model_config_.attn_config.tokens_per_block = 128;
    params.kv_cache_config.seq_size_per_block         = 128;
    params.kv_cache_config.test_block_num             = 5;
    params.runtime_config.warm_up                     = true;
    params.runtime_config.model_warm_up               = true;
    params.runtime_config.max_generate_batch_size     = 2;
    params.runtime_config.fifo_scheduler_config.max_context_batch_size = 1;
    params.concurrency_config.concurrency_limit                        = 2;
    params.hw_kernel_config.enable_cuda_graph                          = true;
    params.hw_kernel_config.decode_capture_batch_sizes                 = {1, 2};
    params.device_resource_config.enable_layer_micro_batch             = 0;
    params.gpt_weights                                                 = Weights{};
    const auto calls                                                   = std::make_shared<LifecycleCalls>();
    params.py_model                                                    = makePythonModel(calls);

    NormalEngine engine(params, nullptr);
    ASSERT_GE(calls->initialize_with_cache.size(), 2u);
    EXPECT_FALSE(calls->initialize_with_cache.front());
    for (size_t index = 1; index < calls->initialize_with_cache.size(); ++index) {
        EXPECT_TRUE(calls->initialize_with_cache[index]);
    }
    EXPECT_GT(calls->cacheless_prefill_forwards, 0);
    EXPECT_GT(calls->cache_backed_graph_prepares, 0);
    ASSERT_NE(engine.resourceContext().cache_manager, nullptr);
    auto* executor = dynamic_cast<NormalExecutor*>(engine.executor_.get());
    ASSERT_NE(executor, nullptr);
    auto* wrapped = dynamic_cast<PyWrappedModel*>(executor->model_.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_TRUE(wrapped->enable_cuda_graph_);
    EXPECT_FALSE(wrapped->prefill_memory_warmup_);
    auto* runner = dynamic_cast<CudaGraphRunner*>(wrapped->graph_runner_);
    ASSERT_NE(runner, nullptr);
    ASSERT_EQ(runner->graph_instances_.size(), 2u);

    auto& held = runner->graph_instances_.at(1).mem_hold_;
    held.py_model_inputs_.input_ids.fill_(7);
    const int before_replay = calls->forwards;
    runner->replayGraph(1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(calls->forwards, before_replay);
    EXPECT_TRUE(held.decoder_layer_hidden_states_.eq(8).all().item<bool>());
    held.py_model_inputs_.input_ids.fill_(11);
    runner->replayGraph(1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(calls->forwards, before_replay);
    EXPECT_TRUE(held.decoder_layer_hidden_states_.eq(12).all().item<bool>());
}

TEST_F(PyWrappedModelWarmupTest, TemporaryWarmupCannotServeRequestsOrDecode) {
    py::gil_scoped_acquire gil;
    auto                   params = cachelessParams(true);
    auto                   calls  = std::make_shared<LifecycleCalls>();
    PyWrappedModel         wrapped(params, makePythonModel(calls));
    EXPECT_TRUE(wrapped.enable_cuda_graph_);
    EXPECT_EQ(wrapped.graph_runner_, nullptr);
    EXPECT_EQ(calls->cache_backed_graph_prepares, 0);
    GptModelInputs input;
    input.sequence_lengths = torch::empty({0}, torch::kInt32);
    input.warmup           = false;
    EXPECT_THROW(wrapped.forward(input), std::exception);
    EXPECT_THROW(wrapped.forwardMicroBatched(input), std::exception);
    EXPECT_THROW(wrapped.prepareAttentionInputs(input), std::exception);
    input.warmup           = true;
    input.sequence_lengths = torch::ones({1}, torch::kInt32);
    EXPECT_THROW(wrapped.forward(input), std::exception);
    EXPECT_THROW(wrapped.prepareAttentionInputs(input), std::exception);
}

TEST_F(PyWrappedModelWarmupTest, MissingServingCacheOrGeometryDoesNotSilentlyDisableGraph) {
    py::gil_scoped_acquire gil;
    auto                   params = cachelessParams(false);
    auto                   calls  = std::make_shared<LifecycleCalls>();
    EXPECT_THROW(PyWrappedModel(params, makePythonModel(calls)), std::exception);
    params.prefill_memory_warmup = true;
    params.tokens_per_block      = 0;
    EXPECT_THROW(PyWrappedModel(params, makePythonModel(calls)), std::exception);
    EXPECT_EQ(calls->cache_backed_graph_prepares, 0);
}

}  // namespace
}  // namespace rtp_llm
