#include <mutex>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);

namespace test {

class V41NativeEngineTest {
public:
    V41NativeEngineTest(py::object py_model,
                       ModelConfig config,
                       const ParallelismConfig& parallelism,
                       torch::Tensor lm_head,
                       bool pdfusion = false) {
        static std::once_flag runtime_once;
        std::call_once(runtime_once, [&]() {
            AUTIL_ROOT_LOG_CONFIG();
            initRuntime(static_cast<int>(parallelism.local_rank), false, false, MlaOpsType::AUTO);
        });
        RTP_LLM_CHECK(config.num_layers == 40 && config.gen_num_per_cycle == 0);
        max_seq_len_ = static_cast<size_t>(config.max_seq_len);
        config.model_type = "deepseek_v41";
        config.input_vocab_size = config.vocab_size;
        config.special_tokens.eos_token_id = -1;
        config.data_type = DataType::TYPE_BF16;
        EngineInitParams params;
        params.model_id = 0;
        params.model_config_ = std::move(config);
        params.parallelism_config = parallelism;
        params.pd_sep_config.role_type = pdfusion ? RoleType::PDFUSION : RoleType::PREFILL;
        params.runtime_config.warm_up = false;
        params.runtime_config.max_generate_batch_size = 2;
        params.runtime_config.fifo_scheduler_config.max_context_batch_size = 2;
        params.runtime_config.fifo_scheduler_config.max_batch_tokens_size = max_seq_len_;
        params.runtime_config.use_batch_decode_scheduler = false;
        params.kv_cache_config.test_block_num = 8 + parallelism.world_rank;
        params.kv_cache_config.seq_size_per_block = params.kv_cache_config.kernel_seq_size_per_block = 256;
        params.kv_cache_config.reuse_cache = false;
        params.hw_kernel_config.enable_cuda_graph = false;
        params.gpt_weights.layers.resize(40);
        auto head = std::make_shared<DenseWeights>();
        head->kernel = std::move(lm_head);
        params.gpt_weights.lm_head = std::move(head);
        params.py_model = std::move(py_model);
        params.py_eplb = params.py_sp_model = params.weight_manager = py::none();
        engine_ = std::make_unique<NormalEngine>(params, nullptr);
        const auto status = engine_->waitStartupResult(std::chrono::seconds(30));
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
        free_before_ = engine_->getCacheManager()->freeBlocksNum();
    }

    void enqueue(const std::vector<std::vector<int32_t>>& requests, int new_tokens = 1) {
        RTP_LLM_CHECK(new_tokens > 0 && new_tokens <= 16);
        for (const auto& tokens : requests) {
            RTP_LLM_CHECK(!tokens.empty() && tokens.size() + new_tokens - 1 <= 512
                          && tokens.size() + new_tokens <= max_seq_len_);
            auto input = std::make_shared<GenerateInput>();
            input->input_ids = torch::tensor(tokens, torch::kInt32);
            input->request_id = ++request_id_;
            input->generate_config = std::make_shared<GenerateConfig>();
            input->generate_config->max_new_tokens = new_tokens;
            input->generate_config->top_k = 1;
            input->generate_config->is_streaming = false;
            streams_.push_back(engine_->enqueue(input));
        }
    }

    py::dict outputs() {
        {
            py::gil_scoped_release release;
            for (const auto& stream : streams_) {
                auto output = stream->nextOutput(60000);
                if (!output.ok()) {
                    throw std::runtime_error("V4.1 engine did not return the requested generation output");
                }
                RTP_LLM_CHECK(output.value().generate_outputs.size() == 1);
                RTP_LLM_CHECK(output.value().generate_outputs.front().aux_info.output_len
                              == stream->generateConfig()->max_new_tokens);
            }
        }
        py::list tokens;
        py::list steps;
        for (const auto& stream : streams_) {
            tokens.append(stream->completeTokenIdsVec());
            steps.append(stream->iterCount());
        }
        streams_.clear();
        py::dict result;
        result["tokens"] = std::move(tokens);
        result["steps"] = std::move(steps);
        return result;
    }

    std::vector<std::string> errors() {
        py::gil_scoped_release release;
        std::vector<std::string> messages;
        for (const auto& stream : streams_) {
            auto output = stream->nextOutput(60000);
            RTP_LLM_CHECK(!output.ok());
            RTP_LLM_CHECK(output.status().code() == ErrorCode::INVALID_PARAMS);
            messages.push_back(output.status().ToString());
        }
        streams_.clear();
        return messages;
    }

    // The test calls this on every rank at an agreed model-forward boundary.
    // This bounds the live loop without racing different collective rounds.
    void requestStop() {
        engine_->running_.store(false);
    }

    py::dict stop() {
        {
            py::gil_scoped_release release;
            const auto status = engine_->stop();
            if (!status.ok()) {
                throw std::runtime_error(status.ToString());
            }
        }
        auto* executor = dynamic_cast<NormalExecutor*>(engine_->executor_.get());
        RTP_LLM_CHECK(executor != nullptr);
        executor->model_->releaseBuffers();
        py::dict result;
        result["free_before"] = free_before_;
        result["free_after"] = engine_->getCacheManager()->freeBlocksNum();
        result["cache_blocks"] = engine_->getCacheManager()->cacheConfig().block_num;
        engine_.reset();
        return result;
    }

private:
    std::unique_ptr<NormalEngine> engine_;
    std::vector<GenerateStreamPtr> streams_;
    int64_t request_id_{0};
    size_t free_before_{0};
    size_t max_seq_len_{0};
};

}  // namespace test
}  // namespace rtp_llm

PYBIND11_MODULE(libth_v41_native_engine_test, m) {
    rtp_llm::registerExecCtxOps(m);
    py::class_<rtp_llm::test::V41NativeEngineTest>(m, "NativeEngine")
        .def(py::init<py::object, rtp_llm::ModelConfig, const rtp_llm::ParallelismConfig&, torch::Tensor, bool>(),
             py::arg("model"), py::arg("config"), py::arg("parallelism"), py::arg("lm_head"),
             py::arg("pdfusion") = false)
        .def("enqueue", &rtp_llm::test::V41NativeEngineTest::enqueue,
             py::arg("requests"), py::arg("new_tokens") = 1)
        .def("outputs", &rtp_llm::test::V41NativeEngineTest::outputs)
        .def("errors", &rtp_llm::test::V41NativeEngineTest::errors)
        .def("request_stop", &rtp_llm::test::V41NativeEngineTest::requestStop)
        .def("stop", &rtp_llm::test::V41NativeEngineTest::stop);
}
