#include <mutex>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/V41PPAdmission.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);

namespace test {

class V41NativeEngineTest {
public:
    V41NativeEngineTest(py::object py_model,
                       ModelConfig config,
                       const ParallelismConfig& parallelism,
                       torch::Tensor lm_head,
                       bool pdfusion = false,
                       int64_t pp_chunk_size = 0) {
        static std::once_flag runtime_once;
        std::call_once(runtime_once, [&]() {
            AUTIL_ROOT_LOG_CONFIG();
            initRuntime(static_cast<int>(parallelism.local_rank), false, false, MlaOpsType::AUTO);
        });
        RTP_LLM_CHECK(config.num_layers == 40 && config.gen_num_per_cycle == 0);
        pipeline_ = pp_chunk_size > 0;
        RTP_LLM_CHECK(pp_chunk_size >= 0 && pp_chunk_size <= 256);
        if (pipeline_) {
            RTP_LLM_CHECK_WITH_INFO(!pdfusion && parallelism.pp_size == 2 && parallelism.tp_size == 4
                                       && parallelism.ep_size == 4 && parallelism.dp_size == 1
                                       && parallelism.world_size == 8 && parallelism.local_cp_enabled()
                                       && (parallelism.pp_stage_layer_counts == std::vector<int64_t>{21, 19}),
                                   "cold V4.1 PP fixture requires the exact CEP4PP2 PREFILL profile");
        } else {
            RTP_LLM_CHECK(parallelism.pp_size == 1);
        }
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
        if (pipeline_) {
            auto& fifo = params.runtime_config.fifo_scheduler_config;
            fifo.cp_force_single_prefill = false;
            fifo.enable_fast_gen = true;
            fifo.fast_gen_max_context_len = pp_chunk_size;
            fifo.max_batch_tokens_size = 512;
        }
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

    std::vector<int64_t> enqueue(const std::vector<std::vector<int32_t>>& requests,
                                 int new_tokens = 1, bool full_outputs = false) {
        RTP_LLM_CHECK(new_tokens > 0 && new_tokens <= 16);
        RTP_LLM_CHECK(!pipeline_ || (new_tokens == 1 && engine_->parallelism_config.world_rank == 0));
        std::vector<std::shared_ptr<GenerateInput>> inputs;
        std::vector<int64_t> ids;
        for (const auto& tokens : requests) {
            RTP_LLM_CHECK(!tokens.empty() && tokens.size() + new_tokens - 1 <= 512
                          && tokens.size() + new_tokens <= max_seq_len_);
            RTP_LLM_CHECK(!pipeline_ || tokens.size() <= 256);
            auto input = std::make_shared<GenerateInput>();
            input->input_ids = torch::tensor(tokens, torch::kInt32);
            input->request_id = ++request_id_;
            input->generate_config = std::make_shared<GenerateConfig>();
            input->generate_config->max_new_tokens = new_tokens;
            input->generate_config->top_k = 1;
            input->generate_config->is_streaming = false;
            if (pipeline_ || full_outputs) {
                input->generate_config->return_logits = true;
                input->generate_config->return_hidden_states = true;
                input->generate_config->return_all_hidden_states = full_outputs;
            }
            ids.push_back(input->request_id);
            if (pipeline_) {
                inputs.push_back(std::move(input));
            } else {
                streams_.push_back(engine_->enqueue(input));
            }
        }
        if (pipeline_) {
            auto admitted = engine_->enqueueMultiple(inputs);
            RTP_LLM_CHECK(admitted.first.size() == inputs.size() && admitted.second.size() == inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                RTP_LLM_CHECK(admitted.first[i]);
                streams_.push_back(std::move(admitted.second[i]));
            }
        }
        return ids;
    }

    py::dict outputs() {
        std::vector<GenerateOutputs> generated;
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
                generated.push_back(std::move(output.value()));
            }
        }
        py::list tokens;
        py::list steps;
        py::list rows;
        for (size_t i = 0; i < streams_.size(); ++i) {
            const auto& stream = streams_[i];
            tokens.append(stream->completeTokenIdsVec());
            steps.append(stream->iterCount());
            const auto& output = generated[i].generate_outputs.front();
            py::dict row;
            row["request_id"] = stream->streamId();
            row["tokens"] = stream->completeTokenIdsVec();
            row["outstanding"] = stream->ppOutstandingResults();
            row["hidden"] = output.hidden_states;
            row["logits"] = output.logits;
            row["all_hidden"] = output.all_hidden_states;
            rows.append(std::move(row));
        }
        streams_.clear();
        py::dict result;
        result["tokens"] = std::move(tokens);
        result["steps"] = std::move(steps);
        result["outputs"] = std::move(rows);
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
        py::dict result;
        if (pipeline_) {
            auto* executor = dynamic_cast<PPExecutor*>(engine_->executor_.get());
            RTP_LLM_CHECK(executor != nullptr && executor->shutdownCompleted());
            result["shutdown"] = executor->shutdownCompleted();
            result["active_requests"] = executor->v41_pp_admission_->activeRequests();
            result["identity_count"] = executor->v41_request_identities_.size();
            size_t retained = 0;
            for (auto& slot : executor->slots_) {
                executor->waitAll(slot.plan_sends, "cold engine plan drain");
                executor->waitAll(slot.activation_sends, "cold engine activation drain");
                executor->waitAll(slot.execution_result_sends, "cold engine result drain");
                retained += slot.v41_cache_batch || slot.v41_model_inputs ? 1 : 0;
            }
            result["retained_slots"] = retained;
            executor->releaseAllModelBuffers();
        } else {
            auto* executor = dynamic_cast<NormalExecutor*>(engine_->executor_.get());
            RTP_LLM_CHECK(executor != nullptr);
            executor->model_->releaseBuffers();
        }
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
    bool pipeline_{false};
};

}  // namespace test
}  // namespace rtp_llm

PYBIND11_MODULE(libth_v41_native_engine_test, m) {
    rtp_llm::registerExecCtxOps(m);
    m.def("native_lm_logits", [](const torch::Tensor& hidden, const torch::Tensor& head) {
        return torch_ext::cublas_gemm_bf16_bf16_fp32(hidden.to(torch::kBFloat16), head);
    });
    py::class_<rtp_llm::test::V41NativeEngineTest>(m, "NativeEngine")
        .def(py::init<py::object, rtp_llm::ModelConfig, const rtp_llm::ParallelismConfig&, torch::Tensor, bool, int64_t>(),
             py::arg("model"), py::arg("config"), py::arg("parallelism"), py::arg("lm_head"),
             py::arg("pdfusion") = false, py::arg("pp_chunk_size") = 0)
        .def("enqueue", &rtp_llm::test::V41NativeEngineTest::enqueue,
             py::arg("requests"), py::arg("new_tokens") = 1, py::arg("full_outputs") = false)
        .def("outputs", &rtp_llm::test::V41NativeEngineTest::outputs)
        .def("errors", &rtp_llm::test::V41NativeEngineTest::errors)
        .def("request_stop", &rtp_llm::test::V41NativeEngineTest::requestStop)
        .def("stop", &rtp_llm::test::V41NativeEngineTest::stop);
}
