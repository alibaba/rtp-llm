#include <mutex>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/normal_engine/Dsv4DpPrefillPolicy.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm::test {
namespace {

CacheConfig smallV41CacheConfig(int gen_num_per_cycle) {
    // Same six-layer source schedule as the Python worker fixture. Byte
    // geometry includes only the configured generation slack.
    RTP_LLM_CHECK(gen_num_per_cycle == 0 || gen_num_per_cycle == 1);
    const int swa_rows = (128 + gen_num_per_cycle + 1) & ~1;
    const int state_rows = (2 + gen_num_per_cycle + 1) & ~1;
    const int swa_bytes = ((swa_rows * 528 + 511) / 512) * 512;
    CacheConfig config;
    config.layer_num = config.layer_all_num = 6;
    config.block_num = 8;
    config.seq_size_per_block = config.kernel_seq_size_per_block = 256;
    config.use_independent_block_pools = true;
    config.use_opaque_kv_cache_store = true;
    config.fromGroupedSpecs(
        {makeResolvedOpaqueSpec(true, "swa_kv", DataType::TYPE_UINT8, swa_bytes, 256),
         makeResolvedOpaqueSpec(true, "csa_state", DataType::TYPE_FP32, state_rows * 4096, 256),
         makeResolvedOpaqueSpec(false, "csa_kv", DataType::TYPE_UINT8, 36864, 256),
         makeResolvedOpaqueSpec(false, "hca_kv", DataType::TYPE_UINT8, 73728, 256),
         makeResolvedOpaqueSpec(false, "indexer_kv", DataType::TYPE_UINT8, 17408, 256)},
        {{0, 1, 2, 3, 4, 5}, {2}, {2}, {4}, {2, 4}},
        {CacheGroupType::SWA, CacheGroupType::SWA, CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::FULL},
        {"swa_kv", "csa_state", "csa_kv", "hca_kv", "indexer_kv"});
    auto groups = config.topology().groups();
    for (auto& group : groups) {
        group.policy.cp_mapping = CpBlockMappingMode::NONE;
        if (group.policy.group_type == CacheGroupType::SWA) {
            group.policy.single_block_per_request = true;
            group.policy.enable_prefix_reuse = false;
        }
    }
    config.setTopology(std::move(groups), config.topology().layers());
    return config;
}

class V41NativePrefillTest {
public:
    V41NativePrefillTest(py::object py_model,
                        ModelConfig model_config,
                        const ParallelismConfig& parallelism,
                        torch::Tensor lm_head):
        model_config_(std::move(model_config)) {
        static std::once_flag runtime_once;
        std::call_once(runtime_once, [&]() {
            initRuntime(static_cast<int>(parallelism.local_rank), false, false, MlaOpsType::AUTO);
        });
        model_config_.model_type = "deepseek_v41";
        model_config_.input_vocab_size = model_config_.vocab_size;
        model_config_.special_tokens.eos_token_id = -1;
        model_config_.data_type = DataType::TYPE_BF16;
        model_config_.attn_config.tokens_per_block = 256;
        model_config_.attn_config.kernel_tokens_per_block = 256;
        CacheConfig config;
        if (model_config_.num_layers == 40) {
            RTP_LLM_CHECK(model_config_.gen_num_per_cycle == 0);
            KVCacheConfig cache_options;
            cache_options.test_block_num = 8;
            cache_options.seq_size_per_block = cache_options.kernel_seq_size_per_block = 256;
            RuntimeConfig runtime;
            runtime.max_generate_batch_size = 2;
            runtime.fifo_scheduler_config.max_context_batch_size = 2;
            config = CacheConfigCreator::createConfig(
                model_config_, parallelism, runtime, cache_options, std::nullopt, SpeculativeExecutionConfig{});
        } else {
            config = smallV41CacheConfig(model_config_.gen_num_per_cycle);
        }
        manager_ = std::make_shared<KVCacheManager>(config);
        if (!manager_->init()) {
            throw std::runtime_error("V4.1 test cache manager initialization failed");
        }
        EngineInitParams params;
        params.model_id = 0;
        params.model_config_ = model_config_;
        params.parallelism_config = parallelism;
        params.pd_sep_config.role_type = RoleType::PREFILL;
        params.runtime_config.max_generate_batch_size = 2;
        params.runtime_config.fifo_scheduler_config.max_context_batch_size = 2;
        params.hw_kernel_config.enable_cuda_graph = false;
        params.gpt_weights.layers.resize(model_config_.num_layers);
        auto head = std::make_shared<DenseWeights>();
        head->kernel = std::move(lm_head);
        params.gpt_weights.lm_head = std::move(head);
        params.py_model = std::move(py_model);
        params.py_eplb = params.py_sp_model = params.weight_manager = py::none();
        executor_ = std::make_unique<NormalExecutor>(params, manager_, false);
    }

    py::dict run(const std::vector<std::vector<int32_t>>& requests, py::function reduce_any_prefill) {
        RTP_LLM_CHECK(requests.size() <= 2);
        const auto free_before = manager_->freeBlocksNum();
        ScheduleOutput schedule;
        for (const auto& tokens : requests) {
            RTP_LLM_CHECK(!tokens.empty() && tokens.size() <= 256);
            schedule.streams.push_back(makeStream(tokens, false));
        }
        const bool selected = alignDsv4DpPrefillPhase(
            schedule.streams,
            [&](bool local) { return reduce_any_prefill(local).cast<bool>(); },
            [&]() { return makeStream({0}, true); });
        // Match NormalEngine's all-idle PREFILL fallback.
        if (schedule.streams.empty()) {
            schedule.streams.push_back(makeStream({0}, true));
        }
        auto status = executor_->process(schedule);
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
        py::list tokens_out;
        py::list steps;
        for (const auto& stream : schedule.streams) {
            steps.append(stream->iterCount());
            if (!stream->isFakeStream()) {
                tokens_out.append(stream->completeTokenIdsVec());
                stream->releaseResource();
            }
        }
        // Drain the wrapper's last input leases before inspecting reclamation.
        executor_->model_->releaseBuffers();
        py::dict result;
        result["tokens"] = std::move(tokens_out);
        result["steps"] = std::move(steps);
        result["selected_prefill"] = selected;
        result["free_before"] = free_before;
        result["free_after"] = manager_->freeBlocksNum();
        return result;
    }

    py::dict runDecode(const std::vector<std::vector<int32_t>>& requests, int new_tokens) {
        RTP_LLM_CHECK(requests.size() <= 2 && new_tokens > 1 && new_tokens <= 16);
        RTP_LLM_CHECK(!executor_->useStreamAsync());
        const auto free_before = manager_->freeBlocksNum();
        std::list<GenerateStreamPtr> live;
        for (const auto& tokens : requests) {
            RTP_LLM_CHECK(!tokens.empty() && tokens.size() + new_tokens - 1 <= 256);
            live.push_back(makeStream(tokens, false, new_tokens));
        }
        for (int step = 0; step < new_tokens; ++step) {
            ScheduleOutput schedule;
            schedule.streams = live;
            if (schedule.streams.empty()) {
                auto fake = makeStream({0}, true);
                fake->setIsContextStream(step == 0);
                schedule.streams.push_back(std::move(fake));
            }
            auto status = executor_->process(schedule);
            if (!status.ok()) {
                throw std::runtime_error(status.ToString());
            }
            for (const auto& stream : schedule.streams) {
                RTP_LLM_CHECK(stream->iterCount() == (stream->isFakeStream() ? 0 : step + 1));
            }
        }
        py::list tokens_out;
        for (const auto& stream : live) {
            tokens_out.append(stream->completeTokenIdsVec());
            stream->releaseResource();
        }
        executor_->model_->releaseBuffers();
        py::dict result;
        result["tokens"] = std::move(tokens_out);
        result["free_before"] = free_before;
        result["free_after"] = manager_->freeBlocksNum();
        return result;
    }

private:
    GenerateStreamPtr makeStream(const std::vector<int32_t>& tokens, bool fake, int new_tokens = 1) {
        ResourceContext context;
        context.cache_manager = manager_;
        context.reuse_cache = false;
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::tensor(tokens, torch::kInt32);
        input->request_id = ++request_id_;
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = new_tokens;
        input->generate_config->top_k = 1;
        input->fake_query = fake;
        auto stream = std::make_shared<NormalGenerateStream>(input, model_config_, RuntimeConfig{}, context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setIsFakeStream(fake);
        stream->setIsContextStream(true);
        if (fake) {
            stream->fakeInitKVBlock();
            stream->initFakeContextChunk();
        } else {
            auto status = stream->streamCacheResource().initKVBlock();
            if (!status.ok()) {
                throw std::runtime_error(status.ToString());
            }
        }
        return stream;
    }

    ModelConfig model_config_;
    std::shared_ptr<KVCacheManager> manager_;
    std::unique_ptr<NormalExecutor> executor_;
    int64_t request_id_{0};
};

}  // namespace
}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_v41_native_prefill_test, m) {
    py::class_<rtp_llm::test::V41NativePrefillTest>(m, "NativePrefill")
        .def(py::init<py::object, rtp_llm::ModelConfig, const rtp_llm::ParallelismConfig&, torch::Tensor>())
        .def("run", &rtp_llm::test::V41NativePrefillTest::run)
        .def("run_decode", &rtp_llm::test::V41NativePrefillTest::runDecode);
}
