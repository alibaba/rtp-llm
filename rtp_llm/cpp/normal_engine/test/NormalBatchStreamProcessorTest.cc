#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <list>
#include "rtp_llm/cpp/cache/test/TestLayoutSpec.h"
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include "torch/csrc/autograd/profiler_kineto.h"
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

using namespace std;

namespace rtp_llm {

template<typename T>
std::vector<T> toVec(const torch::Tensor& t) {
    auto c = t.is_cuda() ? t.cpu().contiguous() : t.contiguous();
    return std::vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

static torch::Tensor hostIntBuffer(std::vector<int32_t> data) {
    return torch::tensor(data, torch::kInt32);
}

static void initFullCacheConfig(CacheConfig& cache_config, int layer_num) {
    auto spec = std::make_shared<MHAKVCacheSpec>();
    spec->tag = "default";
    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.layer_num = static_cast<uint32_t>(layer_num);

    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
}

class NormalBatchStreamProcessorTest: public DeviceTestBase {
protected:
    static EngineInitParams makeExecutorParams(int worker_count) {
        EngineInitParams params;
        params.model_id                                          = 0;
        params.model_config_.max_seq_len                         = 128;
        params.model_config_.vocab_size                          = 128;
        params.model_config_.input_vocab_size                    = 128;
        params.model_config_.num_layers                          = 1;
        params.model_config_.attn_config.head_num                = 2;
        params.model_config_.attn_config.kv_head_num             = 2;
        params.model_config_.attn_config.size_per_head           = 64;
        params.model_config_.hidden_size                         = 128;
        params.model_config_.attn_config.tokens_per_block        = 2;
        params.model_config_.attn_config.kernel_tokens_per_block = 2;
        params.runtime_config.output_dispatcher_worker_count     = worker_count;
        params.py_model                                          = py::none();
        return params;
    }

    static ModelConfig makeOutputVocabModelConfig(std::vector<int64_t> output_vocab_ids = {0, 2, 7},
                                                  int64_t              padded_size      = 0) {
        ModelConfig model_config;
        model_config.max_seq_len      = 8;
        model_config.vocab_size       = 10;
        model_config.num_layers       = 1;
        model_config.output_vocab_ids = std::move(output_vocab_ids);
        model_config.output_vocab_padded_size =
            padded_size > 0 ? padded_size : static_cast<int64_t>(model_config.output_vocab_ids.size());
        return model_config;
    }
};

class OutputDispatchTest: public NormalBatchStreamProcessorTest, public ::testing::WithParamInterface<int> {};

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, OutputDispatchTest, ::testing::Values(0, 2));

TEST_P(OutputDispatchTest, testDispatchPreservesCpuProfilingAndAsyncDisableGuard) {
    namespace tap = torch::autograd::profiler;
    namespace tpi = torch::profiler::impl;

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 1;
    RuntimeConfig          runtime_config;
    NormalOutputDispatcher dispatcher({}, GetParam());
    AsyncRunner            runner(cuda_graph::graphGetStreamFromPool(true));

    // Reuse the same workers across profiler sessions and the disabled async
    // path, checking that TLS is both installed and restored for each task.
    for (bool stream_async : {false, true, false}) {
        SCOPED_TRACE(stream_async);
        std::list<GenerateStreamPtr> streams;
        for (int input_token : {0, 1}) {
            auto query             = make_shared<GenerateInput>();
            query->input_ids       = hostIntBuffer({input_token});
            query->generate_config = make_shared<GenerateConfig>();
            auto stream =
                make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
            stream->generate_status_->status = StreamState::RUNNING;
            streams.push_back(stream);
        }
        StreamGroups stream_groups(streams);
        MergedOutput outputs;
        outputs.sampler_output.token_ids = torch::tensor({1, 0}, torch::kInt32).reshape({2, 1});
        outputs.sampler_output.success   = torch::tensor({true, true}, torch::kBool);

        tpi::ProfilerConfig               config(tpi::ProfilerState::KINETO, /*report_input_shapes=*/false);
        const std::set<tpi::ActivityType> activities{tpi::ActivityType::CPU};
        tap::prepareProfiler(config, activities);
        tap::enableProfiler(config, activities);
        absl::Status status;
        try {
            auto dispatch = [&] { status = dispatcher.dispatch(stream_groups, outputs); };
            if (stream_async) {
                runner.launch(dispatch);
                runner.sync(cuda_graph::graphGetCurrentStream());
            } else {
                dispatch();
            }
        } catch (...) {
            tap::disableProfiler();
            throw;
        }
        auto trace = tap::disableProfiler();
        ASSERT_TRUE(status.ok());
        ASSERT_NE(trace, nullptr);
        size_t update_events = 0;
        for (const auto& event : trace->events()) {
            if (event.name().find("GenerateStream::update(") != std::string::npos) {
                ++update_events;
                EXPECT_EQ(event.startThreadId() == at::RecordFunction::currentThreadId(), GetParam() == 0);
            }
        }
        EXPECT_EQ(update_events, stream_async ? 0u : 2u);
        for (const auto& stream : streams) {
            EXPECT_FALSE(stream->hasError());
            EXPECT_EQ(stream->seqLength(), 2);
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testDispatchWorkersOnlyCreatedOnOutputRank) {
    for (int rank : {0, 1}) {
        SCOPED_TRACE(rank);
        EngineInitParams params;
        params.model_config_.max_seq_len                     = 8;
        params.model_config_.vocab_size                      = 10;
        params.model_config_.num_layers                      = 1;
        params.model_config_.attn_config.head_num            = 2;
        params.model_config_.attn_config.kv_head_num         = 2;
        params.model_config_.attn_config.size_per_head       = 64;
        params.model_config_.hidden_size                     = 128;
        params.parallelism_config.tp_size                    = 2;
        params.parallelism_config.tp_rank                    = rank;
        params.runtime_config.output_dispatcher_worker_count = 2;
        params.py_model                                      = py::none();
        // Exercise the executor constructor without starting TP collectives.
        NormalExecutor executor(params, nullptr);
        const auto&    pool = executor.batch_stream_processor_->output_dispatcher_->thread_pool_;
        if (rank == 0) {
            ASSERT_NE(pool, nullptr);
            EXPECT_EQ(pool->getThreadNum(), 2u);
        } else {
            EXPECT_EQ(pool, nullptr);
        }
    }
}

TEST_P(OutputDispatchTest, testExecutorUsesGatheredModeAndPreservesCaptureErrors) {
    class CaptureModel: public ModelBase {
    public:
        GptModelOutputs forward(const GptModelInputs& inputs) override {
            ++forward_count;
            EXPECT_EQ(inputs.skip_lm_head, expected_prefill_only);
            EXPECT_EQ(inputs.capture_hidden_states, expected_prefill_only);
            GptModelOutputs output;
            output.generation_prefill_cuda_graph_status = GenerationPrefillCudaGraphStatus::REPLAYED;
            // No logits at all for target-only prefill: entering the sampler is a regression.
            if (!inputs.skip_lm_head) {
                output.logits = torch::zeros({inputs.lm_output_indexes.size(0), 128},
                                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            }
            return output;
        }

        std::optional<std::string> takeDeferredHiddenStateCaptureError() override {
            return fail_capture ? std::make_optional<std::string>("capture publication failed") : std::nullopt;
        }

        bool expected_prefill_only = true;
        bool fail_capture          = false;
        int  forward_count         = 0;
    };

    struct TestCase {
        const char* name;
        int         first_max_new_tokens;
        int         second_max_new_tokens;
        bool        first_has_error;
        bool        second_is_decode;
        bool        expected_prefill_only;
        bool        fail_capture;
    };
    const std::vector<TestCase> cases = {
        {"prefill-only", 0, 0, false, false, true, false},
        {"failed-generation-before-prefill", 1, 0, true, false, true, false},
        {"failed-prefill-before-generation", 0, 1, true, false, false, false},
        {"failed-prefill-before-decode-generation", 0, 1, true, true, false, false},
        {"decode-prefill-precedes-context-generation", 1, 0, false, true, true, false},
        {"decode-generation-precedes-context-prefill", 0, 1, false, true, false, false},
        {"capture-failure", 0, 0, false, false, true, true},
        {"capture-failure-preserves-existing-error", 1, 0, true, false, true, true},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto           params = makeExecutorParams(GetParam());
        NormalExecutor executor(params, nullptr);
        // Reuse the warmup gatherer's fake KV layout, but run normal executor dispatch.
        executor.setBatchProcessor(std::make_unique<NormalBatchStreamProcessor>(params.model_config_,
                                                                                params.pd_sep_config,
                                                                                params.profiling_debug_logging_config,
                                                                                CacheConfig(),
                                                                                true,
                                                                                GetParam()));
        auto  model                  = std::make_unique<CaptureModel>();
        auto* observed_model         = model.get();
        model->expected_prefill_only = test_case.expected_prefill_only;
        model->fail_capture          = test_case.fail_capture;
        executor.setModel(std::move(model));

        ResourceContext resource_context;
        auto            make_stream = [&](int max_new_tokens) {
            auto query                             = make_shared<GenerateInput>();
            query->input_ids                       = hostIntBuffer({1, 2});
            query->generate_config                 = make_shared<GenerateConfig>();
            query->generate_config->max_new_tokens = max_new_tokens;
            query->generate_config->top_k          = 1;
            query->generate_config->aux_info       = true;
            auto stream                            = make_shared<NormalGenerateStream>(
                query, params.model_config_, params.runtime_config, resource_context, nullptr);
            stream->generate_status_->status = StreamState::RUNNING;
            return stream;
        };
        auto first  = make_stream(test_case.first_max_new_tokens);
        auto second = make_stream(test_case.second_max_new_tokens);
        if (test_case.first_has_error) {
            first->reportError(ErrorCode::INVALID_PARAMS, "original request error");
        }
        second->setIsContextStream(!test_case.second_is_decode);
        ASSERT_TRUE(executor.process({first, second}).ok());
        EXPECT_EQ(observed_model->forward_count, 1);
        EXPECT_FALSE(first->hasPendingAsyncBookkeeping());
        EXPECT_FALSE(second->hasPendingAsyncBookkeeping());
        if (test_case.first_has_error) {
            EXPECT_EQ(first->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(first->stopReason(), "original request error");
        }
        if (test_case.fail_capture) {
            EXPECT_EQ(second->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
            EXPECT_EQ(second->stopReason(), "capture publication failed");
            EXPECT_FALSE(second->hasOutput());
        } else if (test_case.second_is_decode && !test_case.first_has_error) {
            // Mixed batches still forward for collective alignment, but may not publish success.
            EXPECT_EQ(first->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(second->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(second->stopReason(), kMixedExecutionModeBatchError);
            EXPECT_FALSE(first->hasOutput());
            EXPECT_FALSE(second->hasOutput());
        } else {
            ASSERT_FALSE(second->hasError());
            auto output = second->nextOutput();
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(output.value().generate_outputs.size(), 1);
            EXPECT_TRUE(output.value().generate_outputs[0].finished);
            EXPECT_EQ(second->outputTokenLen(), test_case.second_max_new_tokens);
            EXPECT_EQ(second->seqLength(), 2 + test_case.second_max_new_tokens);
            if (test_case.expected_prefill_only) {
                EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
                EXPECT_EQ(second->generationPrefillCudaGraphStatus(), GenerationPrefillCudaGraphStatus::REPLAYED);
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testZeroDispatchWorkerCount) {
    NormalOutputDispatcher default_dispatcher;
    EXPECT_EQ(default_dispatcher.thread_pool_, nullptr);
    NormalOutputDispatcher zero_worker_dispatcher({}, 0);
    EXPECT_EQ(zero_worker_dispatcher.thread_pool_, nullptr);
}

TEST_F(NormalBatchStreamProcessorTest, testNegativeDispatchWorkerCountIsRejected) {
    EXPECT_THROW(NormalOutputDispatcher({}, -1), std::invalid_argument);
}

TEST_F(NormalBatchStreamProcessorTest, testWarmUpWithoutCacheManager) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len      = 2048;
    model_config.vocab_size       = 2048;
    model_config.input_vocab_size = 2048;
    model_config.num_layers       = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({1, 2, 3});
    query->generate_config = make_shared<GenerateConfig>();
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    StreamGroups stream_groups({stream});

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    EXPECT_EQ(processor.model_input_gatherer_config_.kv_cache_group_nums, 0);
    EXPECT_TRUE(processor.model_input_gatherer_config_.kv_cache_group_types.empty());
    ASSERT_EQ(stream->kvCache().groupNums(), 1);
    EXPECT_EQ(stream->kvCache().cacheResource().soleGroupTagForLayer(0), "__warmup__");
    TensorHolder holder;
    auto         model_input = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input.ok());
    EXPECT_FALSE(model_input->skip_lm_head);
    EXPECT_FALSE(model_input->capture_hidden_states);
    EXPECT_FALSE(model_input->kv_cache_block_id.defined());
    EXPECT_FALSE(model_input->kv_cache_kernel_block_id.defined());
}

TEST_F(NormalBatchStreamProcessorTest, testSpeculativeReserveStepFormula) {
    SpeculativeExecutionConfig config;
    config.type = SP_TYPE_NONE;
    EXPECT_EQ(config.speculativeReserveStep(), 0);

    config.type = SP_TYPE_MTP;
    config.gen_num_per_cycle = 3;
    EXPECT_EQ(config.speculativeReserveStep(), 4);

    config.type = SP_TYPE_DSPARK;
    config.gen_num_per_cycle = 3;
    EXPECT_EQ(config.speculativeReserveStep(), 9);

    config.type = SP_TYPE_MTP;
    config.gen_num_per_cycle = std::numeric_limits<int64_t>::max();
    EXPECT_ANY_THROW((void)config.speculativeReserveStep());

    config.type = SP_TYPE_DSPARK;
    config.gen_num_per_cycle = static_cast<int64_t>(std::numeric_limits<int>::max()) / 3 + 1;
    EXPECT_ANY_THROW((void)config.speculativeReserveStep());
}

TEST_P(OutputDispatchTest, testExecutorPassesFinalGraphWidthOnlyWithLayout) {
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    };

    ModelConfig model;
    model.num_layers                          = 1;
    model.max_seq_len                         = 64;
    model.vocab_size                          = 16;
    model.hidden_size                         = 4;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 4;
    model.attn_config.tokens_per_block        = 8;
    model.attn_config.kernel_tokens_per_block = 8;

    CacheConfig cache_config = makeMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*local_head_num_kv=*/1, /*size_per_head=*/1,
        /*tokens_per_block=*/8, rtp_llm::DataType::TYPE_INT8);
    auto manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(manager->init());

    auto params                                         = makeExecutorParams(GetParam());
    params.model_config_                   = model;
    params.model_config_.hidden_state_capture_layer_ids = {0};
    params.model_config_.hidden_state_capture_fail_open = true;
    params.pd_sep_config.role_type                      = RoleType::PREFILL;
    params.py_model                        = py::none();
    params.hw_kernel_config.enable_cuda_graph = true;
    params.sp_config.type              = SP_TYPE_MTP;
    params.sp_config.gen_num_per_cycle = 1;

    struct CapturedParamsModel: ModelBase {
        GptModelOutputs forward(const GptModelInputs&) override {
            return {};
        }
    };

    int64_t captured_width      = -1;
    bool    captured_has_layout = false;
    std::vector<int64_t> captured_layer_ids;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams& init_params) {
        captured_width      = init_params.kernel_block_table_width;
        captured_has_layout = init_params.kv_cache_layer_layout.has_value();
        captured_layer_ids  = init_params.hidden_state_capture_layer_ids;
        EXPECT_EQ(init_params.hidden_state_capture_dtype, params.model_config_.hidden_state_capture_dtype);
        EXPECT_TRUE(init_params.hidden_state_capture_fail_open);
        return std::make_unique<CapturedParamsModel>();
    };
    FactoryResetGuard factory_reset_guard;

    {
        NormalExecutor executor(params,
                                manager,
                                false,
                                false,
                                0,
                                MlaOpsType::AUTO,
                                nullptr,
                                nullptr);
        EXPECT_EQ(captured_width, 9);
        EXPECT_TRUE(captured_has_layout);
        EXPECT_EQ(captured_layer_ids, (std::vector<int64_t>{0}));
    }

    // The real bound is 11 blocks; an unrelated 1 + gamma fake candidate
    // would inflate this NormalExecutor capture to 17 blocks.
    params.sp_config.gen_num_per_cycle = 16;
    {
        NormalExecutor executor(params, manager, false, false, 0, MlaOpsType::AUTO);
        EXPECT_EQ(captured_width, 11);
        EXPECT_TRUE(captured_has_layout);
        EXPECT_EQ(captured_layer_ids, (std::vector<int64_t>{0}));
    }

    {
        NormalExecutor executor(params,
                                nullptr,
                                false,
                                false,
                                0,
                                MlaOpsType::AUTO,
                                nullptr,
                                nullptr);
        EXPECT_EQ(captured_width, 0);
        EXPECT_FALSE(captured_has_layout);
        EXPECT_EQ(captured_layer_ids, (std::vector<int64_t>{0}));
    }

    for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
        for (const bool warm_up : {false, true}) {
            for (const bool is_propose : {false, true}) {
                SCOPED_TRACE(testing::Message() << "role=" << static_cast<int>(role) << " warm_up=" << warm_up
                                                << " is_propose=" << is_propose);
                params.pd_sep_config.role_type = role;
                // A cacheless draft exercises policy selection without manufacturing an MTP layout.
                NormalExecutor executor(params, is_propose ? nullptr : manager, warm_up, is_propose);
                EXPECT_EQ(captured_width, is_propose ? 0 : 11);
                EXPECT_EQ(captured_has_layout, !is_propose);
                const bool capture = !warm_up && !is_propose && role != RoleType::DECODE;
                EXPECT_EQ(captured_layer_ids, capture ? std::vector<int64_t>{0} : std::vector<int64_t>{});
            }
        }
    }
}

class NormalBatchStreamProcessorPrefillOnlyTest:
    public NormalBatchStreamProcessorTest,
    public testing::WithParamInterface<std::tuple<int, std::optional<GenerationPrefillCudaGraphStatus>>> {};

TEST_P(NormalBatchStreamProcessorPrefillOnlyTest, prefillOnlySkipsLmHeadAndDispatchesOneEmptyOutput) {
    class RecordingGenerateStream: public NormalGenerateStream {
    public:
        using NormalGenerateStream::NormalGenerateStream;

        void updateOutput(const StreamUpdateInfo& update_info) override {
            update_infos.push_back(update_info);
            NormalGenerateStream::updateOutput(update_info);
        }

        std::vector<StreamUpdateInfo> update_infos;
    };

    const auto&     requested_status = std::get<1>(GetParam());
    const auto      graph_status     = requested_status.value_or(GenerationPrefillCudaGraphStatus::NOT_REQUESTED);
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len      = 128;
    model_config.vocab_size       = 128;
    model_config.input_vocab_size = 128;
    model_config.num_layers       = 1;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    auto make_stream = [&](std::vector<int32_t> tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = hostIntBuffer(std::move(tokens));
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 0;
        query->generate_config->aux_info       = true;
        auto stream =
            make_shared<RecordingGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };

    auto                                                                           stream1  = make_stream({1, 2});
    auto                                                                           stream2  = make_stream({3, 4, 5});
    const std::vector<std::pair<std::shared_ptr<RecordingGenerateStream>, size_t>> expected = {
        {stream1, stream1->seqLength()}, {stream2, stream2->seqLength()}};
    StreamGroups               stream_groups({stream1, stream2});
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true, std::get<0>(GetParam()));

    TensorHolder holder;
    auto         model_input = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input.ok());
    EXPECT_TRUE(model_input->skip_lm_head);
    EXPECT_TRUE(model_input->capture_hidden_states);
    EXPECT_EQ(model_input->combo_tokens.numel(), 5);

    for (int i = 0; i < 2; ++i) {
        if (requested_status.has_value()) {
            ASSERT_TRUE(processor.dispatchPrefillOnly(stream_groups, *requested_status).ok());
        } else {
            ASSERT_TRUE(processor.dispatchPrefillOnly(stream_groups).ok());
        }
    }
    for (const auto& [stream, input_length] : expected) {
        ASSERT_EQ(stream->update_infos.size(), 1);
        const auto& update_info = stream->update_infos.front();
        EXPECT_EQ(update_info.generation_prefill_cuda_graph_status, graph_status);
        EXPECT_EQ(update_info.num_new_tokens, 0);
        EXPECT_EQ(update_info.new_tokens.sizes(), (torch::IntArrayRef{1, 0}));
        EXPECT_FALSE(update_info.update_remote_generate);
        EXPECT_FALSE(update_info.force_update_info);
        EXPECT_FALSE(update_info.prompt_logits.has_value());
        EXPECT_FALSE(update_info.error_info.has_value());

        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), 1);
        EXPECT_TRUE(output.value().generate_outputs[0].finished);
        EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
        EXPECT_EQ(output.value().generate_outputs[0].aux_info.generation_prefill_cuda_graph_status,
                  generationPrefillCudaGraphStatusString(graph_status));
        EXPECT_EQ(stream->generationPrefillCudaGraphStatus(), graph_status);
        EXPECT_EQ(stream->seqLength(), input_length);
        EXPECT_FALSE(stream->hasOutput());

        auto finished = stream->nextOutput();
        ASSERT_FALSE(finished.ok());
        EXPECT_EQ(finished.status().code(), ErrorCode::FINISHED);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GenerationPrefillCudaGraphStatus,
    NormalBatchStreamProcessorPrefillOnlyTest,
    testing::Combine(
        testing::Values(0, 2),
        testing::Values(std::optional<GenerationPrefillCudaGraphStatus>{},
                        std::make_optional(GenerationPrefillCudaGraphStatus::NOT_REQUESTED),
                        std::make_optional(GenerationPrefillCudaGraphStatus::REPLAYED),
                        std::make_optional(GenerationPrefillCudaGraphStatus::CAPTURE_UNAVAILABLE),
                        std::make_optional(GenerationPrefillCudaGraphStatus::GRAPH_INPUT_SHAPE_MISMATCH))));

TEST_F(NormalBatchStreamProcessorTest, gatherExecutionModesPreserveFlagsAndErrors) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len      = 128;
    model_config.vocab_size       = 128;
    model_config.input_vocab_size = 128;
    model_config.num_layers       = 1;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    auto make_stream = [&](std::vector<int32_t> tokens, int max_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = hostIntBuffer(std::move(tokens));
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    struct StreamMode {
        bool context;
        int  max_new_tokens;
    };
    struct TestCase {
        const char*             name;
        std::vector<StreamMode> modes;
        bool                    prefill_only;
        bool                    mixed;
        bool                    existing_error = false;
    };
    const std::vector<TestCase> cases = {
        {"empty", {}, false, false},
        {"decode-generation", {{false, 1}, {false, 2}}, false, false},
        {"context-generation", {{true, 1}, {true, 2}}, false, false},
        {"context-prefill-only", {{true, 0}, {true, 0}}, true, false},
        {"decode-prefill-only", {{false, 0}}, true, false},
        {"both-containers-generation", {{true, 1}, {false, 1}}, false, false},
        {"both-containers-prefill-only", {{true, 0}, {false, 0}}, true, false},
        {"context-mixed-prefill-first", {{true, 0}, {true, 1}}, true, true},
        {"context-mixed-generation-first", {{true, 1}, {true, 0}}, false, true},
        {"decode-mixed-prefill-first", {{false, 0}, {false, 1}}, true, true},
        {"decode-mixed-generation-first", {{false, 1}, {false, 0}}, false, true},
        {"cross-container-decode-generation-first", {{false, 1}, {true, 0}}, false, true},
        {"cross-container-context-prefill-first", {{true, 0}, {false, 1}}, false, true},
        {"cross-container-decode-prefill-first", {{false, 0}, {true, 1}}, true, true},
        {"cross-container-context-generation-first", {{true, 1}, {false, 0}}, true, true},
        {"mixed-context-behind-generation-decode", {{false, 1}, {true, 1}, {true, 0}}, false, true},
        {"mixed-decode-ahead-of-generation-context", {{true, 1}, {false, 1}, {false, 0}}, false, true},
        {"existing-error-does-not-define-mode", {{true, 0}, {true, 1}}, false, false, true},
        {"existing-error-in-cross-container-batch", {{true, 0}, {false, 1}}, false, false, true},
        {"existing-error-before-mixed-context", {{false, 1}, {true, 0}, {true, 1}}, true, true, true},
        {"existing-error-in-homogeneous-batch", {{true, 1}, {false, 1}}, false, false, true},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        std::list<GenerateStreamPtr> streams;
        int64_t                      expected_tokens = 0;
        for (const auto& mode : test_case.modes) {
            auto stream = make_stream({1, 2}, mode.max_new_tokens);
            stream->setIsContextStream(mode.context);
            streams.push_back(stream);
            expected_tokens += mode.context ? 2 : 1;
        }
        if (test_case.existing_error) {
            streams.front()->reportError(ErrorCode::EXECUTION_EXCEPTION, "existing request error");
        }
        StreamGroups stream_groups(streams);
        TensorHolder holder;
        auto         model_input = processor.gatherModelInput(stream_groups, holder);
        // Mixed batches still reach TP sync; flags follow the execution mode
        // derived from schedulable (non-errored) streams.
        ASSERT_TRUE(model_input.ok());
        EXPECT_EQ(model_input->combo_tokens.numel(), expected_tokens);
        EXPECT_EQ(model_input->skip_lm_head, test_case.prefill_only);
        EXPECT_EQ(model_input->capture_hidden_states, test_case.prefill_only);
        for (const auto& stream : streams) {
            if (test_case.existing_error && stream == streams.front()) {
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
                EXPECT_EQ(stream->stopReason(), "existing request error");
            } else if (test_case.mixed) {
                EXPECT_TRUE(stream->hasError());
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_EQ(stream->stopReason(), kMixedExecutionModeBatchError);
            } else {
                EXPECT_FALSE(stream->hasError());
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testCacheKeyWidthIndependentOfBlockTable) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 1;

    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PREFILL;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({1, 2, 3});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->num_return_sequences = 2;
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    BatchKVCacheResource resource;
    resource.resetBatchSize(2);
    resource.initGroups(cache_config.topologyPtr());
    resource.setBatchBlocks(0, "default", {1, 2});
    resource.setBatchBlocks(1, "default", {3, 4});
    resource.setBatchCacheKeys(0, CacheKeysType{101, 102, 103});
    resource.setBatchCacheKeys(1, CacheKeysType{201, 202, 203, 204, 205});
    stream->setKVCache(resource);
    stream->generate_status_->status = StreamState::RUNNING;

    StreamGroups stream_groups({stream});
    EXPECT_EQ(stream_groups.curBlocksNum(), 2);
    EXPECT_EQ(stream_groups.maxCacheKeysNum(), 5);

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().pd_separation);
    const auto& cache_keys = merge_input_status.value().cache_keys;
    ASSERT_TRUE(cache_keys.defined());
    EXPECT_EQ(cache_keys.size(0), 2);
    EXPECT_EQ(cache_keys.size(1), 5);
    EXPECT_EQ(toVec<int64_t>(cache_keys), (std::vector<int64_t>{101, 102, 103, 0, 0, 201, 202, 203, 204, 205}));
}

TEST_F(NormalBatchStreamProcessorTest, testModelKernelPageIgnoresLargerStatePool) {
    for (const bool state_first : {false, true}) {
        SCOPED_TRACE(state_first);
        ModelConfig model_config;
        model_config.num_layers = 2;
        CacheConfig cache_config;
        cache_config.layer_num          = 2;
        cache_config.seq_size_per_block = 256;
        auto attention                  = std::make_shared<MHAKVCacheSpec>("attention", 256, 128, 1);
        auto state                      = std::make_shared<FixedStateCacheSpec>("state", 512, 512, 1);
        if (state_first) {
            cache_config.fromGroupedSpecs(
                {state, attention}, {{0}, {1}}, {CacheGroupType::SWA, CacheGroupType::FULL}, {"state", "attention"});
        } else {
            cache_config.fromGroupedSpecs(
                {attention, state}, {{1}, {0}}, {CacheGroupType::FULL, CacheGroupType::SWA}, {"attention", "state"});
        }

        EXPECT_EQ(cache_config.group("attention").kernelSeqSizePerBlock(), 128u);
        EXPECT_EQ(cache_config.group("state").kernelSeqSizePerBlock(), 512u);
        NormalBatchStreamProcessor processor(
            model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache_config, true);
        EXPECT_EQ(processor.model_input_gatherer_config_.seq_size_per_block, 256u);
        EXPECT_EQ(processor.model_input_gatherer_config_.kernel_seq_size_per_block, 0u);
        EXPECT_EQ(processor.model_input_gatherer_config_.kernel_blocks_per_kv_block, 2u);

        const auto topology = cache_config.topologyPtr();
        EXPECT_EQ(CudaGraphRunner::captureKernelBlockTableWidth(*topology, 513, 0), 6);
        EXPECT_EQ(CudaGraphRunner::captureKernelBlockTableWidth(*topology, 1), 2);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testCacheMetadataWithoutBlockCapacity) {
    NormalModelInputGathererConfig config;
    config.kv_cache_group_nums  = 2;
    config.kv_cache_group_tags  = {"swa", "full"};
    config.kv_cache_group_types = {CacheGroupType::SWA, CacheGroupType::FULL};
    StreamGroups             groups(std::list<GenerateStreamPtr>{});
    TensorHolder             holder;
    NormalModelInputGatherer gatherer(config);
    auto                     inputs = gatherer.gather(groups, holder);
    ASSERT_TRUE(inputs.ok());
    EXPECT_EQ(inputs->kv_cache_group_tags, config.kv_cache_group_tags);
    EXPECT_EQ(
        toVec<int32_t>(inputs->kv_cache_group_types),
        (std::vector<int32_t>{static_cast<int32_t>(CacheGroupType::SWA), static_cast<int32_t>(CacheGroupType::FULL)}));
    EXPECT_FALSE(inputs->kv_cache_block_id.defined());
    EXPECT_FALSE(inputs->kv_cache_kernel_block_id.defined());
    auto kernel = gatherer.gatherKvCacheKernelBlockId(groups, {"full", "swa"}, holder);
    ASSERT_TRUE(kernel.ok());
    EXPECT_FALSE(kernel->defined());

    for (const auto& tags : std::vector<std::vector<std::string>>{{}, {"full"}, {"full", "full"}, {"full", ""}}) {
        auto invalid_config                = config;
        invalid_config.kv_cache_group_tags = tags;
        NormalModelInputGatherer invalid(invalid_config);
        EXPECT_ANY_THROW((void)invalid.gather(groups, holder));
    }
    config.kv_cache_group_types.pop_back();
    NormalModelInputGatherer invalid_types(config);
    EXPECT_ANY_THROW((void)invalid_types.gather(groups, holder));
}

TEST_F(NormalBatchStreamProcessorTest, testDistinctKernelPagesRemainGroupLocal) {
    for (const bool reversed : {false, true}) {
        ModelConfig model;
        model.num_layers = 1;
        CacheConfig config;
        config.layer_num          = 1;
        config.seq_size_per_block = 256;  // tokens/cache-key block
        auto first                = std::make_shared<MHAKVCacheSpec>("first", 256, 64, 1);
        auto second               = std::make_shared<MHAKVCacheSpec>("second", 512, 128, 1);
        config.fromGroupedSpecs(reversed ? std::vector<KVCacheSpecPtr>{second, first} :
                                           std::vector<KVCacheSpecPtr>{first, second},
                                {{0}, {0}},
                                {CacheGroupType::FULL, CacheGroupType::FULL});
        NormalBatchStreamProcessor processor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, true);
        EXPECT_EQ(processor.model_input_gatherer_config_.kernel_seq_size_per_block, 0u);
        EXPECT_EQ(config.groupForLayer(0, "first").kernelSeqSizePerBlock(), 64u);
        EXPECT_EQ(config.groupForLayer(0, "second").kernelSeqSizePerBlock(), 128u);

        const auto topology = config.topologyPtr();
        EXPECT_EQ(CudaGraphRunner::captureKernelBlockTableWidth(*topology, 513, 0), 12);
        EXPECT_EQ(CudaGraphRunner::captureKernelBlockTableWidth(*topology, 1), 4);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testSingleGroupGraphUsesSpecGeometry) {
    CacheConfig config;
    config.layer_num          = 1;
    config.seq_size_per_block = 256;  // tokens/cache-key block
    auto spec                 = std::make_shared<MHAKVCacheSpec>("attention", 512, 64, 1);
    config.fromGroupedSpecs({spec}, {{0}}, {CacheGroupType::FULL});

    const auto topology = config.topologyPtr();
    EXPECT_EQ(CudaGraphRunner::captureKernelBlockTableWidth(*topology, 513, 0), 16);
    EXPECT_EQ(CudaGraphRunner::captureKernelBlockTableWidth(*topology, 1), 8);
    config.seq_size_per_block = 384;  // Cache-key granularity does not determine graph table width.
    EXPECT_EQ(CudaGraphRunner::captureKernelBlockTableWidth(*topology, 513, 0), 16);
}

TEST_F(NormalBatchStreamProcessorTest, testMixedGroupBlockWidthsGatherCompleteRows) {
    for (const int full_bpk : {4, 128}) {
        for (const bool full_first : {false, true}) {
            SCOPED_TRACE("full_bpk=" + std::to_string(full_bpk) + " full_first=" + std::to_string(full_first));
            ModelConfig model_config;
            model_config.max_seq_len = 2048;
            model_config.vocab_size  = 2048;
            model_config.num_layers  = 2;
            CacheConfig cache_config;
            cache_config.layer_num = 2;
            auto      full         = std::make_shared<MHAKVCacheSpec>("full", 128, 128 / full_bpk, 1);
            auto      swa          = std::make_shared<MHAKVCacheSpec>("swa", 128, 128, 1);
            const int full_gid     = full_first ? 0 : 1;
            const int swa_gid      = 1 - full_gid;
            if (full_first) {
                cache_config.fromGroupedSpecs(
                    {full, swa}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::SWA}, {"full", "swa"});
            } else {
                cache_config.fromGroupedSpecs(
                    {swa, full}, {{1}, {0}}, {CacheGroupType::SWA, CacheGroupType::FULL}, {"swa", "full"});
            }
            CacheConfig resource_config;
            resource_config.layer_num = 2;
            if (full_first) {
                resource_config.fromGroupedSpecs({swa, full}, {{1}, {0}}, {CacheGroupType::SWA, CacheGroupType::FULL});
            } else {
                resource_config.fromGroupedSpecs({full, swa}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::SWA});
            }
            ResourceContext              resource_context;
            RuntimeConfig                runtime_config;
            std::list<GenerateStreamPtr> streams;
            for (int batch = 0; batch < 2; ++batch) {
                auto query             = std::make_shared<GenerateInput>();
                query->input_ids       = hostIntBuffer({1, 2, 3});
                query->generate_config = std::make_shared<GenerateConfig>();
                auto stream            = std::make_shared<NormalGenerateStream>(
                    query, model_config, runtime_config, resource_context, nullptr);
                BatchKVCacheResource resource;
                resource.resetBatchSize(1);
                resource.initGroups(resource_config.topologyPtr());
                resource.mutableBlockIds(0, "full").assign({10 + batch * 2, NULL_BLOCK_IDX, 11 + batch * 2});
                resource.mutableBlockIds(0, "swa").assign({20 + batch, 30 + batch, 40 + batch, 50 + batch, 60 + batch});
                stream->setKVCache(resource);
                stream->streamCacheResource().block_update_mapping_ = {{"swa", 20 + batch, 30 + batch},
                                                                       {"full", 10 + batch * 2, 40 + batch}};
                stream->generate_status_->status                    = StreamState::RUNNING;
                streams.push_back(stream);
            }
            StreamGroups               groups(streams);
            NormalBatchStreamProcessor processor(
                model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache_config, false);
            NormalModelInputGatherer gatherer(processor.model_input_gatherer_config_);
            TensorHolder             holder;
            const auto&              payload_tags = processor.model_input_gatherer_config_.kv_cache_group_tags;
            auto                     device_table = gatherer.gatherKvCacheKernelBlockId(groups, payload_tags, holder);
            ASSERT_TRUE(device_table.ok());
            auto inputs = processor.gatherModelInput(groups, holder);
            ASSERT_TRUE(inputs.ok());
            EXPECT_EQ(inputs->kv_cache_group_tags, payload_tags);
            EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_update_mapping),
                      (std::vector<int32_t>{swa_gid, 20, 30, full_gid, 10, 40, swa_gid, 21, 31, full_gid, 12, 41}));
            ASSERT_EQ(inputs->kv_cache_block_id.size(2), 5);
            ASSERT_EQ(inputs->kv_cache_kernel_block_id.size(2), 3 * full_bpk);
            EXPECT_EQ(toVec<int32_t>(*device_table), toVec<int32_t>(inputs->kv_cache_kernel_block_id));
            for (int batch = 0; batch < 2; ++batch) {
                std::vector<int32_t> expected_full(3 * full_bpk, NULL_BLOCK_IDX);
                std::iota(expected_full.begin(), expected_full.begin() + full_bpk, (10 + batch * 2) * full_bpk);
                std::iota(expected_full.begin() + 2 * full_bpk, expected_full.end(), (11 + batch * 2) * full_bpk);
                EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_kernel_block_id[full_gid][batch]), expected_full);
                std::vector<int32_t> expected_swa(3 * full_bpk, 0);
                for (int index = 0; index < 5; ++index) {
                    expected_swa[index] = 20 + 10 * index + batch;
                }
                EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_kernel_block_id[swa_gid][batch]), expected_swa);
                EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_block_id[full_gid][batch]),
                          (std::vector<int32_t>{10 + batch * 2, NULL_BLOCK_IDX, 11 + batch * 2, 0, 0}));
                EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_block_id[swa_gid][batch]),
                          (std::vector<int32_t>{20 + batch, 30 + batch, 40 + batch, 50 + batch, 60 + batch}));
            }
            const std::vector<std::string> reversed_tags(payload_tags.rbegin(), payload_tags.rend());
            auto reversed_table = processor.gatherKvCacheKernelBlockId(groups, reversed_tags, holder);
            ASSERT_TRUE(reversed_table.ok());
            EXPECT_EQ(toVec<int32_t>((*reversed_table)[0]), toVec<int32_t>(inputs->kv_cache_kernel_block_id[1]));
            EXPECT_EQ(toVec<int32_t>((*reversed_table)[1]), toVec<int32_t>(inputs->kv_cache_kernel_block_id[0]));
            EXPECT_ANY_THROW((void)gatherer.gatherKvCacheKernelBlockId(groups, {"full", "unknown"}, holder));
            EXPECT_ANY_THROW((void)gatherer.gatherKvCacheKernelBlockId(groups, {"full", "full"}, holder));
            EXPECT_ANY_THROW((void)gatherer.gatherKvCacheKernelBlockId(groups, {"full", ""}, holder));
            EXPECT_ANY_THROW((void)gatherer.gatherKvCacheKernelBlockId(groups, {"full"}, holder));
            auto undersized_config                       = processor.model_input_gatherer_config_;
            undersized_config.kernel_blocks_per_kv_block = 1;
            NormalModelInputGatherer undersized(undersized_config);
            auto                     independent = undersized.gatherKvCacheKernelBlockId(groups, payload_tags, holder);
            ASSERT_TRUE(independent.ok());
            EXPECT_EQ(toVec<int32_t>(*independent), toVec<int32_t>(*device_table));
            auto independent_inputs = undersized.gather(groups, holder);
            ASSERT_TRUE(independent_inputs.ok());
            EXPECT_EQ(toVec<int32_t>(independent_inputs->kv_cache_kernel_block_id), toVec<int32_t>(*device_table));

            // Published shapes, including padding, own the steady stream's widths.
            // Its empty host resource must never be traversed by either gather path.
            auto steady = streams.front();
            steady->setIsContextStream(false);
            GenerateStream::MtpAsyncDeviceState state;
            state.next_seq_len_upper_bound   = 3;
            const auto cuda_i32              = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            state.next_kv_cache_block_id_gpu = torch::arange(14, cuda_i32).reshape({2, 1, 7});
            state.next_kv_cache_kernel_block_id_gpu =
                torch::arange(2 * (3 * full_bpk + 3), cuda_i32).reshape({2, 1, 3 * full_bpk + 3});
            steady->setMtpAsyncDeviceState(state);
            steady->setKVCache(BatchKVCacheResource{});
            StreamGroups snapshot_groups(streams);
            auto         snapshot_inputs = gatherer.gather(snapshot_groups, holder);
            ASSERT_TRUE(snapshot_inputs.ok());
            EXPECT_EQ(snapshot_groups.curBlocksNum(), 7u);
            EXPECT_EQ(snapshot_inputs->kv_cache_block_id.size(2), 7);
            EXPECT_EQ(snapshot_inputs->kv_cache_kernel_block_id.size(2), 3 * full_bpk + 3);
            EXPECT_EQ(toVec<int32_t>(snapshot_inputs->kv_cache_block_id[full_gid][1]),
                      (std::vector<int32_t>{12, NULL_BLOCK_IDX, 13, 0, 0, 0, 0}));
            auto snapshot_kernel = gatherer.gatherKvCacheKernelBlockId(snapshot_groups, reversed_tags, holder);
            ASSERT_TRUE(snapshot_kernel.ok());
            EXPECT_EQ(toVec<int32_t>((*snapshot_kernel)[0][0]),
                      toVec<int32_t>(state.next_kv_cache_kernel_block_id_gpu[1][0]));
            EXPECT_EQ(toVec<int32_t>((*snapshot_kernel)[1][0]),
                      toVec<int32_t>(state.next_kv_cache_kernel_block_id_gpu[0][0]));
            for (int row = 0; row < 2; ++row) {
                EXPECT_EQ(toVec<int32_t>((*snapshot_kernel)[row][1]),
                          toVec<int32_t>(snapshot_inputs->kv_cache_kernel_block_id[1 - row][1]));
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testMixedEmptyAndNonEmptyOrdinaryResources) {
    ModelConfig model;
    model.num_layers  = 1;
    model.vocab_size  = 16;
    model.max_seq_len = 128;

    CacheConfig config;
    config.layer_num = 1;
    initFullCacheConfig(config, model.num_layers);

    auto make_stream = [&](BatchKVCacheResource resource) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({1, 2, 3});
        query->generate_config = std::make_shared<GenerateConfig>();
        auto stream            = std::make_shared<NormalGenerateStream>(
            query, model, RuntimeConfig{}, ResourceContext{}, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setKVCache(std::move(resource));
        return stream;
    };

    BatchKVCacheResource empty_resource;
    empty_resource.resetBatchSize(1);

    BatchKVCacheResource full_resource;
    full_resource.resetBatchSize(1);
    full_resource.initGroups(config.topologyPtr());
    full_resource.setBatchBlocks(0, "default", {1, 2, 3});

    auto empty_stream = make_stream(std::move(empty_resource));
    auto full_stream  = make_stream(std::move(full_resource));
    StreamGroups groups({empty_stream, full_stream});

    NormalBatchStreamProcessor processor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, false);
    TensorHolder holder;

    auto kernel_table = processor.gatherKvCacheKernelBlockId(groups, {"default"}, holder);
    ASSERT_TRUE(kernel_table.ok());
    ASSERT_EQ(kernel_table->size(0), 1);
    ASSERT_EQ(kernel_table->size(1), 2);
    ASSERT_EQ(kernel_table->size(2), 3);
    EXPECT_EQ(toVec<int32_t>((*kernel_table)[0][0]), (std::vector<int32_t>{0, 0, 0}));
    EXPECT_EQ(toVec<int32_t>((*kernel_table)[0][1]), (std::vector<int32_t>{1, 2, 3}));

    auto inputs = processor.gatherModelInput(groups, holder);
    ASSERT_TRUE(inputs.ok());
    ASSERT_EQ(inputs->kv_cache_block_id.size(0), 1);
    ASSERT_EQ(inputs->kv_cache_block_id.size(1), 2);
    ASSERT_EQ(inputs->kv_cache_block_id.size(2), 3);
    EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_block_id[0][0]), (std::vector<int32_t>{0, 0, 0}));
    EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_block_id[0][1]), (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_kernel_block_id[0][0]), (std::vector<int32_t>{0, 0, 0}));
    EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_kernel_block_id[0][1]), (std::vector<int32_t>{1, 2, 3}));
}

TEST_F(NormalBatchStreamProcessorTest, testGatherDistinguishesEmptyAndUnknownGroup) {
    ModelConfig model;
    model.num_layers  = 2;
    model.vocab_size  = 16;
    model.max_seq_len = 128;
    auto        full  = std::make_shared<MHAKVCacheSpec>("full", 128, 64, 1);
    auto        swa   = std::make_shared<MHAKVCacheSpec>("swa", 128, 128, 1);
    CacheConfig config;
    config.layer_num = 2;
    config.fromGroupedSpecs({full, swa}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::SWA});
    NormalBatchStreamProcessor processor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, false);
    auto                       query = std::make_shared<GenerateInput>();
    query->input_ids                 = hostIntBuffer({1, 2, 3});
    query->generate_config           = std::make_shared<GenerateConfig>();
    auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, ResourceContext{}, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    BatchKVCacheResource resource;
    resource.resetBatchSize(1);
    resource.initGroups(config.topologyPtr());
    resource.mutableBlockIds(0, "full").assign({7});
    stream->setKVCache(resource);
    StreamGroups groups({stream});
    TensorHolder holder;
    auto         inputs = processor.gatherModelInput(groups, holder);
    ASSERT_TRUE(inputs.ok());
    EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_block_id[1]), (std::vector<int32_t>{0}));
    EXPECT_EQ(toVec<int32_t>(inputs->kv_cache_kernel_block_id[1]), (std::vector<int32_t>{0, 0}));

    auto        unknown = std::make_shared<MHAKVCacheSpec>("unknown", 128, 128, 1);
    CacheConfig wrong_config;
    wrong_config.layer_num = 2;
    wrong_config.fromGroupedSpecs({full, unknown}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::SWA});
    resource.initGroups(wrong_config.topologyPtr());
    resource.mutableBlockIds(0, "full").assign({7});
    stream->setKVCache(resource);
    StreamGroups wrong_groups({stream});
    EXPECT_ANY_THROW((void)processor.gatherModelInput(wrong_groups, holder));
    EXPECT_ANY_THROW((void)processor.gatherKvCacheKernelBlockId(wrong_groups, {"full", "swa"}, holder));
}

class TestStatefulLogitsProcessor: public BaseLogitsProcessor {
public:
    explicit TestStatefulLogitsProcessor(bool async_device_state): async_device_state_(async_device_state) {}

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        (void)inputs;
        (void)start_idx;
        (void)finish_idx;
        return std::nullopt;
    }

    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override {
        (void)src_batch_indices;
    }

    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override {
        (void)new_tokens;
        accepted_token_len_ += num_new_tokens;
        return std::nullopt;
    }

    bool isStateful() const override {
        return true;
    }

    bool supportsNormalAsyncDeviceState() const override {
        return async_device_state_;
    }

    int64_t acceptedTokenLen() const override {
        return accepted_token_len_;
    }

private:
    bool    async_device_state_;
    int64_t accepted_token_len_ = 0;
};

TEST_F(NormalBatchStreamProcessorTest, testSimpleAssemble) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                = 2048;
    model_config.vocab_size                 = 2048;
    model_config.num_layers                 = 2;
    model_config.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    rtp_llm::test::setGroupBlockLayout(
        cache_config, {"default"}, {cache_config.group("default").block_num}, {4096}, {256});

    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = hostIntBuffer({1, 2});
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    query1->input_ids = hostIntBuffer({1});
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, "default", {1, 2, 3, 4});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = hostIntBuffer({1, 2, 3});
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    query2->input_ids = hostIntBuffer({1, 2});
    BatchKVCacheResource addr2;
    addr2.resetBatchSize(1);
    addr2.initGroups(cache_config.topologyPtr());
    addr2.setBatchBlocks(0, "default", {5, 6, 7, 8});
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = hostIntBuffer({1, 2, 3});
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(cache_config.topologyPtr());
    addr3.setBatchBlocks(0, "default", {9, 10});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = hostIntBuffer({1, 2, 3, 4});
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(cache_config.topologyPtr());
    addr4.setBatchBlocks(0, "default", {11, 12, 13, 14});
    stream4->setKVCache(addr4);
    stream4->setReuseLength(1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    {
        StreamGroups stream_groups(streams);
        TensorHolder holder;

        auto merge_input_status = processor.gatherModelInput(stream_groups, holder);

        EXPECT_TRUE(merge_input_status.ok());
        auto&       model_input       = merge_input_status.value();
        vector<int> combo_tokens      = {2, 3, 1, 2, 3, 2, 3, 4};
        vector<int> input_lengths     = {1, 2, 3, 3};
        vector<int> sequence_lengths  = {1, 2};
        vector<int> prefix_lengths    = {0, 1};
        vector<int> kv_cache_block_id = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 11, 12, 13, 14};
        EXPECT_EQ(combo_tokens, toVec<int>(model_input.combo_tokens));
        EXPECT_EQ(input_lengths, toVec<int>(model_input.input_lengths));
        EXPECT_EQ(sequence_lengths, toVec<int>(model_input.sequence_lengths));
        EXPECT_EQ(prefix_lengths, toVec<int>(model_input.prefix_lengths));
        EXPECT_EQ(kv_cache_block_id, toVec<int>(model_input.kv_cache_block_id));
        EXPECT_EQ(model_input.kv_block_stride_bytes, cache_config.groups().front().kvBlockStrideBytes());
        EXPECT_EQ(model_input.kv_scale_stride_bytes, cache_config.groups().front().kvScaleStrideBytes());
    }
    {
        MMModelConfig mm_model_config;
        model_config.mm_model_config = mm_model_config;
        NormalBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

        StreamGroups stream_groups(streams);
        TensorHolder holder;
        auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
        EXPECT_TRUE(merge_input_status.ok());
        auto& model_input = merge_input_status.value();
        EXPECT_FALSE(model_input.attention_mask.defined());
    }
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathWaitsForBlockingLogitsProcessorState) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 128900;
    RuntimeConfig runtime_config;

    std::shared_ptr<GenerateInput> query          = make_shared<GenerateInput>();
    query->input_ids                              = hostIntBuffer({1, 2, 3});
    query->generate_config                        = make_shared<GenerateConfig>();
    query->generate_config->in_think_mode         = true;
    query->generate_config->max_thinking_tokens   = 10;
    query->generate_config->begin_think_token_ids = {128821};
    query->generate_config->end_think_token_ids   = {128822};

    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });

    std::list<GenerateStreamPtr> streams{stream};
    StreamGroups                 stream_groups(streams);

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    EXPECT_TRUE(executor.gatherCanUseDeviceState(stream_groups));
    stream->logits_processor_list_.push_back(std::make_shared<TestStatefulLogitsProcessor>(false));
    stream->incPendingAsyncBookkeeping();
    EXPECT_FALSE(executor.gatherCanUseDeviceState(stream_groups));
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathAllowsAsyncLogitsProcessorState) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 128900;
    RuntimeConfig runtime_config;

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = hostIntBuffer({1, 2, 3});
    query->generate_config               = make_shared<GenerateConfig>();

    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });
    stream->logits_processor_list_.push_back(std::make_shared<TestStatefulLogitsProcessor>(true));

    std::list<GenerateStreamPtr> streams{stream};
    StreamGroups                 stream_groups(streams);

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    stream->incPendingAsyncBookkeeping();
    EXPECT_TRUE(executor.gatherCanUseDeviceState(stream_groups));
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_P(OutputDispatchTest, testSoftmaxProbs) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1         = make_shared<GenerateInput>();
    query1->input_ids                             = hostIntBuffer({1});
    query1->generate_config                       = make_shared<GenerateConfig>();
    query1->generate_config->return_softmax_probs = true;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, "default", {1});
    stream1->setKVCache(addr1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, GetParam());

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(merge_input_status.ok());

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    auto          hidden_tensor                = torch::tensor({1.0f, 2.0f}).reshape({1, 2}).to(torch::kCUDA);
    auto          logits_tensor                = torch::tensor({1.0f, 2.0f}).reshape({1, 2}).to(torch::kCUDA);
    merge_outputs.model_output.hidden_states   = hidden_tensor;
    merge_outputs.model_output.logits          = logits_tensor;
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1}, torch::kInt32).reshape({1, 2});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f}).to(torch::kCUDA);
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    auto softmax_probs = stream1->getSoftmaxProbs();
    EXPECT_TRUE(softmax_probs.defined());
    EXPECT_EQ(2048, softmax_probs.numel());
    EXPECT_NEAR(0.731058, softmax_probs.data_ptr<float>()[1], 0.0001);
}

TEST_P(OutputDispatchTest, testParallelDispatchMultipleStreams) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    auto make_stream = [&](int input_token) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer({input_token});
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->return_softmax_probs = true;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    auto stream1 = make_stream(0);
    auto stream2 = make_stream(1);

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, GetParam());
    EXPECT_EQ(processor.output_dispatcher_->thread_pool_ != nullptr, GetParam() > 0);

    StreamGroups stream_groups({stream1, stream2});
    MergedOutput merge_outputs;
    merge_outputs.model_output.logits =
        torch::tensor({1.0f, 2.0f, 3.0f, 1.0f}, torch::kFloat32).reshape({2, 2}).to(torch::kCUDA);
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1, 1, 0}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.success       = torch::tensor({true, true}, torch::kBool);
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f}, torch::kFloat32).to(torch::kCUDA);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    ASSERT_FALSE(stream1->hasError());
    ASSERT_FALSE(stream2->hasError());
    EXPECT_EQ(stream1->completeTokenIdsVec(0), (std::vector<int>{0, 1}));
    EXPECT_EQ(stream2->completeTokenIdsVec(0), (std::vector<int>{1, 0}));

    auto stream1_probs = stream1->getSoftmaxProbs();
    auto stream2_probs = stream2->getSoftmaxProbs();
    ASSERT_TRUE(stream1_probs.defined());
    ASSERT_TRUE(stream2_probs.defined());
    EXPECT_NEAR(stream1_probs.data_ptr<float>()[1], 0.731058f, 0.0001f);
    EXPECT_NEAR(stream2_probs.data_ptr<float>()[1], 0.880797f, 0.0001f);
}

TEST_P(OutputDispatchTest, testMixedPromptLengthsAndBeamExpansion) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    auto make_stream = [&](std::vector<int32_t> prompt, bool beam) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer(std::move(prompt));
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens       = 1;
        query->generate_config->return_hidden_states = true;
        query->generate_config->return_logits        = true;
        query->generate_config->return_softmax_probs = true;
        query->generate_config->calculate_loss       = 2;
        if (beam) {
            query->generate_config->variable_num_beams = {2};
        }
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    // Input row offsets are 0/1/2, output row offsets are 0/1/3, and
    // all-logits token offsets are 0/1/3. The last stream catches offset drift.
    auto         first = make_stream({0}, false);
    auto         beam  = make_stream({1, 0}, true);
    auto         last  = make_stream({2, 2, 1}, false);
    StreamGroups groups({first, beam, last});
    ASSERT_EQ(groups.totalSamplerBatchSizeIn(), 3u);
    ASSERT_EQ(groups.totalSamplerBatchSizeOut(), 4u);
    ASSERT_EQ(groups.modelExecuteTokenSize(), 6u);
    ASSERT_EQ(beam->currentBatchSize(), 1);
    ASSERT_EQ(beam->nextBatchSize(), 2);

    const auto logits = torch::tensor({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 9.f, 7.f, 5.f, 3.f, 1.f,
                                       0.f, 2.f, 4.f, 6.f, 8.f, 2.f, 0.f, 4.f, 1.f, 6.f, 3.f, 8.f, 5.f, 9.f, 7.f})
                            .reshape({3, 10});
    const auto hidden = torch::tensor({11.f, 12.f, 21.f, 22.f, 31.f, 32.f}).reshape({3, 2});
    // Distinct distributions per token; adding a row-wise constant would hide
    // incorrect slicing because cross entropy is invariant to that constant.
    const auto all_logits =
        torch::tensor({0.f, 1.f, 2.f, 3.f, 1.f, 0.f, 1.f, 4.f, 2.f, 2.f, 0.f, 5.f, 6.f, 3.f, 1.f, 1.f, 2.f, 7.f})
            .reshape({6, 3});
    MergedOutput merged;
    merged.model_output.logits        = logits.to(torch::kCUDA);
    merged.model_output.hidden_states = hidden.to(torch::kCUDA);
    merged.model_output.all_logits    = all_logits.to(torch::kCUDA);
    merged.sampler_output.token_ids =
        torch::tensor({0, 9, 9, 4, 1, 0, 2, 9, 1, 0, 3, 9, 2, 2, 1, 5}, torch::kInt32).reshape({4, 4});
    merged.sampler_output.beam_index    = torch::tensor({0, 0, 0, 0}, torch::kInt32);
    merged.sampler_output.success       = torch::tensor({true, true, true}, torch::kBool);
    merged.sampler_output.cum_log_probs = torch::tensor({-1.f, -2.f, -3.f, -4.f}).to(torch::kCUDA);
    NormalOutputDispatcher dispatcher({}, GetParam());
    ASSERT_TRUE(dispatcher.dispatch(groups, merged).ok());

    // Computing probabilities must preserve the model logits, including beam rows.
    EXPECT_TRUE(torch::equal(merged.model_output.logits.cpu(), logits));

    EXPECT_EQ(first->completeTokenIdsVec(0), (std::vector<int>{0, 4}));
    EXPECT_EQ(beam->completeTokenIdsVec(0), (std::vector<int>{1, 0, 2}));
    EXPECT_EQ(beam->completeTokenIdsVec(1), (std::vector<int>{1, 0, 3}));
    EXPECT_EQ(last->completeTokenIdsVec(0), (std::vector<int>{2, 2, 1, 5}));
    const std::vector<std::shared_ptr<NormalGenerateStream>> streams{first, beam, last};
    const std::vector<std::vector<int>>                      tokens{{4}, {2, 3}, {5}};
    for (size_t i = 0; i < streams.size(); ++i) {
        SCOPED_TRACE(i);
        ASSERT_FALSE(streams[i]->hasError());
        auto output = streams[i]->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), tokens[i].size());
        auto expected_probs = torch::softmax(logits[i], -1);
        auto probs          = streams[i]->getSoftmaxProbs();
        for (size_t row = 0; row < tokens[i].size(); ++row) {
            const auto& result = output.value().generate_outputs[row];
            ASSERT_TRUE(result.logits.has_value());
            ASSERT_TRUE(result.hidden_states.has_value());
            EXPECT_EQ(toVec<float>(*result.logits), toVec<float>(logits[i]));
            EXPECT_EQ(toVec<float>(*result.hidden_states), toVec<float>(hidden[i]));
            EXPECT_NEAR(probs[row][streams[i]->inputLength()].item<float>(),
                        expected_probs[tokens[i][row]].item<float>(),
                        1e-5);
        }
    }
    EXPECT_FALSE(first->getLoss().defined());
    ASSERT_TRUE(beam->getLoss().defined());
    ASSERT_TRUE(last->getLoss().defined());
    EXPECT_TRUE(torch::allclose(beam->getLoss(), -torch::log_softmax(all_logits[1], -1)[0].reshape({1})));
    EXPECT_TRUE(torch::allclose(
        last->getLoss(),
        torch::stack({-torch::log_softmax(all_logits[3], -1)[2], -torch::log_softmax(all_logits[4], -1)[1]})));
}

TEST_F(NormalBatchStreamProcessorTest, testParallelDispatchWaitsForAllWorkersBeforePropagatingException) {
    class ControlledStream: public NormalGenerateStream {
    public:
        using NormalGenerateStream::NormalGenerateStream;

        void updateOutput(const StreamUpdateInfo& update_info) override {
            before_update();
            NormalGenerateStream::updateOutput(update_info);
            update_completed = true;
        }

        std::function<void()> before_update;
        std::atomic<bool>     update_completed{false};
    };

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;
    auto          make_stream = [&]() {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = hostIntBuffer({0});
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        auto stream = make_shared<ControlledStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    auto failing_stream = make_stream();
    auto delayed_stream = make_stream();

    std::promise<void> release_delayed;
    auto               gate = release_delayed.get_future().share();
    std::promise<void> delayed_started;
    auto               started = delayed_started.get_future().share();
    std::promise<void> worker_throwing;
    auto               throwing   = worker_throwing.get_future();
    delayed_stream->before_update = [&]() {
        delayed_started.set_value();
        gate.wait();
    };
    failing_stream->before_update = [&]() {
        started.wait();
        worker_throwing.set_value();
        throw std::runtime_error("output dispatch worker failed");
    };

    NormalOutputDispatcher dispatcher({}, 2);
    ASSERT_NE(dispatcher.thread_pool_, nullptr);
    StreamGroups stream_groups({failing_stream, delayed_stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({0, 1, 0, 1}, torch::kInt32).reshape({2, 2});
    const auto dispatch_stream             = cuda_graph::graphGetCurrentStream();
    auto       result                      = std::async(std::launch::async, [&]() {
        cuda_graph::GraphStreamGuard stream_guard(dispatch_stream);
        return dispatcher.dispatch(stream_groups, merge_outputs);
    });

    // Always release the gate before any fatal assertion or future destruction.
    EXPECT_EQ(throwing.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(delayed_stream->update_completed.load());
    release_delayed.set_value();

    try {
        const auto status = result.get();
        FAIL() << "dispatch silently ignored the worker exception: " << status.ToString();
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "output dispatch worker failed");
        EXPECT_TRUE(delayed_stream->update_completed.load());
    }
    EXPECT_EQ(delayed_stream->completeTokenIdsVec(0), (std::vector<int>{0, 1}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsGreedyTokenBeforeStreamUpdate) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({2});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({1, 2});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSamplerFailureDoesNotBlockPeerStream) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto make_stream = [&]() {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({2});
        query->generate_config = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    auto failed_stream  = make_stream();
    auto healthy_stream = make_stream();

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({failed_stream, healthy_stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 99, 2, 2}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.success   = torch::tensor({false, true}, torch::kBool);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_TRUE(failed_stream->hasError());
    EXPECT_EQ(failed_stream->completeTokenIdsVec(0), (std::vector<int>{2}));
    EXPECT_FALSE(healthy_stream->hasError());
    EXPECT_EQ(healthy_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testInvalidCompactTokenDoesNotIndexProbabilitiesOrBlockPeer) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto make_stream = [&]() {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer({2});
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens       = 1;
        query->generate_config->return_softmax_probs = true;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    auto failed_stream  = make_stream();
    auto healthy_stream = make_stream();

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({failed_stream, healthy_stream});
    MergedOutput merge_outputs;
    merge_outputs.model_output.logits =
        torch::tensor({0.0f, 1.0f, 9.0f, 0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({2, 3}).to(torch::kCUDA);
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 99, 2, 2}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.success   = torch::tensor({true, true}, torch::kBool);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_TRUE(failed_stream->hasError());
    EXPECT_EQ(failed_stream->completeTokenIdsVec(0), (std::vector<int>{2}));
    EXPECT_FALSE(healthy_stream->hasError());
    EXPECT_EQ(healthy_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_P(OutputDispatchTest, testDynamicBeamRejectsParentOutsidePreviousBatch) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig({0, 1, 2, 4, 7, 9});
    RuntimeConfig   runtime_config;

    auto query                                 = make_shared<GenerateInput>();
    query->input_ids                           = hostIntBuffer({2});
    query->generate_config                     = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams = {2};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->hasError());
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, GetParam());
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids  = torch::tensor({2, 1, 2, 1}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.beam_index = torch::tensor({0, 1}, torch::kInt32);
    merge_outputs.sampler_output.success    = torch::tensor({true}, torch::kBool);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRestoresOnlyCurrentBeamToken) {
    NormalOutputDispatcher dispatcher({0, 2, 4, 7, 9});
    auto batch_token_ids   = torch::tensor({100, 3, 101, 200, 4, 201}, torch::kInt32).reshape({2, 3}).contiguous();
    auto current_token_ids = torch::tensor({3, 4}, torch::kInt32).reshape({2, 1}).contiguous();
    GenerateStreamPtr unused_stream;

    ASSERT_TRUE(dispatcher.restoreCurrentTokenIds(unused_stream, batch_token_ids, current_token_ids, 1));
    EXPECT_EQ(toVec<int32_t>(batch_token_ids), (std::vector<int32_t>{100, 7, 101, 200, 9, 201}));
    EXPECT_EQ(toVec<int32_t>(current_token_ids), (std::vector<int32_t>{7, 9}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSelectedProbabilityUsesCompactToken) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens       = 1;
    query->generate_config->return_softmax_probs = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.model_output.logits =
        torch::tensor({0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({1, 2});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    const auto expected_probability = torch::softmax(torch::tensor({0.0f, 1.0f, 9.0f}), -1)[2].item<float>();
    EXPECT_NEAR(stream->getSoftmaxProbs()[0][1].item<float>(), expected_probability, 1e-6);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabClampsPositiveTopKToLogitsWidth) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig({0, 7});
    RuntimeConfig   runtime_config;

    auto query                    = make_shared<GenerateInput>();
    query->input_ids              = hostIntBuffer({2});
    query->generate_config        = make_shared<GenerateConfig>();
    query->generate_config->top_k = 8;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({1, 2}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_output);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 2);
}

TEST_F(NormalBatchStreamProcessorTest, testDisabledOutputVocabMasksPaddedLogits) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    auto query                               = make_shared<GenerateInput>();
    query->input_ids                         = hostIntBuffer({2});
    query->generate_config                   = make_shared<GenerateConfig>();
    query->generate_config->top_k            = 8;
    query->generate_config->return_all_probs = ReturnAllProbsMode::DEFAULT;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({1, 16}, torch::kFloat32).to(torch::kCUDA);
    model_output.logits.narrow(1, 10, 6).fill_(100.0f);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_output);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 8);
    EXPECT_EQ(sampler_inputs->all_probs.size(1), 16);
    EXPECT_TRUE(torch::isneginf(sampler_inputs->logits.narrow(1, 10, 6)).all().item<bool>());
}

TEST_F(NormalBatchStreamProcessorTest, testPaddedSizeLargerThanKKeepsDispatchAndSamplingOnK) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig({0, 2, 7}, /*padded_size=*/8);
    RuntimeConfig   runtime_config;

    auto query                    = make_shared<GenerateInput>();
    query->input_ids              = hostIntBuffer({2});
    query->generate_config        = make_shared<GenerateConfig>();
    query->generate_config->top_k = 8;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    EXPECT_EQ(stream->outputVocabSize(), 3u);

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});

    // top_k is clamped to the K-wide logits, not to the padded width P
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);
    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_output);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->vocab_size, 3u);
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 3);

    // dispatch consumes K-wide results; compact-id restoration is insensitive to P
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({1, 2});
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabPassesCompactEosOnlyToMultiSeqProcessor) {
    ResourceContext resource_context;
    auto            model_config             = makeOutputVocabModelConfig({0, 2, 4, 7, 9});
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->num_return_sequences = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->logits_processor_list_.size(), 1);
    ASSERT_NE(std::dynamic_pointer_cast<MultiSeqLogitsProcessor>(stream->logits_processor_list_[0]), nullptr);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({2, 5}, torch::kFloat32).to(torch::kCUDA);
    inputs.finished_mask = torch::tensor({false, true}, torch::kBool);
    stream->logits_processor_list_[0]->process(inputs, 0, 2);

    auto processed_logits = inputs.logits.cpu();
    for (int token_id = 0; token_id < 5; ++token_id) {
        if (token_id == 3) {
            EXPECT_FLOAT_EQ(processed_logits[1][token_id].item<float>(), 0.0f);
        } else {
            EXPECT_EQ(processed_logits[1][token_id].item<float>(), -std::numeric_limits<float>::infinity());
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsUnsupportedRequestOnCurrentStream) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto penalty_query                                 = make_shared<GenerateInput>();
    penalty_query->input_ids                           = hostIntBuffer({2});
    penalty_query->generate_config                     = make_shared<GenerateConfig>();
    penalty_query->generate_config->repetition_penalty = 1.1f;
    auto penalty_stream =
        make_shared<NormalGenerateStream>(penalty_query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(penalty_stream->hasError());
    EXPECT_EQ(penalty_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);

    auto beam_query                        = make_shared<GenerateInput>();
    beam_query->input_ids                  = hostIntBuffer({2});
    beam_query->generate_config            = make_shared<GenerateConfig>();
    beam_query->generate_config->num_beams = 2;
    auto beam_stream =
        make_shared<NormalGenerateStream>(beam_query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(beam_stream->hasError());
    EXPECT_EQ(beam_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);

    auto think_query                                  = make_shared<GenerateInput>();
    think_query->input_ids                            = hostIntBuffer({2});
    think_query->generate_config                      = make_shared<GenerateConfig>();
    think_query->generate_config->in_think_mode       = true;
    think_query->generate_config->max_thinking_tokens = 1;
    think_query->generate_config->end_think_token_ids = {7};
    auto think_stream =
        make_shared<NormalGenerateStream>(think_query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(think_stream->hasError());
    EXPECT_EQ(think_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsEachUnsupportedConfigItem) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    struct RejectCase {
        std::string                          message_keyword;
        std::function<void(GenerateConfig&)> mutate;
    };
    std::vector<RejectCase> cases = {
        {"repetition", [](GenerateConfig& c) { c.repetition_penalty = 1.1f; }},
        {"presence", [](GenerateConfig& c) { c.presence_penalty = 0.1f; }},
        {"frequency", [](GenerateConfig& c) { c.frequency_penalty = 0.1f; }},
        {"no_repeat_ngram_size", [](GenerateConfig& c) { c.no_repeat_ngram_size = 2; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.return_logits = true; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.return_prompt_logits = true; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.return_all_probs = ReturnAllProbsMode::DEFAULT; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.calculate_loss = 1; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.select_tokens_id = {2}; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.select_tokens_str = {"a"}; }},
        {"think mode", [](GenerateConfig& c) { c.in_think_mode = true; }},
    };
    for (const auto& reject_case : cases) {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({2});
        query->generate_config = make_shared<GenerateConfig>();
        reject_case.mutate(*query->generate_config);
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        EXPECT_TRUE(stream->hasError()) << "keyword=" << reject_case.message_keyword;
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS) << "keyword=" << reject_case.message_keyword;
        EXPECT_NE(stream->statusInfo().ToString().find(reject_case.message_keyword), std::string::npos)
            << "keyword=" << reject_case.message_keyword;
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsMissingPrimaryEos) {
    ResourceContext resource_context;
    // Default special_tokens.eos_token_id is 0; this vocabulary does not contain it.
    auto          model_config = makeOutputVocabModelConfig({2, 4, 7});
    RuntimeConfig runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({2});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("EOS"), std::string::npos);
}

TEST_P(OutputDispatchTest, testDynamicBeamDispatchReordersAndPlacesTokenAtSeqLength) {
    ResourceContext resource_context;
    ModelConfig     model_config;  // no output vocab: beam layout is orthogonal to pruning
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({5});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams   = {2, 2};
    query->generate_config->max_new_tokens       = 2;
    query->generate_config->return_hidden_states = true;
    query->generate_config->return_logits        = true;
    query->generate_config->return_softmax_probs = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->hasError());

    // Advance one beam step so currentBatchSize == nextBatchSize == 2 and the
    // stream uses the beam token layout (seqLength 1 -> 2).
    int  error_token_id = -1;
    auto first_tokens   = torch::tensor({5, 1, 5, 2}, torch::kInt32).reshape({2, 2});
    ASSERT_TRUE(
        stream->complete_token_ids_->update(first_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_EQ(stream->currentBatchSize(), 2);
    ASSERT_EQ(stream->nextBatchSize(), 2);

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, GetParam());
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    // Per-output-beam rows; the new token sits at column seqLength()==2 while the
    // trailing column holds a different token, so a wrong token_position (last column)
    // would be observable instead of silently passing.
    merge_outputs.sampler_output.token_ids  = torch::tensor({5, 1, 2, 1, 5, 2, 3, 1}, torch::kInt32).reshape({2, 4});
    merge_outputs.sampler_output.beam_index = torch::tensor({1, 0}, torch::kInt32);
    merge_outputs.sampler_output.success    = torch::tensor({true, true}, torch::kBool);
    // Distinct per-row values so parent reordering is observable.
    merge_outputs.model_output.hidden_states =
        torch::tensor({10.0f, 10.0f, 20.0f, 20.0f}).reshape({2, 2}).to(torch::kCUDA);
    merge_outputs.model_output.logits =
        torch::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}).reshape({2, 4}).to(torch::kCUDA);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    ASSERT_FALSE(stream->hasError()) << "code=" << static_cast<int>(stream->statusInfo().code())
                                     << " msg=" << stream->statusInfo().ToString();

    // (1) New tokens land in the seqLength column (index 2), not the last column,
    // and each beam keeps its own parent history.
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{5, 1, 2}));
    EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{5, 2, 3}));

    // (2) Hidden states and logits follow beam_index: output row 0 <- parent row 1,
    // output row 1 <- parent row 0.
    auto outputs_status = stream->nextOutput();
    ASSERT_TRUE(outputs_status.ok());
    auto outputs = std::move(outputs_status.value());
    ASSERT_EQ(outputs.generate_outputs.size(), 2u);
    ASSERT_TRUE(outputs.generate_outputs[0].hidden_states.has_value());
    ASSERT_TRUE(outputs.generate_outputs[1].hidden_states.has_value());
    ASSERT_TRUE(outputs.generate_outputs[0].logits.has_value());
    ASSERT_TRUE(outputs.generate_outputs[1].logits.has_value());
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[0].hidden_states), (std::vector<float>{20.0f, 20.0f}));
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[1].hidden_states), (std::vector<float>{10.0f, 10.0f}));
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[0].logits), (std::vector<float>{4.0f, 5.0f, 6.0f, 7.0f}));
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[1].logits), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f}));

    // (3) Softmax probabilities are gathered from the parent's raw logits row:
    // beam 0 <- raw row 1 at token 2, beam 1 <- raw row 0 at token 3.
    auto probs = stream->getSoftmaxProbs();
    ASSERT_TRUE(probs.defined());
    const auto row1_softmax  = torch::softmax(torch::tensor({4.0f, 5.0f, 6.0f, 7.0f}), -1);
    const auto row0_softmax  = torch::softmax(torch::tensor({0.0f, 1.0f, 2.0f, 3.0f}), -1);
    bool       beam0_matched = false, beam1_matched = false;
    for (int pos = 0; pos < probs.size(1); ++pos) {
        if (std::abs(probs[0][pos].item<float>() - row1_softmax[2].item<float>()) < 1e-5) {
            beam0_matched = true;
        }
        if (std::abs(probs[1][pos].item<float>() - row0_softmax[3].item<float>()) < 1e-5) {
            beam1_matched = true;
        }
    }
    EXPECT_TRUE(beam0_matched);
    EXPECT_TRUE(beam1_matched);
}

TEST_P(OutputDispatchTest, testLoss) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1   = make_shared<GenerateInput>();
    query1->input_ids                       = hostIntBuffer({1});
    query1->generate_config                 = make_shared<GenerateConfig>();
    query1->generate_config->calculate_loss = 1;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, "default", {1});
    stream1->setKVCache(addr1);

    std::shared_ptr<GenerateInput> query3   = make_shared<GenerateInput>();
    query3->input_ids                       = hostIntBuffer({0, 1});
    query3->generate_config                 = make_shared<GenerateConfig>();
    query3->generate_config->calculate_loss = 2;
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(cache_config.topologyPtr());
    addr3.setBatchBlocks(0, "default", {9});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4   = make_shared<GenerateInput>();
    query4->input_ids                       = hostIntBuffer({0, 1, 0});
    query4->generate_config                 = make_shared<GenerateConfig>();
    query4->generate_config->calculate_loss = 1;
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(cache_config.topologyPtr());
    addr4.setBatchBlocks(0, "default", {11, 12});
    stream4->setKVCache(addr4);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false, GetParam());

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().need_all_logits);

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    auto loss_hidden_tensor = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2}).to(torch::kCUDA);
    auto loss_logits_tensor = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2}).to(torch::kCUDA);
    auto loss_all_logits_tensor =
        torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f})
            .reshape({6, 2})
            .to(torch::kCUDA);
    merge_outputs.model_output.hidden_states = loss_hidden_tensor;
    merge_outputs.model_output.logits        = loss_logits_tensor;
    merge_outputs.model_output.all_logits    = loss_all_logits_tensor;
    merge_outputs.sampler_output.token_ids =
        torch::tensor({0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1}, torch::kInt32).reshape({3, 4});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f, 3.0f}).to(torch::kCUDA);
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(stream1->getLoss().defined());
    EXPECT_TRUE(stream3->getLoss().defined());
    auto loss3 = stream3->getLoss();
    EXPECT_EQ(1, loss3.numel());
    EXPECT_NEAR(0.31326, loss3.data_ptr<float>()[0], 0.0001);
    EXPECT_TRUE(stream4->getLoss().defined());
    auto loss4 = stream4->getLoss();
    EXPECT_EQ(2, loss4.numel());
    EXPECT_NEAR(2.25525, *(torch::mean(loss4).exp().data_ptr<float>()), 0.0001);
}

TEST_P(OutputDispatchTest, testCustomOutputDispatch) {
    ModelConfig model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    CacheConfig cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    NormalBatchStreamProcessor processor(model_config, {}, {}, cache_config, false, GetParam());
    auto                       make_stream = [&](bool decode) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer({0, 1});
        query->generate_config                       = make_shared<GenerateConfig>();
        query->custom_output_token_position          = decode ? -1 : 0;
        query->generate_config->is_streaming         = true;
        query->generate_config->num_return_sequences = decode ? 1 : 2;
        query->generate_config->max_new_tokens       = 2;
        auto stream =
            make_shared<NormalGenerateStream>(query, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
        if (decode) {
            query->input_ids = hostIntBuffer({0});
            stream->setIsContextStream(false);
        }
        BatchKVCacheResource addr;
        addr.resetBatchSize(stream->currentBatchSize());
        addr.initGroups(cache_config.topologyPtr());
        for (int i = 0; i < stream->currentBatchSize(); ++i) {
            addr.setBatchBlocks(i, "default", {1, 2});
        }
        stream->setKVCache(addr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    // Mixed decode, cached and selected requests must keep their row correspondence.
    for (const auto& [score_rows, cached_first] : {std::pair{2, false},
                                                   std::pair{2, true},
                                                   std::pair{0, false},
                                                   std::pair{0, true},
                                                   std::pair{1, false},
                                                   std::pair{1, true}}) {
        auto decode  = make_stream(true);
        auto context = make_stream(false);
        auto cached  = make_stream(false);
        cached->setReuseLength(1);  // The scoring token at position 0 is already cached.
        StreamGroups groups(cached_first ? std::list<GenerateStreamPtr>{decode, cached, context} :
                                           std::list<GenerateStreamPtr>{decode, context, cached});
        TensorHolder holder;
        auto         input = processor.gatherModelInput(groups, holder);
        ASSERT_TRUE(input.ok());
        EXPECT_EQ(toVec<int64_t>(input->custom_output_indexes),
                  cached_first ? (std::vector<int64_t>{3, 5}) : (std::vector<int64_t>{1, 3}));
        EXPECT_EQ(cached->reuseLength(), 1);
        MergedOutput outputs;
        outputs.sampler_output.token_ids = torch::tensor({{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}}, torch::kInt32);
        if (score_rows == 0) {
            outputs.model_output.custom_output_error = "handler failure";
        } else {
            outputs.model_output.custom_output =
                torch::tensor({{5.f, 6.f}, {7.f, 8.f}}, torch::kCUDA).narrow(0, 0, score_rows);
        }
        ASSERT_TRUE(processor.dispatch(groups, outputs).ok());
        auto decode_result = decode->nextOutput(100);
        ASSERT_TRUE(decode_result.ok());
        EXPECT_FALSE(decode_result.value().generate_outputs[0].custom_output.has_value());
        auto cached_result = cached->nextOutput(100);
        ASSERT_TRUE(cached_result.ok());  // Another request's head failure must not affect this request.
        for (const auto& output : cached_result.value().generate_outputs) {
            EXPECT_FALSE(output.custom_output.has_value());
            EXPECT_FALSE(output.finished);
        }
        auto result = context->nextOutput(100);
        if (score_rows < 2) {
            EXPECT_FALSE(result.ok());
            EXPECT_FALSE(context->isActive());
            EXPECT_EQ(context->statusInfo().ToString(),
                      score_rows == 0 ? "custom output processor failed: handler failure" :
                                        "custom output row count mismatch");
        } else {
            for (int step = 0; step < 2; ++step) {
                ASSERT_TRUE(result.ok());
                ASSERT_EQ(result.value().generate_outputs.size(), 2u);
                for (int i = 0; i < 2; ++i) {
                    const auto& output = result.value().generate_outputs[i];
                    ASSERT_TRUE(output.custom_output.has_value());
                    EXPECT_FALSE(output.custom_output->is_cuda());
                    EXPECT_EQ(toVec<float>(*output.custom_output), (std::vector<float>{5.f + 2 * i, 6.f + 2 * i}));
                    EXPECT_EQ(output.finished, step == 1);
                }
                if (step == 0) {
                    context->setIsContextStream(false);
                    outputs.model_output             = {};
                    outputs.sampler_output.token_ids = torch::tensor({{2}, {2}}, torch::kInt32);
                    ASSERT_TRUE(processor.dispatch(StreamGroups({context}), outputs).ok());
                    result = context->nextOutput(100);
                }
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testMultimodalGatherBatch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = hostIntBuffer({1, -1, -1, -1, 2});
    query1->generate_config               = make_shared<GenerateConfig>();
    query1->mm_locs                       = torch::tensor({1}, torch::kInt32);
    query1->text_tokens_mask              = torch::tensor({1, 0, 0, 0, 1}, torch::kInt32);
    query1->multimodal_features           = {torch::rand({3, 10}, torch::kFloat16)};
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    stream1->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = hostIntBuffer({3, 4, 5});
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    stream2->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = hostIntBuffer({6, 7, -1, -1, 8});
    query3->generate_config               = make_shared<GenerateConfig>();
    query3->mm_locs                       = torch::tensor({2}, torch::kInt32);
    query3->text_tokens_mask              = torch::tensor({1, 1, 0, 0, 1}, torch::kInt32);
    query3->multimodal_features           = {torch::rand({2, 10}, torch::kFloat16)};
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    stream3->setIsContextStream(true);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    {
        StreamGroups stream_groups(streams);
        TensorHolder holder;

        auto merge_input_status = processor.gatherModelInput(stream_groups, holder);
        EXPECT_TRUE(merge_input_status.ok());

        auto&       model_input      = merge_input_status.value();
        vector<int> combo_tokens     = {1, -1, -1, -1, 2, 3, 4, 5, 6, 7, -1, -1, 8};
        vector<int> input_lengths    = {5, 3, 5};
        vector<int> text_tokens_mask = {1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1};
        vector<int> mm_features_locs = {1, 10};

        EXPECT_EQ(combo_tokens, toVec<int>(model_input.combo_tokens));
        EXPECT_EQ(input_lengths, toVec<int>(model_input.input_lengths));
        EXPECT_EQ(text_tokens_mask, toVec<int>(model_input.text_tokens_mask));
        EXPECT_EQ(mm_features_locs, toVec<int>(model_input.mm_features_locs));

        EXPECT_EQ(model_input.multimodal_features.value().size(), 2);
        EXPECT_EQ(model_input.multimodal_features.value()[0].numel(), 3 * 10);
        EXPECT_EQ(model_input.multimodal_features.value()[1].numel(), 2 * 10);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testPartiallyReusedMultimodalFeatureIsNormalizedWithinItsStream) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    auto query1             = make_shared<GenerateInput>();
    query1->input_ids       = hostIntBuffer({11, 12, 13, 14});
    query1->generate_config = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    stream1->setIsContextStream(true);

    auto fully_reused_feature   = torch::arange(2, torch::kFloat32).reshape({1, 2});
    auto fully_reused_deepstack = torch::arange(4, torch::kFloat32).reshape({2, 1, 2});
    auto feature                = torch::arange(12, torch::kFloat32).reshape({6, 2});
    auto deepstack              = torch::arange(24, torch::kFloat32).reshape({2, 6, 2});
    auto query2                 = make_shared<GenerateInput>();
    query2->input_ids           = hostIntBuffer({-1, -1, -1, -1, -1, -1, -1, 21});
    query2->generate_config     = make_shared<GenerateConfig>();
    query2->mm_locs             = torch::tensor({0, 1}, torch::kInt32);
    query2->text_tokens_mask    = torch::tensor({0, 0, 0, 0, 0, 0, 0, 1}, torch::kInt32);
    query2->multimodal_features = {fully_reused_feature, feature};
    query2->mm_extra_input      = std::vector<torch::Tensor>{fully_reused_deepstack.flatten(), deepstack.flatten()};
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    stream2->setIsContextStream(true);
    stream2->setReuseLength(3);

    std::list<GenerateStreamPtr> streams{stream1, stream2};
    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(merge_input_status.ok());

    auto& model_input = merge_input_status.value();
    EXPECT_EQ(toVec<int>(model_input.combo_tokens), (vector<int>{11, 12, 13, 14, -1, -1, -1, -1, 21}));
    EXPECT_EQ(toVec<int>(model_input.input_lengths), (vector<int>{4, 5}));
    EXPECT_EQ(toVec<int>(model_input.mm_features_locs), (vector<int>{4}));
    EXPECT_EQ(toVec<int>(model_input.text_tokens_mask), (vector<int>{1, 1, 1, 1, 0, 0, 0, 0, 1}));

    ASSERT_TRUE(model_input.multimodal_features.has_value());
    ASSERT_EQ(model_input.multimodal_features.value().size(), 1);
    EXPECT_TRUE(torch::equal(model_input.multimodal_features.value()[0].cpu(), feature.slice(0, 2, 6)));

    ASSERT_TRUE(model_input.mm_extra_input.has_value());
    ASSERT_EQ(model_input.mm_extra_input.value().size(), 1);
    EXPECT_TRUE(torch::equal(model_input.mm_extra_input.value()[0].cpu().reshape({2, 4, 2}), deepstack.slice(1, 2, 6)));

    // The stream retains the complete ViT output for later reuse decisions.
    ASSERT_EQ(stream2->multimodalFeatures().size(), 2);
    EXPECT_TRUE(torch::equal(stream2->multimodalFeatures()[0], fully_reused_feature));
    EXPECT_TRUE(torch::equal(stream2->multimodalFeatures()[1], feature));
}

TEST_F(NormalBatchStreamProcessorTest, testMisalignedMultimodalExtraInputIsRejected) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    auto feature               = torch::arange(12, torch::kFloat32).reshape({6, 2});
    auto query                 = make_shared<GenerateInput>();
    query->input_ids           = hostIntBuffer({-1, -1, -1, -1, -1, -1, 21});
    query->generate_config     = make_shared<GenerateConfig>();
    query->mm_locs             = torch::tensor({0}, torch::kInt32);
    query->text_tokens_mask    = torch::tensor({0, 0, 0, 0, 0, 0, 1}, torch::kInt32);
    query->multimodal_features = {feature};
    query->mm_extra_input      = std::vector<torch::Tensor>{torch::arange(3, torch::kFloat32)};
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(true);
    stream->generate_status_->status = StreamState::RUNNING;

    std::list<GenerateStreamPtr> streams{stream};
    StreamGroups                 stream_groups(streams);
    TensorHolder                 holder;
    bool                         threw = false;
    try {
        (void)processor.gatherModelInput(stream_groups, holder);
    } catch (const std::runtime_error& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("not divisible"), std::string::npos);
    }
    EXPECT_TRUE(threw);
}

}  // namespace rtp_llm
