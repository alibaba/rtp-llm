#include "rtp_llm/cpp/utils/ServingDeviceHealth.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <future>
#include <functional>
#include <limits>
#include <optional>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#define private public
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#undef private

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <limits>
#include <memory>
#include <pybind11/embed.h>

namespace py = pybind11;

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
};

TEST_F(NormalEngineTest, testExecutorDecodesCopyRowsUsingPayloadTags) {
    class CopyPayloadProcessor: public NormalBatchStreamProcessor {
    public:
        CopyPayloadProcessor(const ModelConfig& model, const CacheConfig& config, torch::Tensor mapping):
            NormalBatchStreamProcessor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, false),
            mapping_(std::move(mapping)) {}

        absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& groups,
                                                        TensorHolder&       holder) const override {
            auto result = NormalBatchStreamProcessor::gatherModelInput(groups, holder);
            if (!result.ok()) {
                return result.status();
            }
            auto& inputs = result.value();
            inputs.kv_cache_update_mapping = mapping_;
            return result;
        }

    private:
        torch::Tensor mapping_;
    };
    struct StopBeforeSampling {};

    ModelConfig model;
    model.num_layers                          = 2;
    model.max_seq_len                         = 128;
    model.vocab_size                          = 16;
    model.input_vocab_size                    = 16;
    model.hidden_size                         = 4;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 4;
    model.attn_config.tokens_per_block        = 2;
    model.attn_config.kernel_tokens_per_block = 2;
    CacheConfig config;
    config.layer_num          = 2;
    config.seq_size_per_block = 2;
    config.dtype              = DataType::TYPE_FP16;
    config.fromGroupedSpecs({test::makeResolvedMhaSpec(config.dtype, 1, 4, 2, "second"),
                             test::makeResolvedMhaSpec(config.dtype, 1, 4, 2, "first")},
                            {{1}, {0}},
                            {CacheGroupType::FULL, CacheGroupType::FULL});
    config.finalizeBlockNums(4, RuntimeConfig{});
    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());

    auto blockBytes = [&](int layer, const std::string& tag, int block) {
        auto addr = manager->convertIndexToAddr(layer, tag, block);
        return torch::from_blob(addr.kv_addr,
                                {static_cast<int64_t>(config.group(tag).kvBlockStrideBytes())},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    };
    auto first_src  = blockBytes(0, "first", 1);
    auto first_dst  = blockBytes(0, "first", 2);
    auto second_src = blockBytes(1, "second", 1);
    auto second_dst = blockBytes(1, "second", 2);
    first_src.fill_(11);
    second_src.fill_(22);

    EngineInitParams params;
    params.model_id      = 0;
    params.model_config_ = model;
    params.py_model      = py::none();
    NormalExecutor executor(params, manager, false);
    bool           reached_model = false;
    executor.setModel(std::make_unique<MockModel>(model.vocab_size, [&](const GptModelInputs& inputs) {
        reached_model = true;
        EXPECT_EQ(inputs.kv_cache_group_tags, (std::vector<std::string>{"first", "second"}));
        throw StopBeforeSampling{};
    }));

    for (const bool invalid_row : {false, true}) {
        SCOPED_TRACE(invalid_row);
        first_dst.fill_(33);
        second_dst.fill_(44);
        reached_model = false;
        auto mapping  = invalid_row ? torch::tensor({1, 1, 2, 2, 1, 2}, torch::kInt32).reshape({2, 3}) :
                                      torch::tensor({1, 1, 2}, torch::kInt32).reshape({1, 3});
        executor.setBatchProcessor(std::make_unique<CopyPayloadProcessor>(model, config, mapping));
        auto query                   = std::make_shared<GenerateInput>();
        query->input_ids             = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config       = std::make_shared<GenerateConfig>();
        query->need_release_resource = false;
        auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, ResourceContext{}, nullptr);
        BatchKVCacheResource resource;
        resource.resetBatchSize(1);
        resource.initGroups(config.topologyPtr());
        resource.mutableBlockIds(0, "first").assign({1, 2});
        resource.mutableBlockIds(0, "second").assign({1, 2});
        stream->setKVCache(resource);
        if (invalid_row) {
            EXPECT_ANY_THROW((void)executor.process({stream}));
            EXPECT_FALSE(reached_model);
        } else {
            EXPECT_THROW((void)executor.process({stream}), StopBeforeSampling);
            EXPECT_TRUE(reached_model);
        }
        runtimeSyncAndCheck();
        EXPECT_TRUE(first_src.cpu().eq(11).all().item<bool>());
        EXPECT_TRUE(second_src.cpu().eq(22).all().item<bool>());
        EXPECT_TRUE(first_dst.cpu().eq(33).all().item<bool>());
        EXPECT_TRUE(second_dst.cpu().eq(invalid_row ? 44 : 22).all().item<bool>());
    }
}

TEST_F(NormalEngineTest, testDecodeWarmupReserveTokensAreConvertedToBlocksAfterAddition) {
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/7, /*reserve_tokens=*/1, /*tokens_per_block=*/8), 1u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/8, /*reserve_tokens=*/1, /*tokens_per_block=*/8), 2u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/7, /*reserve_tokens=*/9, /*tokens_per_block=*/8), 2u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/9, /*reserve_tokens=*/8, /*tokens_per_block=*/8), 3u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/0, /*reserve_tokens=*/0, /*tokens_per_block=*/8), 0u);
    EXPECT_EQ(NormalEngine::warmUpReservedBlockCount(/*seq_len=*/8, /*reserve_tokens=*/0, /*tokens_per_block=*/8), 1u);
    EXPECT_ANY_THROW(
        NormalEngine::warmUpReservedBlockCount(/*seq_len=*/1, /*reserve_tokens=*/1, /*tokens_per_block=*/0));
    EXPECT_ANY_THROW(NormalEngine::warmUpReservedBlockCount(
        std::numeric_limits<size_t>::max(), /*reserve_tokens=*/1, /*tokens_per_block=*/8));
}

TEST(NormalEnginePolicyTest, testBatchDecodeSchedulerCannotSuppressMultiDpMtpFakePrefill) {
    EXPECT_FALSE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/true, /*use_batch_decode_scheduler=*/true, /*dp_size=*/2));
    EXPECT_FALSE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/true, /*dp_size=*/1));
    EXPECT_TRUE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/true, /*dp_size=*/2));
    EXPECT_TRUE(NormalEngine::shouldAddMtpFakePrefill(
        /*has_prefill=*/false, /*use_batch_decode_scheduler=*/false, /*dp_size=*/1));
}

TEST(NormalEnginePolicyTest, testPrefillOnlyStreamDoesNotReserveSpeculativeSteps) {
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/4, /*is_prefill_only=*/true), 0);
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/4, /*is_prefill_only=*/false), 4);
    EXPECT_EQ(NormalEngine::reserveStepForStream(/*configured_reserve_step=*/0, /*is_prefill_only=*/true), 0);
}

TEST_F(NormalEngineTest, testMakeStreamReservesKVBeforeEnqueue) {
    struct ReserveCase {
        const char* name;
        int         engine_reserve;
        int         prompt_length;
        bool        prefill_only;
        size_t      expected_blocks;
    };
    const ReserveCase cases[] = {
        {"none_boundary", 0, 8, false, 1},
        {"none_cross_boundary", 0, 9, false, 2},
        {"MTP_boundary", 5, 3, false, 1},
        {"MTP_cross_boundary", 5, 4, false, 2},
        {"DSpARK", 12, 7, false, 3},
        {"prefill_only_boundary", 12, 8, true, 1},
        {"prefill_only_cross_boundary", 12, 9, true, 2},
    };

    for (bool prefill_only : {false, true}) {
        SCOPED_TRACE(testing::Message() << "prefill_only=" << prefill_only);
        CustomConfig config;
        config.role_type        = prefill_only ? RoleType::PREFILL : RoleType::DECODE;
        config.tokens_per_block = 8;
        auto engine             = createMockEngine(config);
        ASSERT_TRUE(engine->stop().ok());
        ASSERT_EQ(engine->getCacheManager()->cacheConfig().seq_size_per_block, 8u);

        for (const auto& test_case : cases) {
            if (test_case.prefill_only != prefill_only) {
                continue;
            }
            SCOPED_TRACE(test_case.name);
            // Inject the constructor's reserve into the real engine after joining its loop;
            // no speculative executor is needed to exercise pre-enqueue KV allocation.
            engine->reserve_step_                  = test_case.engine_reserve;
            auto query                             = make_shared<GenerateInput>();
            query->input_ids                       = torch::arange(1, test_case.prompt_length + 1, torch::kInt32);
            query->generate_config                 = make_shared<GenerateConfig>();
            query->generate_config->max_new_tokens = prefill_only ? 0 : 1;

            auto stream = engine->makeStream(query);
            ASSERT_NE(stream, nullptr);
            ASSERT_FALSE(stream->hasError());
            ASSERT_EQ(stream->seqLength(), test_case.prompt_length);
            ASSERT_EQ(stream->curBlocksNum(), 0u);
            const size_t expected_reserve = prefill_only ? 0 : test_case.engine_reserve;
            EXPECT_EQ(stream->reserveStep(), expected_reserve);
            EXPECT_EQ(stream->completeTokenIdsPtr()->getReserveStep(), expected_reserve);
            EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

            // Match DecodeRpcServer's allocation before P/D handoff and enqueue.
            const auto status = stream->streamCacheResource().initKVBlock();
            ASSERT_TRUE(status.ok()) << status.ToString();
            EXPECT_EQ(stream->curBlocksNum(), test_case.expected_blocks);
            EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
        }
    }
}

TEST_F(NormalEngineTest, testDeviceHealthCachedBeforeStartupAndSurvivesConcurrentStop) {
    class HealthModel: public MockModel {
    public:
        HealthModel(size_t vocab, ServingDeviceHealthPtr health): MockModel(vocab), health_(std::move(health)) {}
        ServingDeviceHealthPtr servingDeviceHealth() const override {
            return health_;
        }

    private:
        ServingDeviceHealthPtr health_;
    };
    for (bool startup_failure : {false, true}) {
        ModelConfig   model;
        RuntimeConfig runtime;
        KVCacheConfig cache;
        auto          params = createEngineInitParams(CustomConfig{}, model, runtime, cache);
        auto          health = std::make_shared<ServingDeviceHealth>();
        if (startup_failure) {
            health->observeRuntimeResult({DeviceRuntimeDomain::CUDA, 700, 0, DeviceRuntimeOperation::EVENT_WAIT});
        }
        NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
            return std::make_unique<HealthModel>(model.vocab_size, health);
        };
        struct FactoryReset {
            ~FactoryReset() {
                NormalExecutor::test_model_factory = nullptr;
            }
        } reset;
        auto engine = std::make_shared<NormalEngine>(params, nullptr);
        EXPECT_EQ(engine->serving_device_health_, health);
        EXPECT_TRUE(engine->EngineBase::checkHealth().ok());
        EXPECT_EQ(engine->checkHealth().ok(), !startup_failure);
        std::atomic<bool> wrong_result{false};
        std::thread       reader([&] {
            for (int i = 0; i < 10000; ++i) {
                if (engine->checkHealth().ok() == startup_failure) {
                    wrong_result.store(true);
                }
            }
        });
        EXPECT_TRUE(engine->stop().ok());
        reader.join();
        EXPECT_FALSE(wrong_result.load());
        EXPECT_EQ(engine->executor_, nullptr);
        EXPECT_EQ(engine->serving_device_health_, health);
        health->observeRuntimeResult({DeviceRuntimeDomain::CUDA, 710, 0, DeviceRuntimeOperation::STREAM_WAIT});
        EXPECT_EQ(engine->checkHealth().code(), ErrorCode::SERVICE_UNHEALTHY);
    }
}

TEST_F(NormalEngineTest, testDeviceHealthRejectsOnlyNewAdmissionAndPreservesErrorsAndReservations) {
    class AdmissionExecutor: public Executor {
    public:
        ServingDeviceHealthPtr           health = std::make_shared<ServingDeviceHealth>();
        HiddenStateCaptureReservationPtr next;
        int                              reservations = 0;
        ServingDeviceHealthPtr           servingDeviceHealth() const override {
            return health;
        }
        HiddenStateCaptureReservationPtr tryReserveHiddenStateCapture(int64_t, int64_t) override {
            ++reservations;
            return std::exchange(next, {});
        }
        absl::Status process(const std::list<GenerateStreamPtr>&, int64_t) override {
            ADD_FAILURE() << "health query/admission must not execute models or collectives";
            return absl::InternalError("unexpected process");
        }
    };
    class AdmissionScheduler: public SchedulerBase {
    public:
        int          enqueues = 0;
        int          groups   = 0;
        int          stops    = 0;
        absl::Status enqueue(const GenerateStreamPtr&) override {
            ++enqueues;
            return absl::OkStatus();
        }
        std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
        enqueueGroup(const std::vector<GenerateStreamPtr>& streams) override {
            ++groups;
            return {std::vector<bool>(streams.size(), true), streams};
        }
        absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
            ADD_FAILURE() << "admission test must not schedule";
            return std::list<GenerateStreamPtr>{};
        }
        absl::Status stop() override {
            ++stops;
            return absl::OkStatus();
        }
        bool empty() override {
            return true;
        }
        int64_t lastScheduleTime() override {
            return 0;
        }
        int64_t onflightStreams() override {
            return enqueues;
        }
    };
    auto engine = createMockEngine(CustomConfig{});
    EXPECT_TRUE(engine->checkHealth().ok());  // No publisher/mock model.
    ASSERT_TRUE(engine->stop().ok());
    auto  executor                                       = std::make_unique<AdmissionExecutor>();
    auto* recorder                                       = executor.get();
    engine->serving_device_health_                       = executor->servingDeviceHealth();
    engine->executor_                                    = std::move(executor);
    auto  scheduler                                      = std::make_unique<AdmissionScheduler>();
    auto* scheduling                                     = scheduler.get();
    engine->scheduler_                                   = std::move(scheduler);
    engine->model_config_.hidden_state_capture_layer_ids = {0};
    auto make_query                                      = [](int64_t id) {
        auto input                                    = std::make_shared<GenerateInput>();
        input->request_id                             = id;
        input->input_ids                              = torch::tensor({1, 2}, torch::kInt32);
        input->generate_config                        = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens        = 1;
        input->generate_config->capture_hidden_states = true;
        input->generate_config->reuse_cache           = false;
        return input;
    };
    auto queued = engine->enqueue(make_query(1));
    ASSERT_FALSE(queued->hasError());
    ASSERT_EQ(scheduling->enqueues, 1);
    int cancel_count = 0;
    recorder->next   = std::make_shared<HiddenStateCaptureReservation>([&]() noexcept { ++cancel_count; });
    auto external    = engine->makeStream(make_query(2));
    ASSERT_TRUE(external->hasHiddenStateCaptureReservation());
    ASSERT_TRUE(external->streamCacheResource().initKVBlock().ok());
    ASSERT_GT(external->curBlocksNum(), 0u);
    int old_error_cancel_count = 0;
    recorder->next = std::make_shared<HiddenStateCaptureReservation>([&]() noexcept { ++old_error_cancel_count; });
    auto old_error = engine->makeStream(make_query(3));
    old_error->reportError(ErrorCode::INVALID_PARAMS, "keep original request error");
    const int reserve_calls = recorder->reservations;
    recorder->health->observeRuntimeResult({DeviceRuntimeDomain::HIP, 719, 0, DeviceRuntimeOperation::STREAM_WAIT});
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(engine->checkHealth().code(), ErrorCode::SERVICE_UNHEALTHY);
        engine->enqueue(external);
        engine->enqueue(old_error);
    }
    EXPECT_EQ(external->statusInfo().code(), ErrorCode::SERVICE_UNHEALTHY);
    EXPECT_EQ(external->curBlocksNum(), 0u);
    EXPECT_FALSE(external->hasHiddenStateCaptureReservation());
    EXPECT_EQ(cancel_count, 1);
    EXPECT_EQ(old_error_cancel_count, 1);
    EXPECT_EQ(old_error->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(old_error->statusInfo().ToString(), "keep original request error");
    auto invalid                             = make_query(4);
    invalid->generate_config->max_new_tokens = 0;
    invalid->generate_config->num_beams      = 2;
    auto rejected                            = engine->makeStream(invalid);
    ASSERT_NE(rejected, nullptr);
    EXPECT_EQ(rejected->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(invalid->generate_config->max_new_tokens, 0);
    EXPECT_EQ(invalid->generate_config->num_beams, 2);
    EXPECT_NE(rejected->generateConfig().get(), invalid->generate_config.get());
    auto single = engine->enqueue(make_query(5));
    EXPECT_EQ(single->statusInfo().code(), ErrorCode::SERVICE_UNHEALTHY);
    auto [accepted, streams] = engine->enqueueMultiple({make_query(6), invalid, make_query(7)});
    EXPECT_EQ(accepted, (std::vector<bool>{false, false, false}));
    ASSERT_EQ(streams.size(), 3u);
    for (size_t i = 0; i < streams.size(); ++i) {
        ASSERT_NE(streams[i], nullptr);
        EXPECT_EQ(streams[i]->statusInfo().code(), i == 1 ? ErrorCode::INVALID_PARAMS : ErrorCode::SERVICE_UNHEALTHY);
        EXPECT_FALSE(streams[i]->hasHiddenStateCaptureReservation());
    }
    EXPECT_EQ(streams[0]->streamId(), 6);
    EXPECT_EQ(streams[1]->streamId(), 4);
    EXPECT_EQ(streams[2]->streamId(), 7);
    EXPECT_EQ(recorder->reservations, reserve_calls);
    EXPECT_EQ(scheduling->enqueues, 1);
    EXPECT_EQ(scheduling->groups, 0);
    EXPECT_EQ(scheduling->stops, 0);
    EXPECT_FALSE(queued->hasError());
}

TEST_F(NormalEngineTest, testDeviceHealthDoesNotChangeInternalDpFakeControlFlow) {
    class TraceExecutor: public Executor {
    public:
        std::vector<std::vector<int64_t>> trace;
        absl::Status                      process(const std::list<GenerateStreamPtr>& streams, int64_t) override {
            EXPECT_EQ(streams.size(), 1u);
            for (const auto& stream : streams) {
                trace.push_back({stream->hasError(),
                                                      stream->isContextStream(),
                                                      stream->seqLength(),
                                                      stream->isFakeStream(),
                                                      stream->generateInput()->fake_query,
                                                      stream->generateConfig()->max_new_tokens,
                                                      static_cast<int64_t>(stream->reserveStep())});
            }
            return absl::OkStatus();
        }
    };
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    engine->parallelism_config.dp_size = 2;
    engine->pd_sep_config.role_type    = RoleType::PDFUSION;
    auto  recorder                     = std::make_unique<TraceExecutor>();
    auto* trace                        = recorder.get();
    engine->executor_                  = std::move(recorder);
    engine->serving_device_health_     = std::make_shared<ServingDeviceHealth>();
    // The real empty scheduler -> mayAddFakeStream -> fake update -> process
    // path is identical before/after health latches; no executor phase is skipped.
    ASSERT_TRUE(engine->step().ok());
    engine->serving_device_health_->observeRuntimeResult(
        {DeviceRuntimeDomain::CUDA, 700, 0, DeviceRuntimeOperation::STREAM_WAIT});
    ASSERT_TRUE(engine->step().ok());
    ASSERT_EQ(trace->trace.size(), 2u);
    EXPECT_EQ(trace->trace[0], trace->trace[1]);
    EXPECT_EQ(trace->trace[1][0], 0);  // no error
    EXPECT_EQ(trace->trace[1][1], 0);  // updated to decode
    EXPECT_EQ(trace->trace[1][2], 2);  // one prompt token + fake update
    EXPECT_EQ(trace->trace[1][3], 1);
    EXPECT_EQ(trace->trace[1][4], 1);
    auto external                             = std::make_shared<GenerateInput>();
    external->input_ids                       = torch::tensor({1}, torch::kInt32);
    external->generate_config                 = std::make_shared<GenerateConfig>();
    external->generate_config->max_new_tokens = 1;
    external->fake_query                      = true;  // This flag is NOT authority for an internal bypass.
    EXPECT_EQ(engine->makeStream(external)->statusInfo().code(), ErrorCode::SERVICE_UNHEALTHY);
}

TEST_F(NormalEngineTest, testParameterPriorityAndHealthyNormalizationKeepOriginalObject) {
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    auto health                    = std::make_shared<ServingDeviceHealth>();
    engine->serving_device_health_ = health;
    auto make_query                = [] {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        return query;
    };
    auto scoring                                   = make_query();
    scoring->generate_config->return_prompt_logits = true;
    scoring->generate_config->max_new_tokens       = 0;
    auto original                                  = scoring->generate_config;
    auto stream                                    = engine->makeStream(scoring);
    ASSERT_FALSE(stream->hasError());
    EXPECT_EQ(stream->generateConfig(), original);
    EXPECT_EQ(scoring->generate_config, original);
    EXPECT_EQ(original->max_new_tokens, 1);
    EXPECT_FALSE(original->is_streaming);
    EXPECT_FALSE(original->reuse_cache);
    health->observeRuntimeResult({DeviceRuntimeDomain::CUDA, 700, 0, DeviceRuntimeOperation::STREAM_WAIT});
    auto invalid                             = make_query();
    invalid->generate_config->max_new_tokens = 0;
    invalid->generate_config->num_beams      = 2;
    EXPECT_EQ(engine->makeStream(invalid)->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(invalid->generate_config->max_new_tokens, 0);
    EXPECT_EQ(invalid->generate_config->num_beams, 2);
    auto valid                             = make_query();
    valid->generate_config->max_new_tokens = 1;
    EXPECT_EQ(engine->makeStream(valid)->statusInfo().code(), ErrorCode::SERVICE_UNHEALTHY);
    auto normalized_copy_only                                   = make_query();
    normalized_copy_only->generate_config->return_prompt_logits = true;
    normalized_copy_only->generate_config->max_new_tokens       = 0;
    EXPECT_EQ(engine->makeStream(normalized_copy_only)->statusInfo().code(), ErrorCode::SERVICE_UNHEALTHY);
    EXPECT_EQ(normalized_copy_only->generate_config->max_new_tokens, 0);
    EXPECT_TRUE(normalized_copy_only->generate_config->reuse_cache);
}

TEST_F(NormalEngineTest, testSchedulerOwnedResubmissionCannotReleaseWaitingOrRunningResources) {
    class HoldingScheduler: public SchedulerBase {
    public:
        std::vector<GenerateStreamPtr>                held;
        std::function<void(const GenerateStreamPtr&)> prepare;
        std::mutex                                    mutex;
        std::condition_variable                       changed;
        bool                                          entered          = false;
        bool                                          continue_handoff = false;
        bool                                          block_handoff    = false;
        int                                           calls            = 0;
        absl::Status                                  enqueue(const GenerateStreamPtr& stream) override {
            accept({stream});
            return absl::OkStatus();
        }
        std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
        enqueueGroup(const std::vector<GenerateStreamPtr>& streams) override {
            accept(streams);
            return {std::vector<bool>(streams.size(), true), streams};
        }
        void accept(const std::vector<GenerateStreamPtr>& streams) {
            std::unique_lock<std::mutex> lock(mutex);
            ++calls;
            for (const auto& stream : streams) {
                EXPECT_EQ(stream->schedulerAdmission(), GenerateStream::SchedulerAdmission::HANDED_OFF);
                prepare(stream);
                held.push_back(stream);
            }
            entered = true;
            changed.notify_all();
            if (block_handoff)
                changed.wait(lock, [&] { return continue_handoff; });
        }
        absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
            return std::list<GenerateStreamPtr>{};
        }
        absl::Status stop() override {
            return absl::OkStatus();
        }
        bool empty() override {
            return held.empty();
        }
        int64_t lastScheduleTime() override {
            return 0;
        }
        int64_t onflightStreams() override {
            return held.size();
        }
    };
    for (bool batch : {false, true}) {
        for (auto state : {StreamState::WAITING, StreamState::RUNNING}) {
            SCOPED_TRACE(batch);
            SCOPED_TRACE(static_cast<int>(state));
            auto engine = createMockEngine(CustomConfig{});
            ASSERT_TRUE(engine->stop().ok());
            auto health                    = std::make_shared<ServingDeviceHealth>();
            engine->serving_device_health_ = health;
            auto  scheduler                = std::make_unique<HoldingScheduler>();
            auto* holder                   = scheduler.get();
            auto  cancellations            = std::make_shared<std::atomic<int>>(0);
            holder->prepare                = [&, cancellations](const GenerateStreamPtr& stream) {
                ASSERT_TRUE(stream->streamCacheResource().initKVBlock().ok());
                stream->setHiddenStateCaptureReservation(
                    std::make_shared<HiddenStateCaptureReservation>([cancellations]() noexcept { ++*cancellations; }));
                stream->generate_status_->status = state;
                if (batch && stream->streamId() == 11) {
                    stream->reportError(ErrorCode::INVALID_PARAMS, "scheduler-owned request error");
                }
            };
            holder->block_handoff = true;
            engine->scheduler_    = std::move(scheduler);
            auto make_query       = [](int id) {
                auto query                             = std::make_shared<GenerateInput>();
                query->request_id                      = id;
                query->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
                query->generate_config                 = std::make_shared<GenerateConfig>();
                query->generate_config->max_new_tokens = 1;
                return query;
            };
            auto                           first    = make_query(10);
            auto                           second   = make_query(11);
            auto                           admitted = std::async(std::launch::async, [&] {
                if (batch)
                    return engine->enqueueMultiple({first, second});
                auto stream = engine->enqueue(first);
                return std::make_pair(std::vector<bool>{!stream->hasError()}, std::vector<GenerateStreamPtr>{stream});
            });
            bool                           entered;
            std::vector<GenerateStreamPtr> owned;
            {
                std::unique_lock<std::mutex> lock(holder->mutex);
                entered = holder->changed.wait_for(lock, std::chrono::seconds(10), [&] { return holder->entered; });
                owned   = holder->held;
            }
            // Even a healthy duplicate cannot enter scheduling a second time.
            for (const auto& original_stream : owned) {
                auto duplicate = original_stream;
                engine->enqueue(duplicate);
                EXPECT_NE(duplicate, original_stream);
                EXPECT_EQ(duplicate->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            }
            // Fault races AFTER the final health check and ownership publication,
            // but BEFORE the scheduler returns. These requests stay admitted.
            health->observeRuntimeResult({DeviceRuntimeDomain::CUDA, 700, 0, DeviceRuntimeOperation::STREAM_WAIT});
            EXPECT_TRUE(entered);
            for (const auto& original_stream : owned) {
                const auto blocks         = original_stream->curBlocksNum();
                const auto original_error = original_stream->statusInfo();
                EXPECT_GT(blocks, 0u);
                auto duplicate = original_stream;
                engine->enqueue(duplicate);
                EXPECT_NE(duplicate, original_stream);
                EXPECT_EQ(duplicate->statusInfo().code(),
                          original_error.ok() ? ErrorCode::SERVICE_UNHEALTHY : original_error.code());
                EXPECT_EQ(original_stream->statusInfo().code(), original_error.code());
                EXPECT_EQ(original_stream->statusInfo().ToString(), original_error.ToString());
                EXPECT_EQ(original_stream->getStatus(), state);
                EXPECT_EQ(original_stream->curBlocksNum(), blocks);
                EXPECT_TRUE(original_stream->hasHiddenStateCaptureReservation());
                const auto free_blocks = engine->resource_context_.cache_manager->freeBlocksNum();
                for (bool copy_on_write : {false, true}) {
                    SCOPED_TRACE(copy_on_write);
                    // Both an implicit derived copy and the base-reference COW
                    // constructor share KV, but neither may release the owner's KV.
                    GenerateStreamPtr wrapper =
                        copy_on_write ? std::make_shared<NormalGenerateStream>(*original_stream) :
                                        std::make_shared<NormalGenerateStream>(
                                            *std::static_pointer_cast<NormalGenerateStream>(original_stream));
                    EXPECT_NE(wrapper.get(), original_stream.get());
                    EXPECT_EQ(&wrapper->streamCacheResource(), &original_stream->streamCacheResource());
                    EXPECT_FALSE(wrapper->ownsResourceRelease());
                    std::weak_ptr<GenerateStream> old_wrapper = wrapper;
                    engine->enqueue(wrapper);  // Replacement destroys the independent wrapper here.
                    EXPECT_TRUE(old_wrapper.expired());
                    EXPECT_EQ(wrapper->statusInfo().code(),
                              original_error.ok() ? ErrorCode::SERVICE_UNHEALTHY : original_error.code());
                    wrapper.reset();
                    EXPECT_EQ(original_stream->curBlocksNum(), blocks);
                    EXPECT_EQ(original_stream->getStatus(), state);
                    EXPECT_EQ(original_stream->statusInfo().ToString(), original_error.ToString());
                    EXPECT_TRUE(original_stream->hasHiddenStateCaptureReservation());
                    EXPECT_EQ(engine->resource_context_.cache_manager->freeBlocksNum(), free_blocks);
                    EXPECT_EQ(cancellations->load(), 0);
                }
            }
            EXPECT_EQ(cancellations->load(), 0);
            {
                std::lock_guard<std::mutex> lock(holder->mutex);
                holder->continue_handoff = true;
            }
            holder->changed.notify_all();
            auto result = admitted.get();
            EXPECT_EQ(result.first, std::vector<bool>(batch ? 2 : 1, true));
            EXPECT_EQ(holder->calls, 1);
            EXPECT_EQ(cancellations->load(), 0);
            // The scheduler, not rejected resubmission, owns eventual cleanup.
            for (const auto& stream : holder->held) {
                EXPECT_TRUE(stream->ownsResourceRelease());
                const auto before_release = engine->resource_context_.cache_manager->freeBlocksNum();
                stream->releaseResource();
                EXPECT_EQ(stream->curBlocksNum(), 0u);
                const auto free_blocks = engine->resource_context_.cache_manager->freeBlocksNum();
                EXPECT_GT(free_blocks, before_release);
                stream->releaseResource();
                EXPECT_EQ(engine->resource_context_.cache_manager->freeBlocksNum(), free_blocks);
            }
            EXPECT_EQ(cancellations->load(), batch ? 2 : 1);
        }
    }
}

TEST_F(NormalEngineTest, testAdmissionExceptionsAndExplicitSchedulerRefusalReleaseOnlyBeforeHandoff) {
    class ThrowingHealthEngine: public NormalEngine {
    public:
        using NormalEngine::NormalEngine;
        mutable std::function<void()> health_hook;
        ErrorInfo                     checkHealth() const override {
            if (health_hook) {
                health_hook();
            }
            return NormalEngine::checkHealth();
        }
    };
    class RefusingScheduler: public SchedulerBase {
    public:
        bool                                          ambiguous_throw = false;
        int                                           calls           = 0;
        std::vector<GenerateStreamPtr>                seen;
        std::function<void(const GenerateStreamPtr&)> prepare;
        absl::Status                                  enqueue(const GenerateStreamPtr& stream) override {
            visit({stream});
            return absl::ResourceExhaustedError("explicit refusal");
        }
        std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
        enqueueGroup(const std::vector<GenerateStreamPtr>& streams) override {
            visit(streams);
            return {std::vector<bool>(streams.size(), false), streams};
        }
        void visit(const std::vector<GenerateStreamPtr>& streams) {
            ++calls;
            seen = streams;
            for (const auto& stream : streams) {
                EXPECT_EQ(stream->schedulerAdmission(), GenerateStream::SchedulerAdmission::HANDED_OFF);
                if (prepare) {
                    prepare(stream);
                }
                if (!ambiguous_throw) {
                    stream->reportError(ErrorCode::INVALID_PARAMS, "scheduler refusal retains original error");
                }
            }
            if (ambiguous_throw) {
                throw std::runtime_error("scheduler may have accepted these streams");
            }
        }
        absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
            return std::list<GenerateStreamPtr>{};
        }
        absl::Status stop() override {
            return absl::OkStatus();
        }
        bool empty() override {
            return seen.empty();
        }
        int64_t lastScheduleTime() override {
            return 0;
        }
        int64_t onflightStreams() override {
            return seen.size();
        }
    };
    ModelConfig   model;
    RuntimeConfig runtime;
    KVCacheConfig cache;
    auto          params               = createEngineInitParams(CustomConfig{}, model, runtime, cache);
    NormalExecutor::test_model_factory = [vocab = model.vocab_size](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab);
    };
    struct FactoryReset {
        ~FactoryReset() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } reset;
    auto engine = std::make_shared<ThrowingHealthEngine>(params, nullptr);
    ASSERT_TRUE(engine->stop().ok());
    auto make_query = [] {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 1;
        return input;
    };
    // Failure comes from the actual virtual checkHealth called AFTER the CAS.
    // Exercise both the first and final single-stream gates, including bad_alloc.
    for (int gate : {0, 1}) {
        auto  scheduler        = std::make_unique<RefusingScheduler>();
        auto* observer         = scheduler.get();
        engine->scheduler_     = std::move(scheduler);
        auto       stream      = engine->makeStream(make_query());
        const auto free_before = engine->resource_context_.cache_manager->freeBlocksNum();
        ASSERT_TRUE(stream->initKVBlock().ok());
        auto cancellations = std::make_shared<int>(0);
        stream->setHiddenStateCaptureReservation(
            std::make_shared<HiddenStateCaptureReservation>([cancellations]() noexcept { ++*cancellations; }));
        int checks          = 0;
        engine->health_hook = [&, gate] {
            EXPECT_EQ(stream->schedulerAdmission(), GenerateStream::SchedulerAdmission::CLAIMED);
            if (checks++ == gate) {
                throw std::bad_alloc();
            }
        };
        EXPECT_THROW(engine->enqueue(stream), std::bad_alloc);
        engine->health_hook = {};
        EXPECT_EQ(checks, gate + 1);
        EXPECT_EQ(observer->calls, 0);
        EXPECT_EQ(stream->schedulerAdmission(), GenerateStream::SchedulerAdmission::REJECTED);
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_EQ(stream->curBlocksNum(), 0u);
        EXPECT_EQ(*cancellations, 1);
        EXPECT_EQ(engine->resource_context_.cache_manager->freeBlocksNum(), free_before);
        auto original = stream;
        engine->enqueue(stream);
        EXPECT_EQ(stream, original);
        EXPECT_EQ(observer->calls, 0);
    }
    for (bool batch : {false, true}) {
        for (bool ambiguous : {false, true}) {
            SCOPED_TRACE(batch);
            SCOPED_TRACE(ambiguous);
            auto  scheduler           = std::make_unique<RefusingScheduler>();
            auto* observer            = scheduler.get();
            observer->ambiguous_throw = ambiguous;
            auto cancellations        = std::make_shared<int>(0);
            auto prepare              = [cancellations](const GenerateStreamPtr& stream) {
                EXPECT_TRUE(stream->initKVBlock().ok());
                stream->setHiddenStateCaptureReservation(
                    std::make_shared<HiddenStateCaptureReservation>([cancellations]() noexcept { ++*cancellations; }));
            };
            engine->scheduler_     = std::move(scheduler);
            const auto free_before = engine->resource_context_.cache_manager->freeBlocksNum();
            if (batch) {
                observer->prepare = prepare;
                if (ambiguous) {
                    EXPECT_THROW(engine->enqueueMultiple({make_query(), make_query()}), std::runtime_error);
                } else {
                    auto result = engine->enqueueMultiple({make_query(), make_query()});
                    EXPECT_EQ(result.first, std::vector<bool>({false, false}));
                    EXPECT_EQ(result.second, observer->seen);
                }
            } else {
                auto stream = engine->makeStream(make_query());
                prepare(stream);  // External stream already owns KV before enqueue.
                if (ambiguous) {
                    EXPECT_THROW(engine->enqueue(stream), std::runtime_error);
                } else {
                    engine->enqueue(stream);
                }
                ASSERT_EQ(observer->seen.size(), 1u);
                EXPECT_EQ(observer->seen[0], stream);
            }
            EXPECT_EQ(observer->calls, 1);
            ASSERT_EQ(observer->seen.size(), batch ? 2u : 1u);
            EXPECT_EQ(*cancellations, ambiguous ? 0 : static_cast<int>(observer->seen.size()));
            for (const auto& stream : observer->seen) {
                EXPECT_EQ(stream->schedulerAdmission(),
                          ambiguous ? GenerateStream::SchedulerAdmission::HANDED_OFF :
                                      GenerateStream::SchedulerAdmission::REJECTED);
                if (ambiguous) {
                    EXPECT_FALSE(stream->hasError());
                    EXPECT_GT(stream->curBlocksNum(), 0u);
                    EXPECT_TRUE(stream->hasHiddenStateCaptureReservation());
                    // The fake scheduler, not unwinding admission, releases accepted work.
                    stream->releaseResource();
                } else {
                    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                    EXPECT_EQ(stream->curBlocksNum(), 0u);
                    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
                }
            }
            EXPECT_EQ(*cancellations, observer->seen.size());
            EXPECT_EQ(engine->resource_context_.cache_manager->freeBlocksNum(), free_before);
        }
    }
    // Final batch health failure unwinds all outstanding claims/reservations.
    class ReservingExecutor: public Executor {
    public:
        std::shared_ptr<int> cancellations = std::make_shared<int>(0);
        int                  reservations  = 0;
        absl::Status         process(const std::list<GenerateStreamPtr>&, int64_t) override {
            ADD_FAILURE() << "admission tests must not execute a model";
            return absl::InternalError("unexpected process");
        }
        HiddenStateCaptureReservationPtr tryReserveHiddenStateCapture(int64_t, int64_t) override {
            ++reservations;
            return std::make_shared<HiddenStateCaptureReservation>([count = cancellations]() noexcept { ++*count; });
        }
    };
    auto  recorder                                       = std::make_unique<ReservingExecutor>();
    auto* reservations                                   = recorder.get();
    engine->executor_                                    = std::move(recorder);
    engine->model_config_.hidden_state_capture_layer_ids = {0};
    auto  scheduler                                      = std::make_unique<RefusingScheduler>();
    auto* observer                                       = scheduler.get();
    engine->scheduler_                                   = std::move(scheduler);
    auto first                                           = make_query();
    auto second                                          = make_query();
    for (const auto& input : {first, second}) {
        input->generate_config->capture_hidden_states = true;
        input->generate_config->reuse_cache           = false;
    }
    int checks          = 0;
    engine->health_hook = [&] {
        if (++checks == 3) {  // two createStream checks, then the batch handoff gate
            throw std::bad_alloc();
        }
    };
    EXPECT_THROW(engine->enqueueMultiple({first, second}), std::bad_alloc);
    engine->health_hook = {};
    EXPECT_EQ(checks, 3);
    EXPECT_EQ(observer->calls, 0);
    EXPECT_EQ(reservations->reservations, 2);
    EXPECT_EQ(*reservations->cancellations, 2);
}

TEST_F(NormalEngineTest, testCaptureAdmissionAndEnqueueCancellationUseRequestAndGlobalGates) {
    class RecordingExecutor: public Executor {
    public:
        absl::Status process(const std::list<GenerateStreamPtr>&, int64_t) override {
            ADD_FAILURE() << "the engine loop must remain stopped for admission-only tests";
            return absl::InternalError("unexpected model execution");
        }

        HiddenStateCaptureReservationPtr tryReserveHiddenStateCapture(int64_t request_id, int64_t tokens) override {
            reserve_calls.emplace_back(request_id, tokens);
            if (reserve_error) {
                std::rethrow_exception(reserve_error);
            }
            return std::exchange(next_reservation, {});
        }

        HiddenStateCaptureReservationPtr         next_reservation;
        std::vector<std::pair<int64_t, int64_t>> reserve_calls;
        std::exception_ptr                       reserve_error;
    };
    enum class Rejection {
        NONE,
        EXISTING_ERROR,
        SCHEDULER_LENGTH
    };
    struct TestCase {
        const char*        name;
        bool               global_capture;
        int                max_new_tokens;
        bool               capture_hidden_states;
        bool               reuse_cache;
        bool               offer_reservation;
        bool               expect_reserve_call;
        Rejection          rejection        = Rejection::NONE;
        std::exception_ptr reserve_error    = {};
        bool               expect_bad_alloc = false;
    };
    const TestCase cases[] = {
        {"global-off", false, 1, true, false, true, false},
        {"generation-default", true, 1, false, true, true, false},
        {"generation-opt-in", true, 1, true, false, true, true},
        {"generation-reuse-conflict", true, 1, true, true, true, false},
        {"zero-token-legacy", true, 0, false, true, true, true},
        {"zero-token-explicit-reuse-conflict", true, 0, true, true, true, false},
        {"admission-miss", true, 1, true, false, false, true},
        {"generation-reserve-error",
         true,
         1,
         true,
         false,
         false,
         true,
         Rejection::NONE,
         std::make_exception_ptr(std::runtime_error("capture temporarily unavailable"))},
        {"prefill-only-reserve-error",
         true,
         0,
         false,
         true,
         false,
         true,
         Rejection::NONE,
         std::make_exception_ptr(std::runtime_error("capture temporarily unavailable"))},
        {"generation-bad-alloc",
         true,
         1,
         true,
         false,
         false,
         true,
         Rejection::NONE,
         std::make_exception_ptr(std::bad_alloc()),
         true},
        {"prefill-only-bad-alloc",
         true,
         0,
         false,
         true,
         false,
         true,
         Rejection::NONE,
         std::make_exception_ptr(std::bad_alloc()),
         true},
        {"enqueue-invalid-stream", true, 1, true, false, true, true, Rejection::EXISTING_ERROR},
        {"enqueue-scheduler-length-rejection", true, 1, true, false, true, true, Rejection::SCHEDULER_LENGTH},
    };
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    auto  executor     = std::make_unique<RecordingExecutor>();
    auto* recorder     = executor.get();
    engine->executor_  = std::move(executor);
    int64_t request_id = 1000;
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        ASSERT_EQ(engine->getScheduler().onflightStreams(), 0);
        engine->model_config_.hidden_state_capture_layer_ids =
            test_case.global_capture ? std::vector<int64_t>{0} : std::vector<int64_t>{};
        engine->reserve_step_                               = 0;
        auto                             cancellation_count = std::make_shared<std::atomic<int>>(0);
        HiddenStateCaptureReservationPtr offered;
        if (test_case.offer_reservation) {
            offered = std::make_shared<HiddenStateCaptureReservation>(
                [cancellation_count]() noexcept { ++*cancellation_count; });
        }
        recorder->next_reservation = offered;
        recorder->reserve_error    = test_case.reserve_error;
        recorder->reserve_calls.clear();
        auto query             = std::make_shared<GenerateInput>();
        query->request_id      = request_id++;
        query->input_ids       = torch::tensor({1, 2}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        ASSERT_FALSE(query->generate_config->capture_hidden_states);
        query->generate_config->max_new_tokens        = test_case.max_new_tokens;
        query->generate_config->capture_hidden_states = test_case.capture_hidden_states;
        query->generate_config->reuse_cache           = test_case.reuse_cache;

        const auto        expected_calls = test_case.expect_reserve_call ?
                                               (std::vector<std::pair<int64_t, int64_t>>{{query->request_id, 2}}) :
                                               (std::vector<std::pair<int64_t, int64_t>>{});
        GenerateStreamPtr stream;
        if (test_case.expect_bad_alloc) {
            EXPECT_THROW(stream = engine->makeStream(query), std::bad_alloc);
            EXPECT_EQ(stream, nullptr);
            EXPECT_EQ(recorder->reserve_calls, expected_calls);
            EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
            continue;
        }
        ASSERT_NO_THROW(stream = engine->makeStream(query));
        ASSERT_NE(stream, nullptr);
        ASSERT_FALSE(stream->hasError());
        EXPECT_EQ(stream->generateConfig()->max_new_tokens, test_case.max_new_tokens);
        EXPECT_EQ(stream->generateConfig()->isPrefillOnly(), test_case.max_new_tokens == 0);
        EXPECT_EQ(recorder->reserve_calls, expected_calls);
        const bool expected_reservation = test_case.expect_reserve_call && test_case.offer_reservation;
        EXPECT_EQ(stream->hasHiddenStateCaptureReservation(), expected_reservation);
        EXPECT_EQ(cancellation_count->load(), 0);
        EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

        if (test_case.rejection == Rejection::EXISTING_ERROR) {
            stream->reportError(ErrorCode::INVALID_PARAMS, "existing request error");
        } else if (test_case.rejection == Rejection::SCHEDULER_LENGTH) {
            // enqueue rewrites reserveStep before the real FIFO length check.
            engine->reserve_step_ = engine->model_config_.max_seq_len;
        }
        engine->enqueue(stream);
        EXPECT_EQ(recorder->reserve_calls, expected_calls);
        if (test_case.rejection == Rejection::NONE) {
            EXPECT_FALSE(stream->hasError());
            EXPECT_EQ(engine->getScheduler().onflightStreams(), 1);
            EXPECT_EQ(stream->hasHiddenStateCaptureReservation(), expected_reservation);
            EXPECT_EQ(cancellation_count->load(), 0);
        } else {
            EXPECT_TRUE(stream->hasError());
            EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
            EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
            EXPECT_EQ(cancellation_count->load(), 1);
            if (test_case.rejection == Rejection::EXISTING_ERROR) {
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_EQ(stream->stopReason(), "existing request error");
            } else {
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::LONG_PROMPT_ERROR);
            }
        }
        stream->cancelHiddenStateCapture();
        stream->cancelHiddenStateCapture();
        EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
        EXPECT_EQ(cancellation_count->load(), expected_reservation ? 1 : 0);
        if (expected_reservation) {
            EXPECT_FALSE(offered->active());
            offered->cancel();
            EXPECT_EQ(cancellation_count->load(), 1);
        }
        ASSERT_TRUE(engine->getScheduler().stop().ok());
        recorder->next_reservation.reset();
    }
}

TEST_F(NormalEngineTest, testStopJoinsAsyncWorkerWithoutMovingExecutorDestructionToWorker) {
    struct ShutdownState {
        std::promise<std::thread::id> callback_entered;
        std::promise<std::thread::id> destructor_entered;
        std::promise<void>            worker_returned;
        std::promise<void>            reserve_entered;
        std::shared_future<void>      resume_callback;
        std::shared_future<void>      resume_reserve;
        std::atomic<int>              cancellation_count{0};
        std::atomic<int>              admission_cancellation_count{0};
        std::atomic<int>              reserve_calls{0};
        std::atomic<bool>             callback_resumed{false};
        std::atomic<bool>             reserve_resumed{false};
    };
    class ObservedStopScheduler: public FIFOScheduler {
    public:
        using FIFOScheduler::FIFOScheduler;

        absl::Status stop() override {
            auto status = FIFOScheduler::stop();
            if (!std::exchange(stop_reported_, true)) {
                stop_entered.set_value();
            }
            return status;
        }

        std::promise<void> stop_entered;

    private:
        bool stop_reported_{false};
    };
    class ObservedExecutor: public NormalExecutor {
    public:
        ObservedExecutor(const EngineInitParams&          params,
                         std::shared_ptr<ShutdownState>   state,
                         HiddenStateCaptureReservationPtr reservation):
            NormalExecutor(params, nullptr), state_(std::move(state)), reservation_(std::move(reservation)) {}

        ~ObservedExecutor() override {
            state_->destructor_entered.set_value(std::this_thread::get_id());
        }

        HiddenStateCaptureReservationPtr tryReserveHiddenStateCapture(int64_t, int64_t) override {
            auto state = state_;
            if (++state->reserve_calls == 1) {
                return std::exchange(reservation_, {});
            }
            // Neither the blocked call nor its returned token owns the executor.
            auto reservation = std::make_shared<HiddenStateCaptureReservation>(
                [state]() noexcept { ++state->admission_cancellation_count; });
            state->reserve_entered.set_value();
            state->reserve_resumed =
                state->resume_reserve.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
            return reservation;
        }

    private:
        std::shared_ptr<ShutdownState>   state_;
        HiddenStateCaptureReservationPtr reservation_;
    };
    auto               state              = std::make_shared<ShutdownState>();
    auto               callback_entered   = state->callback_entered.get_future();
    auto               destructor_entered = state->destructor_entered.get_future();
    auto               worker_returned    = state->worker_returned.get_future();
    auto               reserve_entered    = state->reserve_entered.get_future();
    std::promise<void> resume_callback;
    std::promise<void> resume_reserve;
    state->resume_callback = resume_callback.get_future().share();
    state->resume_reserve  = resume_reserve.get_future().share();
    auto reservation       = std::make_shared<HiddenStateCaptureReservation>([state]() noexcept {
        ++state->cancellation_count;
        state->callback_entered.set_value(std::this_thread::get_id());
        state->callback_resumed =
            state->resume_callback.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
    });
    auto engine            = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    auto scheduler     = std::make_unique<ObservedStopScheduler>(engine->runtime_config,
                                                             engine->model_config_,
                                                             engine->pd_sep_config,
                                                             engine->parallelism_config,
                                                             engine->getCacheManager());
    auto stop_entered  = scheduler->stop_entered.get_future();
    engine->scheduler_ = std::move(scheduler);
    EngineInitParams params;
    params.model_config_      = engine->model_config_;
    params.runtime_config     = engine->runtime_config;
    params.parallelism_config = engine->parallelism_config;
    params.pd_sep_config      = engine->pd_sep_config;
    params.py_model           = py::none();
    auto  executor            = std::make_unique<ObservedExecutor>(params, state, reservation);
    auto* observed_executor   = executor.get();
    engine->executor_         = std::move(executor);
    engine->model_config_.hidden_state_capture_layer_ids = {0};
    auto query                                           = std::make_shared<GenerateInput>();
    query->request_id                                    = 2000;
    query->input_ids                                     = torch::tensor({1, 2}, torch::kInt32);
    query->generate_config                               = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens               = 1;
    query->generate_config->capture_hidden_states        = true;
    query->generate_config->reuse_cache                  = false;
    auto stream                                          = engine->makeStream(query);
    ASSERT_TRUE(stream->hasHiddenStateCaptureReservation());
    stream->incPendingAsyncBookkeeping();
    stream->markDeferredRelease();

    std::optional<py::gil_scoped_release> release;
    if (Py_IsInitialized() && PyGILState_Check()) {
        release.emplace();
    }
    // Use the real runner and last-worker release path. No fake dispatch queue
    // or Executor owner is captured by this pending task or its token.
    observed_executor->dispatch_runner_.launch([stream, state] {
        stream->decPendingAsyncBookkeepingAndMaybeRelease();
        state->worker_returned.set_value();
    });
    const auto callback_ready = callback_entered.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(callback_ready, std::future_status::ready);
    std::thread::id worker_id;
    if (callback_ready == std::future_status::ready) {
        worker_id = callback_entered.get();
    }
    auto admission_query        = std::make_shared<GenerateInput>(*query);
    admission_query->request_id = 2001;
    auto admission =
        std::async(std::launch::async, [engine, admission_query] { return engine->makeStream(admission_query); });
    EXPECT_EQ(reserve_entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    std::promise<absl::Status> stop_result;
    auto                       stopped = stop_result.get_future();
    std::thread                stop_thread([engine, &stop_result] {
        try {
            stop_result.set_value(engine->stop());
        } catch (...) {
            stop_result.set_exception(std::current_exception());
        }
    });
    EXPECT_EQ(stop_entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    // stop has reached the real scheduler shutdown while reserve is still blocked.
    EXPECT_EQ(destructor_entered.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    resume_reserve.set_value();
    const auto destructor_ready = destructor_entered.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(destructor_ready, std::future_status::ready);
    if (destructor_ready == std::future_status::ready) {
        const auto destructor_thread = destructor_entered.get();
        EXPECT_EQ(destructor_thread, stop_thread.get_id());
        EXPECT_NE(destructor_thread, worker_id);
    }
    // Destruction has begun, but the real AsyncRunner must still join the
    // worker whose cancellation callback is blocked on our explicit gate.
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    // Destruction must not hold the admission mutex, even for a caller retaining
    // the GIL when this fixture runs with an initialized Python interpreter.
    auto late_admission = std::async(std::launch::async, [engine, query] {
        std::optional<py::gil_scoped_acquire> gil;
        if (Py_IsInitialized()) {
            gil.emplace();
        }
        return engine->makeStream(query);
    });
    EXPECT_EQ(late_admission.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    resume_callback.set_value();
    auto require_ready = [](auto& future) {
        if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            ADD_FAILURE() << "shutdown/admission did not finish after releasing both gates";
            // Do not hang in thread::join or an async future's destructor on a deadlock regression.
            std::abort();
        }
    };
    require_ready(stopped);
    require_ready(admission);
    require_ready(late_admission);
    stop_thread.join();
    EXPECT_NO_THROW({
        const auto status = stopped.get();
        EXPECT_TRUE(status.ok()) << status.ToString();
    });
    EXPECT_EQ(worker_returned.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_TRUE(state->callback_resumed.load());
    EXPECT_EQ(state->cancellation_count.load(), 1);
    EXPECT_EQ(engine->executor_.get(), nullptr);
    EXPECT_TRUE(state->reserve_resumed.load());
    EXPECT_EQ(state->reserve_calls.load(), 2);
    auto admitted_stream = admission.get();
    ASSERT_NE(admitted_stream, nullptr);
    EXPECT_FALSE(admitted_stream->hasError());
    EXPECT_TRUE(admitted_stream->hasHiddenStateCaptureReservation());
    EXPECT_EQ(state->admission_cancellation_count.load(), 0);
    // The token remains independently cancellable after executor destruction.
    admitted_stream->cancelHiddenStateCapture();
    admitted_stream->cancelHiddenStateCapture();
    EXPECT_EQ(state->admission_cancellation_count.load(), 1);
    auto late_stream = late_admission.get();
    ASSERT_NE(late_stream, nullptr);
    EXPECT_FALSE(late_stream->hasError());
    EXPECT_FALSE(late_stream->hasHiddenStateCaptureReservation());
    for (const bool capture : {true, false}) {
        query->generate_config->capture_hidden_states = capture;
        auto after_stop                               = engine->makeStream(query);
        ASSERT_NE(after_stop, nullptr);
        EXPECT_FALSE(after_stop->hasError());
        EXPECT_FALSE(after_stop->hasHiddenStateCaptureReservation());
    }
    EXPECT_EQ(state->reserve_calls.load(), 2);
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
    EXPECT_FALSE(stream->isDeferredReleasePending());
    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
    stream->cancelHiddenStateCapture();
    EXPECT_EQ(state->cancellation_count.load(), 1);
}

TEST(NormalEnginePolicyTest, testEmptyBatchEarlyReturnPreservesRequiredAlignment) {
    EXPECT_TRUE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/1, /*enable_ffn_disaggregate=*/false));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/1, /*enable_ffn_disaggregate=*/true));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/true, /*tp_size=*/2, /*enable_ffn_disaggregate=*/false));
    EXPECT_FALSE(NormalEngine::shouldEarlyReturnEmptyBatch(
        /*streams_empty=*/false, /*tp_size=*/1, /*enable_ffn_disaggregate=*/false));
}

TEST_F(NormalEngineTest, testWarmUpInputLengthAccountsForReserve) {
    EXPECT_EQ(warmUpInputLength(64, 0), 63u);
    EXPECT_EQ(warmUpInputLength(64, 17), 47u);
    EXPECT_ANY_THROW((void)warmUpInputLength(1, 0));
    EXPECT_ANY_THROW((void)warmUpInputLength(17, 17));
}

TEST_F(NormalEngineTest, testDecodeWarmupUsesWarmupCacheTopology) {
    CustomConfig config;
    config.warm_up = true;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto params                    = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.pd_sep_config.role_type = RoleType::DECODE;

    bool saw_decode_warmup             = false;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<MockModel>(model_config.vocab_size, [&](const GptModelInputs& inputs) {
            if (!inputs.warmup) {
                return;
            }
            saw_decode_warmup = true;
            EXPECT_EQ(inputs.kv_cache_group_tags, (std::vector<std::string>{"full"}));
            ASSERT_TRUE(inputs.kv_cache_block_id.defined());
            ASSERT_TRUE(inputs.kv_cache_kernel_block_id.defined());
            EXPECT_EQ(inputs.kv_cache_block_id.size(0), 1);
            EXPECT_EQ(inputs.kv_cache_kernel_block_id.size(0), 1);
        });
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    auto engine = std::make_shared<NormalEngine>(params, nullptr);

    EXPECT_TRUE(saw_decode_warmup);
}

TEST_F(NormalEngineTest, testCacheManagerInitFailureDoesNotPublishPartialState) {
    ModelConfig model;
    model.num_layers                   = 1;
    model.attn_config.head_num         = 1;
    model.attn_config.kv_head_num      = 1;
    model.attn_config.size_per_head    = 1;
    model.attn_config.tokens_per_block        = 1;
    model.attn_config.kernel_tokens_per_block = 1;
    model.kv_cache_spec_descs          = {{{"default", KVCacheSpecType::MultiHeadAttention}}};
    const auto config                  = CacheConfigCreator::createWarmupConfig(model, {});

    auto            previous  = std::make_shared<KVCacheManager>(config, /*warmup=*/true);
    auto            candidate = std::make_shared<KVCacheManager>(config, /*warmup=*/true);
    ResourceContext resource_context;
    resource_context.cache_manager = previous;
    resource_context.role_type     = RoleType::PREFILL;
    int group_num                  = 17;

    EXPECT_ANY_THROW(NormalEngine::initializeAndPublishCacheManager(
        resource_context, group_num, RoleType::DECODE, candidate, [](KVCacheManager&) { return false; }));
    EXPECT_EQ(resource_context.cache_manager, previous);
    EXPECT_EQ(resource_context.role_type, RoleType::PREFILL);
    EXPECT_EQ(group_num, 17);
}

TEST_F(NormalEngineTest, testRejectGenerationPrefillWithSpeculativeBeforeRunnerCreation) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.hw_kernel_config.enable_cuda_graph                        = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};
    params.parallelism_config.role_type                              = RoleType::PDFUSION;
    params.pd_sep_config.role_type                                   = RoleType::PDFUSION;

    // Exercise the real constructor, not just the wrapper ownership predicate.
    // Both the configured execution mode and a supplied propose model must
    // reject the combination before warmup, KV allocation or runner creation.
    for (const auto speculative_type : {SP_TYPE_MTP, SP_TYPE_DSPARK, SP_TYPE_NONE}) {
        for (const bool has_propose_model : {false, true}) {
            if (speculative_type == SP_TYPE_NONE && !has_propose_model) {
                continue;
            }
            SCOPED_TRACE(::testing::Message() << "speculative_type=" << static_cast<int>(speculative_type)
                                              << " has_propose_model=" << has_propose_model);
            params.sp_config.type = speculative_type;
            auto propose_params =
                has_propose_model ? std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_MTP, 2) : nullptr;
            try {
                NormalEngine engine(params, std::move(propose_params));
                FAIL() << "explicit generation-prefill/speculative combination must fail initialization";
            } catch (const std::exception& error) {
                EXPECT_NE(std::string(error.what())
                              .find("GENERATION_PREFILL_CAPTURE_CONFIG does not support speculative execution"),
                          std::string::npos)
                    << error.what();
            }
        }
    }
}

TEST_F(NormalEngineTest, testRejectMismatchedSpeculativeProposalBeforeWarmup) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.runtime_config.warm_up = false;

    const auto check_error = [&](SpeculativeType config_type,
                                 int64_t         config_gamma,
                                 SpeculativeType proposal_type,
                                 size_t          proposal_gamma,
                                 const char*     expected) {
        params.sp_config.type              = config_type;
        params.sp_config.gen_num_per_cycle = config_gamma;
        try {
            NormalEngine engine(params, std::make_unique<ProposeModelEngineInitParams>(proposal_type, proposal_gamma));
            FAIL() << "mismatched speculative parameters must fail initialization";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find(expected), std::string::npos) << error.what();
        }
    };
    check_error(SP_TYPE_MTP, 2, SP_TYPE_EAGLE, 2, "speculative type mismatch");
    check_error(SP_TYPE_MTP, 2, SP_TYPE_MTP, 3, "speculative gamma mismatch");
    check_error(SP_TYPE_MTP, -1, SP_TYPE_MTP, 2, "speculative gen_num_per_cycle must be non-negative");
}

TEST_F(NormalEngineTest, testPdRolesIgnoreGenerationPrefillWithOrWithoutSpeculativeConfig) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto          params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
    params.hw_kernel_config.enable_cuda_graph                        = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {8, 16};
    params.runtime_config.warm_up                                    = false;

    // Keep the real NormalEngine constructor and its configuration checks.
    // Only model execution is mocked; no draft model is needed to exercise the
    // retained speculative configuration that previously failed at startup.
    NormalExecutor::test_model_factory = [vocab = model_config.vocab_size](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    for (const auto role_type : {RoleType::PREFILL, RoleType::DECODE}) {
        for (const auto speculative_type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_DSPARK}) {
            SCOPED_TRACE(::testing::Message() << "role_type=" << static_cast<int>(role_type)
                                              << " speculative_type=" << static_cast<int>(speculative_type));
            params.parallelism_config.role_type = role_type;
            params.pd_sep_config.role_type      = role_type;
            params.sp_config.type               = speculative_type;
            EXPECT_NO_THROW({ NormalEngine engine(params, nullptr); });
        }
    }
}

TEST_F(NormalEngineTest, testPrefillWarmUpUsesCachelessSingleInput) {
    CustomConfig config;
    config.warm_up = true;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto params            = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);

    const KVCacheSpecDesc default_desc{"default", KVCacheSpecType::MultiHeadAttention};
    KVCacheSpecDesc       indexer_desc{"indexer_kv", KVCacheSpecType::OpaqueKV};
    indexer_desc.entry_dtype       = DataType::TYPE_UINT8;
    indexer_desc.entry_elems       = 132;  // 128 indexer bytes and one FP32 scale per token.
    indexer_desc.entry_count_mode  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    indexer_desc.compression_ratio = 1;
    params.model_config_.kv_cache_spec_descs.assign(static_cast<size_t>(params.model_config_.num_layers),
                                                    {default_desc, indexer_desc});

    bool saw_cacheless_warmup          = false;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams& init_params) {
        if (init_params.cache_manager == nullptr) {
            EXPECT_FALSE(init_params.kv_cache_layer_layout.has_value());
            return std::make_unique<MockModel>(model_config.vocab_size, [&](const GptModelInputs& inputs) {
                EXPECT_FALSE(saw_cacheless_warmup);
                saw_cacheless_warmup = true;
                EXPECT_TRUE(inputs.warmup);
                EXPECT_TRUE(inputs.kv_cache_group_tags.empty());
                EXPECT_FALSE(inputs.kv_cache_block_id.defined());
                EXPECT_FALSE(inputs.kv_cache_kernel_block_id.defined());
            });
        }
        return std::make_unique<MockModel>(model_config.vocab_size);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    auto engine = std::make_shared<NormalEngine>(params, nullptr);

    EXPECT_TRUE(saw_cacheless_warmup);
}

TEST_F(NormalEngineTest, testFp8KVCache) {
    CustomConfig config;
    config.kv_cache_data_type = DataType::TYPE_FP8_E4M3;
    auto engine               = createMockEngine(config);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 5;
    query->generate_config->is_streaming   = false;

    shared_ptr<GenerateStream> stream = engine->enqueue(query);

    ASSERT_TRUE(stream != nullptr);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
    ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    auto output2 = stream->nextOutput();
    ASSERT_TRUE(!output2.ok());
}

TEST_F(NormalEngineTest, testPrefillOnlyReturnsOneEmptyOutput) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto query                             = make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 0;

    auto       stream       = engine->enqueue(query);
    const auto input_length = stream->seqLength();
    auto       output       = stream->nextOutput();

    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1);
    EXPECT_TRUE(output.value().generate_outputs[0].finished);
    EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
    EXPECT_EQ(stream->seqLength(), input_length);
    EXPECT_EQ(stream->outputTokenLen(), 0);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
}

TEST_F(NormalEngineTest, testDecodeRoleRejectsPrefillOnlyBeforeScheduling) {
    CustomConfig config;
    config.role_type = RoleType::DECODE;
    auto engine      = createMockEngine(config);

    auto zero_query                             = make_shared<GenerateInput>();
    zero_query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    zero_query->generate_config                 = make_shared<GenerateConfig>();
    zero_query->generate_config->max_new_tokens = 0;

    auto zero_stream = engine->enqueue(zero_query);
    ASSERT_NE(zero_stream, nullptr);
    EXPECT_TRUE(zero_stream->hasError());
    EXPECT_EQ(zero_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(zero_stream->curBlocksNum(), 0);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

    auto positive_query                             = make_shared<GenerateInput>();
    positive_query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    positive_query->generate_config                 = make_shared<GenerateConfig>();
    positive_query->generate_config->max_new_tokens = 1;

    auto positive_stream = engine->makeStream(positive_query);
    ASSERT_NE(positive_stream, nullptr);
    EXPECT_FALSE(positive_stream->hasError());
    EXPECT_EQ(positive_stream->curBlocksNum(), 0);
}

TEST_F(NormalEngineTest, testDirectEngineEntriesConvertPrefillValidationErrorsToStreamErrors) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto make_invalid_query = [] {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 0;
        query->generate_config->min_new_tokens = 1;
        return query;
    };

    std::shared_ptr<GenerateStream> stream;
    EXPECT_NO_THROW(stream = engine->enqueue(make_invalid_query()));
    ASSERT_NE(stream, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);

    auto make_stream_query = make_invalid_query();
    EXPECT_NO_THROW(stream = engine->makeStream(make_stream_query));
    ASSERT_NE(stream, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);

    std::vector<std::shared_ptr<GenerateInput>>                  inputs{make_invalid_query(), make_invalid_query()};
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>> result;
    EXPECT_NO_THROW(result = engine->enqueueMultiple(inputs));
    EXPECT_EQ(result.first, std::vector<bool>({false, false}));
    ASSERT_EQ(result.second.size(), 2);
    for (const auto& invalid_stream : result.second) {
        ASSERT_NE(invalid_stream, nullptr);
        EXPECT_TRUE(invalid_stream->hasError());
        EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    }
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
}

TEST_F(NormalEngineTest, testDirectEngineEntriesSkipIneligibleCaptureAndPreserveZeroTokenErrors) {
    CustomConfig config;
    auto         engine = createMockEngine(config);
    ASSERT_TRUE(engine->stop().ok());

    struct TestCase {
        const char*      name;
        int              num_beams;
        std::vector<int> variable_num_beams;
        int              num_return_sequences;
        bool             disable_reuse_cache;
        const char*      error;
    };
    const std::vector<TestCase> cases = {
        {"default_reuse_cache", 1, {}, 1, false, ""},
        {"beam_search", 2, {}, 1, true, "max_new_tokens == 0 does not support num_beams > 1"},
        {"variable_beams", 1, {1, 2}, 1, true, "max_new_tokens == 0 does not support non-empty variable_num_beams"},
        {"multiple_sequences", 1, {}, 2, true, "max_new_tokens == 0 does not support num_return_sequences > 1"},
        {"beam_and_reuse_cache", 2, {}, 1, false, "max_new_tokens == 0 does not support num_beams > 1"},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        for (const int max_new_tokens : {0, 1}) {
            SCOPED_TRACE(max_new_tokens);
            auto query                                    = make_shared<GenerateInput>();
            query->input_ids                              = torch::tensor({1, 2, 3, 4}, torch::kInt32);
            query->generate_config                        = make_shared<GenerateConfig>();
            query->generate_config->max_new_tokens        = max_new_tokens;
            query->generate_config->capture_hidden_states = true;
            query->generate_config->num_beams             = test_case.num_beams;
            query->generate_config->variable_num_beams    = test_case.variable_num_beams;
            query->generate_config->num_return_sequences  = test_case.num_return_sequences;
            if (test_case.disable_reuse_cache) {
                query->generate_config->reuse_cache = false;
            }
            const auto original_config = query->generate_config;
            const auto original_fields = original_config->debugString();
            const auto original_ids    = query->input_ids.clone();
            const auto original_data   = query->input_ids.data_ptr();
            const bool invalid_zero    = max_new_tokens == 0 && test_case.error[0] != '\0';
            auto       expect_result   = [&](const GenerateStreamPtr& stream) {
                ASSERT_NE(stream, nullptr);
                if (!invalid_zero) {
                    EXPECT_FALSE(stream->hasError());
                    EXPECT_FALSE(stream->isHiddenStateCaptureCandidate());
                    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
                    EXPECT_EQ(stream->generateInput(), query);
                    EXPECT_EQ(stream->generateConfig(), original_config);
                    EXPECT_EQ(stream->generateConfig()->debugString(), original_fields);
                    EXPECT_EQ(stream->reuseCache(), max_new_tokens > 0 && !test_case.disable_reuse_cache);
                    return;
                }
                EXPECT_TRUE(stream->hasError());
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_NE(stream->stopReason().find(test_case.error), std::string::npos);
                EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
                EXPECT_EQ(stream->curBlocksNum(), 0);
                EXPECT_NE(stream->generateInput(), query);
                EXPECT_NE(stream->generateConfig(), original_config);
                EXPECT_EQ(stream->generateConfig()->max_new_tokens, 1);
                EXPECT_FALSE(stream->generateConfig()->capture_hidden_states);
                EXPECT_EQ(stream->generateConfig()->maxNumBeams(), 1);
                EXPECT_EQ(stream->generateConfig()->num_return_sequences, 1);
                EXPECT_EQ(query->generate_config, original_config);
                EXPECT_EQ(query->generate_config->debugString(), original_fields);
                EXPECT_EQ(query->input_ids.data_ptr(), original_data);
                EXPECT_TRUE(torch::equal(query->input_ids, original_ids));
            };

            std::shared_ptr<GenerateStream> stream;
            EXPECT_NO_THROW(stream = engine->makeStream(query));
            expect_result(stream);
            stream.reset();
            EXPECT_NO_THROW(stream = engine->enqueue(query));
            expect_result(stream);

            std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>> result;
            EXPECT_NO_THROW(result = engine->enqueueMultiple({query, query}));
            EXPECT_EQ(result.first, std::vector<bool>({!invalid_zero, !invalid_zero}));
            ASSERT_EQ(result.second.size(), 2);
            for (const auto& invalid_stream : result.second) {
                expect_result(invalid_stream);
            }
            EXPECT_EQ(engine->getScheduler().onflightStreams(), invalid_zero ? 0 : 3);
            ASSERT_TRUE(engine->getScheduler().stop().ok());
        }
    }
}

TEST_F(NormalEngineTest, testEngineEntriesPreserveValidConfigsWithInvalidZeroInBatch) {
    CustomConfig config;
    auto         engine = createMockEngine(config);
    // Inspect scheduling without executing capture on the logits-only mock model.
    ASSERT_TRUE(engine->stop().ok());

    auto make_query = [](int max_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        return query;
    };
    size_t scheduled = 0;
    for (const bool capture : {false, true}) {
        SCOPED_TRACE(capture);
        for (const int max_new_tokens : {0, 1}) {
            SCOPED_TRACE(max_new_tokens);
            for (const bool invalid_first : {false, true}) {
                SCOPED_TRACE(invalid_first);
                auto valid                                      = make_query(max_new_tokens);
                valid->generate_config->capture_hidden_states   = capture;
                valid->generate_config->reuse_cache             = !capture;
                const auto valid_config                         = valid->generate_config;
                const auto valid_fields                         = valid_config->debugString();
                auto       invalid                              = make_query(0);
                invalid->generate_config->capture_hidden_states = true;
                invalid->generate_config->min_new_tokens        = 1;
                const auto invalid_config                       = invalid->generate_config;
                const auto invalid_fields                       = invalid_config->debugString();

                auto expect_valid = [&](const GenerateStreamPtr& stream) {
                    ASSERT_NE(stream, nullptr);
                    EXPECT_FALSE(stream->hasError());
                    EXPECT_EQ(stream->generateInput(), valid);
                    EXPECT_EQ(stream->generateConfig(), valid_config);
                    EXPECT_EQ(valid->generate_config, valid_config);
                    EXPECT_EQ(valid_config->debugString(), valid_fields);
                };
                std::shared_ptr<GenerateStream> stream;
                EXPECT_NO_THROW(stream = engine->makeStream(valid));
                expect_valid(stream);
                stream.reset();
                EXPECT_NO_THROW(stream = engine->enqueue(valid));
                expect_valid(stream);
                ++scheduled;
                EXPECT_EQ(engine->getScheduler().onflightStreams(), scheduled);

                auto inputs = invalid_first ? std::vector<std::shared_ptr<GenerateInput>>{invalid, valid} :
                                              std::vector<std::shared_ptr<GenerateInput>>{valid, invalid};
                std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>> result;
                EXPECT_NO_THROW(result = engine->enqueueMultiple(inputs));
                EXPECT_EQ(result.first, std::vector<bool>({!invalid_first, invalid_first}));
                ASSERT_EQ(result.second.size(), 2);
                expect_valid(result.second[invalid_first ? 1 : 0]);
                const auto& rejected = result.second[invalid_first ? 0 : 1];
                ASSERT_NE(rejected, nullptr);
                EXPECT_TRUE(rejected->hasError());
                EXPECT_EQ(rejected->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_FALSE(rejected->hasEvent(StreamEvents::CanRun));
                EXPECT_EQ(rejected->curBlocksNum(), 0);
                EXPECT_EQ(invalid->generate_config, invalid_config);
                EXPECT_EQ(invalid_config->debugString(), invalid_fields);
                ++scheduled;
                EXPECT_EQ(engine->getScheduler().onflightStreams(), scheduled);
            }
        }
    }
}

TEST_F(NormalEngineTest, testEnqueueMultipleRejectsMixedModesBeforeRoleFiltering) {
    CustomConfig config;
    config.role_type = RoleType::DECODE;
    auto engine      = createMockEngine(config);

    auto make_query = [](int64_t request_id, int64_t group_id, int max_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->request_id                      = request_id;
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        query->group_id                        = group_id;
        query->group_size                      = 2;
        return query;
    };

    for (bool generation_first : {false, true}) {
        const int64_t group_id = 100 + generation_first;
        auto          prefill  = make_query(10 + generation_first, group_id, 0);
        auto          generate = make_query(20 + generation_first, group_id, 1);
        auto          inputs   = generation_first ? std::vector<std::shared_ptr<GenerateInput>>{generate, prefill} :
                                                    std::vector<std::shared_ptr<GenerateInput>>{prefill, generate};

        auto [enqueue_successes, streams] = engine->enqueueMultiple(inputs);

        EXPECT_EQ(enqueue_successes, std::vector<bool>({false, false}));
        ASSERT_EQ(streams.size(), 2);
        for (const auto& stream : streams) {
            EXPECT_TRUE(stream->hasError());
            EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(stream->stopReason(), kMixedForceBatchGroupError);
        }
        EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
    }
}

TEST_F(NormalEngineTest, testEnqueueMultipleIgnoresInvalidStreamForModeCheck) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    auto make_query = [](int max_new_tokens, int min_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        query->generate_config->min_new_tokens = min_new_tokens;
        return query;
    };

    auto [enqueue_successes, streams] = engine->enqueueMultiple({make_query(0, 1), make_query(1, 0)});

    ASSERT_EQ(streams.size(), 2);
    EXPECT_EQ(enqueue_successes, std::vector<bool>({false, true}));
    EXPECT_TRUE(streams[0]->hasError());
    EXPECT_EQ(streams[0]->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(streams[1]->hasError());
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 1);
}

class NormalEngineCustomOutputTest: public NormalEngineTest {
protected:
    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
        // Observed on the CI ut-sm8x lane: without these two lines the process
        // segfaults during "import torch" (deepest frame: hashlib.py
        // __get_openssl_constructor) or aborts at teardown with glibc heap
        // corruption (corrupted size vs. prev_size) after the tests pass.
        // The binary links 3rdparty/3fs's libcrypto.so.1.1 and the sandbox puts
        // nvidia pip repos on sys.path; blocking the _hashlib import (hashlib
        // falls back to builtin digests) and dropping those sys.path entries
        // before importing torch makes the suite run to a clean exit. Exact
        // in-library corruption mechanics were not traced further.
        py::exec(R"(
import os, sys
sys.path = [p for p in sys.path if not os.path.isdir(os.path.join(p, "nvidia"))]
sys.modules['_hashlib'] = None
import torch)");
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    static std::shared_ptr<GenerateInput> makeQuery(int max_new_tokens, int first_token = 1) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({first_token, 2, 3, 4}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        return query;
    }

    // Keep torch and pybind interpreter state alive across all selector cases.
    inline static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

TEST_F(NormalEngineCustomOutputTest, testDeviceFaultDuringCreationIsRejectedAtSchedulerHandoff) {
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    auto health                     = std::make_shared<ServingDeviceHealth>();
    engine->serving_device_health_  = health;
    int selections                  = 0;
    engine->custom_output_selector_ = py::cpp_function([&](const torch::Tensor&, py::object) {
        ++selections;
        health->observeRuntimeResult({DeviceRuntimeDomain::CUDA, 719, 0, DeviceRuntimeOperation::STREAM_WAIT});
        return 0;
    });
    py::gil_scoped_release release;
    auto [accepted, streams] = engine->enqueueMultiple({makeQuery(1), makeQuery(1)});
    ASSERT_EQ(streams.size(), 2u);
    EXPECT_EQ(selections, 1);
    EXPECT_EQ(accepted, (std::vector<bool>{false, false}));
    for (const auto& stream : streams) {
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::SERVICE_UNHEALTHY);
    }
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
}

TEST_F(NormalEngineCustomOutputTest, testUnhealthyQueryNeedsNoGilAndAdmissionSkipsSelector) {
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    engine->serving_device_health_ = std::make_shared<ServingDeviceHealth>();
    engine->serving_device_health_->observeRuntimeResult(
        {DeviceRuntimeDomain::CUDA, 700, 0, DeviceRuntimeOperation::EVENT_WAIT});
    int selections                  = 0;
    engine->custom_output_selector_ = py::cpp_function([&](const torch::Tensor&, py::object) {
        ++selections;
        return 0;
    });
    auto       result               = std::async(std::launch::async, [&] { return engine->checkHealth().code(); });
    const auto ready                = result.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(ready, std::future_status::ready);
    {
        py::gil_scoped_release release;
        EXPECT_EQ(result.get(), ErrorCode::SERVICE_UNHEALTHY);
        EXPECT_EQ(engine->makeStream(makeQuery(1))->statusInfo().code(), ErrorCode::SERVICE_UNHEALTHY);
        auto [accepted, streams] = engine->enqueueMultiple({makeQuery(1), makeQuery(0)});
        EXPECT_EQ(accepted, (std::vector<bool>{false, false}));
        ASSERT_EQ(streams.size(), 2u);
    }
    EXPECT_EQ(selections, 0);
}

TEST_F(NormalEngineCustomOutputTest, testCustomOutputSelectedOnceBeforeSystemPrefix) {
    CustomConfig config;
    config.multi_task_prompt_tokens = {{"1", {8, 9, 10}}};
    auto engine                     = createMockEngine(config);
    // Join the loop to inspect entry/scheduler state without executing the mock model's logits-only output.
    ASSERT_TRUE(engine->stop().ok());
    engine->reserve_step_           = 4;
    int calls                       = 0;
    engine->custom_output_selector_ = py::cpp_function([&](const torch::Tensor& ids, py::object mask) {
        ++calls;
        EXPECT_EQ(ids.numel(), 4);
        EXPECT_EQ(ids[0].item<int>(), 1);
        EXPECT_TRUE(mask.is_none());
        return 2;
    });
    py::gil_scoped_release release;
    auto                   make_query = [&] {
        auto query                      = makeQuery(2);
        query->generate_config->task_id = "1";
        return query;
    };

    auto stream = engine->makeStream(make_query());
    EXPECT_EQ(calls, 1);
    engine->enqueue(stream);
    EXPECT_EQ(calls, 1);
    auto single = engine->enqueue(make_query());
    EXPECT_EQ(calls, 2);
    auto [successes, batch] = engine->enqueueMultiple({make_query(), make_query()});
    EXPECT_EQ(successes, std::vector<bool>({true, true}));
    ASSERT_EQ(batch.size(), 2);
    EXPECT_EQ(calls, 4);
    for (const auto& selected : {stream, single, batch[0], batch[1]}) {
        ASSERT_FALSE(selected->hasError());
        EXPECT_EQ(selected->reserveStep(), 4u);
        EXPECT_EQ(selected->inputLength(), 7);
        EXPECT_EQ(selected->generateInput()->custom_output_token_position, 5);
        EXPECT_EQ(selected->generateInput()->input_ids[5].item<int>(), 3);
    }
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 4);

    auto fake        = make_query();
    fake->fake_query = true;
    EXPECT_FALSE(engine->makeStream(fake)->hasError());
    EXPECT_EQ(fake->custom_output_token_position, -1);
    EXPECT_EQ(calls, 4);
}

TEST_F(NormalEngineCustomOutputTest, testPrefillOnlySkipsCustomSelectorAndCompletes) {
    auto engine                     = createMockEngine(CustomConfig{});
    int  calls                      = 0;
    engine->custom_output_selector_ = py::cpp_function([&](const torch::Tensor& ids, py::object) {
        ++calls;
        if (ids[0].item<int>() != 7) {
            throw py::value_error("generation prompt requires scoring token");
        }
        return 0;
    });
    py::gil_scoped_release release;

    auto generation = engine->makeStream(makeQuery(1, 7));
    ASSERT_FALSE(generation->hasError());
    EXPECT_EQ(generation->generateInput()->custom_output_token_position, 0);
    auto rejected = engine->enqueue(makeQuery(1));
    ASSERT_TRUE(rejected->hasError());
    EXPECT_EQ(rejected->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(engine->getScheduler().onflightStreams(), 0);
    EXPECT_EQ(calls, 2);

    auto expect_finished = [&](const GenerateStreamPtr& stream) {
        auto output = stream->nextOutput(5000);
        ASSERT_TRUE(output.ok()) << output.status().ToString();
        ASSERT_EQ(output.value().generate_outputs.size(), 1);
        const auto& result = output.value().generate_outputs[0];
        EXPECT_TRUE(result.finished);
        EXPECT_EQ(result.output_ids.sizes(), (torch::IntArrayRef{1, 0}));
        EXPECT_FALSE(result.custom_output.has_value());
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->generateInput()->custom_output_token_position, -1);
        EXPECT_EQ(stream->reserveStep(), 0u);
        EXPECT_EQ(stream->seqLength(), 4);
        EXPECT_EQ(stream->outputTokenLen(), 0);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        EXPECT_FALSE(stream->nextOutput(100).ok());
    };
    expect_finished(engine->enqueue(makeQuery(0)));
    // Failed generation must be filtered before the mixed-mode check, without reordering results.
    const std::vector<std::shared_ptr<GenerateInput>> inputs{makeQuery(0), makeQuery(1), makeQuery(0)};
    auto [successes, batch] = engine->enqueueMultiple(inputs);
    EXPECT_EQ(successes, std::vector<bool>({true, false, true}));
    ASSERT_EQ(batch.size(), inputs.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        EXPECT_EQ(batch[i]->generateInput(), inputs[i]);
        if (i == 1) {
            EXPECT_TRUE(batch[i]->hasError());
            EXPECT_EQ(batch[i]->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(batch[i]->curBlocksNum(), 0);
        } else {
            expect_finished(batch[i]);
        }
    }
    auto invalid                             = makeQuery(0);
    invalid->generate_config->min_new_tokens = 1;
    auto invalid_stream                      = engine->enqueue(invalid);
    EXPECT_TRUE(invalid_stream->hasError());
    EXPECT_EQ(invalid_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(calls, 3);
}

TEST_F(NormalEngineTest, testSimple) {
    CustomConfig config;
    auto         engine = createMockEngine(config);

    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_FALSE(engine->resourceContext().system_prompt);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // test streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 3;
        query->generate_config->is_streaming   = true;
        query->generate_config->gen_timeline   = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.iter_count, 1);

        auto output2 = stream->nextOutput();
        ASSERT_TRUE(output2.ok());
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.output_len, 2);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output2.value().generate_outputs[0].aux_info.iter_count, 2);

        auto output3 = stream->nextOutput();
        ASSERT_TRUE(output3.ok());
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.output_len, 3);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output3.value().generate_outputs[0].aux_info.iter_count, 3);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output4 = stream->nextOutput();
        ASSERT_TRUE(!output4.ok());
    }

    // test non-streaming query
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 5;
        query->generate_config->is_streaming   = false;

        shared_ptr<GenerateStream> stream = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.output_len, 5);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.input_len, 7);
        ASSERT_EQ(output.value().generate_outputs[0].aux_info.iter_count, 5);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testParallelDispatchMultipleRequests) {
    CustomConfig config;
    config.output_dispatcher_worker_count = 2;
    auto engine                           = createMockEngine(config);

    auto make_query = [](int input_token) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({input_token}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 2;
        query->generate_config->is_streaming   = false;
        return query;
    };

    auto [enqueue_successes, streams] = engine->enqueueMultiple({make_query(1), make_query(2)});
    ASSERT_EQ(enqueue_successes, (std::vector<bool>{true, true}));
    ASSERT_EQ(streams.size(), 2u);
    for (auto& stream : streams) {
        ASSERT_NE(stream, nullptr);
        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), 1u);
        EXPECT_EQ(output.value().generate_outputs[0].aux_info.input_len, 1);
        EXPECT_EQ(output.value().generate_outputs[0].aux_info.output_len, 2);
        EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    }

    engine.reset();
}

TEST_F(NormalEngineTest, testChunkedPrefillWarmupStartup) {
    CustomConfig config;
    config.warm_up                  = true;
    config.prefill_chunk_size       = 4;
    config.forward_shapes           = std::make_shared<std::vector<ForwardShape>>();
    config.multi_task_prompt_tokens = {{"system", {1, 2, 3, 4, 5, 6, 7}}};
    auto engine                     = createMockEngine(config);

    // Token-heavy: one row walks the 19-token prompt in budget-4 chunks. Row-heavy: four rows
    // share the budget, each executing one token after a block-aligned 18-token synthetic prefix.
    // The 7-token whole-segment shape runs once during warmup and once for the system prompt.
    ASSERT_EQ(*config.forward_shapes,
              (std::vector<ForwardShape>{
                  {4, 0, 1},
                  {4, 4, 1},
                  {4, 8, 1},
                  {4, 12, 1},
                  {3, 16, 1},
                  {4, 18, 4},
                  {7, 0, 1},
                  {7, 0, 1}}));
    const auto& cache_config = engine->resourceContext().cache_manager->cacheConfig();
    ASSERT_EQ(cache_config.groups().size(), 1u);
    ASSERT_EQ(cache_config.groups().front().block_num, 100u);
}

TEST_F(NormalEngineTest, testChunkedPrefillWarmupCapsIntMaxBudgetByContextBatchSize) {
    CustomConfig config;
    config.warm_up                = true;
    config.prefill_chunk_size     = std::numeric_limits<int>::max();
    config.max_context_batch_size = 8;
    config.forward_shapes         = std::make_shared<std::vector<ForwardShape>>();
    (void)createMockEngine(config);

    // The budget exceeds the prompt length, so all eight rows execute the full 19-token prompt.
    ASSERT_EQ(*config.forward_shapes,
              (std::vector<ForwardShape>{
                  {19, 0, 1},
                  {19 * 8, 0, 8},
              }));
}

TEST_F(NormalEngineTest, testChunkedPrefillWarmupAlignsPrefixWithOddBudget) {
    CustomConfig config;
    config.warm_up            = true;
    config.prefill_chunk_size = 5;
    config.forward_shapes     = std::make_shared<std::vector<ForwardShape>>();
    (void)createMockEngine(config);

    // Middle chunks align to two-token blocks; five row-heavy rows share the budget after prefix 18.
    ASSERT_EQ(*config.forward_shapes,
              (std::vector<ForwardShape>{
                  {4, 0, 1},
                  {4, 4, 1},
                  {4, 8, 1},
                  {4, 12, 1},
                  {3, 16, 1},
                  {5, 18, 5},
              }));
}

TEST_F(NormalEngineTest, testChunkedPrefillGenerationGraphWarmupUsesChunkShapes) {
    CustomConfig config;
    config.warm_up                = true;
    config.prefill_chunk_size     = 4;
    config.max_context_batch_size = 8;

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    auto params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    params.parallelism_config.role_type                             = RoleType::PDFUSION;
    params.hw_kernel_config.enable_cuda_graph                       = true;
    params.hw_kernel_config.generation_prefill_capture_token_buckets = {4, 8};

    auto forward_shapes = std::make_shared<std::vector<ForwardShape>>();
    NormalExecutor::test_model_factory = [vocab = model_config.vocab_size, forward_shapes](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab, forward_shapes);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    auto engine = std::make_shared<NormalEngine>(params, nullptr);

    EXPECT_EQ(*forward_shapes,
              (std::vector<ForwardShape>{
                  {4, 0, 1},
                  {4, 4, 1},
                  {4, 8, 1},
                  {4, 12, 1},
                  {3, 16, 1},
                  {4, 18, 4},
              }));
}

TEST_F(NormalEngineTest, testChunkedPrefillLossWarmupUsesWholeSegment) {
    CustomConfig config;
    config.warm_up             = true;
    config.warm_up_with_loss   = true;
    config.prefill_chunk_size  = 4;
    config.forward_shapes      = std::make_shared<std::vector<ForwardShape>>();
    (void)createMockEngine(config);

    ASSERT_EQ(*config.forward_shapes, (std::vector<ForwardShape>{{19 * 128, 0, 128}}));
}

TEST_F(NormalEngineTest, testSystemPrompt) {
    CustomConfig config;
    vector<int>  prompt_1           = {1, 2, 3};
    vector<int>  prompt_2           = {4, 5, 6, 7, 8, 9};
    config.multi_task_prompt_tokens = {{"1", prompt_1}, {"2", prompt_2}};
    config.reuse_cache              = false;
    config.enable_device_cache      = false;
    auto engine                     = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().cache_manager);
    ASSERT_TRUE(engine->resourceContext().system_prompt);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    ASSERT_TRUE(engine->resourceContext().enable_device_cache);
    const auto block_tree_cache = engine->resourceContext().cache_manager->blockTreeCache();
    ASSERT_NE(block_tree_cache, nullptr);
    ASSERT_TRUE(block_tree_cache->config().enable_device_cache);

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 2);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({10, 20, 30, 40, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->task_id        = "2";
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 6);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testReuseCacheOption) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    config.reuse_cache = false;
    auto engine2       = createMockEngine(config);
    ASSERT_FALSE(engine2->resourceContext().reuse_cache);
}

TEST_F(NormalEngineTest, testReuseCache) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOn) {
    CustomConfig config;
    config.reuse_cache = true;
    auto engine        = createMockEngine(config);
    ASSERT_TRUE(engine->resourceContext().reuse_cache);

    // First query with reuse_cache = true
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Second query with reuse_cache = false (should not reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Third query with reuse_cache = true (should reuse cache)
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len, 4);  // Should be 4 because reuse_cache = true
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testQueryReuseCacheWhenSwitchIsOff) {
    // Test with engine-level reuse_cache = false (master switch off)
    CustomConfig config;
    config.reuse_cache = false;
    auto engine        = createMockEngine(config);
    ASSERT_FALSE(engine->resourceContext().reuse_cache);

    // Query with reuse_cache = true, but should be ignored because engine-level is false
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 5, 6, 7}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = true;  // This should be ignored
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }

    // Query with reuse_cache = false, should also result in no cache reuse
    {
        std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3, 4, 50, 60, 70}, torch::kInt32);
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        query->generate_config->reuse_cache    = false;
        shared_ptr<GenerateStream> stream      = engine->enqueue(query);

        ASSERT_TRUE(stream != nullptr);
        auto output1 = stream->nextOutput();
        ASSERT_TRUE(output1.ok());
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.output_len, 1);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.prefix_len, 0);
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.reuse_len,
                  0);  // Should be 0 because engine-level reuse_cache = false
        ASSERT_EQ(output1.value().generate_outputs[0].aux_info.input_len, 7);

        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        auto output2 = stream->nextOutput();
        ASSERT_TRUE(!output2.ok());
    }
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithPrefillCP) {
    CustomConfig config;
    config.output_vocab_ids   = {0, 2, 7};
    config.prefill_cp_enabled = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithDeviceInput) {
    CustomConfig config;
    config.output_vocab_ids = {0, 2, 7};
    setenv("RTP_LLM_DEVICE_INPUT", "1", 1);
    EXPECT_THROW(createMockEngine(config), std::exception);
    unsetenv("RTP_LLM_DEVICE_INPUT");
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithSpeculative) {
    CustomConfig config;
    config.output_vocab_ids    = {0, 2, 7};
    config.speculative_enabled = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectOutputVocabWithWarmUpWithLoss) {
    CustomConfig config;
    config.output_vocab_ids  = {0, 2, 7};
    config.warm_up_with_loss = true;
    EXPECT_THROW(createMockEngine(config), std::exception);
}

TEST_F(NormalEngineTest, testRejectInvalidOutputVocabIds) {
    CustomConfig unsorted;
    unsorted.output_vocab_ids = {7, 2, 0};
    EXPECT_THROW(createMockEngine(unsorted), std::exception);

    CustomConfig duplicated;
    duplicated.output_vocab_ids = {0, 2, 2, 7};
    EXPECT_THROW(createMockEngine(duplicated), std::exception);

    CustomConfig out_of_range;
    out_of_range.output_vocab_ids = {0, 2, 100};  // vocab_size is 100
    EXPECT_THROW(createMockEngine(out_of_range), std::exception);
}

TEST_F(NormalEngineTest, testAllowsUnsupportedCombosWithoutOutputVocab) {
    CustomConfig config;
    config.prefill_cp_enabled  = true;
    config.speculative_enabled = true;
    config.warm_up_with_loss   = true;
    EXPECT_NO_THROW(createMockEngine(config));
}

}  // namespace rtp_llm
