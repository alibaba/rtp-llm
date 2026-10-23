#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <thread>

#include "absl/status/status.h"
#define private public
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#undef private

#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/utils/Logger.h"
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
#include <optional>
#include <pybind11/embed.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:
    void checkInternalForkHistoryCopy(bool        invalid_row,
                                      const char* remote_address   = nullptr,
                                      const char* failed_block_dir = nullptr,
                                      bool        fail_after_copy  = false);
};

class FailingSaveModel: public ModelBase {
public:
    FailingSaveModel(GenerateStreamPtr stream,
                     bool              concurrent_cancel,
                     std::vector<int64_t> expected_save_positions = {},
                     bool              throw_from_forward = false):
        stream_(std::move(stream)),
        concurrent_cancel_(concurrent_cancel),
        expected_save_positions_(std::move(expected_save_positions)),
        throw_from_forward_(throw_from_forward) {}

    bool supportsSparseCacheSave() const override { return true; }

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        if (!expected_save_positions_.empty()) {
            EXPECT_TRUE(inputs.cache_save_positions.defined() && inputs.cache_save_positions.numel() > 0);
            EXPECT_TRUE(inputs.cache_save_block_ids.defined() && inputs.cache_save_block_ids.numel() > 0);
            if (!inputs.cache_save_positions.defined()) {
                return {};
            }
            const auto positions = inputs.cache_save_positions.to(torch::kCPU).contiguous();
            const auto* data = positions.data_ptr<int64_t>();
            const std::vector<int64_t> observed(data, data + positions.numel());
            for (int64_t expected : expected_save_positions_) {
                EXPECT_NE(std::find(observed.begin(), observed.end(), expected), observed.end());
            }
        }
        // Submit real device work so the post-writer completion boundary
        // and the cancellation guard both cover an in-flight CUDA stream.
        writer_work_ = torch::ones({1024}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        writer_work_submitted_ = true;
        if (concurrent_cancel_) {
            stream_->reportError(ErrorCode::EXECUTION_EXCEPTION, "injected cancellation");
            release_future_ = std::async(std::launch::async, [this]() {
                stream_->releaseResource();
                released_.store(true, std::memory_order_release);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            EXPECT_FALSE(released_.load(std::memory_order_acquire));
        }
        if (throw_from_forward_) {
            throw std::runtime_error("injected selected writer failure");
        }
        return {};
    }

    std::optional<std::string> takeDeferredHiddenStateCaptureError() override {
        if (throw_from_forward_) {
            return std::nullopt;
        }
        return "injected deferred capture failure";
    }

    bool waitForConcurrentRelease() {
        if (!concurrent_cancel_) {
            return true;
        }
        if (release_future_.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
            return false;
        }
        release_future_.get();
        return released_.load(std::memory_order_acquire);
    }

    bool writerWorkSubmitted() const { return writer_work_submitted_; }

private:
    GenerateStreamPtr stream_;
    bool concurrent_cancel_;
    std::vector<int64_t> expected_save_positions_;
    bool throw_from_forward_;
    torch::Tensor writer_work_;
    bool writer_work_submitted_{false};
    std::future<void> release_future_;
    std::atomic<bool> released_{false};
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
            std::reverse(inputs.kv_cache_group_tags.begin(), inputs.kv_cache_group_tags.end());
            inputs.kv_cache_block_id        = inputs.kv_cache_block_id.flip({0});
            inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id.flip({0});
            inputs.kv_cache_group_types     = inputs.kv_cache_group_types.flip({0});
            inputs.kv_cache_update_mapping  = mapping_;
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
    config.fromGroupedSpecs({test::makeResolvedMhaSpec(config.dtype, 1, 4, 2, "first"),
                             test::makeResolvedMhaSpec(config.dtype, 1, 4, 2, "second")},
                            {{0}, {1}},
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
        EXPECT_EQ(inputs.kv_cache_group_tags, (std::vector<std::string>{"second", "first"}));
        throw StopBeforeSampling{};
    }));

    for (const bool invalid_row : {false, true}) {
        SCOPED_TRACE(invalid_row);
        first_dst.fill_(33);
        second_dst.fill_(44);
        reached_model = false;
        auto mapping  = invalid_row ? torch::tensor({0, 1, 2, 2, 1, 2}, torch::kInt32).reshape({2, 3}) :
                                      torch::tensor({0, 1, 2}, torch::kInt32).reshape({1, 3});
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

TEST_F(NormalEngineTest, testSparseSaveRequiresWriterBeforeForward) {
    class SparseSaveProcessor: public NormalBatchStreamProcessor {
    public:
        SparseSaveProcessor(const ModelConfig& model, const CacheConfig& cache):
            NormalBatchStreamProcessor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, false) {}

        absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& groups,
                                                        TensorHolder&       holder) const override {
            auto result = NormalBatchStreamProcessor::gatherModelInput(groups, holder);
            if (result.ok()) {
                result->cache_save_positions = torch::tensor({5}, torch::kInt64).reshape({1, 1});
                result->cache_save_block_ids = torch::tensor({2}, torch::kInt32).reshape({1, 1, 1});
            }
            return result;
        }
    };

    ModelConfig model;
    model.num_layers                          = 1;
    model.max_seq_len                         = 16;
    model.vocab_size                          = 16;
    model.input_vocab_size                    = 16;
    model.hidden_size                         = 4;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 4;
    model.attn_config.tokens_per_block        = 2;
    model.attn_config.kernel_tokens_per_block = 2;
    CacheConfig cache;
    cache.layer_num          = 1;
    cache.seq_size_per_block = 2;
    cache.dtype              = DataType::TYPE_FP16;
    cache.fromGroupedSpecs({test::makeResolvedMhaSpec(cache.dtype, 1, 4, 2, "full")},
                           {{0}}, {CacheGroupType::FULL});
    cache.finalizeBlockNums(4, RuntimeConfig{});
    auto manager = std::make_shared<KVCacheManager>(cache);
    ASSERT_TRUE(manager->init());

    EngineInitParams params;
    params.model_config_ = model;
    params.py_model = py::none();
    NormalExecutor executor(params, manager);
    bool reached_model = false;
    executor.setModel(std::make_unique<MockModel>(model.vocab_size, [&](const GptModelInputs&) {
        reached_model = true;
    }));
    executor.setBatchProcessor(std::make_unique<SparseSaveProcessor>(model, cache));
    auto query = std::make_shared<GenerateInput>();
    query->input_ids = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, ResourceContext{}, nullptr);
    BatchKVCacheResource resource;
    resource.resetBatchSize(1);
    resource.initGroups(cache.topologyPtr());
    resource.mutableBlockIds(0, "full").assign({1, 2, 3});
    stream->setKVCache(resource);
    EXPECT_ANY_THROW((void)executor.process({stream}));
    EXPECT_FALSE(reached_model);
}

TEST_F(NormalEngineTest, testFeaturePrefillWithoutWriterCannotPublishRegistration) {
    class SparseSaveProcessor: public NormalBatchStreamProcessor {
    public:
        SparseSaveProcessor(const ModelConfig& model, const CacheConfig& cache):
            NormalBatchStreamProcessor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache, false) {}

        absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& groups,
                                                        TensorHolder&       holder) const override {
            auto result = NormalBatchStreamProcessor::gatherModelInput(groups, holder);
            if (result.ok()) {
                result->cache_save_positions = torch::tensor({5}, torch::kInt64).reshape({1, 1});
                result->cache_save_block_ids = torch::tensor({2}, torch::kInt32).reshape({1, 1, 1});
            }
            return result;
        }
    };

    ModelConfig model;
    model.num_layers                          = 1;
    model.max_seq_len                         = 16;
    model.vocab_size                          = 16;
    model.input_vocab_size                    = 16;
    model.hidden_size                         = 4;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 4;
    model.attn_config.tokens_per_block        = 2;
    model.attn_config.kernel_tokens_per_block = 2;
    CacheConfig cache;
    cache.layer_num          = 1;
    cache.seq_size_per_block = 2;
    cache.dtype              = DataType::TYPE_FP16;
    cache.fromGroupedSpecs({test::makeResolvedMhaSpec(cache.dtype, 1, 4, 2, "full")},
                           {{0}}, {CacheGroupType::FULL});
    cache.finalizeBlockNums(8, RuntimeConfig{});

    KVCacheConfig kv_config;
    kv_config.reuse_cache                        = true;
    kv_config.enable_prefill_partial_tail_reuse = true;
    PDSepConfig pd_config;
    pd_config.role_type = RoleType::PREFILL;
    auto manager = std::make_shared<KVCacheManager>(cache, false, nullptr, kv_config,
                                                     ParallelismConfig{}, RuntimeConfig{},
                                                     SpeculativeExecutionConfig{}, pd_config);
    ASSERT_TRUE(manager->init());

    EngineInitParams params;
    params.model_config_   = model;
    params.py_model        = py::none();
    params.kv_cache_config = kv_config;
    params.pd_sep_config   = pd_config;
    NormalExecutor executor(params, manager);
    bool reached_model = false;
    executor.setModel(std::make_unique<MockModel>(model.vocab_size, [&](const GptModelInputs&) {
        reached_model = true;
    }));
    executor.setBatchProcessor(std::make_unique<SparseSaveProcessor>(model, cache));

    auto query = std::make_shared<GenerateInput>();
    query->input_ids = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();
    ResourceContext context;
    context.cache_manager = manager;
    context.reuse_cache = true;
    context.role_type = RoleType::PREFILL;
    context.ignore_request_cache_switches = true;
    context.enable_prefill_partial_tail_reuse = true;
    auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, context, nullptr);
    auto& resource = stream->streamCacheResource();
    resource.init(1);
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
    EXPECT_ANY_THROW((void)executor.process({stream}));
    EXPECT_FALSE(reached_model);
    for (const auto& point : resource.prefixCacheRegistration()->points) {
        for (const auto& binding : point.group_bindings) {
            EXPECT_NE(binding.state, PrefixCacheSaveState::CONTENT_COMPLETE);
        }
    }
    stream->releaseResource();
}

TEST_F(NormalEngineTest, testTailReuseCaptureErrorAndConcurrentCancelNeverPublish) {
    for (bool concurrent_cancel : {false, true}) {
        ModelConfig model;
        model.num_layers                          = 1;
        model.max_seq_len                         = 16;
        model.vocab_size                          = 16;
        model.input_vocab_size                    = 16;
        model.hidden_size                         = 4;
        model.attn_config.head_num                = 1;
        model.attn_config.kv_head_num             = 1;
        model.attn_config.size_per_head           = 4;
        model.attn_config.tokens_per_block        = 2;
        model.attn_config.kernel_tokens_per_block = 2;
        CacheConfig cache;
        cache.layer_num          = 1;
        cache.seq_size_per_block = 2;
        cache.dtype              = DataType::TYPE_FP16;
        cache.fromGroupedSpecs({test::makeResolvedMhaSpec(cache.dtype, 1, 4, 2, "full")},
                               {{0}}, {CacheGroupType::FULL});
        cache.finalizeBlockNums(16, RuntimeConfig{});
        KVCacheConfig kv_config;
        kv_config.reuse_cache = true;
        kv_config.enable_device_cache = true;
        kv_config.enable_prefill_partial_tail_reuse = true;
        PDSepConfig pd_config;
        pd_config.role_type = RoleType::PREFILL;
        auto manager = std::make_shared<KVCacheManager>(cache, false, nullptr, kv_config,
                                                         ParallelismConfig{}, RuntimeConfig{},
                                                         SpeculativeExecutionConfig{}, pd_config);
        ASSERT_TRUE(manager->init());

        ResourceContext context;
        context.cache_manager = manager;
        context.reuse_cache = true;
        context.role_type = RoleType::PREFILL;
        context.ignore_request_cache_switches = true;
        context.enable_prefill_partial_tail_reuse = true;
        auto query = std::make_shared<GenerateInput>();
        query->input_ids = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        auto& resource = stream->streamCacheResource();
        resource.init(1);
        ASSERT_TRUE(resource.initKVBlock().ok());
        ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
        ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
        CacheKeysType path;
        for (const auto& point : resource.prefixCacheRegistration()->points) {
            path.push_back(point.endpoint.key);
        }

        EngineInitParams params;
        params.model_config_ = model;
        params.py_model = py::none();
        params.kv_cache_config = kv_config;
        params.pd_sep_config = pd_config;
        NormalExecutor executor(params, manager);
        auto failing_model = std::make_unique<FailingSaveModel>(stream, concurrent_cancel);
        auto* failing_model_ptr = failing_model.get();
        executor.setModel(std::move(failing_model));
        const auto status = executor.process({stream});
        EXPECT_TRUE(status.ok()) << status.ToString();
        EXPECT_TRUE(failing_model_ptr->waitForConcurrentRelease());
        EXPECT_TRUE(stream->hasError());
        EXPECT_FALSE(resource.prefixCacheRegistration()->content_completed);
        const auto found = manager->blockTreeCache()->tree()->findNode(path);
        EXPECT_TRUE(found.size() < path.size() || found.back()->token_end != 5);
        if (!concurrent_cancel) {
            stream->releaseResource();
        }
        EXPECT_TRUE(resource.prefixCacheRegistration()->source_pins.empty());
    }
}

TEST_F(NormalEngineTest, testInternalForkWriterOrCaptureErrorAndConcurrentCancelNeverPublish) {
    const std::vector<std::pair<bool, bool>> failure_cases{{false, false}, {false, true}, {true, false}, {true, true}};
    for (const auto& [writer_failure, concurrent_cancel] : failure_cases) {
        SCOPED_TRACE(::testing::Message() << "writer_failure=" << writer_failure
                                          << " concurrent_cancel=" << concurrent_cancel);
        ModelConfig model;
        model.num_layers                          = 1;
        model.max_seq_len                         = 16;
        model.vocab_size                          = 16;
        model.input_vocab_size                    = 16;
        model.hidden_size                         = 4;
        model.attn_config.head_num                = 1;
        model.attn_config.kv_head_num             = 1;
        model.attn_config.size_per_head           = 4;
        model.attn_config.tokens_per_block        = 2;
        model.attn_config.kernel_tokens_per_block = 2;
        auto cache = test::makeSimpleLinearCacheConfig(1, 16, 2, DataType::TYPE_FP16);
        KVCacheConfig kv_config;
        kv_config.reuse_cache                            = true;
        kv_config.enable_device_cache                    = true;
        kv_config.enable_prefill_partial_tail_reuse      = true;
        kv_config.enable_prefill_common_prefix_recovery  = true;
        PDSepConfig pd_config;
        pd_config.role_type = RoleType::PREFILL;
        auto manager = std::make_shared<KVCacheManager>(cache, false, nullptr, kv_config,
                                                         ParallelismConfig{}, RuntimeConfig{},
                                                         SpeculativeExecutionConfig{}, pd_config);
        ASSERT_TRUE(manager->init());

        ResourceContext context;
        context.cache_manager = manager;
        context.reuse_cache = true;
        context.role_type = RoleType::PREFILL;
        context.ignore_request_cache_switches = true;
        context.enable_prefill_partial_tail_reuse = true;
        context.enable_prefill_common_prefix_recovery = true;
        auto query = std::make_shared<GenerateInput>();
        query->input_ids = torch::tensor({1, 2, 3, 9, 5}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        auto& resource = stream->streamCacheResource();
        resource.init(1);
        auto tokens = stream->completeTokenIdsPtr();

        // The source shares three tokens with B, then diverges inside the
        // second physical block. Matching must discover F3 from the tree.
        const auto prefix_key = cacheKeyAtTokenEnd(*tokens, 0, 2, 2, 0);
        const auto source_key = cacheKeyForSourceSegment(*tokens, {3, 4}, 2, 2, prefix_key);
        auto tree = manager->blockTreeCache()->tree();
        std::vector<std::vector<GroupSetResource>> empty(2, std::vector<GroupSetResource>(1));
        const std::vector<std::vector<int>> source_segments{{1, 2}, {3, 4}};
        const std::vector<size_t> source_ends{2, 4};
        ASSERT_EQ(tree->insertNode({prefix_key, source_key}, empty, false, false,
                                   &source_segments, &source_ends).inserted_nodes.size(), 2u);

        auto pool = manager->blockTreeCache()->groupSets().front()->devicePools().front();
        const size_t free_before = pool->freeBlocksNum();
        ASSERT_TRUE(resource.initKVBlock().ok());
        ASSERT_EQ(stream->initialReuseLength(), 0);
        ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
        resource.prepareRegisteredSave();
        ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
        const auto& registration = *resource.prefixCacheRegistration();
        ASSERT_TRUE(registration.prepared);
        ASSERT_EQ(registration.points.size(), 4u);
        ASSERT_TRUE(registration.farthest_fork.has_value());
        EXPECT_EQ(registration.farthest_fork->pos, 3u);
        EXPECT_EQ(registration.points[2].endpoint.pos, 5u);
        EXPECT_FALSE(registration.points[2].internal_only);
        EXPECT_EQ(registration.points[3].endpoint.pos, 3u);
        EXPECT_TRUE(registration.points[3].internal_only);
        const auto& tail = registration.points[2].group_bindings.front();
        const auto& fork = registration.points[3].group_bindings.front();
        ASSERT_EQ(tail.state, PrefixCacheSaveState::REGISTERED);
        ASSERT_EQ(fork.state, PrefixCacheSaveState::REGISTERED);
        ASSERT_TRUE(tail.resource.hasCompleteDeviceValue());
        ASSERT_TRUE(fork.resource.hasCompleteDeviceValue());
        const auto tail_block = tail.resource.device_blocks.front();
        const auto fork_block = fork.resource.device_blocks.front();
        EXPECT_NE(tail_block, fork_block);

        EngineInitParams params;
        params.model_config_ = model;
        params.py_model = py::none();
        params.kv_cache_config = kv_config;
        params.pd_sep_config = pd_config;
        NormalExecutor executor(params, manager);
        auto failing_model = std::make_unique<FailingSaveModel>(stream, concurrent_cancel,
                                                               std::vector<int64_t>{3, 5}, writer_failure);
        auto* failing_model_ptr = failing_model.get();
        executor.setModel(std::move(failing_model));
        if (writer_failure) {
            EXPECT_THROW((void)executor.process({stream}), std::runtime_error);
        } else {
            const auto status = executor.process({stream});
            EXPECT_TRUE(status.ok()) << status.ToString();
        }
        EXPECT_TRUE(failing_model_ptr->writerWorkSubmitted());
        EXPECT_TRUE(failing_model_ptr->waitForConcurrentRelease());
        if (!writer_failure || concurrent_cancel) {
            EXPECT_TRUE(stream->hasError());
        }
        EXPECT_FALSE(registration.content_completed);
        EXPECT_EQ(fork.state, PrefixCacheSaveState::REGISTERED);
        EXPECT_EQ(tail.state, PrefixCacheSaveState::REGISTERED);
        EXPECT_LT(tree->findNode({prefix_key, registration.points[3].endpoint.key}).size(), 2u);
        EXPECT_LT(tree->findNode({prefix_key, registration.points[1].endpoint.key,
                                  registration.points[2].endpoint.key}).size(), 3u);
        if (!concurrent_cancel) {
            stream->releaseResource();
        }
        EXPECT_TRUE(registration.source_pins.empty());
        EXPECT_FALSE(pool->isAllocated(tail_block));
        EXPECT_FALSE(pool->isAllocated(fork_block));
        EXPECT_EQ(pool->freeBlocksNum(), free_before);
        EXPECT_EQ(tree->findNode({prefix_key, source_key}).size(), 2u);

        auto retry_query = std::make_shared<GenerateInput>();
        retry_query->input_ids = torch::tensor({1, 2, 3, 9, 5}, torch::kInt32);
        retry_query->generate_config = std::make_shared<GenerateConfig>();
        auto retry = std::make_shared<NormalGenerateStream>(retry_query, model, RuntimeConfig{}, context, nullptr);
        auto& retry_resource = retry->streamCacheResource();
        retry_resource.init(1);
        ASSERT_TRUE(retry_resource.initKVBlock().ok());
        ASSERT_TRUE(retry_resource.prepareRequiredContinuation().ok());
        retry_resource.prepareRegisteredSave();
        ASSERT_TRUE(retry_resource.prefixCacheRegistration().has_value());
        const auto& retry_points = retry_resource.prefixCacheRegistration()->points;
        ASSERT_EQ(retry_points.size(), 4u);
        EXPECT_TRUE(retry_points[3].internal_only);
        EXPECT_EQ(retry_points[3].endpoint.pos, 3u);
        EXPECT_TRUE(retry_points[2].group_bindings.front().resource.hasCompleteDeviceValue());
        EXPECT_TRUE(retry_points[3].group_bindings.front().resource.hasCompleteDeviceValue());
        retry->releaseResource();
        EXPECT_EQ(pool->freeBlocksNum(), free_before);
    }
}

void NormalEngineTest::checkInternalForkHistoryCopy(bool        invalid_row,
                                                    const char* remote_address,
                                                    const char* failed_block_dir,
                                                    bool        fail_after_copy) {
    class ForkCopyRowProcessor: public NormalBatchStreamProcessor {
    public:
        ForkCopyRowProcessor(const ModelConfig& model,
                             const CacheConfig& config,
                             BlockIdxType       src,
                             BlockIdxType       dst,
                             bool&              observed,
                             bool               invalid_row):
            NormalBatchStreamProcessor(model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config, false),
            src_(src),
            dst_(dst),
            observed_(observed),
            invalid_row_(invalid_row) {}

        absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& groups,
                                                        TensorHolder&       holder) const override {
            auto result = NormalBatchStreamProcessor::gatherModelInput(groups, holder);
            if (!result.ok()) {
                return result.status();
            }
            auto& inputs = result.value();
            if (!inputs.kv_cache_update_mapping.defined()) {
                return absl::InternalError("F3 history copy mapping was not gathered");
            }
            auto* rows = inputs.kv_cache_update_mapping.data_ptr<int32_t>();
            for (int64_t i = 0; i < inputs.kv_cache_update_mapping.size(0); ++i) {
                const auto group_row = rows[3 * i];
                if (group_row >= 0 && static_cast<size_t>(group_row) < inputs.kv_cache_group_tags.size()
                    && inputs.kv_cache_group_tags[group_row] == "linear" && rows[3 * i + 1] == src_
                    && rows[3 * i + 2] == dst_) {
                    observed_ = true;
                    // Fault only the genuine F3 history-copy row. The existing
                    // copy validator rejects the complete mapping before any copy.
                    if (invalid_row_) {
                        rows[3 * i] = static_cast<int32_t>(inputs.kv_cache_group_tags.size());
                    }
                    return result;
                }
            }
            return absl::InternalError("F3 history copy row missing from gathered mapping");
        }

    private:
        BlockIdxType src_;
        BlockIdxType dst_;
        bool&        observed_;
        bool         invalid_row_;
    };

    ModelConfig model;
    model.num_layers                          = 2;
    model.max_seq_len                         = 16;
    model.vocab_size                          = 16;
    model.input_vocab_size                    = 16;
    model.hidden_size                         = 4;
    model.attn_config.head_num                = 1;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 4;
    model.attn_config.tokens_per_block        = 2;
    model.attn_config.kernel_tokens_per_block = 2;

    CacheConfig cache;
    cache.layer_num          = 2;
    cache.seq_size_per_block = 2;
    cache.linear_step        = 1;
    cache.dtype              = DataType::TYPE_FP16;
    auto full_spec           = test::makeResolvedMhaSpec(cache.dtype, 1, 4, 2, "full");
    auto linear_spec = test::makeResolvedLinearSpec(cache.dtype, 1, 1, 1, 1, 2, 4, cache.dtype, cache.dtype, "linear");
    cache.fromGroupedSpecs(
        {full_spec, linear_spec}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::LINEAR}, {"full", "linear"});
    cache.finalizeBlockNums(16, RuntimeConfig{});
    KVCacheConfig kv_config;
    kv_config.reuse_cache                           = true;
    kv_config.enable_device_cache                   = true;
    kv_config.enable_prefill_partial_tail_reuse     = true;
    kv_config.enable_prefill_common_prefix_recovery = true;
    RuntimeConfig runtime_config;
    if (remote_address) {
        runtime_config.model_name       = "common-prefix-real-remote-history-fixture";
        kv_config.enable_remote_cache   = true;
        kv_config.enable_memory_cache   = false;
        kv_config.enable_disk_cache     = false;
        kv_config.kvcm_server_address   = remote_address;
        kv_config.kvcm_enable_vipserver = false;
        // Isolate remote keys from other runs without changing production identity logic.
        kv_config.kvcm_instance_group   = "default";
        kv_config.kvcm_instance_id_salt = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    }
    PDSepConfig pd_config;
    pd_config.role_type = RoleType::PREFILL;
    auto manager        = std::make_shared<KVCacheManager>(
        cache, false, nullptr, kv_config, ParallelismConfig{}, runtime_config, SpeculativeExecutionConfig{}, pd_config);
    ASSERT_TRUE(manager->init());

    ResourceContext context;
    context.cache_manager                         = manager;
    context.reuse_cache                           = true;
    context.role_type                             = RoleType::PREFILL;
    context.ignore_request_cache_switches         = true;
    context.enable_prefill_partial_tail_reuse     = true;
    context.enable_prefill_common_prefix_recovery = true;
    context.enable_remote_cache                   = remote_address != nullptr;
    auto query                                    = std::make_shared<GenerateInput>();
    query->input_ids                              = torch::tensor({1, 2, 3, 9, 5}, torch::kInt32);
    query->generate_config                        = std::make_shared<GenerateConfig>();
    auto stream = std::make_shared<NormalGenerateStream>(query, model, RuntimeConfig{}, context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    auto& resource                   = stream->streamCacheResource();
    resource.init(1);
    auto        tokens     = stream->completeTokenIdsPtr();
    const auto  prefix_key = cacheKeyAtTokenEnd(*tokens, 0, 2, 2, 0);
    const auto  source_key = cacheKeyForSourceSegment(*tokens, {3, 4}, 2, 2, prefix_key);
    auto        tree       = manager->blockTreeCache()->tree();
    const auto& groups     = manager->blockTreeCache()->groupSets();
    ASSERT_EQ(groups.size(), 2u);
    const auto full_group = std::find_if(
        groups.begin(), groups.end(), [](const auto& group) { return group->groupTags().front() == "full"; });
    const auto linear_group = std::find_if(
        groups.begin(), groups.end(), [](const auto& group) { return group->groupTags().front() == "linear"; });
    ASSERT_NE(full_group, groups.end());
    ASSERT_NE(linear_group, groups.end());
    const size_t linear_group_id = static_cast<size_t>(std::distance(groups.begin(), linear_group));
    auto         linear_pool     = (*linear_group)->devicePools().front();

    std::vector<std::vector<GroupSetResource>> source_resources(remote_address ? 3 : 2,
                                                                std::vector<GroupSetResource>(groups.size()));
    std::vector<std::pair<torch::Tensor, int>> source_payloads;
    for (size_t node = 0; node < source_resources.size(); ++node) {
        for (size_t group = 0; group < groups.size(); ++group) {
            // A's FULL-only suffix keeps the preceding FULL nodes nonleaf during
            // LINEAR eviction. This is ordinary partial-group content, not F state.
            if (node == 2 && groups[group]->groupTags().front() != "full") {
                continue;
            }
            auto       pool   = groups[group]->devicePools().front();
            const auto blocks = pool->malloc(1);
            ASSERT_TRUE(blocks.has_value());
            pool->incRef(*blocks);
            source_resources[node][group].device_blocks = *blocks;
            const auto& tag                             = groups[group]->groupTags().front();
            const int   layer                           = tag == "full" ? 0 : 1;
            auto        payload = torch::from_blob(manager->convertIndexToAddr(layer, tag, blocks->front()).kv_addr,
                                                   {static_cast<int64_t>(cache.group(tag).kvBlockStrideBytes())},
                                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
            const int   pattern = 37 + static_cast<int>(node * groups.size() + group);
            payload.fill_(pattern);
            if (!remote_address || tag == "full") {
                source_payloads.emplace_back(std::move(payload), pattern);
            }
        }
    }
    auto                          source_linear = source_resources[0][linear_group_id].device_blocks.front();
    std::vector<std::vector<int>> source_segments{{1, 2}, {3, 4}};
    std::vector<size_t>           source_ends{2, 4};
    CacheKeysType                 source_keys{prefix_key, source_key};
    if (remote_address) {
        source_segments.push_back({5, 6});
        source_ends.push_back(6);
        source_keys.push_back(cacheKeyForSourceSegment(*tokens, {5, 6}, 4, 2, source_key));
    }
    if (remote_address) {
        // Publishing can start backend export on another thread. Complete source
        // initialization before exposing these buffers to the ordinary storer.
        runtimeSyncAndCheck();
        // The public cache insertion path also registers eviction candidates.
        // A raw tree insertion alone cannot be reclaimed by evictForGroup.
        manager->blockTreeCache()->insert(
            source_keys, source_resources, Tier::DEVICE, false, &source_segments, nullptr, &source_ends);
        ASSERT_EQ(tree->findNode(source_keys).size(), 3u);
    } else {
        ASSERT_EQ(
            tree->insertNode({prefix_key, source_key}, source_resources, false, false, &source_segments, &source_ends)
                .inserted_nodes.size(),
            2u);
    }
    for (size_t node = 0; node < source_resources.size(); ++node) {
        for (size_t group = 0; group < groups.size(); ++group) {
            groups[group]->devicePools().front()->decRef(source_resources[node][group].device_blocks);
        }
    }
    auto blockBytes = [&](BlockIdxType block) {
        return torch::from_blob(manager->convertIndexToAddr(1, "linear", block).kv_addr,
                                {static_cast<int64_t>(cache.group("linear").kvBlockStrideBytes())},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    };
    runtimeSyncAndCheck();
    const auto expected_remote_linear = blockBytes(source_linear).cpu().clone();
    if (remote_address) {
        auto backend = manager->blockTreeCache()->storageBackend();
        ASSERT_NE(backend, nullptr);
        StorageRequest request;
        request.keys = std::make_shared<CacheKeysType>(CacheKeysType{prefix_key});
        request.handles.resize(1);
        for (size_t group = 0; group < groups.size(); ++group) {
            request.handles[0].push_back(
                {groups[group]->groupTags().front(), source_resources[0][group].device_blocks.front()});
        }
        // Normal cache insertion already exports both canonical A nodes. Avoid
        // duplicate writes; poll G2's committed metadata from that real export.
        // Later Stream reads must also prove the actual transferred bytes.
        const auto deadline  = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        bool       committed = false;
        while (!committed && std::chrono::steady_clock::now() < deadline) {
            auto completion = std::make_shared<std::promise<bool>>();
            auto future     = completion->get_future();
            backend->match(request,
                           [completion](size_t count,
                                        std::shared_ptr<StorageBackendMatchMeta>,
                                        bool                         success,
                                        PrefixCacheRemoteQueryStatus status) {
                               completion->set_value(success && count == 1
                                                     && status == PrefixCacheRemoteQueryStatus::HIT);
                           });
            ASSERT_EQ(future.wait_until(deadline), std::future_status::ready);
            committed = future.get();
            if (!committed) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        ASSERT_TRUE(committed);
        // Write-task pins can outlive metadata commit and are released one by
        // one. Both LINEAR buffers must be quiescent before poisoning either.
        for (const auto& node_resources : source_resources) {
            if (node_resources[linear_group_id].device_blocks.empty()) {
                continue;
            }
            const auto block = node_resources[linear_group_id].device_blocks.front();
            while (linear_pool->refCount(block) > 1 && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            ASSERT_EQ(linear_pool->refCount(block), 1u);
        }
        // A recycled allocation must not pass from stale original bytes when the
        // remote transfer is missing. Poison only uploaded LINEAR sources; FULL
        // stays immutable and expected_remote_linear owns detached CPU storage.
        for (size_t node = 0; node < source_resources.size(); ++node) {
            if (source_resources[node][linear_group_id].device_blocks.empty()) {
                continue;
            }
            const auto block = source_resources[node][linear_group_id].device_blocks.front();
            blockBytes(block).fill_(201 + static_cast<int>(node));
        }
        runtimeSyncAndCheck();
        ASSERT_EQ(manager->blockTreeCache()->evictForGroup("linear", 2), 2);
        auto retained = tree->findNode(source_keys);
        ASSERT_EQ(retained.size(), 3u);
        for (const auto* node : retained) {
            ASSERT_TRUE(node->group_set_resources[static_cast<size_t>(std::distance(groups.begin(), full_group))]
                            .hasCompleteDeviceValue());
            ASSERT_FALSE(node->group_set_resources[linear_group_id].hasTier(Tier::DEVICE));
            ASSERT_FALSE(node->group_set_resources[linear_group_id].hasTier(Tier::HOST));
            ASSERT_FALSE(node->group_set_resources[linear_group_id].hasTier(Tier::DISK));
        }
        // Never dereference old LINEAR addresses after eviction: the expected bytes
        // above own CPU storage, and only retained FULL tensors remain in source_payloads.
    }
    const size_t linear_free_before = linear_pool->freeBlocksNum();
    const auto   source_refs_before = remote_address ? 0u : linear_pool->refCount(source_linear);
    if (failed_block_dir) {
        // The server for this opt-in run owns one fresh storage root. Match the
        // exact uploaded G2 LINEAR bytes before taking only that file offline;
        // metadata remains committed so the real SDK read must fail downstream.
        namespace fs = std::filesystem;
        const fs::path block_root(failed_block_dir);
        ASSERT_TRUE(block_root.is_absolute());
        ASSERT_EQ(block_root.filename(), "block");
        ASSERT_TRUE(fs::is_directory(block_root));
        ASSERT_FALSE(fs::is_symlink(block_root));
        std::vector<fs::path> instances;
        const std::string      instance_prefix = kv_config.kvcm_instance_id_salt + "_";
        for (const auto& entry : fs::directory_iterator(block_root)) {
            ASSERT_TRUE(entry.is_directory());
            ASSERT_FALSE(entry.is_symlink());
            if (entry.path().filename().string().rfind(instance_prefix, 0) == 0) {
                instances.push_back(entry.path());
            }
        }
        ASSERT_EQ(instances.size(), 1u);
        const fs::path linear_dir = instances.front() / "tp0_Llinear";
        ASSERT_TRUE(fs::is_directory(linear_dir));
        ASSERT_FALSE(fs::is_symlink(linear_dir));
        std::vector<fs::path> matching_files;
        size_t                regular_files = 0;
        const auto*           expected      = expected_remote_linear.data_ptr<uint8_t>();
        for (const auto& entry : fs::directory_iterator(linear_dir)) {
            ASSERT_TRUE(entry.is_regular_file());
            ASSERT_FALSE(entry.is_symlink());
            ++regular_files;
            std::ifstream input(entry.path(), std::ios::binary);
            ASSERT_TRUE(input.is_open());
            const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            ASSERT_FALSE(input.bad());
            if (bytes.size() == static_cast<size_t>(expected_remote_linear.numel())
                && std::equal(bytes.begin(), bytes.end(), expected)) {
                matching_files.push_back(entry.path());
            }
        }
        ASSERT_EQ(regular_files, 2u);
        ASSERT_EQ(matching_files.size(), 1u);
        const fs::path missing_path = matching_files.front();
        const fs::path held_path    = missing_path.string() + ".unavailable";
        ASSERT_FALSE(fs::exists(held_path));
        std::error_code rename_error;
        fs::rename(missing_path, held_path, rename_error);
        ASSERT_FALSE(rename_error) << rename_error.message();
        ASSERT_FALSE(fs::exists(missing_path));
        ASSERT_TRUE(fs::is_regular_file(held_path));
        std::cerr << "real KVCM read fault: held G2 LINEAR object " << held_path << std::endl;

        // The object is gone, but the real metadata service must still offer
        // the committed G2 key. Otherwise this would only test a remote miss.
        auto backend = manager->blockTreeCache()->storageBackend();
        ASSERT_NE(backend, nullptr);
        StorageRequest request;
        request.keys = std::make_shared<CacheKeysType>(CacheKeysType{prefix_key});
        request.handles.resize(1);
        for (const auto& group : groups) {
            request.handles[0].push_back({group->groupTags().front(), NULL_BLOCK_IDX});
        }
        auto completion = std::make_shared<std::promise<bool>>();
        auto future     = completion->get_future();
        backend->match(request,
                       [completion](size_t count,
                                    std::shared_ptr<StorageBackendMatchMeta>,
                                    bool                         success,
                                    PrefixCacheRemoteQueryStatus status) {
                           completion->set_value(success && count == 1 && status == PrefixCacheRemoteQueryStatus::HIT);
                       });
        ASSERT_EQ(future.wait_for(std::chrono::seconds(60)), std::future_status::ready);
        ASSERT_TRUE(future.get());
    }
    ASSERT_TRUE(resource.initKVBlock().ok());
    if (failed_block_dir) {
        ASSERT_TRUE(resource.asyncLoadCache());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!resource.loadCacheDone() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_TRUE(resource.loadCacheDone());
        EXPECT_FALSE(resource.asyncLoadCache());
        EXPECT_EQ(stream->remoteReuseLength(), 0);
        EXPECT_EQ(stream->initialReuseLength(), 0);
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->kvCache().cacheResource(0).storageBackendReuseBlockNum(), 0u);
        ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
        resource.prepareRegisteredSave();
        std::vector<std::pair<size_t, BlockIdxType>> optional_blocks;
        std::vector<CacheKeyType>                    fork_keys;
        bool                                         saw_internal_fork = false;
        bool                                         saw_fork_linear_binding = false;
        EXPECT_TRUE(resource.prefixCacheRegistration().has_value());
        if (resource.prefixCacheRegistration()) {
            const auto& registration = *resource.prefixCacheRegistration();
            EXPECT_TRUE(registration.prepared);
            EXPECT_TRUE(registration.farthest_fork.has_value());
            if (registration.farthest_fork) {
                EXPECT_EQ(registration.farthest_fork->pos, 3u);
            }
            for (const auto& point : registration.points) {
                if (point.internal_only) {
                    EXPECT_EQ(point.endpoint.pos, 3u);
                    saw_internal_fork = true;
                    fork_keys.push_back(point.endpoint.key);
                    EXPECT_LT(tree->findNode({prefix_key, point.endpoint.key}).size(), 2u);
                }
                for (const auto& binding : point.group_bindings) {
                    // FULL can be an existing cached reference or a zero-state
                    // demand at F. Only LINEAR's missing exact state must stay
                    // unready after the failed remote byte read.
                    if (point.internal_only && binding.group_set_id == linear_group_id) {
                        saw_fork_linear_binding = true;
                        EXPECT_EQ(binding.state, PrefixCacheSaveState::REGISTERED);
                        EXPECT_EQ(binding.ownership, PrefixCacheSaveOwnership::OPTIONAL_REQUEST);
                        EXPECT_TRUE(binding.resource.hasCompleteDeviceValue());
                    }
                    if (binding.ownership == PrefixCacheSaveOwnership::OPTIONAL_REQUEST
                        && binding.resource.hasCompleteDeviceValue()) {
                        for (const auto block : binding.resource.device_blocks) {
                            optional_blocks.emplace_back(binding.group_set_id, block);
                        }
                    }
                }
            }
        }
        EXPECT_TRUE(saw_internal_fork);
        EXPECT_TRUE(saw_fork_linear_binding);
        EXPECT_FALSE(optional_blocks.empty());
        for (const auto& [payload, pattern] : source_payloads) {
            EXPECT_TRUE(payload.cpu().eq(pattern).all().item<bool>());
        }
        stream->releaseResource();
        if (resource.prefixCacheRegistration()) {
            EXPECT_TRUE(resource.prefixCacheRegistration()->source_pins.empty());
        }
        for (const auto& [group_id, block] : optional_blocks) {
            EXPECT_FALSE(groups.at(group_id)->devicePools().front()->isAllocated(block));
        }
        for (const auto fork_key : fork_keys) {
            EXPECT_LT(tree->findNode({prefix_key, fork_key}).size(), 2u);
        }
        EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before);
        ASSERT_EQ(tree->findNode(source_keys).size(), 3u);
        for (const auto& [payload, pattern] : source_payloads) {
            EXPECT_TRUE(payload.cpu().eq(pattern).all().item<bool>());
        }
        return;
    }
    if (remote_address) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!resource.loadCacheDone() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_TRUE(resource.loadCacheDone());
        ASSERT_EQ(stream->remoteReuseLength(), 2);
        ASSERT_EQ(stream->kvCache().cacheResource(0).storageBackendReuseBlockNum(), 1u);
        const auto& restored = stream->kvCache().cacheResource(0).blockIds("linear").blocks();
        ASSERT_FALSE(restored.empty());
        source_linear = restored.front();
        runtimeSyncAndCheck();
        const auto restored_linear_bytes = blockBytes(source_linear).cpu();
        ASSERT_TRUE(torch::equal(restored_linear_bytes, expected_remote_linear));
        EXPECT_FALSE(restored_linear_bytes.eq(201).any().item<bool>());
        EXPECT_FALSE(restored_linear_bytes.eq(202).any().item<bool>());
        for (const auto& [payload, pattern] : source_payloads) {
            EXPECT_TRUE(payload.cpu().eq(pattern).all().item<bool>());
        }
        const auto& restored_full = stream->kvCache().cacheResource(0).blockIds("full").blocks();
        ASSERT_FALSE(restored_full.empty());
        auto      full_payload = torch::from_blob(manager->convertIndexToAddr(0, "full", restored_full.front()).kv_addr,
                                                  {static_cast<int64_t>(cache.group("full").kvBlockStrideBytes())},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        const int full_pattern = 37 + static_cast<int>(std::distance(groups.begin(), full_group));
        EXPECT_TRUE(full_payload.cpu().eq(full_pattern).all().item<bool>());
    }
    ASSERT_EQ(stream->initialReuseLength(), 2);
    ASSERT_TRUE(resource.prepareRequiredContinuation().ok());
    resource.prepareRegisteredSave();
    ASSERT_TRUE(resource.prefixCacheRegistration().has_value());
    const auto& registration = *resource.prefixCacheRegistration();
    ASSERT_TRUE(registration.prepared);
    ASSERT_TRUE(registration.farthest_fork.has_value());
    ASSERT_EQ(registration.farthest_fork->pos, 3u);
    const auto fork_point = std::find_if(registration.points.begin(), registration.points.end(), [](const auto& point) {
        return point.internal_only && point.endpoint.pos == 3;
    });
    ASSERT_NE(fork_point, registration.points.end());
    const auto tail_point = std::find_if(registration.points.begin(), registration.points.end(), [](const auto& point) {
        return point.endpoint.pos == 5 && !point.internal_only;
    });
    ASSERT_NE(tail_point, registration.points.end());
    const auto& fork_binding = fork_point->group_bindings[linear_group_id];
    const auto& tail_binding = tail_point->group_bindings[linear_group_id];
    ASSERT_EQ(fork_binding.state, PrefixCacheSaveState::REGISTERED);
    ASSERT_EQ(tail_binding.state, PrefixCacheSaveState::REGISTERED);
    ASSERT_EQ(fork_binding.ownership, PrefixCacheSaveOwnership::OPTIONAL_REQUEST);
    ASSERT_TRUE(fork_binding.resource.hasCompleteDeviceValue());
    const auto fork_destination = fork_binding.resource.device_blocks.front();
    ASSERT_TRUE(tail_binding.resource.hasCompleteDeviceValue());
    ASSERT_NE(fork_destination, tail_binding.resource.device_blocks.front());
    const auto history_copy =
        std::find_if(registration.copy_mappings.begin(), registration.copy_mappings.end(), [&](const auto& mapping) {
            return mapping.tag == "linear" && mapping.src == source_linear && mapping.dst == fork_destination;
        });
    ASSERT_NE(history_copy, registration.copy_mappings.end());

    ASSERT_NE(fork_destination, source_linear);
    auto destination_payload = blockBytes(fork_destination);
    destination_payload.fill_(211);
    runtimeSyncAndCheck();
    auto source_before = blockBytes(source_linear).cpu().clone();
    ASSERT_FALSE(source_before.eq(211).any().item<bool>());
    bool observed_fork_copy = false;
    if (invalid_row) {
        EngineInitParams params;
        params.model_config_   = model;
        params.py_model        = py::none();
        params.kv_cache_config = kv_config;
        params.pd_sep_config   = pd_config;
        NormalExecutor executor(params, manager);
        auto  failing_model     = std::make_unique<FailingSaveModel>(stream, false, std::vector<int64_t>{3, 5}, true);
        auto* failing_model_ptr = failing_model.get();
        executor.setModel(std::move(failing_model));
        executor.setBatchProcessor(std::make_unique<ForkCopyRowProcessor>(
            model, cache, source_linear, fork_destination, observed_fork_copy, true));
        EXPECT_ANY_THROW((void)executor.process({stream}));
        EXPECT_FALSE(failing_model_ptr->writerWorkSubmitted());
        runtimeSyncAndCheck();
        EXPECT_TRUE(destination_payload.cpu().eq(211).all().item<bool>());
    } else if (fail_after_copy) {
        // The executor performs the real registered F3 history transfer before
        // entering the model. A bounded forward fault must leave the copied
        // history unpublished and release both optional destinations.
        EngineInitParams params;
        params.model_config_   = model;
        params.py_model        = py::none();
        params.kv_cache_config = kv_config;
        params.pd_sep_config   = pd_config;
        NormalExecutor executor(params, manager);
        auto failing_model = std::make_unique<FailingSaveModel>(stream, false, std::vector<int64_t>{3, 5}, true);
        auto* failing_model_ptr = failing_model.get();
        executor.setModel(std::move(failing_model));
        executor.setBatchProcessor(std::make_unique<ForkCopyRowProcessor>(
            model, cache, source_linear, fork_destination, observed_fork_copy, false));
        EXPECT_THROW((void)executor.process({stream}), std::runtime_error);
        EXPECT_TRUE(failing_model_ptr->writerWorkSubmitted());
        runtimeSyncAndCheck();
        EXPECT_TRUE(torch::equal(destination_payload.cpu(), source_before));
    } else {
        // Exercise the unmodified gathered mapping and real device-copy path.
        // History alone is not completed F3 state: do not run a fake writer
        // or mark/publish this registration as a successful model prefill.
        ForkCopyRowProcessor processor(model, cache, source_linear, fork_destination, observed_fork_copy, false);
        StreamGroups         stream_groups({stream});
        TensorHolder         holder;
        auto                 inputs = processor.gatherModelInput(stream_groups, holder);
        ASSERT_TRUE(inputs.ok());
        ASSERT_TRUE(observed_fork_copy);
        ASSERT_NO_THROW(
            manager->blockBatchCopyForForward(inputs->kv_cache_update_mapping, inputs->kv_cache_group_tags, true));
        runtimeSyncAndCheck();
        EXPECT_TRUE(torch::equal(destination_payload.cpu(), source_before));
    }
    EXPECT_TRUE(observed_fork_copy);
    EXPECT_TRUE(torch::equal(blockBytes(source_linear).cpu(), source_before));
    EXPECT_EQ(fork_binding.state, PrefixCacheSaveState::REGISTERED);
    EXPECT_EQ(tail_binding.state, PrefixCacheSaveState::REGISTERED);
    EXPECT_LT(tree->findNode({prefix_key, fork_point->endpoint.key}).size(), 2u);
    EXPECT_LT(tree->findNode(registration.base_keys).size(), registration.base_keys.size());
    for (const auto& [payload, pattern] : source_payloads) {
        EXPECT_TRUE(payload.cpu().eq(pattern).all().item<bool>());
    }
    stream->releaseResource();
    EXPECT_TRUE(registration.source_pins.empty());
    EXPECT_FALSE(linear_pool->isAllocated(fork_destination));
    if (!remote_address) {
        EXPECT_EQ(linear_pool->refCount(source_linear), source_refs_before);
    }
    for (const auto& [payload, pattern] : source_payloads) {
        EXPECT_TRUE(payload.cpu().eq(pattern).all().item<bool>());
    }
    EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before);

    if (remote_address) {
        // Backend reads target request allocations, unlike HOST/DISK load-desc
        // adoption. No completed prefill was stored, so original source nodes still
        // have no LINEAR device resources and all request allocations are free.
        const auto retained = tree->findNode(source_keys);
        ASSERT_EQ(retained.size(), 3u);
        for (const auto* node : retained) {
            EXPECT_FALSE(node->group_set_resources[linear_group_id].hasTier(Tier::DEVICE));
        }
        return;
    }

    auto retry_query             = std::make_shared<GenerateInput>();
    retry_query->input_ids       = torch::tensor({1, 2, 3, 9, 5}, torch::kInt32);
    retry_query->generate_config = std::make_shared<GenerateConfig>();
    auto  retry = std::make_shared<NormalGenerateStream>(retry_query, model, RuntimeConfig{}, context, nullptr);
    auto& retry_resource = retry->streamCacheResource();
    retry_resource.init(1);
    ASSERT_TRUE(retry_resource.initKVBlock().ok());
    ASSERT_EQ(retry->initialReuseLength(), 2);
    ASSERT_TRUE(retry_resource.prepareRequiredContinuation().ok());
    retry_resource.prepareRegisteredSave();
    ASSERT_TRUE(retry_resource.prefixCacheRegistration().has_value());
    const auto& retry_points = retry_resource.prefixCacheRegistration()->points;
    EXPECT_TRUE(std::any_of(retry_points.begin(), retry_points.end(), [&](const auto& point) {
        return point.internal_only && point.endpoint.pos == 3 && point.group_bindings.size() > linear_group_id
               && point.group_bindings[linear_group_id].resource.hasCompleteDeviceValue();
    }));
    retry->releaseResource();
    EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before);
    EXPECT_EQ(linear_pool->refCount(source_linear), source_refs_before);
}

TEST_F(NormalEngineTest, testInternalForkHistoryCopyRowValidationFailureDoesNotPublishOrDamageSource) {
    checkInternalForkHistoryCopy(true);
}

TEST_F(NormalEngineTest, testInternalForkHistoryCopyCopiesSourceWithoutPrematurePublication) {
    checkInternalForkHistoryCopy(false);
}

TEST_F(NormalEngineTest, testInternalForkRealHistoryCopyThenForwardFailureDoesNotPublish) {
    checkInternalForkHistoryCopy(false, nullptr, nullptr, true);
}

TEST_F(NormalEngineTest, testRealRemoteHistoryRestoresStreamAndPreparesInternalFork) {
    const char* address = std::getenv("COMMON_PREFIX_REAL_KVCM_ADDRESS");
    if (address == nullptr || *address == '\0') {
        GTEST_SKIP() << "manual integration requires COMMON_PREFIX_REAL_KVCM_ADDRESS";
    }
    // Preserve initialization diagnostics even if a fatal assertion returns from
    // the helper or an exception unwinds this test. This does not change retry
    // timing or the success criteria.
    struct FlushRemoteTestLogs {
        ~FlushRemoteTestLogs() {
            alog::Logger::flushAll();
        }
    } flush_remote_test_logs;
    checkInternalForkHistoryCopy(false, address);
}

TEST_F(NormalEngineTest, testRealRemoteReadFailureDoesNotPublishInternalFork) {
    const char* address   = std::getenv("COMMON_PREFIX_REAL_KVCM_ADDRESS");
    const char* block_dir = std::getenv("COMMON_PREFIX_REAL_KVCM_BLOCK_DIR");
    if (address == nullptr || *address == '\0' || block_dir == nullptr || *block_dir == '\0') {
        GTEST_SKIP() << "manual integration requires COMMON_PREFIX_REAL_KVCM_ADDRESS and "
                        "COMMON_PREFIX_REAL_KVCM_BLOCK_DIR";
    }
    struct FlushRemoteTestLogs {
        ~FlushRemoteTestLogs() {
            alog::Logger::flushAll();
        }
    } flush_remote_test_logs;
    checkInternalForkHistoryCopy(false, address, block_dir);
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

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    runtime_config.warm_up         = true;
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
    ASSERT_NE(engine->resourceContext().cache_manager, nullptr);
    EXPECT_EQ(engine->resourceContext().cache_manager->reuseUnitTokens(),
              engine->resourceContext().cache_manager->cacheConfig().seq_size_per_block);
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

TEST_F(NormalEngineTest, testWarmupBindingRestoresManagerAfterException) {
    ModelConfig model;
    model.num_layers = 1;
    model.attn_config.head_num = 1;
    model.attn_config.kv_head_num = 1;
    model.attn_config.size_per_head = 1;
    model.attn_config.tokens_per_block = 1;
    model.kv_cache_spec_descs = {{{"default", KVCacheSpecType::MultiHeadAttention}}};
    const auto config = CacheConfigCreator::createWarmupConfig(model, {});
    auto previous_manager = std::make_shared<KVCacheManager>(config, true);
    auto            temporary_manager  = std::make_shared<KVCacheManager>(config, true);
    ResourceContext context;
    context.cache_manager = previous_manager;
    ResourceContext stream_snapshot;
    try {
        ScopedWarmUpCacheManagerBinding binding(context, temporary_manager);
        stream_snapshot = context;
        EXPECT_EQ(stream_snapshot.cache_manager, temporary_manager);
        throw std::runtime_error("simulate warmup preRun failure");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(context.cache_manager, previous_manager);
    EXPECT_EQ(stream_snapshot.cache_manager, temporary_manager);
}

TEST_F(NormalEngineTest, testCacheStoreProducerFactsFollowBoundWarmupCache) {
    ModelConfig model;
    model.num_layers                       = 1;
    model.max_seq_len                      = 16;
    model.vocab_size                       = 16;
    model.hidden_size                      = 4;
    model.attn_config.head_num             = 1;
    model.attn_config.kv_head_num          = 1;
    model.attn_config.size_per_head        = 4;
    model.attn_config.tokens_per_block     = 2;
    model.kv_cache_spec_descs              = {{{"default", KVCacheSpecType::MultiHeadAttention}}};

    ParallelismConfig parallelism;
    parallelism.tp_rank                            = 1;
    parallelism.tp_size                            = 2;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    auto       cache_config = CacheConfigCreator::createConfig(model, parallelism, KVCacheConfig{});
    const auto draft_config = cache_config;
    cache_config.mtp_sub_configs.push_back(cache_config.mergeMTPModule(draft_config, 0, cache_config.layer_num));
    cache_config.finalizeBlockNums(/*global_block_num=*/8, RuntimeConfig{});
    auto       warmup_config       = CacheConfigCreator::createWarmupConfig(model, parallelism);
    const auto warmup_draft_config = warmup_config;
    warmup_config.mtp_sub_configs.push_back(
        warmup_config.mergeMTPModule(warmup_draft_config, 0, warmup_config.layer_num));
    warmup_config.finalizeBlockNums(/*global_block_num=*/2, RuntimeConfig{});
    auto live_manager = std::make_shared<KVCacheManager>(
        cache_config, /*warmup=*/false, nullptr, KVCacheConfig{}, parallelism);
    auto fake_manager = std::make_shared<KVCacheManager>(warmup_config, /*warmup=*/true);
    ASSERT_TRUE(live_manager->init());
    ASSERT_TRUE(fake_manager->init());
    ASSERT_TRUE(live_manager->cacheConfig().request_projection_required);
    EXPECT_EQ(live_manager->cacheConfig().storage_cp_size, 2);
    ASSERT_FALSE(fake_manager->cacheConfig().request_projection_required);
    EXPECT_EQ(fake_manager->cacheConfig().storage_cp_size, 1);

    EngineInitParams params;
    params.model_config_       = model;
    params.parallelism_config  = parallelism;
    params.pd_sep_config.role_type = RoleType::PREFILL;
    params.py_model            = py::none();
    std::vector<std::optional<std::pair<int, int>>> observed;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams& init) {
        observed.push_back(init.cache_store_producer_cp_topology);
        return std::make_unique<MockModel>(model.vocab_size);
    };
    struct FactoryResetGuard {
        ~FactoryResetGuard() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } factory_reset_guard;

    NormalExecutor live(params, live_manager, /*warm_up=*/false);
    NormalExecutor warmup(params, fake_manager, /*warm_up=*/true);
    ASSERT_EQ(observed.size(), 2u);
    ASSERT_TRUE(observed[0].has_value());
    ASSERT_TRUE(observed[1].has_value());
    EXPECT_EQ(*observed[0], (std::pair<int, int>{1, 2}));
    EXPECT_EQ(*observed[1], (std::pair<int, int>{0, 1}));

    for (const auto speculative_type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        params.sp_config.type = speculative_type;
        NormalExecutor draft(params, live_manager, /*warm_up=*/false, /*is_propose=*/true, /*index=*/0);
        NormalExecutor draft_warmup(params, fake_manager, /*warm_up=*/true, /*is_propose=*/true, /*index=*/0);
    }
    ASSERT_EQ(observed.size(), 6u);
    for (size_t i = 2; i < observed.size(); i += 2) {
        ASSERT_TRUE(observed[i].has_value());
        ASSERT_TRUE(observed[i + 1].has_value());
        EXPECT_EQ(*observed[i], (std::pair<int, int>{1, 2}));
        EXPECT_EQ(*observed[i + 1], (std::pair<int, int>{0, 1}));
    }
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

    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    runtime_config.warm_up = true;
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
