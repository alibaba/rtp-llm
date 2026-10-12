#include <limits>
#include <memory>
#include <numeric>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/normal_engine/EngramTokenHistory.h"

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
    cache_config.layer_num     = static_cast<uint32_t>(layer_num);
    cache_config.layer_all_num = static_cast<uint32_t>(layer_num);
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
}

class NormalBatchStreamProcessorTest: public DeviceTestBase {
protected:
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
    EXPECT_FALSE(model_input->kv_cache_block_id.defined());
    EXPECT_FALSE(model_input->kv_cache_kernel_block_id.defined());
}

TEST(EngramTokenHistoryTest, ChunkDecodeAndVerifyUseExplicitHistory) {
    const std::vector<int> history{11, 12, 13, 14, 15, 16};
    std::vector<int32_t>   rows(8);
    fillEngramTokenWindows(history, 0, 2, rows.data());
    EXPECT_EQ(rows, (std::vector<int32_t>{11, -1, -1, -1, 12, 11, -1, -1}));
    fillEngramTokenWindows(history, 4, 2, rows.data());
    EXPECT_EQ(rows, (std::vector<int32_t>{15, 14, 13, 12, 16, 15, 14, 13}));
    const int32_t anchor[]{16, 15, 14, 13};
    const int32_t candidates[]{99, 20, 21, 22, 23};
    rows.resize(20);
    extendEngramVerifyWindows(anchor, candidates, 5, rows.data());
    EXPECT_EQ(rows,
              (std::vector<int32_t>{99, 15, 14, 13, 20, 99, 15, 14, 21, 20, 99, 15, 22, 21, 20, 99, 23, 22, 21, 20}));
    EXPECT_EQ(history.back(), 16);
    EXPECT_EQ(anchor[0], 16);
}

TEST(EngramTokenHistoryTest, InvalidRangesFailBeforeWriting) {
    const std::vector<int> history{1, 2, 3};
    std::vector<int32_t>   rows(8, 777);
    EXPECT_THROW(fillEngramTokenWindows(history, -1, 1, rows.data()), std::invalid_argument);
    EXPECT_THROW(fillEngramTokenWindows(history, 2, 2, rows.data()), std::invalid_argument);
    EXPECT_THROW(fillEngramTokenWindows(history, 1, std::numeric_limits<size_t>::max(), rows.data()),
                 std::invalid_argument);
    EXPECT_THROW(fillEngramTokenWindows(history, 0, 1, nullptr), std::invalid_argument);
    EXPECT_THROW(extendEngramVerifyWindows(nullptr, nullptr, 1, rows.data()), std::invalid_argument);
    EXPECT_EQ(rows, std::vector<int32_t>(8, 777));
    EXPECT_NO_THROW(fillEngramTokenWindows(history, 3, 0, nullptr));
    EXPECT_NO_THROW(extendEngramVerifyWindows(nullptr, nullptr, 0, nullptr));
}

TEST_F(NormalBatchStreamProcessorTest, EngramGatherUsesRequestHistoryAndMasksFakeRows) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.model_type  = "deepseek_v41";
    model_config.max_seq_len = 32;
    model_config.vocab_size = model_config.input_vocab_size = 128;
    model_config.num_layers                                 = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;
    for (int mode = 0; mode < 6; ++mode) {
        const bool fake        = mode >= 3;
        const bool decode      = mode % 3 == 1;
        const bool chunk       = mode % 3 == 2;
        auto       query       = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({11, 12, 13, 14, 15, 16});
        query->generate_config = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setIsFakeStream(fake);
        stream->setReuseLength(4);
        stream->setIsContextStream(!decode);
        if (chunk) {
            stream->setReuseLength(1);
            stream->enable_fast_gen_   = true;
            stream->last_chunk_len_    = 4;
            stream->current_chunk_len_ = 5;
            stream->max_chunk_len_     = 6;
        }
        NormalBatchStreamProcessor processor(model_config, pd_sep_config, logging_config, cache_config, true);
        TensorHolder               holder;
        const auto                 input = processor.gatherModelInput(StreamGroups({stream}), holder);
        ASSERT_TRUE(input.ok()) << input.status().ToString();
        const int64_t rows = decode || chunk ? 1 : 2;
        EXPECT_EQ(input->engram_token_windows.sizes().vec(), (std::vector<int64_t>{rows, 4}));
        const std::vector<int32_t> expected = decode ? std::vector<int32_t>{16, 15, 14, 13} :
                                                       (chunk ? std::vector<int32_t>{15, 14, 13, 12} :
                                                                std::vector<int32_t>{15, 14, 13, 12, 16, 15, 14, 13});
        EXPECT_EQ(toVec<int32_t>(input->engram_token_windows), fake ? std::vector<int32_t>(rows * 4, -1) : expected);
    }
}

class V41CacheDispatchTest: public NormalBatchStreamProcessorTest {
protected:
    void SetUp() override {
        NormalBatchStreamProcessorTest::SetUp();
        config_                             = test::makeSimpleHybridMhaCacheConfig(4, 33, 2, DataType::TYPE_FP16, 2);
        config_.use_independent_block_pools = true;
        manager_                            = std::make_shared<KVCacheManager>(config_);
        ASSERT_TRUE(manager_->init());
        model_config_.model_type  = "deepseek_v41";
        model_config_.max_seq_len = 32;
        model_config_.vocab_size = model_config_.input_vocab_size = 128;
        model_config_.num_layers                                  = 4;
        model_config_.attn_config.tokens_per_block                = 2;
        processor_                                                = std::make_unique<NormalBatchStreamProcessor>(
            model_config_, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config_, false);
    }

    GenerateStreamPtr makeStream(int64_t id) {
        ResourceContext context;
        context.cache_manager  = manager_;
        context.reuse_cache    = false;
        auto query             = std::make_shared<GenerateInput>();
        query->request_id      = id;
        query->input_ids       = hostIntBuffer({11, 12, 13, 14, 15, 16});
        query->generate_config = std::make_shared<GenerateConfig>();
        auto stream = std::make_shared<NormalGenerateStream>(query, model_config_, RuntimeConfig{}, context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        EXPECT_TRUE(stream->streamCacheResource().initKVBlock().ok());
        return stream;
    }

    void chunk(const GenerateStreamPtr& stream, int start, int end) {
        stream->setIsContextStream(true);
        stream->enable_fast_gen_   = true;
        stream->last_chunk_len_    = start;
        stream->current_chunk_len_ = end;
        stream->max_chunk_len_     = 6;
    }

    absl::StatusOr<GptModelInputs> gather(std::list<GenerateStreamPtr> streams) {
        return processor_->gatherModelInput(StreamGroups(streams), holder_);
    }

    CacheConfig                                 config_;
    ModelConfig                                 model_config_;
    std::shared_ptr<KVCacheManager>             manager_;
    std::unique_ptr<NormalBatchStreamProcessor> processor_;
    TensorHolder                                holder_;
};

TEST_F(V41CacheDispatchTest, ChunkDecodeAndFakeRowsKeepIdentityAndPhysicalLeases) {
    const auto initially_free = manager_->freeBlocksNum();
    auto       stream         = makeStream(41);
    chunk(stream, 0, 2);
    auto first = gather({stream});
    ASSERT_TRUE(first.ok()) << first.status();
    const auto incarnation = stream->streamCacheResource().cacheIncarnation();
    EXPECT_EQ(toVec<int64_t>(first->v41_cache_spans), (std::vector<int64_t>{41, 1, incarnation, 0, 0, 2}));
    ASSERT_EQ(first->v41_cache_leases.size(), 1u);
    ASSERT_TRUE(first->v41_cache_leases[0]);
    EXPECT_FALSE(gather({stream}).ok());

    chunk(stream, 2, 5);
    auto second = gather({stream});
    ASSERT_TRUE(second.ok()) << second.status();
    EXPECT_EQ(toVec<int64_t>(second->v41_cache_spans), (std::vector<int64_t>{41, 1, incarnation, 1, 2, 3}));
    stream->setIsContextStream(false);
    auto fake = makeStream(99);
    fake->setIsFakeStream(true);
    auto decode = gather({fake, stream});
    ASSERT_TRUE(decode.ok()) << decode.status();
    EXPECT_EQ(toVec<int64_t>(decode->v41_cache_spans),
              (std::vector<int64_t>{41, 1, incarnation, 2, 5, 1, -1, -1, -1, -1, -1, -1}));
    ASSERT_EQ(decode->v41_cache_leases.size(), 2u);
    EXPECT_FALSE(decode->v41_cache_leases[1]);
    GptModelInputs micro;
    sliceV41CacheInputs(*decode, micro, 0, 1);
    ASSERT_EQ(micro.v41_cache_leases.size(), 1u);
    EXPECT_EQ(micro.v41_cache_leases[0], decode->v41_cache_leases[0]);
    fake->setIsFakeStream(false);
    fake->releaseResource();
    stream->releaseResource();
    EXPECT_LT(manager_->freeBlocksNum(), initially_free);
    first  = absl::CancelledError("drop first input");
    second = absl::CancelledError("drop second input");
    EXPECT_LT(manager_->freeBlocksNum(), initially_free);
    decode = absl::CancelledError("drop last consumer");
    EXPECT_LT(manager_->freeBlocksNum(), initially_free);
    micro.v41_cache_leases.clear();
    EXPECT_EQ(manager_->freeBlocksNum(), initially_free);
}

TEST_F(V41CacheDispatchTest, FailedGatherDoesNotConsumeGenerationsOrLeakLeases) {
    const auto initially_free                                   = manager_->freeBlocksNum();
    auto       first                                            = makeStream(41);
    auto       second                                           = makeStream(42);
    processor_->model_input_gatherer_->config_.input_vocab_size = 12;
    EXPECT_FALSE(gather({first, second}).ok());
    processor_->model_input_gatherer_->config_.input_vocab_size = 128;
    auto retry                                                  = gather({first, second});
    ASSERT_TRUE(retry.ok()) << retry.status();
    EXPECT_EQ(toVec<int64_t>(retry->v41_cache_spans), (std::vector<int64_t>{41, 1, 1, 0, 0, 6, 42, 2, 1, 0, 0, 6}));
    first->releaseResource();
    second->releaseResource();
    retry = absl::CancelledError("drop consumers");
    EXPECT_EQ(manager_->freeBlocksNum(), initially_free);
}

TEST_F(V41CacheDispatchTest, IdlePhasesHaveNoRequestsAndPreserveLiveGeneration) {
    auto live = makeStream(41);
    chunk(live, 0, 2);
    auto first = gather({live});
    ASSERT_TRUE(first.ok()) << first.status();
    auto fake = makeStream(99);
    fake->setIsFakeStream(true);
    const auto free_blocks = manager_->freeBlocksNum();
    const auto fake_step = fake->iterCount();
    for (int round = 0; round < 4; ++round) {
        fake->setIsContextStream(round % 2 == 0);
        auto idle = gather({fake});
        ASSERT_TRUE(idle.ok()) << idle.status();
        EXPECT_TRUE(idle->is_fake_stream);
        EXPECT_EQ(idle->is_idle_decode, round % 2 != 0);
        for (const auto& tensor : {idle->combo_tokens, idle->input_lengths, idle->prefix_lengths,
                                  idle->sequence_lengths, idle->lm_output_indexes, idle->request_id}) {
            ASSERT_TRUE(tensor.defined());
            EXPECT_EQ(tensor.numel(), 0);
        }
        EXPECT_EQ(idle->engram_token_windows.sizes().vec(), (std::vector<int64_t>{0, 4}));
        EXPECT_EQ(idle->v41_cache_spans.sizes().vec(), (std::vector<int64_t>{0, 6}));
        EXPECT_TRUE(idle->v41_cache_leases.empty());
        EXPECT_EQ(idle->kv_cache_block_id.sizes().vec(),
                  (std::vector<int64_t>{static_cast<int64_t>(config_.groupNums()), 0, 0}));
        EXPECT_EQ(idle->kv_cache_kernel_block_id.numel(), 0);
        EXPECT_FALSE(idle->kv_cache_blocks_to_zero.defined());
        EXPECT_FALSE(idle->kv_cache_update_mapping.defined());
        EXPECT_EQ(manager_->freeBlocksNum(), free_blocks);
        EXPECT_EQ(fake->iterCount(), fake_step);
    }
    chunk(live, 2, 4);
    auto next = gather({live});
    ASSERT_TRUE(next.ok()) << next.status();
    EXPECT_EQ(next->v41_cache_spans[0][1].item<int64_t>(), first->v41_cache_spans[0][1].item<int64_t>());
    EXPECT_EQ(next->v41_cache_spans[0][3].item<int64_t>(), 1);
    EXPECT_EQ(toVec<int32_t>(next->engram_token_windows), (std::vector<int32_t>{13, 12, 11, -1, 14, 13, 12, 11}));
    fake->setIsFakeStream(false);
    fake->releaseResource();
}

TEST_F(V41CacheDispatchTest, ReusedRequestIdAndResetCacheRequireNewGeneration) {
    auto stream = makeStream(41);
    auto old    = gather({stream});
    ASSERT_TRUE(old.ok()) << old.status();
    auto duplicate = makeStream(41);
    EXPECT_FALSE(gather({duplicate}).ok());
    stream->releaseResource();
    auto fresh = gather({duplicate});
    ASSERT_TRUE(fresh.ok()) << fresh.status();
    EXPECT_EQ(fresh->v41_cache_spans[0][1].item<int64_t>(), 2);
    EXPECT_NE(old->v41_cache_leases[0]->blockTables(), fresh->v41_cache_leases[0]->blockTables());
    duplicate->streamCacheResource().tryReleaseKVBlock(duplicate->curBlocksNum());
    ASSERT_TRUE(duplicate->streamCacheResource().initKVBlock().ok());
    auto reset = gather({duplicate});
    ASSERT_TRUE(reset.ok()) << reset.status();
    EXPECT_EQ(reset->v41_cache_spans[0][1].item<int64_t>(), 3);
    EXPECT_GT(reset->v41_cache_spans[0][2].item<int64_t>(), fresh->v41_cache_spans[0][2].item<int64_t>());
}

TEST_F(V41CacheDispatchTest, RejectsGapsPrefixBootstrapAndDuplicateIdsBeforeGather) {
    auto stream = makeStream(41);
    chunk(stream, 2, 3);
    EXPECT_FALSE(gather({stream}).ok());
    chunk(stream, 0, 2);
    EXPECT_FALSE(gather({stream, stream}).ok());
    auto first = gather({stream});
    ASSERT_TRUE(first.ok()) << first.status();
    chunk(stream, 3, 4);
    EXPECT_FALSE(gather({stream}).ok());
    chunk(stream, 2, 4);
    auto next = gather({stream});
    ASSERT_TRUE(next.ok()) << next.status();
    EXPECT_EQ(next->v41_cache_spans[0][3].item<int64_t>(), 1);
}

TEST_F(V41CacheDispatchTest, AppendOnlyGrowthPreservesGenerationAndEarlierSnapshot) {
    config_.fromGroupedSpecs({test::makeMhaSpec("full0", 2, DataType::TYPE_FP16, 1, 1),
                              test::makeMhaSpec("full1", 2, DataType::TYPE_FP16, 1, 1)},
                             {{0, 1}, {2, 3}},
                             {CacheGroupType::FULL, CacheGroupType::FULL},
                             {"full0", "full1"});
    manager_ = std::make_shared<KVCacheManager>(config_);
    ASSERT_TRUE(manager_->init());
    processor_ = std::make_unique<NormalBatchStreamProcessor>(
        model_config_, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config_, false);
    auto stream = makeStream(41);
    chunk(stream, 0, 2);
    auto first = gather({stream});
    ASSERT_TRUE(first.ok()) << first.status();
    const auto tables      = first->v41_cache_leases[0]->blockTables();
    const auto incarnation = stream->streamCacheResource().cacheIncarnation();
    ASSERT_TRUE(stream->streamCacheResource().incrKVBlock(12).ok());
    EXPECT_EQ(stream->streamCacheResource().cacheIncarnation(), incarnation);
    chunk(stream, 2, 4);
    auto next = gather({stream});
    ASSERT_TRUE(next.ok()) << next.status();
    EXPECT_EQ(first->v41_cache_leases[0]->blockTables(), tables);
    EXPECT_EQ(next->v41_cache_spans[0][1].item<int64_t>(), 1);
    EXPECT_GT(next->v41_cache_leases[0]->blockTables().at("full1").size(), tables.at("full1").size());
}

TEST_F(V41CacheDispatchTest, LegacyLinearRotationIsRejectedWithoutChangingEarlierSnapshot) {
    auto stream = makeStream(41);
    chunk(stream, 0, 2);
    auto first = gather({stream});
    ASSERT_TRUE(first.ok()) << first.status();
    const auto tables = first->v41_cache_leases[0]->blockTables();
    ASSERT_TRUE(stream->streamCacheResource().incrKVBlock(12).ok());
    chunk(stream, 2, 4);
    auto next = gather({stream});
    ASSERT_FALSE(next.ok());
    EXPECT_NE(std::string(next.status().message()).find("append-only"), std::string::npos);
    EXPECT_EQ(first->v41_cache_leases[0]->blockTables(), tables);
}

TEST_F(V41CacheDispatchTest, PersistentStateSurvivesPagedGrowthAndStreamRelease) {
    config_.seq_size_per_block = config_.kernel_seq_size_per_block = 256;
    config_.fromGroupedSpecs(
        {test::makeResolvedOpaqueSpec(true, "swa_kv", DataType::TYPE_UINT8, 69120, 256),
         test::makeResolvedOpaqueSpec(true, "csa_state", DataType::TYPE_FP32, 16384, 256),
         test::makeResolvedOpaqueSpec(false, "csa_kv", DataType::TYPE_UINT8, 36864, 256),
         test::makeResolvedOpaqueSpec(false, "hca_kv", DataType::TYPE_UINT8, 73728, 256),
         test::makeResolvedOpaqueSpec(false, "indexer_kv", DataType::TYPE_UINT8, 17408, 256)},
        {{0, 1, 2, 3}, {1}, {1}, {2}, {1, 2}},
        {CacheGroupType::SWA, CacheGroupType::SWA, CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::FULL},
        {"swa_kv", "csa_state", "csa_kv", "hca_kv", "indexer_kv"});
    auto groups = config_.topology().groups();
    for (auto& group : groups) {
        group.policy.cp_mapping = CpBlockMappingMode::NONE;
        if (group.policy.group_type == CacheGroupType::SWA) {
            group.policy.single_block_per_request = true;
            group.policy.enable_prefix_reuse      = false;
        }
    }
    config_.setTopology(std::move(groups), config_.topology().layers());
    manager_ = std::make_shared<KVCacheManager>(config_);
    ASSERT_TRUE(manager_->init());
    model_config_.max_seq_len                  = 1024;
    model_config_.attn_config.tokens_per_block = 256;
    processor_                                 = std::make_unique<NormalBatchStreamProcessor>(
        model_config_, PDSepConfig{}, ProfilingDebugLoggingConfig{}, config_, false);
    const auto      initially_free = manager_->freeBlocksNum();
    ResourceContext context;
    context.cache_manager  = manager_;
    context.reuse_cache    = false;
    auto query             = std::make_shared<GenerateInput>();
    query->request_id      = 41;
    query->input_ids       = hostIntBuffer(std::vector<int32_t>(513, 11));
    query->generate_config = std::make_shared<GenerateConfig>();
    auto stream = std::make_shared<NormalGenerateStream>(query, model_config_, RuntimeConfig{}, context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    stream->completeTokenIdsPtr()->setSeqLength(128);
    ASSERT_TRUE(stream->streamCacheResource().initKVBlock().ok());
    std::vector<KVCacheAllocationLeasePtr> consumers;
    int                                    start       = 0;
    int64_t                                chunk_id    = 0;
    const auto                             incarnation = stream->streamCacheResource().cacheIncarnation();
    for (int end : {128, 256, 257, 385, 512, 513}) {
        stream->completeTokenIdsPtr()->setSeqLength(end);
        ASSERT_TRUE(stream->streamCacheResource().incrKVBlock(end).ok());
        chunk(stream, start, end);
        stream->max_chunk_len_ = 513;
        auto input             = gather({stream});
        ASSERT_TRUE(input.ok()) << input.status();
        EXPECT_EQ(toVec<int64_t>(input->v41_cache_spans),
                  (std::vector<int64_t>{41, 1, incarnation, chunk_id++, start, end - start}));
        consumers.push_back(input->v41_cache_leases.at(0));
        const auto& tables = consumers.back()->blockTables();
        EXPECT_EQ(tables.at("swa_kv"), consumers.front()->blockTables().at("swa_kv"));
        EXPECT_EQ(tables.at("csa_state"), consumers.front()->blockTables().at("csa_state"));
        EXPECT_EQ(tables.at("swa_kv").size(), 1u);
        EXPECT_EQ(tables.at("csa_state").size(), 1u);
        for (const auto* tag : {"csa_kv", "hca_kv", "indexer_kv"}) {
            EXPECT_EQ(tables.at(tag).size(), static_cast<size_t>((end + 255) / 256));
            EXPECT_EQ(tables.at(tag).front(), consumers.front()->blockTables().at(tag).front());
        }
        start = end;
    }
    stream->releaseResource();
    EXPECT_EQ(manager_->freeBlocksNum(), initially_free - 11);
    auto final_consumer = consumers.back();
    consumers.clear();
    EXPECT_EQ(manager_->freeBlocksNum(), initially_free - 11);
    final_consumer.reset();
    EXPECT_EQ(manager_->freeBlocksNum(), initially_free);
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
    resource.setBatchBlocks(0, 0, {1, 2});
    resource.setBatchBlocks(1, 0, {3, 4});
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
    cache_config.kv_block_stride_bytes = 4096;
    cache_config.kv_scale_stride_bytes = 256;

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
    addr1.setBatchBlocks(0, 0, {1, 2, 3, 4});
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
    addr2.setBatchBlocks(0, 0, {5, 6, 7, 8});
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
    addr3.setBatchBlocks(0, 0, {9, 10});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = hostIntBuffer({1, 2, 3, 4});
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(cache_config.topologyPtr());
    addr4.setBatchBlocks(0, 0, {11, 12, 13, 14});
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
        EXPECT_EQ(model_input.kv_block_stride_bytes, cache_config.kv_block_stride_bytes);
        EXPECT_EQ(model_input.kv_scale_stride_bytes, cache_config.kv_scale_stride_bytes);
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
    stream->sampling_state_.logits_processors.push_back(std::make_shared<TestStatefulLogitsProcessor>(false));
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
    stream->sampling_state_.logits_processors.push_back(std::make_shared<TestStatefulLogitsProcessor>(true));

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

TEST_F(NormalBatchStreamProcessorTest, testSoftmaxProbs) {
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
    query1->generate_config->return_logits        = true;
    query1->generate_config->return_softmax_probs = true;
    query1->generate_config->is_streaming         = true;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

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
    auto output_status = stream1->nextOutput(1000);
    ASSERT_TRUE(output_status.ok());
    const auto& outputs = output_status.value().generate_outputs;
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].logits.has_value());
    EXPECT_EQ(toVec<float>(*outputs[0].logits), (std::vector<float>{1.0f, 2.0f}));

    auto softmax_probs = stream1->getSoftmaxProbs();
    EXPECT_TRUE(softmax_probs.defined());
    EXPECT_EQ(2048, softmax_probs.numel());
    EXPECT_NEAR(0.731058, softmax_probs.data_ptr<float>()[1], 0.0001);
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

TEST_F(NormalBatchStreamProcessorTest, testDynamicBeamRejectsParentOutsidePreviousBatch) {
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
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
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

TEST_F(NormalBatchStreamProcessorTest, testDisabledOutputVocabPreservesTopK) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

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
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 8);
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
    auto& logits_processor_list_ = stream->sampling_state_.logits_processors;
    ASSERT_EQ(logits_processor_list_.size(), 1);
    ASSERT_NE(std::dynamic_pointer_cast<MultiSeqLogitsProcessor>(logits_processor_list_[0]), nullptr);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({2, 5}, torch::kFloat32).to(torch::kCUDA);
    inputs.finished_mask = torch::tensor({false, true}, torch::kBool);
    logits_processor_list_[0]->process(inputs, 0, 2);

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

TEST_F(NormalBatchStreamProcessorTest, testDynamicBeamDispatchReordersAndPlacesTokenAtSeqLength) {
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
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
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

TEST_F(NormalBatchStreamProcessorTest, testLoss) {
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
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);

    std::shared_ptr<GenerateInput> query3         = make_shared<GenerateInput>();
    query3->input_ids                             = hostIntBuffer({0, 1});
    query3->generate_config                       = make_shared<GenerateConfig>();
    query3->generate_config->calculate_loss       = 2;
    query3->generate_config->max_new_tokens       = 1;
    query3->generate_config->return_hidden_states = true;
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(cache_config.topologyPtr());
    addr3.setBatchBlocks(0, 0, {9});
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
    addr4.setBatchBlocks(0, 0, {11, 12});
    stream4->setKVCache(addr4);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().need_all_logits);

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    auto loss_hidden_tensor = torch::tensor({1.0f, 2.0f, 5.0f, 6.0f, 11.0f, 12.0f}).reshape({3, 2}).to(torch::kCUDA);
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
    auto output3 = stream3->nextOutput(1000);
    ASSERT_TRUE(output3.ok());
    ASSERT_EQ(output3.value().generate_outputs.size(), 1);
    ASSERT_TRUE(output3.value().generate_outputs[0].hidden_states.has_value());
    EXPECT_EQ(toVec<float>(*output3.value().generate_outputs[0].hidden_states), (std::vector<float>{5.0f, 6.0f}));
    EXPECT_TRUE(stream4->getLoss().defined());
    auto loss4 = stream4->getLoss();
    EXPECT_EQ(2, loss4.numel());
    EXPECT_NEAR(2.25525, *(torch::mean(loss4).exp().data_ptr<float>()), 0.0001);
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
