// pp_size=1 equivalence baseline for gatherModelInput's KV block table
// assembly: any future change must keep these tests green under pp_size=1.
//
// What is pinned (block tables are deterministic from setBatchBlocks):
//  - kv_cache_block_id / kv_cache_kernel_block_id contents and shapes
//  - row order: decode streams first, then context streams
//  - row width: max block count across the batch, zero padding
//  - kernel expansion: kernel_id = block_id * bpk + j (bpk=2 case)
//  - context-only full gather output (combo tokens / lengths / block tables)

#include <memory>
#include <numeric>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

using namespace std;

namespace rtp_llm {

template<typename T>
static std::vector<T> tensorToVec(const torch::Tensor& t) {
    auto c = t.is_cuda() ? t.cpu().contiguous() : t.contiguous();
    return std::vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

static torch::Tensor intTensor(std::vector<int32_t> data) {
    return torch::tensor(data, torch::kInt32);
}

static void initFullCacheConfig(CacheConfig& cache_config, int layer_num, uint32_t spec_seq_size = 0) {
    auto spec = std::make_shared<MHAKVCacheSpec>();
    spec->tag = "default";
    if (spec_seq_size > 0) {
        // KVCacheSpecBase defaults seq_size_per_block to 1, and setTopology prefers
        // the spec value over the CacheConfig global — override it explicitly for
        // kernel-expansion (bpk > 1) scenarios.
        spec->seq_size_per_block = spec_seq_size;
    }
    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.layer_num     = static_cast<uint32_t>(layer_num);
    cache_config.layer_all_num = static_cast<uint32_t>(layer_num);
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
}

class PPKvBlockTableEquivalenceTest: public DeviceTestBase {
protected:
    // Builds a stream with a manually assigned block list (single group).
    GenerateStreamPtr makeStream(const ResourceContext&  resource_context,
                                 const ModelConfig&      model_config,
                                 const RuntimeConfig&    runtime_config,
                                 const CacheConfig&      cache_config,
                                 std::vector<int32_t>    input_ids,
                                 const BlockIndicesType& blocks,
                                 bool                    is_context) {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = intTensor(std::move(input_ids));
        query->generate_config = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

        BatchKVCacheResource resource;
        resource.resetBatchSize(1);
        resource.initGroups(cache_config.topologyPtr());
        resource.setBatchBlocks(0, 0, blocks);
        stream->setKVCache(resource);
        stream->setIsContextStream(is_context);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }
};

// Mixed decode+context batch, bpk=1: block table rows are decode-first then
// context, width = max block count, zero padded; kernel view mirrors block ids.
TEST_F(PPKvBlockTableEquivalenceTest, MixedBatchBlockTableBaseline) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig runtime_config;

    auto decode_stream =
        makeStream(resource_context, model_config, runtime_config, cache_config, {1, 2}, {5, 6}, false);
    auto context_stream =
        makeStream(resource_context, model_config, runtime_config, cache_config, {3, 4, 5}, {7, 8, 9}, true);

    std::list<GenerateStreamPtr> streams{decode_stream, context_stream};
    StreamGroups                 stream_groups(streams);
    NormalBatchStreamProcessor   processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    TensorHolder holder;
    auto         status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(status.ok());
    auto& mi = status.value();

    // [group=1, batch=2, max_blocks=3]; decode row first, zero padding.
    ASSERT_TRUE(mi.kv_cache_block_id.defined());
    EXPECT_EQ(mi.kv_cache_block_id.sizes().vec(), (std::vector<int64_t>{1, 2, 3}));
    EXPECT_EQ((std::vector<int32_t>{5, 6, 0, 7, 8, 9}), tensorToVec<int32_t>(mi.kv_cache_block_id));

    // bpk=1: kernel view is a mirror of the block ids.
    ASSERT_TRUE(mi.kv_cache_kernel_block_id.defined());
    EXPECT_EQ(mi.kv_cache_kernel_block_id.sizes().vec(), (std::vector<int64_t>{1, 2, 3}));
    EXPECT_EQ((std::vector<int32_t>{5, 6, 0, 7, 8, 9}), tensorToVec<int32_t>(mi.kv_cache_kernel_block_id));
}

// Kernel expansion with bpk=2 (seq_size_per_block=4, kernel_seq_size_per_block=2):
// kernel_id = block_id * 2 + j, tensor width = max_blocks * 2.
TEST_F(PPKvBlockTableEquivalenceTest, KernelBlockExpansionBpk2) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.seq_size_per_block        = 4;
    cache_config.kernel_seq_size_per_block = 2;
    initFullCacheConfig(cache_config, model_config.num_layers, /*spec_seq_size=*/4);
    ASSERT_EQ(cache_config.kernelBlocksPerKvBlock(), 2u);
    RuntimeConfig runtime_config;

    auto context_stream =
        makeStream(resource_context, model_config, runtime_config, cache_config, {1, 2, 3}, {5, 7}, true);

    std::list<GenerateStreamPtr> streams{context_stream};
    StreamGroups                 stream_groups(streams);
    NormalBatchStreamProcessor   processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    TensorHolder holder;
    auto         status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(status.ok());
    auto& mi = status.value();

    // Block ids unchanged: {5, 7}.
    ASSERT_TRUE(mi.kv_cache_block_id.defined());
    EXPECT_EQ(mi.kv_cache_block_id.sizes().vec(), (std::vector<int64_t>{1, 1, 2}));
    EXPECT_EQ((std::vector<int32_t>{5, 7}), tensorToVec<int32_t>(mi.kv_cache_block_id));

    // Kernel view expanded: 5 -> {10, 11}, 7 -> {14, 15}; width = 2 * 2 = 4.
    ASSERT_TRUE(mi.kv_cache_kernel_block_id.defined());
    EXPECT_EQ(mi.kv_cache_kernel_block_id.sizes().vec(), (std::vector<int64_t>{1, 1, 4}));
    EXPECT_EQ((std::vector<int32_t>{10, 11, 14, 15}), tensorToVec<int32_t>(mi.kv_cache_kernel_block_id));
}

// Context-only batch: pins the full gather output (combo tokens, lengths,
// block tables) as the prefill-side baseline.
TEST_F(PPKvBlockTableEquivalenceTest, ContextOnlyFullBaseline) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig runtime_config;

    auto ctx1 = makeStream(resource_context, model_config, runtime_config, cache_config, {1, 2, 3}, {5, 6}, true);
    auto ctx2 = makeStream(resource_context, model_config, runtime_config, cache_config, {4, 5}, {7}, true);

    std::list<GenerateStreamPtr> streams{ctx1, ctx2};
    StreamGroups                 stream_groups(streams);
    NormalBatchStreamProcessor   processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    TensorHolder holder;
    auto         status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(status.ok());
    auto& mi = status.value();

    EXPECT_EQ((std::vector<int32_t>{1, 2, 3, 4, 5}), tensorToVec<int32_t>(mi.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{3, 2}), tensorToVec<int32_t>(mi.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{0, 0}), tensorToVec<int32_t>(mi.prefix_lengths));
    ASSERT_TRUE(mi.sequence_lengths.defined());
    EXPECT_EQ(mi.sequence_lengths.numel(), 0);  // no decode streams

    ASSERT_TRUE(mi.kv_cache_block_id.defined());
    EXPECT_EQ(mi.kv_cache_block_id.sizes().vec(), (std::vector<int64_t>{1, 2, 2}));
    EXPECT_EQ((std::vector<int32_t>{5, 6, 7, 0}), tensorToVec<int32_t>(mi.kv_cache_block_id));
}

}  // namespace rtp_llm
