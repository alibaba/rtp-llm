// Focused tests for PyWrappedModel::prepareWriteCacheParams (compiled with
// -fno-access-control like the sibling tests, so the private static method is
// callable without constructing the python-backed model).
#include "rtp_llm/cpp/models/PyWrappedModel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

using namespace rtp_llm;

namespace {

// Minimal eligible PD-prefill input: two context requests, no decode rows.
GptModelInputs makePdPrefillInputs() {
    GptModelInputs inputs;
    inputs.pd_separation         = true;
    inputs.input_lengths         = torch::tensor({4, 6}, torch::kInt32);
    inputs.sequence_lengths      = torch::zeros({0}, torch::kInt32);
    inputs.prefix_lengths        = torch::tensor({0, 0}, torch::kInt32);
    inputs.kv_cache_block_id     = torch::tensor({{0}, {1}}, torch::kInt32);
    inputs.request_id            = torch::tensor({int64_t(7), int64_t(8)}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true, true}, torch::kBool);
    inputs.cache_keys            = torch::tensor({{int64_t(100)}, {int64_t(200)}}, torch::kInt64);
    return inputs;
}

std::optional<torch_ext::PyCacheStoreInputs> prepare(const GptModelInputs& inputs) {
    return PyWrappedModel::prepareWriteCacheParams(inputs, inputs.input_lengths);
}

// Asserts the rejection reason, not just that something threw: every check in
// prepareWriteCacheParams names the field it guards, so a test that only expects
// std::exception would still pass if an unrelated check fired first.
#define EXPECT_PREPARE_REJECTS_WITH(call, expected_substring)                                                          \
    do {                                                                                                               \
        try {                                                                                                          \
            (void)(call);                                                                                              \
            FAIL() << "expected prepareWriteCacheParams to reject: " << (expected_substring);                          \
        } catch (const std::exception& e) {                                                                            \
            EXPECT_NE(std::string(e.what()).find(expected_substring), std::string::npos)                               \
                << "actual message: " << e.what();                                                                     \
        }                                                                                                              \
    } while (0)

#define EXPECT_PREPARE_REJECTS(inputs, expected_substring)                                                             \
    EXPECT_PREPARE_REJECTS_WITH(prepare(inputs), expected_substring)

TEST(PrepareWriteCacheParamsTest, WarmupReturnsNullopt) {
    auto inputs   = makePdPrefillInputs();
    inputs.warmup = true;
    EXPECT_FALSE(prepare(inputs).has_value());
}

TEST(PrepareWriteCacheParamsTest, NonPdSeparationReturnsNullopt) {
    auto inputs          = makePdPrefillInputs();
    inputs.pd_separation = false;
    EXPECT_FALSE(prepare(inputs).has_value());
}

TEST(PrepareWriteCacheParamsTest, DecodeOnlyBatchReturnsNullopt) {
    auto inputs             = makePdPrefillInputs();
    inputs.input_lengths    = torch::tensor({4, 6}, torch::kInt32);
    inputs.sequence_lengths = torch::tensor({3, 5}, torch::kInt32);
    EXPECT_FALSE(prepare(inputs).has_value());
}

TEST(PrepareWriteCacheParamsTest, MixedDecodeAndContextBatchMapsContextRowsOnly) {
    // Production batches are usually mixed: input_lengths spans decode + context rows
    // while request_id / cache_keys / prefix_lengths cover only the context tail. This
    // offset is what runtimeWriteCacheStore relies on (decoder_batch_size + batch_id),
    // so lock it with a positive case rather than only the decode-only nullopt path.
    auto inputs             = makePdPrefillInputs();
    inputs.input_lengths    = torch::tensor({3, 5, 4, 6}, torch::kInt32);  // 2 decode + 2 context
    inputs.sequence_lengths = torch::tensor({3, 5}, torch::kInt32);

    const auto result = prepare(inputs);
    ASSERT_TRUE(result.has_value());
    // Context-scoped tensors keep one entry per context row...
    EXPECT_EQ(result->request_id.numel(), 2);
    EXPECT_EQ(result->prefix_lengths_host.numel(), 2);
    EXPECT_EQ(result->cache_keys.size(0), 2);
    // ...while the published lengths still span the whole batch.
    EXPECT_EQ(result->input_lengths_host.numel(), 4);
    EXPECT_TRUE(result->input_lengths_host.is_same(inputs.input_lengths));
}

TEST(PrepareWriteCacheParamsTest, DecodeBatchLargerThanTotalThrows) {
    auto inputs             = makePdPrefillInputs();
    inputs.sequence_lengths = torch::tensor({1, 2, 3}, torch::kInt32);
    EXPECT_PREPARE_REJECTS(inputs, "smaller than decode batch");
}

TEST(PrepareWriteCacheParamsTest, InvalidRequestIdThrows) {
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::Tensor();
        EXPECT_PREPARE_REJECTS(inputs, "request_id must be defined");
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::tensor({{int64_t(7)}, {int64_t(8)}}, torch::kInt64);
        EXPECT_PREPARE_REJECTS(inputs, "request_id must be 1-D");
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::tensor({7, 8}, torch::kInt32);
        EXPECT_PREPARE_REJECTS(inputs, "request_id must use int64");
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::tensor({int64_t(7)}, torch::kInt64);
        EXPECT_PREPARE_REJECTS(inputs, "request_id count=");
    }
}

// CPU-device branch: a CUDA request_id must be rejected before the writer sees it.
// Separate case so a CUDA-less runner reports SKIPPED instead of silently passing
// a test whose only assertion was compiled out by the runtime guard.
TEST(PrepareWriteCacheParamsTest, CudaRequestIdThrows) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "requires CUDA to build a non-CPU request_id";
    }
    auto inputs       = makePdPrefillInputs();
    inputs.request_id = inputs.request_id.to(torch::kCUDA);
    EXPECT_PREPARE_REJECTS(inputs, "request_id must be a CPU tensor");
}

TEST(PrepareWriteCacheParamsTest, WholeBatchMissingCacheKeysThrows) {
    {
        auto inputs       = makePdPrefillInputs();
        inputs.cache_keys = torch::Tensor();
        EXPECT_PREPARE_REJECTS(inputs, "cache_keys missing");
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.cache_keys = torch::empty({2, 0}, torch::kInt64);
        EXPECT_PREPARE_REJECTS(inputs, "cache_keys missing");
    }
}

TEST(PrepareWriteCacheParamsTest, BatchCountMismatchesThrow) {
    {
        auto inputs       = makePdPrefillInputs();
        inputs.cache_keys = torch::tensor({{int64_t(100)}}, torch::kInt64);  // rows != context batch
        EXPECT_PREPARE_REJECTS(inputs, "cache_keys rows=");
    }
    {
        auto inputs           = makePdPrefillInputs();
        inputs.prefix_lengths = torch::tensor({0}, torch::kInt32);  // one entry short
        EXPECT_PREPARE_REJECTS(inputs, "prefix_lengths must have one entry per context request");
    }
    {
        const auto inputs = makePdPrefillInputs();
        EXPECT_PREPARE_REJECTS_WITH(PyWrappedModel::prepareWriteCacheParams(inputs, torch::Tensor()),
                                    "input lengths must be defined");
        EXPECT_PREPARE_REJECTS_WITH(PyWrappedModel::prepareWriteCacheParams(inputs, torch::tensor({4}, torch::kInt32)),
                                    "input length count=");
    }
}

TEST(PrepareWriteCacheParamsTest, CpPreChunkLengthsPassThroughByIdentity) {
    // CP contract: the published lengths must be the pre-chunk originals handed in
    // by the caller, not the rank-local input_lengths already on the model inputs.
    // This locks the helper-level mapping; the caller-side wiring (forward() re-runs
    // prepareWriteCacheParams with cp_params.prefill_actual_input_lengths_cpu) is
    // enforced by the is_same postcondition in PyWrappedModel::forward, which fails
    // deterministically if the second prepare is ever fed the chunked lengths.
    const auto inputs            = makePdPrefillInputs();
    const auto pre_chunk_lengths = torch::tensor({8, 12}, torch::kInt32);

    const auto result = PyWrappedModel::prepareWriteCacheParams(inputs, pre_chunk_lengths);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->input_lengths_host.is_same(pre_chunk_lengths));
    EXPECT_FALSE(result->input_lengths_host.is_same(inputs.input_lengths));
}

TEST(PrepareWriteCacheParamsTest, EligibleInputsPassTensorsThroughByIdentity) {
    const auto inputs = makePdPrefillInputs();
    const auto result = prepare(inputs);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->input_lengths_host.is_same(inputs.input_lengths));
    EXPECT_TRUE(result->prefix_lengths_host.is_same(inputs.prefix_lengths));
    EXPECT_TRUE(result->host_kv_cache_offset.is_same(inputs.kv_cache_block_id));
    EXPECT_TRUE(result->request_id.is_same(inputs.request_id));
    EXPECT_TRUE(result->request_pd_separation.is_same(inputs.request_pd_separation));
    EXPECT_TRUE(result->cache_keys.is_same(inputs.cache_keys));
}

}  // namespace
