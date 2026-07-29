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

TEST(PrepareWriteCacheParamsTest, DecodeBatchLargerThanTotalThrows) {
    auto inputs             = makePdPrefillInputs();
    inputs.sequence_lengths = torch::tensor({1, 2, 3}, torch::kInt32);
    EXPECT_THROW((void)prepare(inputs), std::exception);
}

TEST(PrepareWriteCacheParamsTest, InvalidRequestIdThrows) {
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::Tensor();
        EXPECT_THROW((void)prepare(inputs), std::exception);
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::tensor({{int64_t(7)}, {int64_t(8)}}, torch::kInt64);
        EXPECT_THROW((void)prepare(inputs), std::exception);
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::tensor({7, 8}, torch::kInt32);
        EXPECT_THROW((void)prepare(inputs), std::exception);
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.request_id = torch::tensor({int64_t(7)}, torch::kInt64);
        EXPECT_THROW((void)prepare(inputs), std::exception);
    }
}

TEST(PrepareWriteCacheParamsTest, WholeBatchMissingCacheKeysSkips) {
    {
        auto inputs       = makePdPrefillInputs();
        inputs.cache_keys = torch::Tensor();
        EXPECT_FALSE(prepare(inputs).has_value());
    }
    {
        auto inputs       = makePdPrefillInputs();
        inputs.cache_keys = torch::empty({2, 0}, torch::kInt64);
        EXPECT_FALSE(prepare(inputs).has_value());
    }
}

TEST(PrepareWriteCacheParamsTest, InvalidInputLengthsThrow) {
    const auto inputs = makePdPrefillInputs();
    EXPECT_THROW((void)PyWrappedModel::prepareWriteCacheParams(inputs, torch::Tensor()), std::exception);
    EXPECT_THROW((void)PyWrappedModel::prepareWriteCacheParams(inputs, torch::tensor({4, 6}, torch::kInt64)),
                 std::exception);
    EXPECT_THROW((void)PyWrappedModel::prepareWriteCacheParams(inputs, torch::tensor({4}, torch::kInt32)),
                 std::exception);
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
