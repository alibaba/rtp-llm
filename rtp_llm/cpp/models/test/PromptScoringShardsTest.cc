#include <gtest/gtest.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/models/PromptScoringShards.h"

using namespace rtp_llm;

namespace {

// Simulates the tp_size-way contiguous vocab split plus the compact all-gather on one device:
// per shard it slices its [T, vocab_l] columns out of the padded full logits, computes the
// partials, packs them, and stacks the payloads rank-major exactly like the production
// all-gather's output layout ([tp_size, payload_len]).
std::pair<torch::Tensor, int64_t> gatherPayloads(const torch::Tensor& full_logits,  // [T, padded]
                                                 const torch::Tensor& rows,         // int64 [R]
                                                 const torch::Tensor& labels,       // int32 [R]
                                                 int64_t              k_max,
                                                 int64_t              tp_size,
                                                 int64_t              vocab,
                                                 int64_t              chunk_rows) {
    const int64_t              vocab_l = full_logits.size(1) / tp_size;
    std::vector<torch::Tensor> payloads;
    int64_t                    shard_k = 0;
    for (int64_t r = 0; r < tp_size; ++r) {
        auto local    = full_logits.narrow(1, r * vocab_l, vocab_l);
        auto partials = computeShardScoringPartials(
            local, rows, labels, k_max, r * vocab_l, vocab, chunk_rows, /*check_nan=*/false);
        shard_k = partials.topk_values.size(1);
        payloads.push_back(packScoringPartials(partials));
    }
    return {torch::stack(payloads, 0), shard_k};
}

torch::Device device() {
    return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
}

}  // namespace

// Global top-k ids/values and target logits must equal the full-vocab reference bitwise;
// lse may differ only by reduction-order ULPs. Covers target ids on both shards, a -1
// (unlabeled) row, k_max > per-shard width, and a padded tail column excluded everywhere.
TEST(PromptScoringShardsTest, MergedPartialsMatchFullVocabReference) {
    torch::manual_seed(0);
    const auto    dev     = device();
    const int64_t rows    = 7;
    const int64_t tp      = 2;
    const int64_t vocab_l = 6;
    const int64_t vocab   = 11;  // one padded column (index 11) on shard 1
    const int64_t k_max   = 7;   // > vocab_l and > shard-1 valid width

    auto full = torch::randn({rows, vocab_l * tp}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    // Make every value distinct so the reference topk ordering is unambiguous.
    full += torch::arange(vocab_l * tp, full.options()).unsqueeze(0) * 1e-3f;
    // Poison the padded column: a huge value that must never leak into any output.
    full.select(1, vocab).fill_(1e30f);

    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({3, 9, -1, 0, 10, 5, 7}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, k_max, tp, vocab, /*chunk_rows=*/512);
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, k_max, vocab);

    auto valid    = full.narrow(1, 0, vocab);
    auto expected = valid.topk(std::min(k_max, vocab), -1);

    EXPECT_TRUE(torch::equal(merged.topk_ids.cpu(), std::get<1>(expected).to(torch::kInt32).cpu()));
    EXPECT_TRUE(torch::equal(merged.topk_values.cpu(), std::get<0>(expected).cpu()));

    auto expected_target = torch::full({rows}, -std::numeric_limits<float>::infinity(), full.options());
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t lbl = labels[r].item<int32_t>();
        if (lbl >= 0) {
            expected_target[r] = full[r][lbl];
        }
    }
    EXPECT_TRUE(torch::equal(merged.target.cpu(), expected_target.cpu()));

    auto expected_lse = valid.logsumexp(-1);
    auto lse_diff     = (merged.lse - expected_lse).abs().max().item<float>();
    EXPECT_LT(lse_diff, 1e-5f);
    EXPECT_TRUE(merged.lse.defined());
}

// Row chunking must not change results: R=8 rows reduced at chunk_rows=3 cross two
// chunk boundaries, and duplicate values must merge deterministically with the smaller
// global token id first.
TEST(PromptScoringShardsTest, ChunkBoundariesAndDuplicateTieBreak) {
    const auto    dev   = device();
    const int64_t rows  = 8;
    const int64_t tp    = 2;
    const int64_t vocab = 8;
    const int64_t k_max = 3;

    auto full = torch::zeros({rows, vocab}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    // Row 0: value 5.0 duplicated at global ids 1 (shard0) and 4 (shard1) -> ids must come
    // out (1, 4) in that order before anything smaller.
    full[0][1] = 5.0f;
    full[0][4] = 5.0f;
    full[0][2] = 4.0f;
    // Other rows get distinct descending values for a fixed expectation.
    for (int64_t r = 1; r < rows; ++r) {
        full[r] = torch::arange(vocab, 0, -1, full.options());
    }

    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::full({rows}, -1, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, k_max, tp, vocab, /*chunk_rows=*/3);
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, k_max, vocab);

    auto ids0 = merged.topk_ids[0].cpu();
    EXPECT_EQ(ids0[0].item<int32_t>(), 1);
    EXPECT_EQ(ids0[1].item<int32_t>(), 4);
    EXPECT_EQ(ids0[2].item<int32_t>(), 2);
    EXPECT_FLOAT_EQ(merged.topk_values[0][0].item<float>(), 5.0f);
    EXPECT_FLOAT_EQ(merged.topk_values[0][1].item<float>(), 5.0f);
    // Chunks crossed at rows 3 and 6; a couple of spot rows prove the row mapping held.
    for (int64_t r : {1, 3, 6, 7}) {
        EXPECT_EQ(merged.topk_ids[r][0].item<int32_t>(), 0);
        EXPECT_FLOAT_EQ(merged.topk_values[r][0].item<float>(), static_cast<float>(vocab));
    }
}

// Zero selected rows is a legal synchronized state (e.g. no consumer this step): partials
// stay empty and the merge must not trip on the degenerate payload.
TEST(PromptScoringShardsTest, EmptyRowSet) {
    const auto    dev   = device();
    const int64_t tp    = 2;
    const int64_t vocab = 8;

    auto full    = torch::randn({4, vocab}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto row_idx = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/3, tp, vocab, 512);
    auto merged              = mergeScoringPartials(gathered, /*rows=*/0, shard_k, /*k_max=*/3, vocab);
    EXPECT_EQ(merged.lse.numel(), 0);
    EXPECT_EQ(merged.topk_ids.size(0), 0);
}

// A loss-only batch (k_max = 0) still needs lse and target; the top-k payload degenerates
// to zero width and the merged tensors must come back [rows, 0].
TEST(PromptScoringShardsTest, LossOnlyZeroTopK) {
    const auto    dev   = device();
    const int64_t tp    = 2;
    const int64_t vocab = 8;
    const int64_t rows  = 3;

    auto full    = torch::randn({rows, vocab}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({2, 7, 4}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/0, tp, vocab, 512);
    EXPECT_EQ(shard_k, 0);
    auto merged = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/0, vocab);

    EXPECT_EQ(merged.topk_ids.size(1), 0);
    auto expected_lse = full.logsumexp(-1);
    EXPECT_LT((merged.lse - expected_lse).abs().max().item<float>(), 1e-5f);
    for (int64_t r = 0; r < rows; ++r) {
        EXPECT_FLOAT_EQ(merged.target[r].item<float>(), full[r][labels[r].item<int32_t>()].item<float>());
    }
}

// tp_size == 1 degenerates to the unsharded computation: the single "shard" covers the
// whole vocab and the merge must reproduce the direct topk/logsumexp exactly.
TEST(PromptScoringShardsTest, SingleShardIsIdentity) {
    const auto    dev   = device();
    const int64_t rows  = 4;
    const int64_t vocab = 8;

    auto full    = torch::randn({rows, vocab}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({0, 7, -1, 3}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/3, /*tp=*/1, vocab, 512);
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/3, vocab);

    auto expected = full.topk(3, -1);
    EXPECT_TRUE(torch::equal(merged.topk_ids.cpu(), std::get<1>(expected).to(torch::kInt32).cpu()));
    EXPECT_TRUE(torch::equal(merged.lse.cpu(), full.logsumexp(-1).cpu()));
}
