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
        auto partials = computeShardScoringPartials(local, rows, labels, k_max, r * vocab_l, vocab, chunk_rows);
        shard_k       = partials.topk_values.size(1);
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
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, k_max, vocab, 512);

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
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, k_max, vocab, 512);

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
    auto merged              = mergeScoringPartials(gathered, /*rows=*/0, shard_k, /*k_max=*/3, vocab, 512);
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
    auto merged = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/0, vocab, 512);

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
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/3, vocab, 512);

    auto expected = full.topk(3, -1);
    EXPECT_TRUE(torch::equal(merged.topk_ids.cpu(), std::get<1>(expected).to(torch::kInt32).cpu()));
    EXPECT_TRUE(torch::equal(merged.lse.cpu(), full.logsumexp(-1).cpu()));
}

// Cross-shard value ties must resolve by ascending global token id deterministically:
// equal keys (value, id) collide only for identical entries, so (value desc, id asc)
// makes the merged order reproducible regardless of rank count.
TEST(PromptScoringShardsTest, CrossShardValueTiesResolveByTokenId) {
    const auto    dev   = device();
    const int64_t rows  = 1;
    const int64_t vocab = 12;
    const int64_t tp    = 2;

    auto full = torch::zeros({rows, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    // ids 11, 5, 2 all carry the tied value 7.5 across shard boundary at vocab_l=8.
    full[0][11]  = 7.5f;
    full[0][2]   = 7.5f;
    full[0][5]   = 7.5f;
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({-1}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/3, tp, vocab, 512);
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/3, vocab, 512);

    auto ids = merged.topk_ids.cpu();
    EXPECT_EQ(ids[0][0].item<int32_t>(), 2);
    EXPECT_EQ(ids[0][1].item<int32_t>(), 5);
    EXPECT_EQ(ids[0][2].item<int32_t>(), 11);
}

// A tie straddling a shard's local top-k cutoff is resolved by torch.topk's own
// (unspecified) rule: a candidate that loses the local cutoff can be a member a global
// top-k would have kept, so the merged id SET may differ from a full-vocab topk — this
// matches the legacy per-shard gather contract and is not a new tie guarantee. What MUST
// hold: every emitted candidate carries its true value, and the merge is deterministic
// for a fixed shard layout.
TEST(PromptScoringShardsTest, LocalCutoffTieKeepsTrueValue) {
    const auto    dev   = device();
    const int64_t rows  = 1;
    const int64_t vocab = 8;
    const int64_t tp    = 2;  // vocab_l = 4 per shard: shard0 cols 0-3, shard1 cols 4-7

    auto full  = torch::zeros({rows, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    full[0][0] = 9.0f;
    // FOUR-way tie at value 5.0: three candidates live on shard0 (cols 1,2,3) and one on
    // shard1 (col 5). k_max=3 lets shard0 keep only {9.0} plus two of its three 5.0s —
    // torch.topk picks them by unspecified order, so one tied candidate is genuinely
    // dropped by the LOCAL cutoff. The merge then orders the surviving candidates by
    // (value desc, id asc), so the two surviving shard0 ids — both < 5 — take the
    // remaining top-3 slots and shard1's col-5 candidate is the one left out.
    full[0][1]   = 5.0f;
    full[0][2]   = 5.0f;
    full[0][3]   = 5.0f;
    full[0][5]   = 5.0f;
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({-1}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/3, tp, vocab, 512);
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/3, vocab, 512);

    auto vals = merged.topk_values.cpu();
    auto ids  = merged.topk_ids.cpu();
    // Values are exact regardless of which tied candidate each shard kept.
    EXPECT_FLOAT_EQ(vals[0][0].item<float>(), 9.0f);
    EXPECT_FLOAT_EQ(vals[0][1].item<float>(), 5.0f);
    EXPECT_FLOAT_EQ(vals[0][2].item<float>(), 5.0f);
    // The two 5.0 slots must be the two shard0 survivors — an unspecified pair out of
    // {1,2,3} whichever way the local cutoff fell — emitted in ascending id order by
    // the merge's (value desc, id asc) key. Shard1's col-5 candidate sorts after both
    // and cannot reach the merged top-3.
    EXPECT_EQ(ids[0][0].item<int32_t>(), 0);
    const int32_t id1 = ids[0][1].item<int32_t>();
    const int32_t id2 = ids[0][2].item<int32_t>();
    EXPECT_TRUE(id1 == 1 || id1 == 2 || id1 == 3) << id1;
    EXPECT_TRUE(id2 == 1 || id2 == 2 || id2 == 3) << id2;
    EXPECT_LT(id1, id2);
    // Same shard layout must merge identically on a repeat run (determinism).
    auto merged2 = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/3, vocab, 512);
    EXPECT_TRUE(torch::equal(merged2.topk_ids.cpu(), merged.topk_ids.cpu()));
}

// Merge-time handling of special floats. The merge applies its own deterministic
// ordering to the candidate set, so exact slot order vs torch.topk (NaN-sign order,
// -0.0/+0.0 placement) is NOT part of the contract — same as the legacy local topk's
// unspecified tie rule. What MUST hold: the emitted value multiset equals the row's
// surviving candidates, every emitted id is a true carrier of its value, and a NaN
// in the row data poisons that row's lse. The collective-uniform abort lives in
// tpReduceScoringPartials (not exercised by this merge test).
TEST(PromptScoringShardsTest, SpecialFloatMergeKeepsCandidateMultiset) {
    const auto    dev   = device();
    const int64_t rows  = 1;
    const int64_t vocab = 7;
    const int64_t tp    = 1;

    auto full    = torch::zeros({rows, vocab}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    full[0][0]   = 1.0f;
    full[0][1]   = std::numeric_limits<float>::infinity();
    full[0][2]   = -0.0f;
    full[0][3]   = 0.0f;
    full[0][4]   = -std::numeric_limits<float>::infinity();
    full[0][5]   = std::numeric_limits<float>::quiet_NaN();
    full[0][6]   = -std::numeric_limits<float>::quiet_NaN();
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({-1}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    // Exercise both k == vocab and k < vocab: which NaN/special candidates survive the
    // local cutoff is unspecified, so the reference multiset is recomputed per k.
    const float   neg_inf   = -std::numeric_limits<float>::infinity();
    const int64_t k_cases[] = {4, vocab};
    for (const int64_t k_max : k_cases) {
        auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, k_max, tp, vocab, 512);
        auto merged              = mergeScoringPartials(gathered, rows, shard_k, k_max, vocab, 512);

        const int64_t k_out = std::min(k_max, vocab);
        EXPECT_EQ(merged.topk_values.size(1), k_out);
        EXPECT_EQ(merged.topk_ids.size(1), k_out);
        auto expected = full.topk(k_out, -1);
        auto mv       = merged.topk_values.cpu();
        auto ev       = std::get<0>(expected).cpu();
        // Same value multiset: identical NaN counts, and the non-NaN values are
        // identical after a canonical sort (NaN slots masked to -inf before sorting).
        EXPECT_EQ(mv.isnan().sum().item<int64_t>(), ev.isnan().sum().item<int64_t>());
        auto mv_sorted = std::get<0>(mv.masked_fill(mv.isnan(), neg_inf).sort(-1));
        auto ev_sorted = std::get<0>(ev.masked_fill(ev.isnan(), neg_inf).sort(-1));
        EXPECT_TRUE(torch::equal(mv_sorted, ev_sorted));
        // Every emitted id is a true carrier of its emitted value (NaN-safe compare).
        auto carried = full.cpu().gather(1, merged.topk_ids.cpu().to(torch::kLong));
        EXPECT_TRUE(torch::equal(carried.isnan(), mv.isnan()));
        EXPECT_TRUE(torch::equal(carried.masked_fill(carried.isnan(), 0), mv.masked_fill(mv.isnan(), 0)));
        // The merge is deterministic for a fixed payload.
        auto merged2 = mergeScoringPartials(gathered, rows, shard_k, k_max, vocab, 512);
        EXPECT_TRUE(torch::equal(merged2.topk_ids.cpu(), merged.topk_ids.cpu()));
        // A NaN in the row data poisons that row's lse.
        EXPECT_TRUE(std::isnan(merged.lse.cpu()[0].item<float>()));
    }
}

// k_max larger than the served vocab produces at most `vocab` entries — the legacy
// min(top_k, vocab) behavior; no cap is imposed on accepted k.
TEST(PromptScoringShardsTest, KExceedsVocabYieldsVocabEntries) {
    const auto    dev   = device();
    const int64_t rows  = 2;
    const int64_t vocab = 11;
    const int64_t tp    = 2;

    auto full    = torch::randn({rows, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({0, 10}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/50, tp, vocab, 512);
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/50, vocab, 512);

    EXPECT_EQ(merged.topk_ids.size(1), vocab);
    auto expected = full.narrow(1, 0, vocab).topk(vocab, -1);
    EXPECT_TRUE(torch::equal(merged.topk_ids.cpu(), std::get<1>(expected).to(torch::kInt32).cpu()));
    EXPECT_TRUE(torch::equal(merged.topk_values.cpu(), std::get<0>(expected).cpu()));
}

// tp=3 with a vocab not divisible by the shard count: the last shard carries both real
// columns and padding, and padded columns must stay out of the merged results.
TEST(PromptScoringShardsTest, ThreeShardsWithPaddedTail) {
    const auto    dev    = device();
    const int64_t rows   = 3;
    const int64_t vocab  = 11;
    const int64_t padded = 12;
    const int64_t tp     = 3;

    auto full    = torch::randn({rows, padded}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels  = torch::tensor({10, 0, 5}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/4, tp, vocab, 512);
    auto merged              = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/4, vocab, 512);

    auto expected = full.narrow(1, 0, vocab).topk(4, -1);
    EXPECT_TRUE(torch::equal(merged.topk_ids.cpu(), std::get<1>(expected).to(torch::kInt32).cpu()));
    EXPECT_TRUE(torch::equal(merged.topk_values.cpu(), std::get<0>(expected).cpu()));
    EXPECT_LT((merged.lse - full.narrow(1, 0, vocab).logsumexp(-1)).abs().max().item<float>(), 1e-5f);
    for (int64_t r = 0; r < rows; ++r) {
        EXPECT_FLOAT_EQ(merged.target[r].item<float>(), full[r][labels[r].item<int32_t>()].item<float>());
    }
}

// Merge row-chunking must be bitwise-equivalent to an unchunked merge: it only bounds
// the sort temporaries and never changes the result.
TEST(PromptScoringShardsTest, MergeChunkingIsEquivalent) {
    const auto    dev   = device();
    const int64_t rows  = 9;
    const int64_t vocab = 23;
    const int64_t tp    = 2;

    auto full    = torch::randn({rows, 24}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto row_idx = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels =
        torch::tensor({0, 22, -1, 7, 11, 3, 19, 1, 8}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    auto [gathered, shard_k] = gatherPayloads(full, row_idx, labels, /*k_max=*/5, tp, vocab, 512);
    auto whole    = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/5, vocab, /*merge_chunk_rows=*/512);
    auto by_one   = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/5, vocab, /*merge_chunk_rows=*/1);
    auto by_three = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/5, vocab, /*merge_chunk_rows=*/3);

    EXPECT_TRUE(torch::equal(by_one.lse.cpu(), whole.lse.cpu()));
    EXPECT_TRUE(torch::equal(by_one.topk_ids.cpu(), whole.topk_ids.cpu()));
    EXPECT_TRUE(torch::equal(by_one.topk_values.cpu(), whole.topk_values.cpu()));
    EXPECT_TRUE(torch::equal(by_three.lse.cpu(), whole.lse.cpu()));
    EXPECT_TRUE(torch::equal(by_three.topk_ids.cpu(), whole.topk_ids.cpu()));
}

// bf16 GEMM output is upcast to fp32 inside the partials path, so merged results must
// equal the fp32 reference computed on the upcast input (no fp16/bf16 merging loss
// beyond the GEMM's own precision).
TEST(PromptScoringShardsTest, Bf16InputMatchesUpcastReference) {
    // bf16's 8-bit mantissa can quantize distinct randn draws into exact ties; a fixed
    // seed keeps the comparison deterministic instead of relying on draw luck.
    torch::manual_seed(7);
    const auto    dev   = device();
    const int64_t rows  = 2;
    const int64_t vocab = 16;
    const int64_t tp    = 2;

    auto full_bf16 = torch::randn({rows, vocab}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    auto full      = full_bf16.to(torch::kFloat32);
    auto row_idx   = torch::arange(rows, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto labels    = torch::tensor({3, 15}, torch::TensorOptions().dtype(torch::kInt32).device(dev));

    // gatherPayloads takes whatever dtype full_logits is; feed bf16 to exercise the
    // computeShardScoringPartials upcast on the same shard split.
    const int64_t              vocab_l = vocab / tp;
    std::vector<torch::Tensor> payloads;
    int64_t                    shard_k = 0;
    for (int64_t r = 0; r < tp; ++r) {
        auto local = full_bf16.narrow(1, r * vocab_l, vocab_l);
        auto p     = computeShardScoringPartials(local, row_idx, labels, 4, r * vocab_l, vocab, 512);
        shard_k    = p.topk_values.size(1);
        payloads.push_back(packScoringPartials(p));
    }
    auto gathered = torch::stack(payloads, 0);
    auto merged   = mergeScoringPartials(gathered, rows, shard_k, /*k_max=*/4, vocab, 512);

    auto expected    = full.topk(4, -1);
    auto merged_ids  = merged.topk_ids.cpu();
    auto merged_vals = merged.topk_values.cpu();
    // Values must equal the global fp32 reference bitwise; ids only need to be true
    // carriers of those values — a bf16-quantized tie may pick a different valid id.
    EXPECT_TRUE(torch::equal(merged_vals, std::get<0>(expected).cpu()));
    EXPECT_TRUE(torch::equal(full.cpu().gather(1, merged_ids.to(torch::kLong)), merged_vals));
}
