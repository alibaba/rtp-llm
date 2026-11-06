#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// Per-row scoring partials produced on one TP vocab shard and merged back into global results.
// Together with GptModelInputs::all_logits_row_labels this is the compact payload the
// tp_size>1 row-selected scoring path all-gathers instead of [rows, vocab] logits:
//   lse          [rows]      fp32 logsumexp over this shard's valid columns
//   target       [rows]      fp32 raw logit at the row's label (-inf when the label is absent
//                            or lives on another shard)
//   topk_values  [rows, K]   fp32 local top-k values over valid columns
//   topk_ids     [rows, K]   int32 global token ids of the local top-k
struct ScoringPartials {
    torch::Tensor lse;
    torch::Tensor target;
    torch::Tensor topk_values;
    torch::Tensor topk_ids;
};

// Reduces the [rows, vocab_l] selection out of this rank's local logits to per-row partials.
// row_indexes/labels index the rows (labels hold global next-token ids, -1 = no label);
// shard_lo is this rank's first global vocab id; vocab is the real (unpadded) vocab size so
// padded tail columns are excluded exactly like the gathered path's output_vocab_size narrow.
// Rows are processed in chunk_rows-sized batches to bound the [c, vocab_l] temporaries.
inline ScoringPartials computeShardScoringPartials(const torch::Tensor& local_logits,
                                                   const torch::Tensor& row_indexes,
                                                   const torch::Tensor& labels,
                                                   int64_t              k_max,
                                                   int64_t              shard_lo,
                                                   int64_t              vocab,
                                                   int64_t              chunk_rows,
                                                   bool                 check_nan) {
    const int64_t rows    = row_indexes.numel();
    const int64_t vocab_l = local_logits.size(1);
    const int64_t n_valid = std::clamp(vocab - shard_lo, (int64_t)0, vocab_l);
    k_max                 = std::clamp(k_max, (int64_t)0, vocab);
    const int64_t shard_k = std::min(k_max, vocab_l);
    RTP_LLM_CHECK_WITH_INFO(chunk_rows > 0, "scoring partial chunk_rows must be positive, got %ld", chunk_rows);

    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(local_logits.device());
    const auto i32 = f32.dtype(torch::kInt32);

    ScoringPartials out;
    out.lse         = torch::empty({rows}, f32);
    out.target      = torch::empty({rows}, f32);
    out.topk_values = torch::empty({rows, shard_k}, f32);
    out.topk_ids    = torch::empty({rows, shard_k}, i32);

    for (int64_t c0 = 0; c0 < rows; c0 += chunk_rows) {
        const int64_t c   = std::min(chunk_rows, rows - c0);
        auto          sel = local_logits.index_select(0, row_indexes.narrow(0, c0, c)).to(torch::kFloat32);
        if (check_nan) {
            RTP_LLM_CHECK_WITH_INFO(!sel.isnan().any().item<bool>(), "NAN detected in scoring logits");
        }
        const auto valid = n_valid < vocab_l ? sel.narrow(1, 0, n_valid) : sel;
        out.lse.narrow(0, c0, c).copy_(valid.logsumexp(-1));
        if (shard_k > 0) {
            // Each shard contributes its local top-min(k_max, n_valid): every entry the global
            // top-k_max could take from this shard is among them, so the merge is exact.
            const int64_t cnt = std::min(shard_k, n_valid);
            if (cnt > 0) {
                auto topk = valid.topk(cnt, -1);
                out.topk_values.narrow(0, c0, c).narrow(1, 0, cnt).copy_(std::get<0>(topk));
                out.topk_ids.narrow(0, c0, c).narrow(1, 0, cnt).copy_((std::get<1>(topk) + shard_lo).to(torch::kInt32));
            }
            if (cnt < shard_k) {
                out.topk_values.narrow(0, c0, c)
                    .narrow(1, cnt, shard_k - cnt)
                    .fill_(-std::numeric_limits<float>::infinity());
                out.topk_ids.narrow(0, c0, c).narrow(1, cnt, shard_k - cnt).fill_(0);
            }
        }
        const auto lbl   = labels.narrow(0, c0, c).to(torch::kLong);
        const auto owner = (lbl >= shard_lo) & (lbl < shard_lo + vocab_l);
        const auto col   = (lbl - shard_lo).clamp(0, vocab_l - 1);
        const auto tv    = sel.gather(1, col.unsqueeze(1)).squeeze(1);
        out.target.narrow(0, c0, c).copy_(tv.masked_fill(~owner, -std::numeric_limits<float>::infinity()));
    }
    return out;
}

// Packed fp32 payload [lse | target | topk_values | topk_ids bit-cast to f32]; the width is a
// pure function of (rows, shard_k), so all ranks produce identical all-gather sizes.
inline torch::Tensor packScoringPartials(const ScoringPartials& partials) {
    std::vector<torch::Tensor> parts = {partials.lse, partials.target};
    if (partials.topk_values.defined() && partials.topk_values.size(1) > 0) {
        parts.push_back(partials.topk_values.reshape({-1}));
        parts.push_back(partials.topk_ids.view(torch::kFloat32).reshape({-1}));
    }
    return torch::cat(parts, 0);
}

// Merges one rank-major [tp_size, payload_len] all-gathered payload into the global per-row
// results: logsumexp over the shard lses (valid shards only contribute real columns, so the
// global lse is exact up to reduction order), max over the -inf-padded target column, and a
// deterministic top-k merge ordered by value desc then global token id asc.
inline ScoringPartials
mergeScoringPartials(const torch::Tensor& gathered, int64_t rows, int64_t shard_k, int64_t k_max, int64_t vocab) {
    const int64_t   tp_size = gathered.size(0);
    ScoringPartials out;
    out.lse    = gathered.narrow(1, 0, rows).t().logsumexp(-1);
    out.target = std::get<0>(gathered.narrow(1, rows, rows).max(0));
    if (shard_k > 0) {
        int64_t k_out = std::min(k_max, tp_size * shard_k);
        if (vocab > 0) {
            // Same clamp the gathered path applies via .narrow(1, 0, output_vocab_size): a top-k
            // wider than the real vocab must not emit padded-column placeholders.
            k_out = std::min(k_out, vocab);
        }
        auto vals = gathered.narrow(1, 2 * rows, rows * shard_k)
                        .view({tp_size, rows, shard_k})
                        .permute({1, 0, 2})
                        .reshape({rows, tp_size * shard_k});
        auto ids = gathered.view(torch::kInt32)
                       .narrow(1, 2 * rows + rows * shard_k, rows * shard_k)
                       .view({tp_size, rows, shard_k})
                       .permute({1, 0, 2})
                       .reshape({rows, tp_size * shard_k});
        // Pack (value desc, global id asc) into one int64 key: flip the float bits into the
        // order-preserving mapping, then append the id complement as tie-breaker. Keys are
        // unique across shards, so the merge order is deterministic regardless of sort
        // stability.
        const int64_t kHalf = int64_t{1} << 31;
        const int64_t kSpan = int64_t{1} << 32;
        const auto    b     = vals.view(torch::kInt32).to(torch::kInt64);
        const auto    skey  = torch::where(b >= 0, b + kHalf, ~b) - kHalf;
        const auto    key   = skey * kSpan + (kSpan - 1 - ids.to(torch::kInt64));
        const auto    ord   = std::get<1>(key.sort(1, /*descending=*/true)).narrow(1, 0, k_out);
        out.topk_values     = vals.gather(1, ord);
        out.topk_ids        = ids.gather(1, ord);
    } else {
        out.topk_values = torch::empty({rows, 0}, gathered.options());
        out.topk_ids    = torch::empty({rows, 0}, gathered.options().dtype(torch::kInt32));
    }
    return out;
}

}  // namespace rtp_llm
