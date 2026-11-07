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
// NOTE: no NaN check happens here. The merged results are checked after the all-gather
// (PyWrappedModel::tpReduceScoringPartials) because a rank-local pre-collective abort could
// desynchronize the TP group: merged values are identical on every rank, so the abort is
// symmetric by construction.
inline ScoringPartials computeShardScoringPartials(const torch::Tensor& local_logits,
                                                   const torch::Tensor& row_indexes,
                                                   const torch::Tensor& labels,
                                                   int64_t              k_max,
                                                   int64_t              shard_lo,
                                                   int64_t              vocab,
                                                   int64_t              chunk_rows) {
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
        const int64_t c     = std::min(chunk_rows, rows - c0);
        auto          sel   = local_logits.index_select(0, row_indexes.narrow(0, c0, c)).to(torch::kFloat32);
        const auto    valid = n_valid < vocab_l ? sel.narrow(1, 0, n_valid) : sel;
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

// Width of the merged top-k output for a scoring batch: each shard contributes shard_k
// local candidates, so the merge can emit up to tp_size*shard_k, capped at k_max and —
// when the real (unpadded) vocab is known — at vocab. mergeScoringPartials allocates its
// outputs with this width; any caller pre-allocating merged buffers (the forward's
// [lm prefix ++ scoring rows] outputs) MUST use the same value or copy_ sees a
// non-broadcastable mismatch for k_max > shard_k.
inline int64_t mergedScoringTopkWidth(int64_t shard_k, int64_t k_max, int64_t tp_size, int64_t vocab) {
    if (shard_k <= 0) {
        return 0;
    }
    int64_t k_out = std::min(k_max, tp_size * shard_k);
    if (vocab > 0) {
        k_out = std::min(k_out, vocab);
    }
    return k_out;
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
// global lse is exact up to reduction order) and max over the -inf-padded target column.
// Top-k ordering contract (documented honestly — same "unspecified ties" class as the legacy
// gathered path's torch.topk):
//   * merged topk_values are exact: every shard contributes its local top-min(k_max, n_valid),
//     which provably contains all global top-k_max candidates, and the merge sorts by the
//     order-preserving float bit map.
//   * the merge order is a total deterministic order: value descending under the bit map
//     (+NaN largest, -NaN smallest, -0.0 below +0.0), then global token id ascending among the
//     candidates that survive to the merge.
//   * candidates that tie at a shard's LOCAL top-k cutoff may be dropped by torch.topk before
//     they ever reach the merge (torch's tie choice is unspecified), so a smaller-id tied
//     token is not guaranteed to appear. Callers must not rely on a globally stable
//     smallest-id rule for ties — matching the unspecified tie behavior the legacy
//     [rows, vocab] torch.topk already had.
// Rows are merged in chunk_rows-sized batches so the [c, tp_size*shard_k] value/id/key
// temporaries stay bounded.
inline ScoringPartials mergeScoringPartials(
    const torch::Tensor& gathered, int64_t rows, int64_t shard_k, int64_t k_max, int64_t vocab, int64_t chunk_rows) {
    const int64_t tp_size = gathered.size(0);
    RTP_LLM_CHECK_WITH_INFO(chunk_rows > 0, "scoring merge chunk_rows must be positive, got %ld", chunk_rows);
    ScoringPartials out;
    out.lse    = torch::empty({rows}, gathered.options());
    out.target = torch::empty({rows}, gathered.options());
    // Same clamp the gathered path applies via .narrow(1, 0, output_vocab_size): a top-k
    // wider than the real vocab must not emit padded-column placeholders.
    const int64_t k_out = mergedScoringTopkWidth(shard_k, k_max, tp_size, vocab);
    if (k_out > 0) {
        out.topk_values = torch::empty({rows, k_out}, gathered.options());
        out.topk_ids    = torch::empty({rows, k_out}, gathered.options().dtype(torch::kInt32));
    } else {
        out.topk_values = torch::empty({rows, 0}, gathered.options());
        out.topk_ids    = torch::empty({rows, 0}, gathered.options().dtype(torch::kInt32));
    }
    // Pack (value desc, global id asc) into one int64 key: flip the float bits into the
    // order-preserving mapping, then append the id complement as tie-breaker. Keys are
    // unique across shards whenever ids are unique (duplicate keys only become possible
    // through padded or pruned vocab columns, which scoring requests cannot reach), so
    // the merge order is deterministic regardless of sort stability.
    const int64_t kHalf        = int64_t{1} << 31;
    const int64_t kSpan        = int64_t{1} << 32;
    const auto    gathered_i32 = gathered.view(torch::kInt32);
    for (int64_t c0 = 0; c0 < rows; c0 += chunk_rows) {
        const int64_t c = std::min(chunk_rows, rows - c0);
        out.lse.narrow(0, c0, c).copy_(gathered.narrow(1, c0, c).t().logsumexp(-1));
        out.target.narrow(0, c0, c).copy_(std::get<0>(gathered.narrow(1, rows + c0, c).max(0)));
        if (shard_k == 0 || k_out == 0) {
            continue;
        }
        auto vals = gathered.narrow(1, 2 * rows + c0 * shard_k, c * shard_k)
                        .view({tp_size, c, shard_k})
                        .permute({1, 0, 2})
                        .reshape({c, tp_size * shard_k});
        auto ids = gathered_i32.narrow(1, 2 * rows + rows * shard_k + c0 * shard_k, c * shard_k)
                       .view({tp_size, c, shard_k})
                       .permute({1, 0, 2})
                       .reshape({c, tp_size * shard_k});
        const auto b    = vals.view(torch::kInt32).to(torch::kInt64);
        const auto skey = torch::where(b >= 0, b + kHalf, ~b) - kHalf;
        const auto key  = skey * kSpan + (kSpan - 1 - ids.to(torch::kInt64));
        const auto ord  = std::get<1>(key.sort(1, /*descending=*/true)).narrow(1, 0, k_out);
        out.topk_values.narrow(0, c0, c).copy_(vals.gather(1, ord));
        out.topk_ids.narrow(0, c0, c).copy_(ids.gather(1, ord));
    }
    return out;
}

}  // namespace rtp_llm
