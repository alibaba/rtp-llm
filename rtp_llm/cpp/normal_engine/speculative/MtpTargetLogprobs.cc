#include "rtp_llm/cpp/normal_engine/speculative/MtpTargetLogprobs.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <algorithm>
#include <limits>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#endif

namespace rtp_llm {
namespace {

void recordSpecTensorUseOnCurrentStream(const torch::Tensor& tensor) {
#if USING_CUDA
    if (tensor.defined() && tensor.is_cuda()) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(),
                                                      at::cuda::getCurrentCUDAStream(tensor.device().index()));
    }
#else
    (void)tensor;
#endif
}
void validateMtpTargetLogprobRows(const MtpTargetLogprobs&    target_logprobs,
                                  const std::vector<int64_t>& source_row_indices) {
    RTP_LLM_CHECK(target_logprobs.raw_logits.defined());
    const int64_t requested_row_count = static_cast<int64_t>(source_row_indices.size());
    RTP_LLM_CHECK_WITH_INFO(requested_row_count <= target_logprobs.raw_logits.size(0),
                            "requested MTP logprob rows %ld exceed captured logits rows %ld",
                            requested_row_count,
                            target_logprobs.raw_logits.size(0));
    for (const int64_t row : source_row_indices) {
        RTP_LLM_CHECK_WITH_INFO(row >= 0 && row < target_logprobs.raw_logits.size(0),
                                "requested MTP logprob row %ld is outside [0, %ld)",
                                row,
                                target_logprobs.raw_logits.size(0));
    }
}

bool isIdentityMtpTargetLogprobRows(const std::vector<int64_t>& source_row_indices, int64_t row_count) {
    if (static_cast<int64_t>(source_row_indices.size()) != row_count) {
        return false;
    }
    for (int64_t row = 0; row < row_count; ++row) {
        if (source_row_indices[row] != row) {
            return false;
        }
    }
    return true;
}

torch::Tensor materializeMtpTargetLogprobRowIndices(MtpTargetLogprobs&          target_logprobs,
                                                    const std::vector<int64_t>& source_row_indices) {
    validateMtpTargetLogprobRows(target_logprobs, source_row_indices);
    const int64_t requested_row_count = static_cast<int64_t>(source_row_indices.size());
    if (target_logprobs.raw_logits.is_cuda()) {
        target_logprobs.source_row_indices_cpu_owner = torch::empty(
            {requested_row_count}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
        std::copy(source_row_indices.begin(),
                  source_row_indices.end(),
                  target_logprobs.source_row_indices_cpu_owner.data_ptr<int64_t>());
        target_logprobs.source_row_indices = target_logprobs.source_row_indices_cpu_owner.to(
            target_logprobs.raw_logits.device(), torch::kInt64, /*non_blocking=*/true, /*copy=*/true);
    } else {
        target_logprobs.source_row_indices =
            torch::tensor(source_row_indices, torch::TensorOptions().dtype(torch::kInt64));
    }
    return target_logprobs.source_row_indices;
}

void selectMtpTargetLogprobRows(MtpTargetLogprobs& target_logprobs, const std::vector<int64_t>& source_row_indices) {
    if (source_row_indices.empty()) {
        return;
    }
    validateMtpTargetLogprobRows(target_logprobs, source_row_indices);

    if (isIdentityMtpTargetLogprobRows(source_row_indices, target_logprobs.raw_logits.size(0))) {
        return;
    }

    materializeMtpTargetLogprobRowIndices(target_logprobs, source_row_indices);
    target_logprobs.raw_logits =
        target_logprobs.raw_logits.index_select(/*dim=*/0, target_logprobs.source_row_indices).detach();
    target_logprobs.retains_full_lm_head_storage = false;
}

std::vector<int64_t> mapDenseMtpLogprobRowsToCapturedRows(const MtpTargetLogprobs&    target_logprobs,
                                                          const std::vector<int64_t>& selected_dense_row_indices) {
    RTP_LLM_CHECK_WITH_INFO(target_logprobs.dense_row_count > 0, "MTP target logprobs have no dense capture row count");

    if (target_logprobs.captured_dense_row_indices.empty()) {
        std::vector<int64_t> compact_rows;
        compact_rows.reserve(selected_dense_row_indices.size());
        for (const int64_t dense_row : selected_dense_row_indices) {
            RTP_LLM_CHECK_WITH_INFO(dense_row >= 0 && dense_row < target_logprobs.dense_row_count,
                                    "selected dense MTP logprob row %ld is outside [0, %ld)",
                                    dense_row,
                                    target_logprobs.dense_row_count);
            compact_rows.push_back(dense_row);
        }
        return compact_rows;
    }

    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(target_logprobs.captured_dense_row_indices.size())
                                == target_logprobs.raw_logits.size(0),
                            "captured MTP dense-row map has %ld entries for %ld compact logits rows",
                            static_cast<int64_t>(target_logprobs.captured_dense_row_indices.size()),
                            target_logprobs.raw_logits.size(0));
    std::vector<int64_t> dense_to_compact(target_logprobs.dense_row_count, -1);
    for (int64_t compact_row = 0; compact_row < target_logprobs.raw_logits.size(0); ++compact_row) {
        const int64_t dense_row = target_logprobs.captured_dense_row_indices[compact_row];
        RTP_LLM_CHECK_WITH_INFO(dense_row >= 0 && dense_row < target_logprobs.dense_row_count,
                                "captured dense MTP logprob row %ld is outside [0, %ld)",
                                dense_row,
                                target_logprobs.dense_row_count);
        RTP_LLM_CHECK_WITH_INFO(
            dense_to_compact[dense_row] < 0, "captured dense MTP logprob row %ld appears more than once", dense_row);
        dense_to_compact[dense_row] = compact_row;
    }

    std::vector<int64_t> compact_rows;
    compact_rows.reserve(selected_dense_row_indices.size());
    for (const int64_t dense_row : selected_dense_row_indices) {
        RTP_LLM_CHECK_WITH_INFO(dense_row >= 0 && dense_row < target_logprobs.dense_row_count,
                                "selected dense MTP logprob row %ld is outside [0, %ld)",
                                dense_row,
                                target_logprobs.dense_row_count);
        RTP_LLM_CHECK_WITH_INFO(
            dense_to_compact[dense_row] >= 0, "selected dense MTP logprob row %ld was not captured", dense_row);
        compact_rows.push_back(dense_to_compact[dense_row]);
    }
    return compact_rows;
}

torch::Tensor alignMtpEmittedTokenIdsToCapturedRows(MtpTargetLogprobs&   target_logprobs,
                                                    const torch::Tensor& emitted_token_ids) {
    auto flattened_ids = emitted_token_ids.reshape({-1}).contiguous();
    RTP_LLM_CHECK_WITH_INFO(flattened_ids.numel() == target_logprobs.dense_row_count,
                            "MTP emitted token rows %ld do not match dense capture rows %ld",
                            flattened_ids.numel(),
                            target_logprobs.dense_row_count);
    RTP_LLM_CHECK_WITH_INFO(flattened_ids.device() == target_logprobs.raw_logits.device(),
                            "MTP target logits and emitted token IDs must share a device");
    if (target_logprobs.captured_dense_row_indices.empty()) {
        return flattened_ids;
    }

    RTP_LLM_CHECK_WITH_INFO(target_logprobs.source_row_indices.defined(),
                            "compact MTP capture is missing its dense-row index tensor");
    RTP_LLM_CHECK_WITH_INFO(target_logprobs.source_row_indices.numel() == target_logprobs.raw_logits.size(0),
                            "compact MTP capture index count %ld does not match logits rows %ld",
                            target_logprobs.source_row_indices.numel(),
                            target_logprobs.raw_logits.size(0));
    recordSpecTensorUseOnCurrentStream(flattened_ids);
    recordSpecTensorUseOnCurrentStream(target_logprobs.source_row_indices);
    return flattened_ids.index_select(/*dim=*/0, target_logprobs.source_row_indices);
}

void computeMtpTargetLogprobRowStatistics(MtpTargetLogprobs& target_logprobs) {
    RTP_LLM_CHECK(target_logprobs.raw_logits.defined());
    RTP_LLM_CHECK(target_logprobs.raw_logits.dim() == 2);
    RTP_LLM_CHECK_WITH_INFO(target_logprobs.raw_logits.size(0) > 0,
                            "MTP target logprobs require at least one emitted row");
    RTP_LLM_CHECK(!target_logprobs.row_max.defined());
    RTP_LLM_CHECK(!target_logprobs.row_shifted_logsumexp.defined());
    RTP_LLM_CHECK(!target_logprobs.top_logits.defined());
    RTP_LLM_CHECK(!target_logprobs.top_logprob_token_ids.defined());

    target_logprobs.row_max =
        torch::empty({target_logprobs.raw_logits.size(0)},
                     target_logprobs.raw_logits.options().dtype(torch::kFloat32).requires_grad(false));
    target_logprobs.row_shifted_logsumexp = torch::empty_like(target_logprobs.row_max);
#if USING_CUDA
    if (target_logprobs.raw_logits.is_cuda()) {
        invokeMtpRowLogSoftmaxStats(target_logprobs.raw_logits,
                                    target_logprobs.row_max,
                                    target_logprobs.row_shifted_logsumexp,
                                    target_logprobs.raw_logits.size(1),
                                    cuda_graph::graphGetCurrentStream().stream());
    } else
#endif
    {
        auto fp32_logits        = target_logprobs.raw_logits.to(torch::kFloat32);
        target_logprobs.row_max = std::get<0>(fp32_logits.max(/*dim=*/-1));
        target_logprobs.row_shifted_logsumexp =
            torch::logsumexp(fp32_logits - target_logprobs.row_max.unsqueeze(1), /*dim=*/-1);
    }

    const int64_t top_k = std::min<int64_t>(target_logprobs.requested_top_logprobs, target_logprobs.raw_logits.size(1));
    if (top_k > 0) {
        auto topk_result                      = target_logprobs.raw_logits.topk(top_k, -1, true, true);
        target_logprobs.top_logits            = std::get<0>(topk_result);
        target_logprobs.top_logprob_token_ids = std::get<1>(topk_result).to(torch::kInt32);
    } else {
        target_logprobs.top_logits =
            torch::empty({target_logprobs.raw_logits.size(0), 0}, target_logprobs.raw_logits.options());
        target_logprobs.top_logprob_token_ids = torch::empty({target_logprobs.raw_logits.size(0), 0},
                                                             target_logprobs.raw_logits.options().dtype(torch::kInt32));
    }
}

}  // namespace

MtpTargetLogprobs captureMtpTargetLogprobs(const torch::Tensor&        logits,
                                           int64_t                     max_top_logprobs,
                                           int64_t                     real_vocab_size,
                                           const std::vector<int64_t>& captured_dense_row_indices) {
    RTP_LLM_CHECK(logits.defined());
    RTP_LLM_CHECK(logits.dim() == 2);
    RTP_LLM_CHECK_WITH_INFO(real_vocab_size > 0 && real_vocab_size <= logits.size(1),
                            "real vocab size %ld must be in [1, logits width %ld]",
                            real_vocab_size,
                            logits.size(1));

    MtpTargetLogprobs result;
    result.raw_logits                   = logits.narrow(/*dim=*/1, /*start=*/0, real_vocab_size);
    result.dense_row_count              = logits.size(0);
    result.retains_full_lm_head_storage = true;
    result.requested_top_logprobs       = std::min<int64_t>(std::max<int64_t>(max_top_logprobs, 0), real_vocab_size);

    // An explicit full identity has the same zero-copy semantics as the empty
    // sentinel returned when every stream asks for logprobs.
    if (!captured_dense_row_indices.empty()
        && !isIdentityMtpTargetLogprobRows(captured_dense_row_indices, result.dense_row_count)) {
        result.captured_dense_row_indices = captured_dense_row_indices;
        selectMtpTargetLogprobRows(result, captured_dense_row_indices);
    }
    return result;
}

MtpTargetLogprobs
captureMtpDecodeTargetLogprobs(const torch::Tensor& logits, int64_t max_top_logprobs, int64_t real_vocab_size) {
    // Acceptance is the first point where decode knows both which streams
    // requested logprobs and which of their P+1 rows were emitted. Deferring
    // selection avoids a pre-acceptance [R,V] index_select for dense mixed
    // batches while preserving the original LM-head storage as an O(1) view.
    return captureMtpTargetLogprobs(logits, max_top_logprobs, real_vocab_size);
}

MtpTargetLogprobs computeMtpTargetLogprobs(const torch::Tensor&        logits,
                                           int64_t                     max_top_logprobs,
                                           int64_t                     real_vocab_size,
                                           const std::vector<int64_t>& source_row_indices) {
    auto result = captureMtpTargetLogprobs(logits, max_top_logprobs, real_vocab_size, source_row_indices);
    computeMtpTargetLogprobRowStatistics(result);
    return result;
}

void finalizeMtpTargetLogprobs(MtpTargetLogprobs&   target_logprobs,
                               const torch::Tensor& emitted_token_ids,
                               const torch::Tensor& raw_logits_override) {
    RTP_LLM_CHECK_WITH_INFO(target_logprobs.defined(), "cannot finalize undefined MTP target logprobs");
    RTP_LLM_CHECK_WITH_INFO(!target_logprobs.finalized(), "MTP target logprobs were finalized more than once");
    RTP_LLM_CHECK(target_logprobs.raw_logits.defined());
    RTP_LLM_CHECK(target_logprobs.row_max.defined());
    RTP_LLM_CHECK(target_logprobs.row_shifted_logsumexp.defined());
    RTP_LLM_CHECK(target_logprobs.top_logits.defined());
    RTP_LLM_CHECK(target_logprobs.top_logprob_token_ids.defined());
    RTP_LLM_CHECK(emitted_token_ids.defined());

    auto flattened_ids = emitted_token_ids.reshape({-1});
    if (target_logprobs.source_row_indices.defined()) {
        RTP_LLM_CHECK(target_logprobs.source_row_indices.device() == target_logprobs.raw_logits.device());
        flattened_ids = flattened_ids.to(target_logprobs.raw_logits.device(), torch::kLong, false, false)
                            .index_select(/*dim=*/0, target_logprobs.source_row_indices);
    } else {
        RTP_LLM_CHECK_WITH_INFO(flattened_ids.numel() == target_logprobs.raw_logits.size(0),
                                "MTP emitted token rows %ld do not match target rows %ld",
                                flattened_ids.numel(),
                                target_logprobs.raw_logits.size(0));
        flattened_ids = flattened_ids.to(target_logprobs.raw_logits.device(), torch::kLong, false, false);
    }
    RTP_LLM_CHECK(flattened_ids.numel() == target_logprobs.raw_logits.size(0));

    const auto& raw_logits = raw_logits_override.defined() ? raw_logits_override : target_logprobs.raw_logits;
    if (raw_logits_override.defined()) {
        RTP_LLM_CHECK_WITH_INFO(raw_logits_override.dim() == 2,
                                "MTP raw logits override must be 2-D, got dim=%ld",
                                raw_logits_override.dim());
        RTP_LLM_CHECK_WITH_INFO(raw_logits_override.sizes() == target_logprobs.raw_logits.sizes(),
                                "MTP raw logits override shape must match captured rows");
        RTP_LLM_CHECK_WITH_INFO(raw_logits_override.device() == target_logprobs.raw_logits.device(),
                                "MTP raw logits override device must match captured rows");
        RTP_LLM_CHECK_WITH_INFO(raw_logits_override.scalar_type() == target_logprobs.raw_logits.scalar_type(),
                                "MTP raw logits override dtype must match captured rows");
    }

    // A failed sampler row or an unused speculative tail slot may contain -1
    // or an otherwise uninitialized token ID. Gather must still remain device
    // safe; downstream success/accept-length handling retains the original IDs
    // and decides whether the row is emitted or reported as an error.
    auto safe_flattened_ids = flattened_ids.clamp(/*min=*/0, /*max=*/raw_logits.size(1) - 1);
    auto selected_logits    = raw_logits.gather(/*dim=*/1, safe_flattened_ids.unsqueeze(1)).squeeze(1);
    target_logprobs.token_logprobs =
        (selected_logits.to(torch::kFloat32) - target_logprobs.row_max) - target_logprobs.row_shifted_logsumexp;
    target_logprobs.top_logprobs =
        (target_logprobs.top_logits.to(torch::kFloat32) - target_logprobs.row_max.unsqueeze(1))
        - target_logprobs.row_shifted_logsumexp.unsqueeze(1);

    // Do not let async bookkeeping extend the lifetime of the full LM-head
    // output. From this point onward the payload is O(requested_rows * K).
    target_logprobs.raw_logits                   = torch::Tensor();
    target_logprobs.row_max                      = torch::Tensor();
    target_logprobs.row_shifted_logsumexp        = torch::Tensor();
    target_logprobs.top_logits                   = torch::Tensor();
    target_logprobs.source_row_indices           = torch::Tensor();
    target_logprobs.retains_full_lm_head_storage = false;
    RTP_LLM_CHECK(target_logprobs.finalized());
}

void finalizeSelectedMtpTargetLogprobs(MtpTargetLogprobs&          target_logprobs,
                                       const torch::Tensor&        emitted_token_ids,
                                       const std::vector<int64_t>& selected_dense_row_indices) {
    RTP_LLM_CHECK_WITH_INFO(target_logprobs.defined(), "cannot finalize undefined MTP target logprobs");
    RTP_LLM_CHECK_WITH_INFO(!target_logprobs.finalized(), "MTP target logprobs were finalized more than once");
    RTP_LLM_CHECK(target_logprobs.raw_logits.defined());
    RTP_LLM_CHECK(!target_logprobs.row_max.defined());
    RTP_LLM_CHECK(!target_logprobs.row_shifted_logsumexp.defined());
    RTP_LLM_CHECK(!target_logprobs.top_logits.defined());
    RTP_LLM_CHECK(!target_logprobs.top_logprob_token_ids.defined());
    RTP_LLM_CHECK(emitted_token_ids.defined());

    // A decode packet may contain only reasoning tokens.  Treat that as a
    // successfully finalized empty payload: no logsumexp/top-k kernel is
    // launched and, crucially, the zero-copy raw LM-head owner is released as
    // soon as rejection sampling has finished with it.
    if (selected_dense_row_indices.empty()) {
        const int64_t top_k            = target_logprobs.requested_top_logprobs;
        auto          fp32_options     = target_logprobs.raw_logits.options().dtype(torch::kFloat32);
        target_logprobs.token_logprobs = torch::empty({0}, fp32_options);
        target_logprobs.top_logprob_token_ids =
            torch::empty({0, top_k}, target_logprobs.raw_logits.options().dtype(torch::kInt32));
        target_logprobs.top_logprobs                 = torch::empty({0, top_k}, fp32_options);
        target_logprobs.raw_logits                   = torch::Tensor();
        target_logprobs.source_row_indices           = torch::Tensor();
        target_logprobs.retains_full_lm_head_storage = false;
        RTP_LLM_CHECK(target_logprobs.finalized());
        return;
    }

    // raw_logits may already contain only the requesting streams. Align the
    // dense emitted-token matrix to that compact capture first, then translate
    // final accepted dense rows to compact raw rows. The fused CUDA kernel can
    // continue indexing both tensors with one compact-row vector.
    auto compact_emitted_token_ids  = alignMtpEmittedTokenIdsToCapturedRows(target_logprobs, emitted_token_ids);
    auto compact_source_row_indices = mapDenseMtpLogprobRowsToCapturedRows(target_logprobs, selected_dense_row_indices);

#if USING_CUDA
    if (target_logprobs.raw_logits.is_cuda()) {
        auto row_indices = materializeMtpTargetLogprobRowIndices(target_logprobs, compact_source_row_indices);
        RTP_LLM_CHECK_WITH_INFO(compact_emitted_token_ids.is_cuda(),
                                "CUDA MTP target logits require CUDA emitted token IDs");

        const int64_t selected_rows    = row_indices.numel();
        const int64_t top_k            = target_logprobs.requested_top_logprobs;
        auto          fp32_options     = target_logprobs.raw_logits.options().dtype(torch::kFloat32);
        target_logprobs.token_logprobs = torch::empty({selected_rows}, fp32_options);
        target_logprobs.top_logprob_token_ids =
            torch::empty({selected_rows, top_k}, target_logprobs.raw_logits.options().dtype(torch::kInt32));
        target_logprobs.top_logprobs = torch::empty({selected_rows, top_k}, fp32_options);

        // These tensors were produced on the main compute stream and are read
        // by the bookkeeping stream. Record that use before dropping the raw
        // owner so the caching allocator cannot recycle either input early.
        recordSpecTensorUseOnCurrentStream(target_logprobs.raw_logits);
        recordSpecTensorUseOnCurrentStream(compact_emitted_token_ids);
        invokeMtpSelectedRowLogProbs(target_logprobs.raw_logits,
                                     row_indices,
                                     compact_emitted_token_ids,
                                     target_logprobs.token_logprobs,
                                     target_logprobs.top_logprob_token_ids,
                                     target_logprobs.top_logprobs,
                                     top_k,
                                     cuda_graph::graphGetCurrentStream().stream());

        // The fused kernel reads selected source rows directly. No [R,V]
        // index_select or full-vocabulary probability tensor is materialized.
        target_logprobs.raw_logits                   = torch::Tensor();
        target_logprobs.source_row_indices           = torch::Tensor();
        target_logprobs.retains_full_lm_head_storage = false;
        RTP_LLM_CHECK(target_logprobs.finalized());
        return;
    }
#endif

    // CPU-only tests/non-CUDA builds use the simple reference implementation;
    // production CUDA decode takes the indexed fused path above.
    const bool selected_every_captured_row =
        isIdentityMtpTargetLogprobRows(compact_source_row_indices, target_logprobs.raw_logits.size(0));
    selectMtpTargetLogprobRows(target_logprobs, compact_source_row_indices);
    computeMtpTargetLogprobRowStatistics(target_logprobs);
    if (selected_every_captured_row) {
        // selectMtpTargetLogprobRows intentionally preserves the capture-time
        // mapping on an identity selection. The token IDs are already aligned
        // to compact raw rows, so finalization must consume them directly.
        target_logprobs.source_row_indices = torch::Tensor();
    }
    finalizeMtpTargetLogprobs(target_logprobs, compact_emitted_token_ids);
}

torch::Tensor reshapeMtpTargetAllProbs(const torch::Tensor& all_probs,
                                       int64_t              batch_size,
                                       int64_t              positions_per_batch,
                                       int64_t              logits_width) {
    RTP_LLM_CHECK(all_probs.defined());
    RTP_LLM_CHECK(batch_size > 0);
    RTP_LLM_CHECK(positions_per_batch > 0);
    RTP_LLM_CHECK(logits_width > 0);
    RTP_LLM_CHECK_WITH_INFO(all_probs.numel() == batch_size * positions_per_batch * logits_width,
                            "MTP target all_probs layout mismatch: numel=%ld, batch=%ld, positions=%ld, width=%ld",
                            all_probs.numel(),
                            batch_size,
                            positions_per_batch,
                            logits_width);
    return all_probs.reshape({batch_size, positions_per_batch, logits_width});
}

}  // namespace rtp_llm
