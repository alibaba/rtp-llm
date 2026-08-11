#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>
#include <array>
#include <cstring>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

void fillAllAllow(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
        std::fill_n(tensor.data_ptr<int32_t>(), tensor.numel(), SpecLogitsProcessor::kBitmaskAllowAll);
    }
}

void bitwiseAndInplace(int32_t* dst, const int32_t* src, size_t words) {
    for (size_t i = 0; i < words; ++i) {
        dst[i] &= src[i];
    }
}

using MaskedByteLut = std::array<std::array<bool, 8>, 256>;

const MaskedByteLut& maskedByteLut() {
    static const MaskedByteLut lut = []() {
        MaskedByteLut result{};
        for (size_t value = 0; value < result.size(); ++value) {
            for (size_t bit = 0; bit < result[value].size(); ++bit) {
                result[value][bit] = (value & (1u << bit)) == 0u;
            }
        }
        return result;
    }();
    return lut;
}

}  // namespace

SpecLogitsVerifyRunner::SpecLogitsVerifyRunner(): copy_stream_(cuda_graph::graphGetStreamFromPool(true)) {}

void SpecLogitsVerifyRunner::ensureBuffersFit(
    size_t total_streams, size_t active_streams, int propose_step, size_t vocab_size, size_t bitmask_words) {
    const int64_t B    = static_cast<int64_t>(total_streams);
    const int64_t A    = static_cast<int64_t>(active_streams);
    const int64_t P    = static_cast<int64_t>(propose_step);
    const int64_t rows = A * (P + 1);
    const int64_t W    = static_cast<int64_t>(bitmask_words);
    (void)vocab_size;

    auto cpu_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32 = cpu_i32.pinned_memory(true);

    if (!draft_tokens_cpu_.defined() || draft_tokens_cpu_.numel() < B * P) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
    if (!processor_bitmask_cpu_.defined() || processor_bitmask_cpu_.numel() < (P + 1) * W) {
        processor_bitmask_cpu_ = torch::empty({P + 1, W}, cpu_i32);
    }
    if (!merged_bitmask_cpu_.defined() || merged_bitmask_cpu_.numel() < rows * W) {
        merged_bitmask_cpu_ = torch::empty({rows, W}, cpu_i32);
    }
    if (!spec_cap_cpu_.defined() || spec_cap_cpu_.numel() < B) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
    }
}

void SpecLogitsVerifyRunner::materializeDraftTokensToCpu(const LaunchTask& task) {
    const int64_t B = static_cast<int64_t>(task.total_streams);
    const int64_t P = static_cast<int64_t>(task.propose_step);
    if (B == 0 || P == 0) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(task.draft_tokens.defined(), "spec logits runner requires draft tokens");
    RTP_LLM_CHECK_WITH_INFO(task.draft_tokens.numel() >= B * P && task.draft_tokens.numel() % B == 0,
                            "spec logits runner draft token shape mismatch");
    const int64_t draft_cols = task.draft_tokens.numel() / B;
    RTP_LLM_CHECK_WITH_INFO(draft_cols == P || draft_cols == P + 1,
                            "spec logits runner requires P proposal columns with an optional leading anchor");
    const int64_t draft_offset = draft_cols == P + 1 ? 1 : 0;
    auto draft = task.draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
    auto dst   = draft_tokens_cpu_.flatten().narrow(0, 0, B * P).view({B, P});
    if (!draft.is_cuda()) {
        auto draft_i32 =
            draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
        dst.copy_(draft_i32);
        return;
    }

    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(copy_stream_));
    if (task.draft_tokens_ready_event) {
        task.draft_tokens_ready_event->block(copy_stream_);
    }
    auto draft_i32 = draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
    dst.copy_(draft_i32, /*non_blocking=*/true);
    copy_stream_.synchronize();
}

void SpecLogitsVerifyRunner::unpackMergedBitmaskToVocabMask(const torch::Tensor& mask_cpu,
                                                            size_t               rows,
                                                            size_t               vocab_size,
                                                            size_t               bitmask_words) {
    const auto* merged = merged_bitmask_cpu_.data_ptr<int32_t>();
    auto*       mask   = mask_cpu.data_ptr<bool>();
    static_assert(sizeof(bool) == 1, "torch bool mask requires one-byte C++ bool");
    const auto& lut        = maskedByteLut();
    const auto  full_words = vocab_size / 32;
    const auto  tail_bits  = vocab_size % 32;
    for (size_t row = 0; row < rows; ++row) {
        const auto* row_bits = merged + row * bitmask_words;
        auto*       row_mask = mask + row * vocab_size;
        for (size_t word_idx = 0; word_idx < full_words; ++word_idx) {
            const uint32_t word = static_cast<uint32_t>(row_bits[word_idx]);
            auto*          out  = row_mask + word_idx * 32;
            std::memcpy(out, lut[word & 0xffu].data(), 8 * sizeof(bool));
            std::memcpy(out + 8, lut[(word >> 8) & 0xffu].data(), 8 * sizeof(bool));
            std::memcpy(out + 16, lut[(word >> 16) & 0xffu].data(), 8 * sizeof(bool));
            std::memcpy(out + 24, lut[(word >> 24) & 0xffu].data(), 8 * sizeof(bool));
        }
        if (tail_bits > 0) {
            const uint32_t word = static_cast<uint32_t>(row_bits[full_words]);
            auto*          out  = row_mask + full_words * 32;
            for (size_t bit = 0; bit < tail_bits; ++bit) {
                out[bit] = (word & (1u << bit)) == 0u;
            }
        }
    }
}

SpecLogitsVerifyRunner::LaunchResult SpecLogitsVerifyRunner::buildInline(const LaunchTask& task) {
    RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.build_inline");
    LaunchResult result;
    if (task.active.empty()) {
        return result;
    }

    const size_t B    = task.total_streams;
    const int    P    = task.propose_step;
    const size_t V    = task.vocab_size;
    const size_t W    = SpecLogitsProcessor::bitmaskWordCount(V);
    const size_t rows = B * static_cast<size_t>(P + 1);
    RTP_LLM_CHECK_WITH_INFO(B > 0 && P > 0 && V > 0, "invalid spec logits runner task");

    std::vector<const ActiveProcessor*> eligible_items;
    eligible_items.reserve(task.active.size());
    std::vector<size_t> active_stream_indices;
    std::vector<bool>   active_stream_seen(B, false);
    for (const auto& item : task.active) {
        if (!item.processor || !item.processor->isSpecVerifyEligible()) {
            // The stream behind this processor already failed (e.g. grammar
            // rejected a token) or opted out of spec verify. Leave its mask
            // rows all-allow and its cap untouched instead of dropping the
            // whole batch artifact: the stream error terminates that request
            // on its own, and other streams still need their verify masks.
            ++result.skipped_ineligible_processors;
            RTP_LLM_LOG_WARNING("spec logits verify skips ineligible processor, stream=%lu processor_idx=%zu",
                                item.stream_id,
                                item.processor_idx);
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(item.stream_idx < B, "spec logits processor stream index out of range");
        eligible_items.push_back(&item);
        if (!active_stream_seen[item.stream_idx]) {
            active_stream_seen[item.stream_idx] = true;
            active_stream_indices.push_back(item.stream_idx);
        }
    }
    if (eligible_items.empty()) {
        return result;
    }
    std::sort(active_stream_indices.begin(), active_stream_indices.end());
    const size_t active_streams = active_stream_indices.size();
    const size_t active_rows    = active_streams * static_cast<size_t>(P + 1);
    RTP_LLM_CHECK_WITH_INFO(active_rows > 0, "spec logits runner has no active rows");

    std::vector<size_t> compact_stream_indices(B, 0);
    for (size_t compact_idx = 0; compact_idx < active_streams; ++compact_idx) {
        compact_stream_indices[active_stream_indices[compact_idx]] = compact_idx;
    }

    ensureBuffersFit(B, active_streams, P, V, W);
    materializeDraftTokensToCpu(task);

    auto merged = merged_bitmask_cpu_.flatten().narrow(0, 0, static_cast<int64_t>(active_rows * W));
    fillAllAllow(merged);
    std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);

    const size_t proc_words = static_cast<size_t>(P + 1) * W;
    auto*        proc_mask  = processor_bitmask_cpu_.data_ptr<int32_t>();
    for (const auto* item_ptr : eligible_items) {
        const auto& item = *item_ptr;
        std::fill_n(proc_mask, proc_words, SpecLogitsProcessor::kBitmaskAllowAll);

        SpecLogitsProcessorRequest request;
        request.draft_tokens       = draft_tokens_cpu_.data_ptr<int32_t>() + item.stream_idx * P;
        request.propose_step       = P;
        request.bitmask_cpu_out    = proc_mask;
        request.bitmask_size_int32 = W;
        request.vocab_size         = V;
        request.stream_id          = item.stream_id;
        request.base_seq_len       = item.base_seq_len;
        request.base_output_len    = item.base_output_len;

        int cap = item.processor->tryAcceptAndFillBitmask(request);
        cap     = std::max(0, std::min(cap, P));

        const size_t compact_idx = compact_stream_indices[item.stream_idx];
        auto*        merged_row  = merged_bitmask_cpu_.data_ptr<int32_t>() + compact_idx * (P + 1) * W;
        bitwiseAndInplace(merged_row, proc_mask, proc_words);
        auto* cap_ptr            = spec_cap_cpu_.data_ptr<int32_t>();
        cap_ptr[item.stream_idx] = std::min<int32_t>(cap_ptr[item.stream_idx], cap);
        result.applied_processors.push_back({item.stream_id, item.processor_idx});
    }

    auto* cpu_slot = static_cast<CpuArtifactSlot*>(nullptr);
    for (auto& slot : cpu_artifact_slots_) {
        if (!slot.ready_event || slot.ready_event->query()) {
            cpu_slot = &slot;
            break;
        }
    }
    if (cpu_slot == nullptr) {
        cpu_artifact_slots_.emplace_back();
        cpu_slot = &cpu_artifact_slots_.back();
    }

    const int64_t mask_elements = static_cast<int64_t>(active_rows * V);
    const int64_t cap_elements  = static_cast<int64_t>(B);
    auto pinned_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true);
    auto pinned_i32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    if (!cpu_slot->mask.defined() || cpu_slot->mask.numel() < mask_elements) {
        cpu_slot->mask = torch::empty({mask_elements}, pinned_bool);
    }
    if (!cpu_slot->cap.defined() || cpu_slot->cap.numel() < cap_elements) {
        cpu_slot->cap = torch::empty({cap_elements}, pinned_i32);
    }
    auto mask_cpu =
        cpu_slot->mask.narrow(0, 0, mask_elements).view({static_cast<int64_t>(active_rows), static_cast<int64_t>(V)});
    auto cap_cpu = cpu_slot->cap.narrow(0, 0, cap_elements);

    unpackMergedBitmaskToVocabMask(mask_cpu, active_rows, V, W);
    cap_cpu.copy_(spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(B)));

    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(copy_stream_));
    auto cuda_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    auto cuda_i32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto                         mask_gpu  = active_rows == rows ?
                                                 torch::empty({static_cast<int64_t>(rows), static_cast<int64_t>(V)}, cuda_bool) :
                                                 torch::zeros({static_cast<int64_t>(rows), static_cast<int64_t>(V)}, cuda_bool);
    auto cap_gpu = torch::empty({static_cast<int64_t>(B)}, cuda_i32);
    if (active_rows == rows) {
        mask_gpu.copy_(mask_cpu, /*non_blocking=*/true);
    } else {
        const int64_t rows_per_stream = static_cast<int64_t>(P + 1);
        for (size_t compact_idx = 0; compact_idx < active_streams; ++compact_idx) {
            const int64_t source_row = static_cast<int64_t>(compact_idx) * rows_per_stream;
            const int64_t target_row = static_cast<int64_t>(active_stream_indices[compact_idx]) * rows_per_stream;
            mask_gpu.narrow(0, target_row, rows_per_stream)
                .copy_(mask_cpu.narrow(0, source_row, rows_per_stream), /*non_blocking=*/true);
        }
    }
    cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);
    auto ready = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    ready->record(copy_stream_);
    cpu_slot->ready_event = ready;
    auto consumed = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());

    result.spec_vocab_mask_gpu       = mask_gpu;
    result.spec_cap_gpu              = cap_gpu;
    result.ready_event               = ready;
    result.consumed_event            = consumed;
    result.has_active_processor      = true;
    result.spec_vocab_mask_cpu_owner = mask_cpu;
    result.spec_cap_cpu_owner        = cap_cpu;
    return result;
}

}  // namespace rtp_llm
