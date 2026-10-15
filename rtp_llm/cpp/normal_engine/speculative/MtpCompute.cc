#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <cstring>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace rtp_llm {
namespace mtp {
namespace {
torch::Tensor
shiftedMtpTokenField(const torch::Tensor& field, const torch::Tensor& input_lengths_cpu, int32_t append_value) {
    if (!field.defined() || field.numel() == 0) {
        return field;
    }
    RTP_LLM_CHECK_WITH_INFO(field.scalar_type() == torch::kInt32,
                            "MTP multimodal token field must be int32, got %s",
                            c10::toString(field.scalar_type()));
    RTP_LLM_CHECK_WITH_INFO(input_lengths_cpu.device().is_cpu() && input_lengths_cpu.dim() == 1
                                && input_lengths_cpu.scalar_type() == torch::kInt32,
                            "MTP input_lengths must be a CPU 1-D int32 tensor");
    auto source                = field.is_cuda() ? field.cpu() : field;
    source                     = source.contiguous();
    const int64_t total_tokens = source.numel();
    int64_t       length_sum   = 0;
    const auto*   lengths      = input_lengths_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < input_lengths_cpu.numel(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(
            lengths[i] > 0, "MTP input length must be positive, got %d at request %ld", lengths[i], i);
        length_sum += lengths[i];
    }
    RTP_LLM_CHECK_WITH_INFO(length_sum == total_tokens,
                            "MTP token field length mismatch: field=%ld input_lengths=%ld",
                            total_tokens,
                            length_sum);

    auto output =
        torch::empty({total_tokens}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    const auto* src    = source.data_ptr<int32_t>();
    auto*       dst    = output.data_ptr<int32_t>();
    int64_t     offset = 0;
    for (int64_t request = 0; request < input_lengths_cpu.numel(); ++request) {
        const int64_t length = lengths[request];
        if (length > 1) {
            std::memcpy(dst + offset, src + offset + 1, static_cast<size_t>(length - 1) * sizeof(int32_t));
        }
        dst[offset + length - 1] = append_value;
        offset += length;
    }
    return field.is_cuda() ? output.to(field.device(), /*non_blocking=*/true) : output;
}

void shiftMtpMultimodalLocations(GptModelInputs& model_input, const torch::Tensor& input_lengths_cpu) {
    const bool has_features = model_input.multimodal_features.has_value() && !model_input.multimodal_features->empty();
    const bool has_locs     = model_input.mm_features_locs.defined() && model_input.mm_features_locs.numel() > 0;
    if (!has_features && !has_locs) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(has_features && has_locs,
                            "MTP multimodal features and mm_features_locs must be provided together");
    RTP_LLM_CHECK_WITH_INFO(input_lengths_cpu.device().is_cpu() && input_lengths_cpu.scalar_type() == torch::kInt32
                                && input_lengths_cpu.dim() == 1,
                            "MTP input_lengths must be a CPU 1-D int32 tensor");

    auto locs =
        model_input.mm_features_locs.is_cuda() ? model_input.mm_features_locs.cpu() : model_input.mm_features_locs;
    locs = locs.contiguous();
    RTP_LLM_CHECK_WITH_INFO(locs.scalar_type() == torch::kInt32 && locs.dim() == 1,
                            "MTP mm_features_locs must be a 1-D int32 tensor");
    const auto& features = model_input.multimodal_features.value();
    RTP_LLM_CHECK_WITH_INFO(locs.numel() == static_cast<int64_t>(features.size()),
                            "MTP multimodal feature/location count mismatch: features=%zu locs=%ld",
                            features.size(),
                            locs.numel());
    const bool  has_extra_input = model_input.mm_extra_input.has_value() && !model_input.mm_extra_input->empty();
    const auto* extra_input     = has_extra_input ? &model_input.mm_extra_input.value() : nullptr;
    RTP_LLM_CHECK_WITH_INFO(!has_extra_input || extra_input->size() == features.size(),
                            "MTP mm_extra_input count mismatch: extra_input=%zu features=%zu",
                            extra_input ? extra_input->size() : 0,
                            features.size());

    std::vector<int64_t> request_starts(input_lengths_cpu.numel() + 1, 0);
    const auto*          lengths = input_lengths_cpu.data_ptr<int32_t>();
    for (int64_t request = 0; request < input_lengths_cpu.numel(); ++request) {
        RTP_LLM_CHECK_WITH_INFO(lengths[request] > 0,
                                "MTP input length must be positive, got %d at request %ld",
                                lengths[request],
                                request);
        request_starts[request + 1] = request_starts[request] + lengths[request];
    }

    std::vector<torch::Tensor> shifted_features;
    std::vector<torch::Tensor> shifted_extra_input;
    std::vector<int32_t>       shifted_locs;
    shifted_features.reserve(features.size());
    shifted_locs.reserve(features.size());
    if (has_extra_input) {
        shifted_extra_input.reserve(extra_input->size());
    }

    const auto* src_locs = locs.data_ptr<int32_t>();
    for (int64_t feature_idx = 0; feature_idx < locs.numel(); ++feature_idx) {
        const auto& feature = features[feature_idx];
        RTP_LLM_CHECK_WITH_INFO(feature.defined() && feature.dim() == 2,
                                "MTP multimodal feature %ld must be a defined 2-D tensor",
                                feature_idx);
        const int64_t feature_start = src_locs[feature_idx];
        const int64_t feature_len   = feature.size(0);
        const int64_t feature_end   = feature_start + feature_len;
        RTP_LLM_CHECK_WITH_INFO(feature_start >= 0 && feature_len > 0,
                                "MTP multimodal feature %ld has invalid range [%ld,%ld)",
                                feature_idx,
                                feature_start,
                                feature_end);
        int64_t owner = -1;
        for (int64_t request = 0; request < input_lengths_cpu.numel(); ++request) {
            if (feature_start >= request_starts[request] && feature_start < request_starts[request + 1]) {
                owner = request;
                break;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(owner >= 0,
                                "MTP multimodal feature %ld [%ld,%ld) does not overlap any request",
                                feature_idx,
                                feature_start,
                                feature_end);
        RTP_LLM_CHECK_WITH_INFO(feature_end <= request_starts[owner + 1],
                                "MTP multimodal feature %ld [%ld,%ld) crosses request %ld boundary [%ld,%ld)",
                                feature_idx,
                                feature_start,
                                feature_end,
                                owner,
                                request_starts[owner],
                                request_starts[owner + 1]);

        /** MTP removes the first token of every request and writes the sampled
         * token into that request's last slot. A feature that starts at the
         * request boundary therefore loses its first row as well; all other
         * features simply move left by one packed position. The request owner
         * is used only to detect this boundary and must not be subtracted from
         * the packed global location. */
        const bool    starts_at_request = feature_start == request_starts[owner];
        const int64_t feature_offset    = starts_at_request ? 1 : 0;
        const int64_t shifted_len       = feature_len - feature_offset;
        if (shifted_len <= 0) {
            continue;
        }

        shifted_features.emplace_back(feature.slice(0, feature_offset, feature_len).contiguous());
        shifted_locs.emplace_back(static_cast<int32_t>(starts_at_request ? feature_start : feature_start - 1));
        if (has_extra_input) {
            shifted_extra_input.emplace_back(
                sliceMultimodalExtraInput((*extra_input)[feature_idx], feature, feature_offset, feature_len));
        }
    }

    auto shifted_locs_cpu = torch::empty({static_cast<int64_t>(shifted_locs.size())},
                                         torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    if (!shifted_locs.empty()) {
        std::memcpy(shifted_locs_cpu.data_ptr<int32_t>(), shifted_locs.data(), shifted_locs.size() * sizeof(int32_t));
    }
    model_input.multimodal_features = std::move(shifted_features);
    if (has_extra_input) {
        model_input.mm_extra_input = std::move(shifted_extra_input);
    }
    model_input.mm_features_locs =
        model_input.mm_features_locs.is_cuda() ?
            shifted_locs_cpu.to(model_input.mm_features_locs.device(), /*non_blocking=*/true) :
            shifted_locs_cpu;
}

torch::Tensor toCudaInt32(const torch::Tensor& tensor, TensorHolder& host_holder) {
    if (!tensor.defined()) {
        return tensor;
    }
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }
    host_holder.hold_host(tensor);
    return tensor.to(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA), /*non_blocking=*/true);
}

void applySpecLogitsAcceptLenCap(const SpecLogitsVerifyRunner::LaunchResult& verify_result,
                                torch::Tensor                              target_token_ids,
                                speculative::SpeculativeSamplerOutput&     accepted) {
    auto cap_src = verify_result.spec_cap_gpu;
    if (!cap_src.defined()) {
        cap_src = verify_result.spec_cap_cpu;
    }
    if (!cap_src.defined()) {
        return;
    }
    auto&      accept_tokens = accepted.accept_tokens;
    auto&      accept_len    = accepted.accept_len;
    const auto batch_size    = accept_tokens.size(0);
    const auto propose_step  = accept_tokens.size(1) - 1;
    RTP_LLM_CHECK_WITH_INFO(accept_len.defined() && accept_len.is_cuda(),
                            "spec logits cap requires CUDA accept_len");

    if (verify_result.ready_event) {
        verify_result.ready_event->block(cuda_graph::graphGetCurrentStream());
    }
    if (!cap_src.is_cuda()) {
        cap_src = cap_src.to(accept_len.device());
    }
#if USING_CUDA
    c10::cuda::CUDACachingAllocator::recordStream(cap_src.storage().data_ptr(),
                                                at::cuda::getCurrentCUDAStream(cap_src.device().index()));
#endif
    auto cap_gpu      = cap_src.to(accept_len.options());
    auto cap_plus_one = cap_gpu + 1;
    accept_len = torch::minimum(accept_len, cap_plus_one);

    RTP_LLM_CHECK_WITH_INFO(accept_tokens.defined() && accept_tokens.is_cuda(),
                            "spec logits cap requires CUDA accept_tokens");
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.defined(),
                            "spec logits cap requires target sampler token_ids");
    if (!target_token_ids.is_cuda()) {
        target_token_ids = target_token_ids.to(accept_tokens.device(), /*non_blocking=*/true);
    }
    const int64_t token_stride  = target_token_ids.size(1);
    auto          target_tokens = target_token_ids.reshape({batch_size, propose_step + 1, token_stride})
                             .select(2, token_stride - 1)
                             .to(accept_tokens.options());
    auto cap_index   = cap_src.to(torch::TensorOptions().device(accept_tokens.device()).dtype(torch::kLong));
    auto replacement = target_tokens.gather(1, cap_index.unsqueeze(1));

    auto cols = torch::arange(propose_step + 1,
                              torch::TensorOptions().device(accept_tokens.device()).dtype(torch::kLong))
                    .unsqueeze(0)
                    .expand({batch_size, propose_step + 1});
    auto replace_mask = (cap_gpu < propose_step).unsqueeze(1) & (accept_len > cap_gpu).unsqueeze(1)
                        & (cols == cap_index.unsqueeze(1));
    accept_tokens =
        torch::where(replace_mask, replacement.expand({batch_size, propose_step + 1}), accept_tokens);
}

torch::Tensor compactAcceptedPositionIds(const torch::Tensor&     combo_position_ids,
                                        const std::vector<int>& accept_lens,
                                        size_t                  total_accept_len,
                                        size_t                  position_id_len_factor,
                                        int64_t                 tokens_per_request) {
    if (!combo_position_ids.defined()) {
        return torch::Tensor();
    }

    auto         compact_position_ids =
        torch::empty({(int64_t)(total_accept_len * position_id_len_factor)}, torch::kInt32).pin_memory();
    const int* src_position_ids = combo_position_ids.data_ptr<int>();
    int*       dst_position_ids = compact_position_ids.data_ptr<int>();

    int token_offset = 0;
    for (size_t i = 0; i < accept_lens.size(); ++i) {
        for (int step = 0; step < accept_lens[i]; ++step) {
            memcpy(dst_position_ids + (token_offset + step) * position_id_len_factor,
                   src_position_ids + (i * tokens_per_request + step) * position_id_len_factor,
                   position_id_len_factor * sizeof(int));
        }
        token_offset += accept_lens[i];
    }

    return compact_position_ids;
}

bool reduceDSparkCacheStoreStatus(bool                 local_ok,
                                 int                  tp_size,
                                 const torch::Stream& sync_stream,
                                 torch::Tensor&       status) {
    if (tp_size <= 1) {
        return local_ok;
    }

    RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(dspark_cache_store_tp_sync)");
    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(sync_stream));
    RTP_LLM_CHECK_WITH_INFO(status.defined(),
                            "DSpARK cache-store TP status buffer is not initialized");
    status.fill_(local_ok ? 1 : 0);
    auto global_status = execAllReduce({status, ReduceOp::Min, false, ParallelMode::TP}).buffer;
    return global_status.item<int32_t>() != 0;
}

}  // namespace

std::string finishDSparkCachePublication(ModelBase*                       target,
                                       ModelBase*                       draft,
                                       const ParallelismConfig&         parallelism_config,
                                       const torch::Stream&             sync_stream,
                                       torch::Tensor&                   status,
                                       const std::function<bool(bool)>& reduce_status) {
    const int tp_rank = parallelism_config.tp_rank;
    auto wait_for_publication = [tp_rank](ModelBase* model, const char* role) {
        if (model == nullptr) {
            return std::string(role) + " cache-store publication model is unavailable";
        }
        try {
            return model->waitCacheStorePublication();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR(
                "DSpARK %s cache-store publication wait threw on TP rank %d: %s", role, tp_rank, e.what());
            return std::string("wait threw: ") + e.what();
        } catch (...) {
            RTP_LLM_LOG_ERROR(
                "DSpARK %s cache-store publication wait threw on TP rank %d: unknown exception", role, tp_rank);
            return std::string("wait threw an unknown exception");
        }
    };

    /** Both waits must finish before every TP rank enters the status reduction. */
    const std::string target_error = wait_for_publication(target, "target");
    const std::string draft_error  = wait_for_publication(draft, "draft");
    const bool        local_ok     = target_error.empty() && draft_error.empty();
    if (!local_ok) {
        RTP_LLM_LOG_ERROR("DSpARK cache-store publication failed on TP rank %d: target=[%s], draft=[%s]",
                          tp_rank,
                          target_error.c_str(),
                          draft_error.c_str());
    }

    const bool global_ok =
        reduce_status ? reduce_status(local_ok) :
                        reduceDSparkCacheStoreStatus(local_ok, parallelism_config.tp_size, sync_stream, status);
    if (global_ok) {
        return {};
    }
    std::string error = "DSpARK cache-store publication failed before decode dispatch";
    if (!target_error.empty()) {
        error += "; target: " + target_error;
    }
    if (!draft_error.empty()) {
        error += "; draft: " + draft_error;
    }
    return error;
}

DSparkCacheStoreDrainGuard::~DSparkCacheStoreDrainGuard() {
    if (!armed) {
        return;
    }
    auto drain = [](ModelBase* model, const char* name) {
        if (model == nullptr) {
            return;
        }
        try {
            const auto error = model->waitCacheStorePublication();
            if (!error.empty()) {
                RTP_LLM_LOG_ERROR(
                    "DSpARK local %s cache-store drain on prefill exit failed: %s", name, error.c_str());
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("DSpARK local %s cache-store drain threw on prefill exit: %s", name, e.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("DSpARK local %s cache-store drain threw an unknown exception on prefill exit",
                              name);
        }
    };
    drain(target, "target");
    drain(draft, "draft");
}

DraftInputLayout selectDraftInputLayout(bool device_state_enabled) {
    return device_state_enabled ? DraftInputLayout::FIXED_WIDTH : DraftInputLayout::COMPACT;
}

AcceptedTokens getDeviceAcceptedTokens(const speculative::SpeculativeSamplerOutput& output) {
    return {output.accept_tokens, output.accept_len};
}

AcceptedTokens getHostAcceptedTokens(const speculative::SpeculativeSamplerOutput& output, bool wait_for_transfer) {
    if (wait_for_transfer) {
        output.transfer_done_event->synchronize();
    }
    return {output.accept_tokens_cpu.defined() ? output.accept_tokens_cpu : output.accept_tokens.cpu(),
            output.accept_len_cpu.defined() ? output.accept_len_cpu : output.accept_len.cpu()};
}

void maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output,
                                         ModelBase&       source,
                                         int64_t          hidden_rows) {
    /** Prefer this forward's explicit output; graph replay must not depend on mutable model-side buffers. */
    if (model_output.mtp_target_hidden_states.defined()) {
        RTP_LLM_CHECK_WITH_INFO(hidden_rows < 0 || model_output.mtp_target_hidden_states.size(0) == hidden_rows,
                                "MTP target hidden output rows mismatch: got %ld, expected %ld",
                                model_output.mtp_target_hidden_states.size(0),
                                hidden_rows);
        model_output.all_hidden_states = model_output.mtp_target_hidden_states;
        return;
    }
    if (hidden_rows == 0) {
        if (!model_output.all_hidden_states.defined() || model_output.all_hidden_states.size(0) == 0) {
            return;
        }
        hidden_rows = model_output.all_hidden_states.size(0);
    }
    auto hidden_states = source.getMtpTargetHiddenStates(hidden_rows);
    if (hidden_states.defined() && hidden_states.numel() > 0) {
        model_output.all_hidden_states = std::move(hidden_states);
    }
}

void prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_input,
                                          const torch::Tensor& target_hidden_states,
                                          const torch::Tensor& sampled_token_ids,
                                          const torch::Tensor& next_position_ids,
                                          size_t               position_id_len_factor,
                                          TensorHolder&        host_holder) {
    const auto batch_size           = sampled_token_ids.size(0);
    const auto sampler_token_stride = sampled_token_ids.size(1);
    const auto sampled_tokens_cpu   = sampled_token_ids.cpu().contiguous();
    const auto input_lengths_cpu    = draft_input.input_lengths.cpu().contiguous();
    const auto target_tokens_cpu    = draft_input.combo_tokens.cpu().contiguous();
    const auto host_options         = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto       combo_tokens         = torch::empty(target_tokens_cpu.sizes(), host_options);

    const bool has_multimodal =
        draft_input.multimodal_features.has_value() && !draft_input.multimodal_features->empty();
    if (has_multimodal) {
        draft_input.text_tokens_mask = shiftedMtpTokenField(draft_input.text_tokens_mask, input_lengths_cpu, 1);
        draft_input.combo_tokens_type_ids =
            shiftedMtpTokenField(draft_input.combo_tokens_type_ids, input_lengths_cpu, 0);
        shiftMtpMultimodalLocations(draft_input, input_lengths_cpu);
    }

    torch::Tensor target_position_ids_cpu;
    torch::Tensor next_position_ids_cpu;
    torch::Tensor combo_position_ids;
    if (draft_input.combo_position_ids.defined()) {
        target_position_ids_cpu = draft_input.combo_position_ids.cpu().contiguous();
        next_position_ids_cpu   = next_position_ids.cpu().contiguous();
        combo_position_ids      = torch::empty(target_position_ids_cpu.sizes(), host_options);
    }

    const auto* input_lengths_ptr  = input_lengths_cpu.data_ptr<int32_t>();
    const auto* target_tokens_ptr  = target_tokens_cpu.data_ptr<int32_t>();
    const auto* sampled_tokens_ptr = sampled_tokens_cpu.data_ptr<int32_t>();
    auto*       combo_tokens_ptr   = combo_tokens.data_ptr<int32_t>();
    int64_t     token_offset       = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto input_length = input_lengths_ptr[i];
        memcpy(combo_tokens_ptr + token_offset,
               target_tokens_ptr + token_offset + 1,
               (input_length - 1) * sizeof(int32_t));
        combo_tokens_ptr[token_offset + input_length - 1] =
            sampled_tokens_ptr[i * sampler_token_stride + sampler_token_stride - 1];

        if (combo_position_ids.defined()) {
            auto* request_positions = combo_position_ids.data_ptr<int32_t>() + token_offset * position_id_len_factor;
            memcpy(request_positions,
                   target_position_ids_cpu.data_ptr<int32_t>() + (token_offset + 1) * position_id_len_factor,
                   (input_length - 1) * position_id_len_factor * sizeof(int32_t));
            memcpy(request_positions + (input_length - 1) * position_id_len_factor,
                   next_position_ids_cpu.data_ptr<int32_t>() + i * position_id_len_factor,
                   position_id_len_factor * sizeof(int32_t));
        }
        token_offset += input_length;
    }

    draft_input.is_target_verify   = false;
    draft_input.last_hidden_states = target_hidden_states;
    draft_input.input_lengths      = toCudaInt32(draft_input.input_lengths, host_holder);
    draft_input.combo_tokens       = toCudaInt32(combo_tokens, host_holder);
    draft_input.combo_position_ids = std::move(combo_position_ids);
}

void prepareDraftPrefillAfterVerify(GptModelInputs&      draft_input,
                                    const torch::Tensor& target_hidden_states,
                                    const torch::Tensor& accepted_token_ids,
                                    const torch::Tensor& accepted_lengths,
                                    DraftInputLayout     layout,
                                    size_t               position_id_len_factor,
                                    TensorHolder&        host_holder) {
    const size_t batch_size         = accepted_lengths.numel();
    const auto   tokens_per_request = accepted_token_ids.size(1);
    draft_input.is_target_verify    = false;
    if (layout == DraftInputLayout::FIXED_WIDTH) {
        const auto total_tokens = accepted_token_ids.numel();
        draft_input.combo_tokens = toCudaInt32(accepted_token_ids.reshape({total_tokens}), host_holder);
        auto accepted_lengths_gpu = toCudaInt32(accepted_lengths, host_holder);
        draft_input.lm_output_indexes =
            torch::arange(0, total_tokens, tokens_per_request, accepted_lengths_gpu.options()) + (accepted_lengths_gpu - 1);
        draft_input.last_hidden_states = target_hidden_states;
        return;
    }

    const auto      accepted_token_ids_cpu = accepted_token_ids.cpu().contiguous();
    const auto      accepted_lengths_cpu   = accepted_lengths.cpu().contiguous();
    const auto*     accepted_lengths_ptr   = accepted_lengths_cpu.data_ptr<int32_t>();
    std::vector<int> accepted_lengths_per_request(batch_size);
    size_t          total_accepted_tokens = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        accepted_lengths_per_request[i] = accepted_lengths_ptr[i];
        total_accepted_tokens += accepted_lengths_per_request[i];
    }

    auto combo_tokens =
        torch::empty({static_cast<int64_t>(total_accepted_tokens)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto input_lengths =
        torch::empty({static_cast<int64_t>(batch_size)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto lm_output_indexes =
        torch::empty({static_cast<int64_t>(batch_size)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    const auto* accepted_token_ids_ptr = accepted_token_ids_cpu.data_ptr<int32_t>();
    auto*       combo_tokens_ptr       = combo_tokens.data_ptr<int32_t>();
    auto*       input_lengths_ptr      = input_lengths.data_ptr<int32_t>();
    auto*       output_indexes_ptr     = lm_output_indexes.data_ptr<int32_t>();

    size_t                    token_offset = 0;
    std::vector<torch::Tensor> accepted_hidden_states;
    accepted_hidden_states.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        memcpy(combo_tokens_ptr + token_offset,
               accepted_token_ids_ptr + i * tokens_per_request,
               accepted_lengths_per_request[i] * sizeof(int32_t));
        accepted_hidden_states.push_back(
            target_hidden_states.narrow(0, i * tokens_per_request, accepted_lengths_per_request[i]));
        input_lengths_ptr[i] = accepted_lengths_per_request[i];
        token_offset += accepted_lengths_per_request[i];
        output_indexes_ptr[i] = static_cast<int32_t>(token_offset - 1);
    }

    draft_input.combo_tokens       = std::move(combo_tokens);
    draft_input.input_lengths      = std::move(input_lengths);
    draft_input.lm_output_indexes  = std::move(lm_output_indexes);
    draft_input.last_hidden_states = torch::cat(accepted_hidden_states).contiguous();
    auto verify_position_ids = draft_input.combo_position_ids.defined() && draft_input.combo_position_ids.is_cuda() ?
                                   draft_input.combo_position_ids.cpu().contiguous() :
                                   draft_input.combo_position_ids;
    auto compact_position_ids = compactAcceptedPositionIds(verify_position_ids,
                                                           accepted_lengths_per_request,
                                                           total_accepted_tokens,
                                                           position_id_len_factor,
                                                           tokens_per_request);
    if (compact_position_ids.defined()) {
        draft_input.combo_position_ids = std::move(compact_position_ids);
    }
}

void syncDraftPrefillAfterVerify(GptModelInputs&          draft_input,
                                 DraftInputLayout         layout,
                                 const ParallelismConfig& parallelism_config) {
    if (parallelism_config.tp_size <= 1) {
        return;
    }

    if (layout == DraftInputLayout::FIXED_WIDTH) {
        execBroadcast({{draft_input.combo_tokens}, 0});
        execBroadcast({{draft_input.last_hidden_states}, 0});
        execBroadcast({{draft_input.lm_output_indexes}, 0});
    } else {
        tpSyncModelInputs(draft_input, parallelism_config);
    }
}

void prepareDraftDecodeAfterPrefill(GptModelInputs&      draft_input,
                                    const torch::Tensor& draft_hidden_states,
                                    const torch::Tensor& draft_token_ids,
                                    size_t               position_id_len_factor) {
    const auto batch_size       = draft_input.input_lengths.numel();
    const auto output_indexes   = draft_input.lm_output_indexes.to(torch::kLong);
    const auto physical_lengths = draft_input.input_lengths.to(torch::kLong);
    const auto starts           = physical_lengths.cumsum(0) - physical_lengths;
    const auto valid_lengths    = output_indexes.to(starts.device()) - starts + 1;

    draft_input.combo_tokens       = draft_token_ids.reshape({batch_size});
    draft_input.last_hidden_states =
        draft_hidden_states.index_select(0, output_indexes.to(draft_hidden_states.device()));
    draft_input.sequence_lengths =
        (draft_input.prefix_lengths.to(valid_lengths.device()) + valid_lengths).to(torch::kInt32);
    draft_input.input_lengths           = torch::ones_like(draft_input.input_lengths);
    draft_input.prefix_lengths          = torch::empty({0}, draft_input.prefix_lengths.options());
    draft_input.sequence_lengths_plus_1 = torch::Tensor();
    draft_input.lm_output_indexes       = torch::arange(batch_size, draft_input.lm_output_indexes.options());
    draft_input.request_id             = torch::Tensor();
    draft_input.request_pd_separation  = torch::Tensor();
    draft_input.cache_keys             = torch::Tensor();
    if (draft_input.combo_position_ids.defined()) {
        const auto positions = draft_input.combo_position_ids.reshape({-1, static_cast<int64_t>(position_id_len_factor)});
        draft_input.combo_position_ids =
            (positions.index_select(0, output_indexes.to(positions.device())) + 1).flatten().cpu().pin_memory();
    }
}

void advanceDraftInput(GptModelInputs&      draft_input,
                       const torch::Tensor& draft_hidden_states,
                       const torch::Tensor& draft_token_ids,
                       size_t               position_id_len_factor,
                       TensorHolder&        host_holder) {
    int batch_size                 = draft_input.combo_tokens.size(0);
    draft_input.last_hidden_states = draft_hidden_states.clone();
    draft_input.combo_tokens       = draft_token_ids.reshape({batch_size});

    if (draft_input.combo_position_ids.defined()) {
        auto         next_position_ids =
            torch::empty({(int64_t)(batch_size * position_id_len_factor)}, torch::kInt32).pin_memory();
        int*       next_position_ids_ptr = next_position_ids.data_ptr<int>();
        const auto position_ids_cpu      = draft_input.combo_position_ids.cpu().contiguous();
        const int* position_ids_ptr      = position_ids_cpu.data_ptr<int>();
        for (int64_t i = 0; i < next_position_ids.numel(); ++i) {
            next_position_ids_ptr[i] = position_ids_ptr[i] + 1;
        }
        draft_input.combo_position_ids = std::move(next_position_ids);
    }

    if (draft_input.sequence_lengths.is_cuda()) {
        draft_input.sequence_lengths = (draft_input.sequence_lengths + 1).to(torch::kInt32);
    } else {
        auto sequence_lengths_cpu = draft_input.sequence_lengths.cpu().clone().pin_memory();
        for (int i = 0; i < batch_size; i++) {
            sequence_lengths_cpu.data_ptr<int>()[i]++;
        }
        draft_input.sequence_lengths = toCudaInt32(sequence_lengths_cpu, host_holder);
    }
}

void prepareDSparkProposeInput(GptModelInputs&            draft_input,
                               const torch::Tensor&       anchors,
                               const torch::Tensor&       committed_ends,
                               size_t                     propose_step,
                               int32_t                    mask_token_id,
                               bool                       sample_from_anchor,
                               DSparkProposeInputBuffers& buffers,
                               TensorHolder&              host_holder) {
    RTP_LLM_CHECK_WITH_INFO(anchors.defined() && anchors.dim() == 1, "dspark propose anchors must be a 1-D tensor");
    RTP_LLM_CHECK_WITH_INFO(committed_ends.defined() && committed_ends.numel() == anchors.numel(),
                            "dspark propose committed ends must contain one value per anchor");
    RTP_LLM_CHECK_WITH_INFO(propose_step > 0, "dspark draft width must be positive");
    RTP_LLM_CHECK_WITH_INFO(mask_token_id >= 0, "dspark requires a non-negative noise token id, got %d", mask_token_id);

    const int64_t batch_size  = anchors.numel();
    const int64_t query_width = static_cast<int64_t>(propose_step) + static_cast<int64_t>(!sample_from_anchor);
    const int64_t logits_rows = batch_size * static_cast<int64_t>(propose_step);
    const auto    cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const bool    width_changed =
        buffers.combo_tokens.defined() && (buffers.combo_tokens.dim() != 2 || buffers.combo_tokens.size(1) != query_width);
    if (!buffers.combo_tokens.defined() || width_changed || buffers.combo_tokens.size(0) < batch_size) {
        buffers.combo_tokens = torch::full({batch_size, query_width}, mask_token_id, cuda_i32);
    }
    if (!buffers.input_lengths.defined() || width_changed || buffers.input_lengths.size(0) < batch_size) {
        buffers.input_lengths = torch::full({batch_size}, query_width, cuda_i32);
    }
    if (!buffers.lm_output_indexes.defined() || width_changed || buffers.lm_output_indexes.numel() < logits_rows) {
        if (sample_from_anchor) {
            buffers.lm_output_indexes = torch::arange(logits_rows, cuda_i32);
        } else {
            buffers.lm_output_indexes = torch::arange(batch_size * query_width, cuda_i32)
                                                .view({batch_size, query_width})
                                                .narrow(1, 1, propose_step)
                                                .contiguous()
                                                .view({-1});
        }
    }

    auto combo_tokens = buffers.combo_tokens.narrow(0, 0, batch_size);
    combo_tokens.select(1, 0).copy_(toCudaInt32(anchors, host_holder));

    draft_input.combo_tokens            = combo_tokens.reshape({-1});
    draft_input.combo_position_ids      = torch::Tensor();
    draft_input.last_hidden_states      = torch::Tensor();
    draft_input.prefix_lengths          = toCudaInt32(committed_ends, host_holder).contiguous();
    draft_input.input_lengths           = buffers.input_lengths.narrow(0, 0, batch_size);
    draft_input.sequence_lengths        = torch::empty({0}, cuda_i32);
    draft_input.sequence_lengths_plus_1 = torch::Tensor();
    draft_input.lm_output_indexes       = buffers.lm_output_indexes.narrow(0, 0, logits_rows);
    draft_input.is_target_verify        = true;
}

void runRejectionSampling(speculative::SpeculativeSampler&               sampler,
                          const speculative::SpeculativeSamplingParams& params,
                          SamplerOutput&                                draft_sampler_output,
                          SamplerOutput&                                target_sampler_output,
                          const SpecLogitsVerifyRunner::LaunchResult&    verify_result,
                          speculative::SpeculativeSamplerOutput&        output) {
    output                  = sampler.forward(params, draft_sampler_output, target_sampler_output);
    output.processor_errors = verify_result.processor_errors;
    applySpecLogitsAcceptLenCap(verify_result, target_sampler_output.token_ids, output);
    output.accept_tokens_cpu = output.accept_tokens.to(torch::kCPU, /*non_blocking=*/true);
    output.accept_len_cpu    = output.accept_len.to(torch::kCPU, /*non_blocking=*/true);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
    if (verify_result.consumed_event) {
        verify_result.consumed_event->record(cuda_graph::graphGetCurrentStream());
    }
}

}  // namespace mtp
}  // namespace rtp_llm
