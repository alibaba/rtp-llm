#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"

namespace rtp_llm {

std::string masterEnqueuedHandoffUniqueKey(int64_t request_id) {
    return "master_enqueued_" + std::to_string(request_id);
}

PDSupportDecision checkPDSupport(const GenerateInputPB& request) {
    const auto& config = request.generate_config();
    if (!config.can_use_pd_separation()) {
        return {false, "PD disabled by request"};
    }
    if (config.return_prompt_logits() || config.max_new_tokens() <= 1) {
        return {false, "PD requires more than one generated token"};
    }
    if (config.num_beams() > 1 || config.variable_num_beams_size() != 0) {
        return {false, "PD does not support beam search"};
    }
    if (config.num_return_sequences() > 1) {
        return {false, "PD does not support multiple return sequences"};
    }
    return {true, ""};
}

ErrorInfo validatePDHandoff(const GenerateInputPB& request) {
    if (request.generate_config().timeout_ms() <= 0) {
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "invalid P2P request timeout");
    }
    if (request.generate_config().unique_key().empty()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "PD handoff requires non-empty unique_key");
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo preprocessForPD(std::shared_ptr<GenerateInput>& input,
                          MultimodalProcessor*            processor,
                          bool                            is_mtp_eagle,
                          RoleType                        role) {
    input->generate_config->pd_separation        = true;
    input->generate_config->force_disable_sp_run = !is_mtp_eagle;
    if (role == RoleType::DECODE) {
        // Decode consumes the FE-expanded IDs. Only Prefill fetches embeddings;
        // its handoff carries the position IDs needed by subsequent decode steps.
        if (input->multimodal_token_layout) {
            const auto& ids   = input->input_ids;
            const auto& spans = input->multimodal_token_layout->spans;
            if (!ids.defined() || !ids.device().is_cpu() || ids.scalar_type() != torch::kInt32 || ids.dim() != 1
                || !ids.is_contiguous() || spans.empty()) {
                return {ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid frontend multimodal token layout"};
            }
            int64_t previous_end = 0;
            for (const auto& [offset, length] : spans) {
                const int64_t end = int64_t(offset) + length;
                if (offset < previous_end || length <= 0 || end > ids.numel()) {
                    return {ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid frontend multimodal span"};
                }
                previous_end = end;
            }
            auto mask = torch::ones_like(ids);
            auto locs = torch::empty({static_cast<int64_t>(spans.size())}, torch::kInt32);
            for (size_t i = 0; i < spans.size(); ++i) {
                const auto& [offset, length] = spans[i];
                mask.narrow(0, offset, length).zero_();
                locs.data_ptr<int32_t>()[i] = offset;
            }
            input->text_tokens_mask = std::move(mask);
            input->mm_locs          = std::move(locs);
        } else if (input->multimodal_inputs && !input->multimodal_inputs->empty() && !input->v41_inputs) {
            return {ErrorCode::MM_WRONG_FORMAT_ERROR, "Decode requires frontend-expanded multimodal token IDs"};
        }
        return ErrorInfo::OkStatus();
    }
    if (input->multimodal_inputs) {
        if (!processor) {
            return {ErrorCode::MM_EMPTY_ENGINE_ERROR, "PD multimodal request requires a multimodal processor"};
        }
        return processor->updateMultimodalFeatures(input);
    }
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
