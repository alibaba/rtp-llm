#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"

namespace rtp_llm {

PDSupportDecision checkPDSupport(const GenerateInputPB& request) {
    const auto& config = request.generate_config();
    if (!config.can_use_pd_separation()) {
        return {false, "PD disabled by request"};
    }
    if (config.max_new_tokens() <= 1) {
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

ErrorInfo checkPDBatchSupport(const BatchGenerateInputPB& request, bool& pd_separation) {
    pd_separation = request.inputs_size() > 0 && checkPDSupport(request.inputs(0)).supported;
    for (const auto& input : request.inputs()) {
        if (checkPDSupport(input).supported != pd_separation) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS,
                             "mixing PD and non-PD requests in one atomic batch is not supported");
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo validatePDHandoff(const GenerateInputPB& request) {
    if (request.generate_config().timeout_ms() <= 0) {
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "invalid P2P request timeout");
    }
    if (request.generate_config().unique_key().empty()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "decode_entrance handoff requires non-empty unique_key");
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo preprocessForPD(std::shared_ptr<GenerateInput>& input, MultimodalProcessor* processor, bool is_mtp_eagle) {
    input->generate_config->pd_separation        = true;
    input->generate_config->force_disable_sp_run = !is_mtp_eagle;
    if (input->multimodal_inputs) {
        if (!processor) {
            return {ErrorCode::MM_EMPTY_ENGINE_ERROR, "PD multimodal request requires a multimodal processor"};
        }
        return processor->updateMultimodalFeatures(input);
    }
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
