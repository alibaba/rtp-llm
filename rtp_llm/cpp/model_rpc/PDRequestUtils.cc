#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"

namespace rtp_llm {
namespace {

void snapshotTensor(PDTensorSnapshotPB* output, const torch::Tensor& tensor, bool include_data = true) {
    output->set_defined(tensor.defined());
    if (!tensor.defined()) {
        return;
    }
    output->set_dtype(c10::toString(tensor.scalar_type()));
    for (auto dim : tensor.sizes()) {
        output->add_shape(dim);
    }
    if (include_data && tensor.numel() > 0) {
        auto cpu = tensor.cpu().contiguous();
        output->set_data(static_cast<const char*>(cpu.data_ptr()), cpu.numel() * cpu.element_size());
    }
}

void snapshotTensors(PDTensorListSnapshotPB*           output,
                     const std::vector<torch::Tensor>& tensors,
                     bool                              include_data = true) {
    for (const auto& tensor : tensors) {
        snapshotTensor(output->add_tensors(), tensor, include_data);
    }
}

}  // namespace

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

PDInputSnapshotPB snapshotPDInput(const GenerateInput& input) {
    PDInputSnapshotPB output;
    snapshotTensor(output.mutable_input_ids(), input.input_ids);
    if (input.text_tokens_mask) {
        snapshotTensor(output.mutable_text_tokens_mask(), *input.text_tokens_mask);
    }
    if (input.mm_locs) {
        snapshotTensor(output.mutable_mm_locs(), *input.mm_locs);
    }
    if (input.mm_position_ids) {
        snapshotTensors(output.mutable_mm_position_ids(), *input.mm_position_ids);
    }
    if (input.mm_extra_input) {
        snapshotTensors(output.mutable_mm_extra_input(), *input.mm_extra_input);
    }
    if (input.multimodal_features) {
        // Feature values stay on the existing MM path; compare their shapes and
        // dtypes along with all expanded tokens and positional metadata.
        snapshotTensors(output.mutable_mm_feature_layouts(), *input.multimodal_features, false);
    }
    return output;
}

ErrorInfo validatePDInput(const GenerateInput& input, const GenerateInputPB& request) {
    const auto context = ", request_id=" + std::to_string(request.request_id())
                         + ", unique_key=" + request.generate_config().unique_key();
    if (!request.has_pd_input_snapshot()) {
        return {ErrorCode::MM_WRONG_FORMAT_ERROR, "missing PD input snapshot" + context};
    }
    const auto  actual   = snapshotPDInput(input);
    const auto& expected = request.pd_input_snapshot();
    // These messages contain no maps. Compare exact dtype, shape, presence and
    // bytes, including ordered tensor lists; no lossy hash or numeric cast.
#define CHECK_PD_FIELD(field)                                                                                          \
    if (actual.has_##field() != expected.has_##field()                                                                 \
        || actual.field().SerializeAsString() != expected.field().SerializeAsString()) {                               \
        return {ErrorCode::MM_WRONG_FORMAT_ERROR, "PD input mismatch: " #field + context};                             \
    }
    CHECK_PD_FIELD(input_ids);
    CHECK_PD_FIELD(text_tokens_mask);
    CHECK_PD_FIELD(mm_locs);
    CHECK_PD_FIELD(mm_position_ids);
    CHECK_PD_FIELD(mm_extra_input);
    CHECK_PD_FIELD(mm_feature_layouts);
#undef CHECK_PD_FIELD
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
