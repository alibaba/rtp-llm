
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTokenUtils.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace py = pybind11;

namespace rtp_llm {

namespace {

ErrorInfo pinMultimodalTensors(std::vector<torch::Tensor>& tensors, const char* tensor_name) {
#if USING_CUDA
    try {
        for (auto& tensor : tensors) {
            if (!tensor.defined() || tensor.is_pinned()) {
                continue;
            }
            if (tensor.is_cuda()) {
                auto options =
                    torch::TensorOptions().dtype(tensor.scalar_type()).device(torch::kCPU).pinned_memory(true);
                tensor = tensor.to(options, /*non_blocking=*/true);
            } else {
                tensor = tensor.pin_memory();
            }
        }
    } catch (const std::exception& e) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                         std::string("failed to move multimodal ") + tensor_name + " to pinned CPU: " + e.what());
    }
#else
    (void)tensors;
    (void)tensor_name;
#endif
    return ErrorInfo::OkStatus();
}

}  // namespace

ErrorInfo MultimodalProcessor::getFeatureHash(int32_t* token_ids, const torch::Tensor& mm_emb) {
    try {
        auto hashes = getMultimodalFeatureHash(mm_emb);
        memcpy(token_ids, hashes.data_ptr<int32_t>(), hashes.nbytes());
    } catch (const std::exception& error) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
    }
    return ErrorInfo::OkStatus();
}

ErrorResult<ExpandedOutput>
MultimodalProcessor::expandTokenIds(const std::vector<torch::Tensor>&                mm_embedding,
                                    const torch::Tensor&                             token_ids,
                                    const std::vector<rtp_llm::MultimodalInput>      mm_inputs,
                                    torch::Tensor                                    token_type_ids,
                                    const std::optional<std::vector<torch::Tensor>>& feature_hashes) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids);
    }

    assert(token_ids.dim() == 1);
    int              expanded_len = token_ids.size(0);
    std::vector<int> embed_len    = {};
    CHECK_AND_RETURN_REF(locs, getMultimodalTags(token_ids));
    torch::Tensor expanded_token_type_ids;
    int           mm_num = mm_embedding.size();
    if (feature_hashes.has_value() && feature_hashes->size() != mm_embedding.size()) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "multimodal feature hash segment count mismatch");
    }
    if (locs.size() != mm_num) {
        std::stringstream exception_str;
        exception_str << "number of multimodal tags and multimodal input not matched, expect " << locs.size()
                      << ", get " << mm_num;
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, exception_str.str());
    }
    for (int i = 0; i < mm_num; i++) {
        expanded_len += mm_embedding[i].sizes()[0] - locs[i].second + locs[i].first;
    }

    auto expanded_ids = torch::empty({(int64_t)expanded_len}, torch::kInt32);
    auto token_masks  = torch::empty({(int64_t)expanded_len}, torch::kInt32);
    auto new_locs     = torch::empty({(int64_t)mm_num}, torch::kInt32);
    memset(expanded_ids.data_ptr(), -1, expanded_ids.nbytes());
    std::fill(token_masks.data_ptr<int32_t>(), token_masks.data_ptr<int32_t>() + token_masks.numel(), 1);
    if (token_type_ids.defined()) {
        expanded_token_type_ids = torch::empty({(int64_t)expanded_len}, torch::kInt32);
        std::fill(expanded_token_type_ids.data_ptr<int32_t>(),
                  expanded_token_type_ids.data_ptr<int32_t>() + expanded_token_type_ids.numel(),
                  0);
    }
    int new_loc_idx = 0, old_loc_idx = 0;
    for (int i = 0; i < mm_num; i++) {
        auto& loc      = locs[i];
        int   copy_len = loc.first - old_loc_idx;
        memcpy(expanded_ids.data_ptr<int32_t>() + new_loc_idx,
               token_ids.data_ptr<int32_t>() + old_loc_idx,
               sizeof(int32_t) * copy_len);
        memset(
            token_masks.data_ptr<int32_t>() + new_loc_idx + copy_len, 0, mm_embedding[i].sizes()[0] * sizeof(int32_t));
        if (token_type_ids.defined()) {
            memcpy(expanded_token_type_ids.data_ptr<int32_t>() + new_loc_idx,
                   token_type_ids.data_ptr<int32_t>() + old_loc_idx,
                   sizeof(int32_t) * copy_len);
        }
        *(new_locs.data_ptr<int32_t>() + i) = copy_len + new_loc_idx;

        auto* target = expanded_ids.data_ptr<int32_t>() + new_loc_idx + copy_len;
        if (feature_hashes.has_value()) {
            const auto& hashes = feature_hashes->at(i);
            if (!hashes.defined() || !hashes.device().is_cpu() || hashes.scalar_type() != torch::kInt32
                || hashes.dim() != 1 || hashes.numel() != mm_embedding[i].size(0)) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid multimodal feature hashes");
            }
            auto contiguous_hashes = hashes.contiguous();
            memcpy(target, contiguous_hashes.data_ptr<int32_t>(), contiguous_hashes.nbytes());
        } else {
            RETURN_IF_STATUS_ERROR(getFeatureHash(target, mm_embedding[i]));
        }

        new_loc_idx += copy_len + mm_embedding[i].sizes()[0];
        old_loc_idx = loc.second;
    }
    if (expanded_ids.size(0) - new_loc_idx != token_ids.size(0) - old_loc_idx) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "expanded length calculate error");
    }
    memcpy(expanded_ids.data_ptr<int32_t>() + new_loc_idx,
           token_ids.data_ptr<int32_t>() + old_loc_idx,
           sizeof(int32_t) * (expanded_ids.size(0) - new_loc_idx));
    if (token_type_ids.defined()) {
        memcpy(expanded_token_type_ids.data_ptr<int32_t>() + new_loc_idx,
               token_type_ids.data_ptr<int32_t>() + old_loc_idx,
               sizeof(int32_t) * (expanded_ids.size(0) - new_loc_idx));
    }
    return ExpandedOutput(
        std::move(expanded_ids), std::move(expanded_token_type_ids), std::move(token_masks), std::move(new_locs));
}

ErrorResult<std::vector<std::pair<int32_t, int32_t>>>
MultimodalProcessor::getMultimodalTags(const torch::Tensor& token_ids) {
    try {
        const auto* data = token_ids.data_ptr<int32_t>();
        return getMultimodalTokenSpans({data, data + token_ids.numel()}, sep_token_ids_, include_sep_tokens_);
    } catch (const std::exception& error) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, error.what());
    }
}

ErrorInfo MultimodalProcessor::checkExpandLength(const ExpandedOutput& expand_output) {
    if (expand_output.expanded_ids.numel() >= max_seq_len_) {
        std::stringstream exception_str;
        exception_str << "input after multimodal process is " << expand_output.expanded_ids.numel() << " > max_seq_len("
                      << max_seq_len_ << ")";
        return ErrorInfo(ErrorCode::MM_LONG_PROMPT_ERROR, exception_str.str());
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::GenerateInput>& input,
                                                        grpc::ServerContext*                     server_context) {
    if (input->generate_config && input->generate_config->calculate_loss) {
        return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "cannot calculate loss in multimodal query");
    }
    std::string ip_port = "";
    if (input->generate_config) {
        for (auto& role_addr : input->generate_config->role_addrs) {
            if (role_addr.role == RoleType::VIT) {
                ip_port = role_addr.ip + ":" + std::to_string(role_addr.grpc_port);
                break;
            }
        }
    }
    CHECK_AND_RETURN_REF(
        mm_embedding_res,
        MultimodalEmbedding(input->multimodal_inputs.value(), ip_port, input->request_id, server_context));
    auto                       mm_features = std::move(mm_embedding_res.mm_features);
    std::vector<torch::Tensor> mm_extra_input;
    if (mm_embedding_res.mm_extra_input.has_value()) {
        mm_extra_input = std::move(mm_embedding_res.mm_extra_input.value());
    }
    input->mm_position_ids = std::move(mm_embedding_res.mm_position_ids);
    CHECK_AND_RETURN_REF(
        expanded_ids,
        expandTokenIds(
            mm_features, input->input_ids, input->multimodal_inputs.value(), {}, mm_embedding_res.mm_feature_hashes));
    RETURN_IF_STATUS_ERROR(checkExpandLength(expanded_ids));
    RETURN_IF_STATUS_ERROR(pinMultimodalTensors(mm_features, "embedding"));
    RETURN_IF_STATUS_ERROR(pinMultimodalTensors(mm_extra_input, "extra input"));
    input->multimodal_features = std::move(mm_features);
    input->mm_extra_input      = std::move(mm_extra_input);
    input->input_ids           = expanded_ids.expanded_ids;
    input->text_tokens_mask    = expanded_ids.text_tokens_mask;
    input->mm_locs             = expanded_ids.locs;
    return ErrorInfo::OkStatus();
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::EmbeddingInput>&    input,
                                                        const std::vector<rtp_llm::MultimodalInput>& mm_inputs,
                                                        const std::string&                           vit_role_addr,
                                                        grpc::ServerContext*                         server_context) {
    CHECK_AND_RETURN_REF(mm_embedding_res,
                         MultimodalEmbedding(mm_inputs, vit_role_addr, input->request_id, server_context));
    MultimodalFeature mm_features;
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(mm_features.features,
                                        input->token_ids,
                                        mm_inputs,
                                        input->token_type_ids,
                                        mm_embedding_res.mm_feature_hashes));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    input->multimodal_features.emplace(mm_features);
    input->token_ids      = expanded_ids.expanded_ids;
    input->token_type_ids = expanded_ids.token_type_ids;
    if (input->input_lengths.numel() == 1 && expanded_ids.expanded_ids.defined()) {
        input->input_lengths.data_ptr<int32_t>()[0] = expanded_ids.expanded_ids.size(0);
        input->total_length                         = expanded_ids.expanded_ids.size(0);
    }
    return ErrorInfo::OkStatus();
}

ErrorResult<MultimodalFeature>
MultimodalProcessor::getMultimodalFeatures(const torch::Tensor&                         input_ids,
                                           const std::vector<rtp_llm::MultimodalInput>& mm_inputs) {
    MultimodalFeature mm_features;
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(mm_inputs));
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(
        expanded_ids,
        expandTokenIds(mm_features.features, input_ids, mm_inputs, {}, mm_embedding_res.mm_feature_hashes));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    return mm_features;
}

}  // namespace rtp_llm
