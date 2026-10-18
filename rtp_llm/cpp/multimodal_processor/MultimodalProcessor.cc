
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTokenUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

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
                // Downstream TP serialization can read the host tensor immediately.
                tensor = tensor.to(options, /*non_blocking=*/false);
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

ErrorResult<ExpandedOutput>
MultimodalProcessor::expandTokenIds(const std::vector<torch::Tensor>&                mm_embedding,
                                    const torch::Tensor&                             token_ids,
                                    torch::Tensor                                    token_type_ids,
                                    const std::optional<std::vector<torch::Tensor>>& feature_hashes) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids);
    }
    if (semantic_mm_token_id_ < -1 || semantic_mm_token_id_ > std::numeric_limits<int32_t>::max()) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "semantic multimodal token ID is outside int32 range");
    }

    assert(token_ids.dim() == 1);
    int expanded_len = token_ids.size(0);
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
    auto cache_spans = semantic_mm_token_id_ >= 0 ? std::make_shared<MultimodalCacheSpans>() : nullptr;
    if (cache_spans) {
        cache_spans->reserve(mm_num);
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

        torch::Tensor hashes;
        try {
            hashes = feature_hashes.has_value() ? feature_hashes->at(i) : getMultimodalFeatureHash(mm_embedding[i]);
        } catch (const std::exception& error) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
        }
        if (!hashes.defined() || !hashes.device().is_cpu() || hashes.scalar_type() != torch::kInt32 || hashes.dim() != 1
            || hashes.numel() != mm_embedding[i].size(0)) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid multimodal feature hashes");
        }
        hashes = hashes.contiguous();

        const int32_t output_offset = new_loc_idx + copy_len;
        auto*         target        = expanded_ids.data_ptr<int32_t>() + output_offset;
        if (cache_spans) {
            std::fill(target, target + hashes.numel(), static_cast<int32_t>(semantic_mm_token_id_));
            const auto* begin = hashes.data_ptr<int32_t>();
            cache_spans->push_back({output_offset, std::vector<int32_t>(begin, begin + hashes.numel())});
        } else {
            memcpy(target, hashes.data_ptr<int32_t>(), hashes.nbytes());
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
    return ExpandedOutput(std::move(expanded_ids),
                          std::move(expanded_token_type_ids),
                          std::move(token_masks),
                          std::move(new_locs),
                          std::move(cache_spans));
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
    if (!input->multimodal_inputs || input->multimodal_inputs->empty()) {
        return ErrorInfo::OkStatus();
    }
    CHECK_AND_RETURN_REF(
        mm_embedding_res,
        MultimodalEmbedding(input->multimodal_inputs.value(), ip_port, input->request_id, server_context));
    auto                       mm_features = std::move(mm_embedding_res.mm_features);
    std::vector<torch::Tensor> mm_extra_input;
    if (mm_embedding_res.mm_extra_input.has_value()) {
        mm_extra_input = std::move(mm_embedding_res.mm_extra_input.value());
    }
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(mm_features, input->input_ids, {}, mm_embedding_res.mm_feature_hashes));
    RETURN_IF_STATUS_ERROR(checkExpandLength(expanded_ids));
    RETURN_IF_STATUS_ERROR(pinMultimodalTensors(mm_features, "embedding"));
    RETURN_IF_STATUS_ERROR(pinMultimodalTensors(mm_extra_input, "extra input"));
    input->multimodal_features    = std::move(mm_features);
    input->mm_position_ids        = std::move(mm_embedding_res.mm_position_ids);
    input->mm_extra_input         = std::move(mm_extra_input);
    input->input_ids              = expanded_ids.expanded_ids;
    input->text_tokens_mask       = expanded_ids.text_tokens_mask;
    input->mm_locs                = expanded_ids.locs;
    input->multimodal_cache_spans = std::move(expanded_ids.cache_spans);
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
    CHECK_AND_RETURN_REF(
        expanded_ids,
        expandTokenIds(
            mm_features.features, input->token_ids, input->token_type_ids, mm_embedding_res.mm_feature_hashes));
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
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(mm_features.features, input_ids, {}, mm_embedding_res.mm_feature_hashes));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    return mm_features;
}

}  // namespace rtp_llm
