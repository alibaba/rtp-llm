
#include <functional>
#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace py = pybind11;

namespace rtp_llm {

ErrorInfo MultimodalProcessor::getFeatureHash(int32_t* token_ids, const torch::Tensor& mm_emb) {
    // Derive one cache-key hash per multimodal token from the content of its feature row.
    // This makes the prefix cache key reflect the actual image/video embedding, so only
    // identical content reuses cached blocks.
    //
    // We compute the hash on the GPU using lightweight per-row statistics and copy only
    // the resulting [num_tokens] int32 hashes to the host. This avoids blocking on a full
    // D2H transfer of the embedding tensor while still producing different keys for
    // different feature content.
    if (mm_emb.dim() < 1 || mm_emb.size(0) <= 0) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "multimodal feature tensor is empty");
    }
    const int64_t num_tokens = mm_emb.size(0);
    const int64_t row_elems  = mm_emb.numel() / num_tokens;

    auto flat     = mm_emb.reshape({num_tokens, row_elems}).to(torch::kFloat32);
    auto mean     = flat.mean(1, /*keepdim=*/true);
    auto centered = flat - mean;
    auto m1       = flat.sum(1);
    auto m2       = flat.pow(2).sum(1);
    auto m3       = centered.pow(3).sum(1);

    const int64_t max_int32 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    auto hash = ((m1 * 1000.0).to(torch::kInt64) + m2.to(torch::kInt64) + (m3 * 1000.0).to(torch::kInt64))
                    .remainder(max_int32)
                    .to(torch::kInt32);

    auto hash_cpu = hash.to(torch::kCPU);
    std::memcpy(token_ids, hash_cpu.data_ptr<int32_t>(), num_tokens * sizeof(int32_t));
    return ErrorInfo::OkStatus();
}

ErrorResult<ExpandedOutput> MultimodalProcessor::expandTokenIds(const std::vector<torch::Tensor>& mm_embedding,
                                                                const torch::Tensor&              token_ids,
                                                                const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                                torch::Tensor token_type_ids) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids);
    }

    assert(token_ids.dim() == 1);
    int              expanded_len = token_ids.size(0);
    std::vector<int> embed_len    = {};
    CHECK_AND_RETURN_REF(locs, getMultimodalTags(token_ids));
    torch::Tensor expanded_token_type_ids;
    int           mm_num = mm_embedding.size();
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

        auto hash_status = getFeatureHash(expanded_ids.data_ptr<int32_t>() + new_loc_idx + copy_len, mm_embedding[i]);
        if (!hash_status.ok()) {
            return hash_status;
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
    int32_t*                                 data = token_ids.data_ptr<int32_t>();
    std::vector<std::pair<int32_t, int32_t>> locs;
    auto                                     num_tokens = token_ids.numel();
    for (const auto& sep_token_id : sep_token_ids_) {
        if (sep_token_id.size() == 1) {
            for (int i = 0; i < num_tokens; i++) {
                auto now_id = *(data + i);
                if (now_id == sep_token_id[0]) {
                    locs.emplace_back(i, i + 1);
                }
            }
        } else if (sep_token_id.size() == 2) {
            std::vector<int32_t> left, right;

            for (int i = 0; i < num_tokens; i++) {
                auto now_id = *(data + i);
                if (now_id == sep_token_id[0]) {
                    if (right.size() != left.size()) {
                        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "unmatched multimodal tag pairs");
                    }
                    if (!include_sep_tokens_) {
                        left.emplace_back(i + 1);
                    } else {
                        left.emplace_back(i);
                    }
                } else if (now_id == sep_token_id[1]) {
                    if (!include_sep_tokens_) {
                        right.emplace_back(i);
                    } else {
                        right.emplace_back(i + 1);
                    }
                    if (right.size() != left.size()) {
                        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "unmatched multimodal tag pairs");
                    }
                }
            }
            if (left.size() != right.size()) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "unclosed multimodal tag pairs");
            }
            for (int i = 0; i < left.size(); i++) {
                locs.emplace_back(left[i], right[i]);
            }
        } else {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                             "more than 2 sep tokens or no sep tokens for multimodal model is not supported");
        }
    }
    std::sort(locs.begin(), locs.end());
    return locs;
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

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::GenerateInput>& input) {
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
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(input->multimodal_inputs.value(), ip_port));
    input->multimodal_features = std::move(mm_embedding_res.mm_features);
    input->mm_position_ids     = std::move(mm_embedding_res.mm_position_ids);
    input->mm_extra_input      = std::move(mm_embedding_res.mm_extra_input);
    CHECK_AND_RETURN_REF(
        expanded_ids,
        expandTokenIds(input->multimodal_features.value(), input->input_ids, input->multimodal_inputs.value()));
    RETURN_IF_STATUS_ERROR(checkExpandLength(expanded_ids));
    input->input_ids        = expanded_ids.expanded_ids;
    input->text_tokens_mask = expanded_ids.text_tokens_mask;
    input->mm_locs          = expanded_ids.locs;
    return ErrorInfo::OkStatus();
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::EmbeddingInput>&    input,
                                                        const std::vector<rtp_llm::MultimodalInput>& mm_inputs,
                                                        const std::string&                           vit_role_addr) {
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(mm_inputs, vit_role_addr));
    MultimodalFeature mm_features;
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(mm_features.features, input->token_ids, mm_inputs, input->token_type_ids));
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
    CHECK_AND_RETURN_REF(expanded_ids, expandTokenIds(mm_features.features, input_ids, mm_inputs));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    return mm_features;
}

}  // namespace rtp_llm
