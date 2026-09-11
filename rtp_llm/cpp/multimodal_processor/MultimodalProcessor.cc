#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace py = pybind11;

namespace rtp_llm {

ErrorInfo MultimodalProcessor::getStrHash(int32_t* token_ids, std::string& url, int mm_emb_len) {
    int url_len = url.length(), data_size_scale = std::max(int(sizeof(int32_t) / sizeof(int32_t)), 1);
    if (mm_emb_len / data_size_scale <= 0) {
        std::stringstream exception_str;
        exception_str << "length of multimodal input is too short, at least " << data_size_scale << ", get "
                      << mm_emb_len;
        return ErrorInfo(ErrorCode::MM_LONG_PROMPT_ERROR, exception_str.str());
    }
    int                    substr_len = (url_len - 1) / (mm_emb_len / data_size_scale) + 1;
    int                    now_idx    = 0;
    std::hash<std::string> hasher;
    while (now_idx * substr_len < url_len && now_idx * data_size_scale < mm_emb_len) {
        int32_t hash_res =
            hasher(url.substr(now_idx * substr_len, std::min(url_len - now_idx * substr_len, substr_len)));
        memcpy(token_ids + now_idx * data_size_scale, &hash_res, sizeof(int32_t));
        now_idx++;
    }
    return ErrorInfo::OkStatus();
}

ErrorResult<ExpandedOutput> MultimodalProcessor::expandTokenIds(const std::vector<torch::Tensor>& mm_embedding,
                                                                const torch::Tensor&              token_ids,
                                                                const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                                torch::Tensor token_type_ids) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids);
    }
    std::vector<std::string> urls;
    for (auto& mm_input : mm_inputs) {
        urls.push_back(mm_input.url);
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
    int  new_loc_idx = 0, old_loc_idx = 0;
    bool hash_urls = urls.size() == mm_num;
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

        if (hash_urls) {
            auto hash_status = getStrHash(
                expanded_ids.data_ptr<int32_t>() + new_loc_idx + copy_len, urls[i], mm_embedding[i].sizes()[0]);
            if (!hash_status.ok()) {
                return hash_status;
            }
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
    if (input->v41_inputs) {
        const auto& prepared = *input->v41_inputs;
        prepared.validate(input->input_ids);
        if (input->multimodal_inputs && !input->multimodal_inputs->empty()) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 prepared input cannot use URL token expansion");
        }
        if (input->generate_config) {
            for (const auto& role : input->generate_config->role_addrs) {
                if (role.role == RoleType::VIT) {
                    return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR,
                                     "V4.1 prepared-image remote ViT RPC is not connected");
                }
            }
        }
        std::vector<torch::Tensor> features;
        if (!prepared.images.empty()) {
            if (mm_process_engine_.is_none()) {
                return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "V4.1 prepared images require a local ViT engine");
            }
            try {
                py::gil_scoped_acquire acquire;
                py::list               images;
                for (const auto& image : prepared.images) {
                    py::dict record;
                    record["start"]              = image.start;
                    record["n_vit_h"]            = image.n_vit_h;
                    record["n_vit_w"]            = image.n_vit_w;
                    record["patches"]            = image.patches;
                    record["types"]              = image.types;
                    record["content_sha256"]     = image.content_sha256;
                    record["processor_identity"] = image.processor_identity;
                    images.append(std::move(record));
                }
                auto result = mm_process_engine_.attr("submit_v41")(images);
                for (auto item : result.attr("embeddings")) {
                    features.push_back(py::cast<torch::Tensor>(item));
                }
            } catch (const py::error_already_set& error) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
            }
        }
        RTP_LLM_CHECK_WITH_INFO(features.size() == prepared.images.size(),
                                "V4.1 requires exactly one full ViT result per image");
        auto locs = torch::empty({static_cast<int64_t>(features.size())}, torch::kInt32);
        for (size_t index = 0; index < features.size(); ++index) {
            RTP_LLM_CHECK_WITH_INFO(
                features[index].dim() == 2 && features[index].size(0) == prepared.images[index].types.numel()
                    && features[index].scalar_type() == torch::kBFloat16 && features[index].is_contiguous(),
                "V4.1 ViT result must preserve all image and delimiter rows");
            locs.data_ptr<int32_t>()[index] = prepared.images[index].start;
        }
        input->multimodal_features = std::move(features);
        input->text_tokens_mask    = prepared.image_mask.logical_not().to(torch::kInt32);
        input->mm_locs             = std::move(locs);
        input->mm_position_ids.reset();
        return ErrorInfo::OkStatus();
    }
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
                                                        const std::vector<rtp_llm::MultimodalInput>& mm_inputs) {
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(mm_inputs, ""));
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
