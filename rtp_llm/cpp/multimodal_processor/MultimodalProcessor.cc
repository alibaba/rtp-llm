#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/utils/Logger.h"

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
                                                                rtp_llm::BufferPtr                token_ids,
                                                                const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                                rtp_llm::BufferPtr token_type_ids) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids);
    }
    RTP_LLM_LOG_INFO("DEBUG: expandTokenIds called. token_ids size: %lu, mm_embedding size: %lu",
                     token_ids->shape()[0],
                     mm_embedding.size());

    std::vector<std::string> urls;
    for (auto& mm_input : mm_inputs) {
        urls.push_back(mm_input.url);
    }

    assert(token_ids->shape().size() == 1);
    int              expanded_len = token_ids->shape()[0];
    std::vector<int> embed_len    = {};
    CHECK_AND_RETURN_REF(locs, getMultimodalTags(token_ids));
    BufferPtr expanded_token_type_ids = nullptr;
    int       mm_num                  = mm_embedding.size();
    if (locs.size() != mm_num) {
        std::stringstream exception_str;
        exception_str << "number of multimodal tags and multimodal input not matched, expect " << locs.size()
                      << ", get " << mm_num;
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, exception_str.str());
    }
    RTP_LLM_LOG_INFO("DEBUG: expandTokenIds locs found: %lu", locs.size());
    for (int i = 0; i < mm_num; i++) {
        // mm embedding is supposed to be a tensor of [expand_len, hidden_dim]
        size_t emb_len = mm_embedding[i].sizes()[0];
        RTP_LLM_LOG_INFO("DEBUG: mm_embedding[%d] size: %lu", i, emb_len);
        expanded_len += emb_len - locs[i].second + locs[i].first;
    }
    RTP_LLM_LOG_INFO("DEBUG: expandTokenIds final expanded_len: %d", expanded_len);

    auto               device = rtp_llm::DeviceFactory::getDefaultDevice();
    rtp_llm::BufferPtr expanded_ids =
        device->allocateBuffer({token_ids->type(), {(size_t)expanded_len}, rtp_llm::AllocationType::HOST}, {});
    rtp_llm::BufferPtr token_masks =
        device->allocateBuffer({token_ids->type(), {(size_t)expanded_len}, rtp_llm::AllocationType::HOST}, {});
    rtp_llm::BufferPtr new_locs =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)mm_num}, rtp_llm::AllocationType::HOST}, {});
    memset(expanded_ids->data(), -1, expanded_ids->sizeBytes());
    std::fill(token_masks->data<int32_t>(), token_masks->dataWithOffset<int32_t>(token_masks->size()), 1);
    if (token_type_ids != nullptr) {
        expanded_token_type_ids =
            device->allocateBuffer({token_type_ids->type(), {(size_t)expanded_len}, rtp_llm::AllocationType::HOST}, {});
        std::fill(expanded_token_type_ids->data<int32_t>(),
                  expanded_token_type_ids->dataWithOffset<int32_t>(token_masks->size()),
                  0);
    }
    int  new_loc_idx = 0, old_loc_idx = 0;
    bool hash_urls = urls.size() == mm_num;
    for (int i = 0; i < mm_num; i++) {
        auto& loc      = locs[i];
        int   copy_len = loc.first - old_loc_idx;
        memcpy(expanded_ids->dataWithOffset<int32_t>(new_loc_idx),
               token_ids->dataWithOffset<int32_t>(old_loc_idx),
               token_ids->typeSize() * copy_len);
        memset(token_masks->dataWithOffset<int32_t>(new_loc_idx + copy_len),
               0,
               mm_embedding[i].sizes()[0] * token_masks->typeSize());
        if (token_type_ids != nullptr) {
            memcpy(expanded_token_type_ids->dataWithOffset<int32_t>(new_loc_idx),
                   token_type_ids->dataWithOffset<int32_t>(old_loc_idx),
                   token_type_ids->typeSize() * copy_len);
        }
        *(new_locs->dataWithOffset<int32_t>(i)) = copy_len + new_loc_idx;

        if (hash_urls) {
            auto hash_status = getStrHash(
                expanded_ids->dataWithOffset<int32_t>(new_loc_idx + copy_len), urls[i], mm_embedding[i].sizes()[0]);
            if (!hash_status.ok()) {
                return hash_status;
            }
        }

        new_loc_idx += copy_len + mm_embedding[i].sizes()[0];
        old_loc_idx = loc.second;
    }
    if (expanded_ids->shape()[0] - new_loc_idx != token_ids->shape()[0] - old_loc_idx) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "expanded length calculate error");
    }
    memcpy(expanded_ids->dataWithOffset<int32_t>(new_loc_idx),
           token_ids->dataWithOffset<int32_t>(old_loc_idx),
           token_ids->typeSize() * (expanded_ids->shape()[0] - new_loc_idx));
    if (token_type_ids != nullptr) {
        memcpy(expanded_token_type_ids->dataWithOffset<int32_t>(new_loc_idx),
               token_type_ids->dataWithOffset<int32_t>(old_loc_idx),
               token_type_ids->typeSize() * (expanded_ids->shape()[0] - new_loc_idx));
    }
    return ExpandedOutput(
        std::move(expanded_ids), std::move(expanded_token_type_ids), std::move(token_masks), std::move(new_locs));
}

ErrorResult<std::vector<std::pair<int32_t, int32_t>>>
MultimodalProcessor::getMultimodalTags(rtp_llm::BufferPtr token_ids) {
    // sep_tokens will split input tokens to text part and multimodal part
    int32_t*                                 data = token_ids->data<int32_t>();
    std::vector<std::pair<int32_t, int32_t>> locs;
    for (const auto& sep_token_id : sep_token_ids_) {
        // when sep token size == 1, just simply split
        if (sep_token_id.size() == 1) {
            for (int i = 0; i < token_ids->size(); i++) {
                auto now_id = *(data + i);
                if (now_id == sep_token_id[0]) {
                    locs.emplace_back(i, i + 1);
                }
            }
        }
        // when sep token size == 2, multimodal part is between but not include them
        else if (sep_token_id.size() == 2) {
            std::vector<int32_t> left, right;

            for (int i = 0; i < token_ids->size(); i++) {
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
    if (expand_output.expanded_ids->size() >= max_seq_len_) {
        std::stringstream exception_str;
        exception_str << "input after multimodal process is " << expand_output.expanded_ids->size() << " > max_seq_len("
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
    RTP_LLM_LOG_INFO("DEBUG: updateMultimodalFeatures (EmbeddingInput) called. mm_inputs size: %lu", mm_inputs.size());
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(mm_inputs, ""));
    MultimodalFeature mm_features;
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(mm_features.features, input->token_ids, mm_inputs, input->token_type_ids));
    RTP_LLM_LOG_INFO("DEBUG: updateMultimodalFeatures (EmbeddingInput) expansion done. New token_ids shape: %lu",
                     expanded_ids.expanded_ids->shape()[0]);
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    input->multimodal_features.emplace(mm_features);
    input->token_ids      = expanded_ids.expanded_ids;
    input->token_type_ids = expanded_ids.token_type_ids;
    if (input->input_lengths->shape().size() == 1 && input->input_lengths->shape()[0] == 1
        && expanded_ids.expanded_ids->shape().size() == 1) {
        input->input_lengths->data<int32_t>()[0] = expanded_ids.expanded_ids->shape()[0];
        input->total_length                      = expanded_ids.expanded_ids->shape()[0];
    }
    return ErrorInfo::OkStatus();
}

ErrorResult<MultimodalFeature>
MultimodalProcessor::getMultimodalFeatures(const rtp_llm::BufferPtr&           input_ids,
                                           const std::vector<MultimodalInput>& mm_inputs) {
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