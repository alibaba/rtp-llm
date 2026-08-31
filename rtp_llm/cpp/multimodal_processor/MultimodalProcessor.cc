
#include <functional>
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace py = pybind11;

namespace rtp_llm {

namespace {

constexpr int32_t kGlm53InterleavedLayoutMagic = -53530053;

struct InterleavedFeatureLayout {
    bool                 group_start = false;
    std::vector<int32_t> prefix_ids;
    std::vector<int32_t> suffix_ids;
};

}  // namespace

ErrorInfo MultimodalProcessor::getFeatureHash(int32_t* token_ids, const torch::Tensor& mm_emb) {
    // Derive one cache-key hash per multimodal token from the content of its feature row.
    // This makes the prefix cache key reflect the actual image/video embedding, so only
    // identical content reuses cached blocks.
    //
    // NOTE on the GPU->CPU sync below: hashing must inspect every byte of the embedding,
    // so we have to materialize it on the host. This is a deliberate blocking step on the
    // prefill-prep path (NOT the decode hot path). Without it the cache key would either
    // (a) require a GPU hash kernel — adds significant complexity for the marginal benefit
    // of avoiding one extra prefill-time D2H, or (b) fall back to URL-based hashing, which
    // would over-share cache blocks between requests whose URLs match but whose actual
    // embedding bytes differ (e.g. dynamic image transforms). Keep this sync.
    if (mm_emb.dim() < 1 || mm_emb.size(0) <= 0) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "multimodal feature tensor is empty");
    }
    auto          emb        = mm_emb.to(torch::kCPU).contiguous();
    const int64_t num_tokens = emb.size(0);
    const int64_t row_bytes  = emb.numel() / num_tokens * emb.element_size();
    const char*   base       = static_cast<const char*>(emb.data_ptr());

    std::hash<std::string_view> hasher;
    for (int64_t j = 0; j < num_tokens; ++j) {
        std::string_view row(base + j * row_bytes, static_cast<size_t>(row_bytes));
        int32_t          hash_res = static_cast<int32_t>(hasher(row));
        memcpy(token_ids + j, &hash_res, sizeof(int32_t));
    }
    return ErrorInfo::OkStatus();
}

ErrorResult<ExpandedOutput>
MultimodalProcessor::expandTokenIds(const std::vector<torch::Tensor>&                mm_embedding,
                                    const torch::Tensor&                             token_ids,
                                    const std::vector<rtp_llm::MultimodalInput>      mm_inputs,
                                    torch::Tensor                                    token_type_ids,
                                    const std::optional<std::vector<torch::Tensor>>& mm_extra_input) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids, {}, {}, mm_inputs);
    }

    assert(token_ids.dim() == 1);
    CHECK_AND_RETURN_REF(locs, getMultimodalTags(token_ids));

    bool has_interleaved_layout = false;
    if (mm_extra_input.has_value() && !mm_extra_input->empty()) {
        const auto& first = mm_extra_input->front();
        if (first.defined() && first.numel() > 0) {
            auto first_cpu         = first.to(torch::kInt32).cpu().contiguous();
            has_interleaved_layout = first_cpu.data_ptr<int32_t>()[0] == kGlm53InterleavedLayoutMagic;
        }
    }

    if (has_interleaved_layout) {
        if (mm_extra_input->size() != mm_embedding.size()) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                             "interleaved multimodal layout count must match feature count");
        }

        std::vector<InterleavedFeatureLayout> layouts;
        std::vector<int32_t>                  feature_groups;
        layouts.reserve(mm_embedding.size());
        feature_groups.reserve(mm_embedding.size());
        int32_t group_index  = -1;
        int64_t expanded_len = token_ids.size(0);
        for (size_t i = 0; i < mm_extra_input->size(); ++i) {
            auto layout = mm_extra_input->at(i).to(torch::kInt32).cpu().contiguous();
            if (layout.dim() != 1 || layout.numel() < 4) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                                 "interleaved multimodal layout must be a flat tensor with a four-value header");
            }
            const auto* values = layout.data_ptr<int32_t>();
            if (values[0] != kGlm53InterleavedLayoutMagic || (values[1] != 0 && values[1] != 1) || values[2] < 0
                || values[3] < 0 || layout.numel() != 4L + values[2] + values[3]) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid interleaved multimodal layout header");
            }
            if (values[1] == 1) {
                ++group_index;
            } else if (group_index < 0) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                                 "interleaved multimodal layout must start a media group");
            }

            InterleavedFeatureLayout parsed;
            parsed.group_start = values[1] == 1;
            parsed.prefix_ids.assign(values + 4, values + 4 + values[2]);
            parsed.suffix_ids.assign(values + 4 + values[2], values + layout.numel());
            layouts.emplace_back(std::move(parsed));
            feature_groups.push_back(group_index);

            if (mm_embedding[i].dim() < 1 || mm_embedding[i].size(0) <= 0) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                                 "interleaved multimodal feature tensor must not be empty");
            }
            expanded_len += mm_embedding[i].size(0) + values[2] + values[3];
        }

        const int64_t group_count = static_cast<int64_t>(group_index) + 1;
        if (group_count != static_cast<int64_t>(mm_inputs.size()) || group_count != static_cast<int64_t>(locs.size())) {
            std::stringstream exception_str;
            exception_str << "interleaved multimodal media groups, tags and inputs must match: groups=" << group_count
                          << ", tags=" << locs.size() << ", inputs=" << mm_inputs.size();
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, exception_str.str());
        }
        for (const auto& loc : locs) {
            expanded_len -= loc.second - loc.first;
        }

        auto expanded_ids = torch::empty({expanded_len}, torch::kInt32);
        auto token_masks  = torch::empty({expanded_len}, torch::kInt32);
        auto new_locs     = torch::empty({static_cast<int64_t>(mm_embedding.size())}, torch::kInt32);
        memset(expanded_ids.data_ptr(), -1, expanded_ids.nbytes());
        std::fill(token_masks.data_ptr<int32_t>(), token_masks.data_ptr<int32_t>() + token_masks.numel(), 1);

        torch::Tensor expanded_token_type_ids;
        if (token_type_ids.defined()) {
            expanded_token_type_ids = torch::zeros({expanded_len}, torch::kInt32);
        }

        std::vector<MultimodalInput> expanded_mm_inputs;
        expanded_mm_inputs.reserve(mm_embedding.size());
        int64_t new_index     = 0;
        int64_t old_index     = 0;
        size_t  feature_index = 0;
        for (size_t input_index = 0; input_index < mm_inputs.size(); ++input_index) {
            const auto&   loc      = locs[input_index];
            const int64_t copy_len = loc.first - old_index;
            if (copy_len < 0) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                                 "interleaved multimodal tags are not in document order");
            }
            if (copy_len > 0) {
                memcpy(expanded_ids.data_ptr<int32_t>() + new_index,
                       token_ids.data_ptr<int32_t>() + old_index,
                       sizeof(int32_t) * copy_len);
                if (token_type_ids.defined()) {
                    memcpy(expanded_token_type_ids.data_ptr<int32_t>() + new_index,
                           token_type_ids.data_ptr<int32_t>() + old_index,
                           sizeof(int32_t) * copy_len);
                }
            }
            new_index += copy_len;
            old_index = loc.second;

            while (feature_index < mm_embedding.size()
                   && feature_groups[feature_index] == static_cast<int32_t>(input_index)) {
                const auto append_text_ids = [&](const std::vector<int32_t>& ids) {
                    if (!ids.empty()) {
                        memcpy(expanded_ids.data_ptr<int32_t>() + new_index, ids.data(), sizeof(int32_t) * ids.size());
                        new_index += ids.size();
                    }
                };
                append_text_ids(layouts[feature_index].prefix_ids);
                new_locs.data_ptr<int32_t>()[feature_index] = static_cast<int32_t>(new_index);
                auto hash_status =
                    getFeatureHash(expanded_ids.data_ptr<int32_t>() + new_index, mm_embedding[feature_index]);
                if (!hash_status.ok()) {
                    return hash_status;
                }
                memset(token_masks.data_ptr<int32_t>() + new_index,
                       0,
                       mm_embedding[feature_index].size(0) * sizeof(int32_t));
                new_index += mm_embedding[feature_index].size(0);
                append_text_ids(layouts[feature_index].suffix_ids);
                expanded_mm_inputs.emplace_back(mm_inputs[input_index]);
                ++feature_index;
            }
        }

        const int64_t tail_len = token_ids.size(0) - old_index;
        if (tail_len > 0) {
            memcpy(expanded_ids.data_ptr<int32_t>() + new_index,
                   token_ids.data_ptr<int32_t>() + old_index,
                   sizeof(int32_t) * tail_len);
            if (token_type_ids.defined()) {
                memcpy(expanded_token_type_ids.data_ptr<int32_t>() + new_index,
                       token_type_ids.data_ptr<int32_t>() + old_index,
                       sizeof(int32_t) * tail_len);
            }
        }
        new_index += tail_len;
        if (feature_index != mm_embedding.size() || new_index != expanded_len) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "interleaved multimodal expansion length mismatch");
        }
        return ExpandedOutput(std::move(expanded_ids),
                              std::move(expanded_token_type_ids),
                              std::move(token_masks),
                              std::move(new_locs),
                              std::move(expanded_mm_inputs),
                              true);
    }

    int           expanded_len = token_ids.size(0);
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
    return ExpandedOutput(std::move(expanded_ids),
                          std::move(expanded_token_type_ids),
                          std::move(token_masks),
                          std::move(new_locs),
                          mm_inputs);
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
    input->multimodal_features = std::move(mm_embedding_res.mm_features);
    input->mm_position_ids     = std::move(mm_embedding_res.mm_position_ids);
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(input->multimodal_features.value(),
                                        input->input_ids,
                                        input->multimodal_inputs.value(),
                                        {},
                                        mm_embedding_res.mm_extra_input));
    RETURN_IF_STATUS_ERROR(checkExpandLength(expanded_ids));
    input->input_ids         = expanded_ids.expanded_ids;
    input->text_tokens_mask  = expanded_ids.text_tokens_mask;
    input->mm_locs           = expanded_ids.locs;
    input->multimodal_inputs = std::move(expanded_ids.multimodal_inputs);
    if (expanded_ids.consumed_mm_layout) {
        input->mm_extra_input.reset();
    } else {
        input->mm_extra_input = std::move(mm_embedding_res.mm_extra_input);
    }
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
            mm_features.features, input->token_ids, mm_inputs, input->token_type_ids, mm_embedding_res.mm_extra_input));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    mm_features.inputs           = std::move(expanded_ids.multimodal_inputs);
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
        expanded_ids, expandTokenIds(mm_features.features, input_ids, mm_inputs, {}, mm_embedding_res.mm_extra_input));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    mm_features.inputs           = std::move(expanded_ids.multimodal_inputs);
    return mm_features;
}

}  // namespace rtp_llm
