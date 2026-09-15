
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
MultimodalProcessor::expandTokenIds(const std::vector<torch::Tensor>&               mm_embedding,
                                    const torch::Tensor&                            token_ids,
                                    const std::vector<rtp_llm::MultimodalInput>     mm_inputs,
                                    torch::Tensor                                   token_type_ids,
                                    const std::vector<std::pair<int32_t, int32_t>>* resolved_locs) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids);
    }

    assert(token_ids.dim() == 1);
    int                                      expanded_len = token_ids.size(0);
    std::vector<int>                         embed_len    = {};
    std::vector<std::pair<int32_t, int32_t>> locs;
    if (resolved_locs) {
        locs = *resolved_locs;
    } else {
        CHECK_AND_RETURN_REF(parsed_locs, getMultimodalTags(token_ids));
        locs = std::move(parsed_locs);
    }
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

ErrorResult<ExpandedOutput> MultimodalProcessor::expandTokenIdsWithLayout(MultimodalOutput&                   output,
                                                                          const torch::Tensor&                token_ids,
                                                                          const std::vector<MultimodalInput>& mm_inputs,
                                                                          torch::Tensor token_type_ids) {
    if (output.mm_token_layouts.empty()) {
        return expandTokenIds(output.mm_features, token_ids, mm_inputs, token_type_ids);
    }
    try {
        CHECK_AND_RETURN_REF(locs, getMultimodalTags(token_ids));
        const size_t count     = output.mm_features.size();
        auto         malformed = [](const std::string& message) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "multimodal token layout: " + message);
        };
        if (locs.size() != count || output.mm_token_layouts.size() != count
            || (output.mm_position_ids && output.mm_position_ids->size() != count)
            || (output.mm_extra_input && output.mm_extra_input->size() != count)) {
            return malformed("media count mismatch");
        }
        std::vector<int32_t>                     ids, types;
        std::vector<std::pair<int32_t, int32_t>> compact_locs;
        MultimodalOutput                         split;
        if (output.mm_position_ids)
            split.mm_position_ids.emplace();
        if (output.mm_extra_input)
            split.mm_extra_input.emplace();
        auto        source   = token_ids.contiguous();
        const auto* original = source.data_ptr<int32_t>();
        if (token_type_ids.defined())
            token_type_ids = token_type_ids.contiguous();
        auto append_original = [&](int64_t begin, int64_t end) {
            ids.insert(ids.end(), original + begin, original + end);
            if (token_type_ids.defined()) {
                auto* ptr = token_type_ids.data_ptr<int32_t>();
                types.insert(types.end(), ptr + begin, ptr + end);
            }
        };
        int64_t cursor = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto& layout_tensor = output.mm_token_layouts[i];
            const auto& features      = output.mm_features[i];
            if (!layout_tensor.defined() || layout_tensor.numel() == 0) {
                const int32_t start = ids.size() + locs[i].first - cursor;
                compact_locs.emplace_back(start, start + locs[i].second - locs[i].first);
                append_original(cursor, locs[i].second);
                cursor = locs[i].second;
                split.mm_features.push_back(features);
                if (mm_inputs.size() == count)
                    split.mm_feature_types.push_back(mm_inputs[i].mm_type);
                if (split.mm_position_ids)
                    split.mm_position_ids->push_back(output.mm_position_ids->at(i));
                if (split.mm_extra_input)
                    split.mm_extra_input->push_back(output.mm_extra_input->at(i));
                continue;
            }
            if (include_sep_tokens_ || locs[i].first <= cursor || locs[i].second >= source.numel()
                || layout_tensor.dim() != 1 || layout_tensor.scalar_type() != torch::kInt32) {
                return malformed("replacement requires int32 layout and enclosing vision tags");
            }
            const int64_t begin = locs[i].first - 1, end = locs[i].second + 1;
            const int32_t start_tag = original[begin], end_tag = original[end - 1];
            bool          paired = false;
            for (const auto& tags : sep_token_ids_) {
                paired |= tags.size() == 2 && tags[0] == start_tag && tags[1] == end_tag;
            }
            if (!paired || features.dim() < 1 || features.size(0) <= 0) {
                return malformed("invalid vision tags or features");
            }
            if (output.mm_extra_input && output.mm_extra_input->at(i).numel() != 0) {
                return malformed("per-frame replacement with opaque extra input is unsupported");
            }
            if (output.mm_position_ids && output.mm_position_ids->at(i).size(0) != features.size(0)) {
                return malformed("position count differs from feature count");
            }
            // vLLM replaces the video placeholder inside the original outer
            // vision pair, retaining that pair as text around timestamped frames.
            append_original(cursor, locs[i].first);
            auto        layout         = layout_tensor.to(torch::kCPU).contiguous();
            const auto* values         = layout.data_ptr<int32_t>();
            int64_t     feature_offset = 0;
            for (int64_t j = 0; j < layout.numel(); ++j) {
                const int32_t value = values[j];
                if (value >= 0) {
                    ids.push_back(value);
                } else {
                    const int64_t length = -static_cast<int64_t>(value);
                    if (j == 0 || j + 1 == layout.numel() || values[j - 1] != start_tag || values[j + 1] != end_tag
                        || length > features.size(0) - feature_offset) {
                        return malformed("invalid visual span");
                    }
                    // Explicit spans distinguish frame features from the nested
                    // outer vision tags, which remain ordinary text tokens.
                    compact_locs.emplace_back(ids.size(), ids.size() + 1);
                    ids.push_back(-1);
                    split.mm_features.push_back(features.narrow(0, feature_offset, length));
                    if (mm_inputs.size() == count)
                        split.mm_feature_types.push_back(mm_inputs[i].mm_type);
                    if (split.mm_position_ids) {
                        split.mm_position_ids->push_back(
                            output.mm_position_ids->at(i).narrow(0, feature_offset, length));
                    }
                    if (split.mm_extra_input)
                        split.mm_extra_input->push_back(output.mm_extra_input->at(i));
                    feature_offset += length;
                }
                if (token_type_ids.defined())
                    types.push_back(token_type_ids.data_ptr<int32_t>()[begin]);
            }
            if (feature_offset != features.size(0))
                return malformed("visual spans do not cover features");
            cursor = locs[i].second;
        }
        append_original(cursor, source.numel());
        auto compact_ids   = torch::tensor(ids, torch::kInt32);
        auto compact_types = token_type_ids.defined() ? torch::tensor(types, torch::kInt32) : torch::Tensor();
        CHECK_AND_RETURN_REF(expanded,
                             expandTokenIds(split.mm_features, compact_ids, mm_inputs, compact_types, &compact_locs));
        output = std::move(split);
        return std::move(expanded);
    } catch (const std::exception& e) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, std::string("invalid multimodal token layout: ") + e.what());
    }
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
    CHECK_AND_RETURN_REF(
        expanded_ids, expandTokenIdsWithLayout(mm_embedding_res, input->input_ids, input->multimodal_inputs.value()));
    input->multimodal_features      = std::move(mm_embedding_res.mm_features);
    input->multimodal_feature_types = std::move(mm_embedding_res.mm_feature_types);
    input->mm_position_ids          = std::move(mm_embedding_res.mm_position_ids);
    input->mm_extra_input           = std::move(mm_embedding_res.mm_extra_input);
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
    CHECK_AND_RETURN_REF(
        expanded_ids, expandTokenIdsWithLayout(mm_embedding_res, input->token_ids, mm_inputs, input->token_type_ids));
    mm_features.features         = std::move(mm_embedding_res.mm_features);
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
    CHECK_AND_RETURN_REF(expanded_ids, expandTokenIdsWithLayout(mm_embedding_res, input_ids, mm_inputs));
    mm_features.features         = std::move(mm_embedding_res.mm_features);
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    return mm_features;
}

}  // namespace rtp_llm
