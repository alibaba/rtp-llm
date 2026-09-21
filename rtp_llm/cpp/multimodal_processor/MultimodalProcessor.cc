#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"

namespace py = pybind11;

namespace rtp_llm {

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

        // A ViT row depends on the complete image. URL fragments cannot
        // identify an image prefix safely when reusing KV inside its block.
        try {
            auto hashes = getMultimodalFeatureHash(mm_embedding[i]);
            memcpy(
                expanded_ids.data_ptr<int32_t>() + new_loc_idx + copy_len, hashes.data_ptr<int32_t>(), hashes.nbytes());
        } catch (const std::exception& error) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
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

ErrorResult<std::vector<rtp_llm::MultimodalInput>>
MultimodalProcessor::setMMPaddingSize(const torch::Tensor&                         token_ids,
                                      const std::vector<rtp_llm::MultimodalInput>& mm_inputs) {
    if (padding_size_ == 0) {
        return std::vector<rtp_llm::MultimodalInput>(mm_inputs);
    }
    CHECK_AND_RETURN_REF(locs, getMultimodalTags(token_ids));
    if (locs.size() != mm_inputs.size()) {
        std::stringstream exception_str;
        exception_str << "number of multimodal tags and multimodal input not matched, expect " << locs.size()
                      << ", get " << mm_inputs.size();
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, exception_str.str());
    }

    auto padded_inputs = mm_inputs;
    for (size_t i = 0; i < padded_inputs.size(); ++i) {
        int32_t phase = locs[i].first;
        if (i > 0) {
            RTP_LLM_CHECK_WITH_INFO(
                locs[i].first >= locs[i - 1].second,
                "multimodal tag ranges must be sorted and non-overlapping: previous_end=%d, start=%d",
                locs[i - 1].second,
                locs[i].first);
            // The image body is alignment-padded and followed by one end token.
            // Its successor is at offset 1; only the text gap remains to account for.
            phase = 1 + locs[i].first - locs[i - 1].second;
        }
        padded_inputs[i].mm_preprocess_config.mm_padding_size = padding_size_ - 1 - phase % padding_size_;
    }
    return padded_inputs;
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

ErrorResult<MultimodalOutput>
MultimodalProcessor::V41MultimodalEmbedding(const V41RequestInputs& inputs, const std::string&, grpc::ClientContext*) {
    py::gil_scoped_acquire acquire;
    if (mm_process_engine_.is_none()) {
        return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "V4.1 prepared images require a local ViT engine");
    }
    try {
        py::list images;
        for (const auto& image : inputs.images) {
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
        auto             result = mm_process_engine_.attr("submit_v41")(images);
        MultimodalOutput output;
        for (auto item : result.attr("embeddings")) {
            output.mm_features.push_back(py::cast<torch::Tensor>(item));
        }
        return output;
    } catch (const py::error_already_set& error) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
    }
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::GenerateInput>& input,
                                                       grpc::ClientContext* rpc_context) {
    grpc::ClientContext local_context;
    if (!rpc_context) {
        rpc_context = &local_context;
    }
    if (input->generate_config && input->generate_config->timeout_ms > 0) {
        const auto begin = input->begin_time_us > 0 ?
                               std::chrono::system_clock::time_point(std::chrono::microseconds(input->begin_time_us)) :
                               std::chrono::system_clock::now();
        rpc_context->set_deadline(
            std::min(rpc_context->deadline(), begin + std::chrono::milliseconds(input->generate_config->timeout_ms)));
    }
    std::string ip_port;
    if (input->generate_config) {
        for (const auto& role_addr : input->generate_config->role_addrs) {
            if (role_addr.role == RoleType::VIT) {
                ip_port = role_addr.ip + ":" + std::to_string(role_addr.grpc_port);
                break;
            }
        }
    }
    if (input->v41_inputs) {
        const auto& prepared   = *input->v41_inputs;
        const auto  cpu_vector = [](const torch::Tensor& tensor, torch::ScalarType dtype) {
            return tensor.defined() && tensor.device().is_cpu() && tensor.dim() == 1 && tensor.scalar_type() == dtype;
        };
        if (!cpu_vector(input->input_ids, torch::kInt32) || !cpu_vector(prepared.token_types, torch::kInt32)
            || !cpu_vector(prepared.image_mask, torch::kBool)
            || prepared.token_types.numel() != input->input_ids.numel()
            || prepared.image_mask.numel() != input->input_ids.numel()) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 token metadata must match the input token vector");
        }
        const int64_t token_count = input->input_ids.numel();
        if (token_count >= max_seq_len_) {
            return ErrorInfo(ErrorCode::MM_LONG_PROMPT_ERROR, "V4.1 prepared input exceeds the model sequence length");
        }
        auto        token_types = prepared.token_types.contiguous();
        auto        image_mask  = prepared.image_mask.contiguous();
        const auto* kinds       = token_types.data_ptr<int32_t>();
        const auto* mask        = image_mask.data_ptr<bool>();
        for (int64_t i = 0; i < token_count; ++i) {
            if (kinds[i] < -1 || kinds[i] > 3 || mask[i] != (kinds[i] != -1)) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "Invalid V4.1 token type or image mask");
            }
        }
        int64_t previous_end = 0;
        for (const auto& image : prepared.images) {
            if (!cpu_vector(image.types, torch::kInt32) || image.types.numel() == 0 || image.start < previous_end
                || image.types.numel() > token_count || image.start > token_count - image.types.numel()) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "Invalid V4.1 image span");
            }
            for (int64_t i = previous_end; i < image.start; ++i) {
                if (kinds[i] != -1) {
                    return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 image spans do not cover token metadata");
                }
            }
            auto        image_types = image.types.contiguous();
            const auto* image_kinds = image_types.data_ptr<int32_t>();
            for (int64_t i = 0; i < image_types.numel(); ++i) {
                if (image_kinds[i] < 0 || image_kinds[i] != kinds[image.start + i]) {
                    return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 image types do not match token metadata");
                }
            }
            previous_end = image.start + image_types.numel();
        }
        for (int64_t i = previous_end; i < token_count; ++i) {
            if (kinds[i] != -1) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 image spans do not cover token metadata");
            }
        }
        std::vector<torch::Tensor> features;
        if (!prepared.images.empty()) {
            CHECK_AND_RETURN_REF(result, V41MultimodalEmbedding(prepared, ip_port, rpc_context));
            features = std::move(result.mm_features);
        }
        if (features.size() != prepared.images.size()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "V4.1 ViT returned an unexpected image count");
        }
        auto locs = torch::empty({static_cast<int64_t>(features.size())}, torch::kInt32);
        for (size_t index = 0; index < features.size(); ++index) {
            if (features[index].dim() != 2 || features[index].size(0) != prepared.images[index].types.numel()) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "V4.1 ViT returned an invalid image span");
            }
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
    CHECK_AND_RETURN_REF(padded_inputs, setMMPaddingSize(input->input_ids, input->multimodal_inputs.value()));
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(padded_inputs, ip_port, rpc_context));
    input->multimodal_features = std::move(mm_embedding_res.mm_features);
    input->mm_position_ids     = std::move(mm_embedding_res.mm_position_ids);
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(input->multimodal_features.value(), input->input_ids, padded_inputs));
    RETURN_IF_STATUS_ERROR(checkExpandLength(expanded_ids));
    input->input_ids        = expanded_ids.expanded_ids;
    input->text_tokens_mask = expanded_ids.text_tokens_mask;
    input->mm_locs          = expanded_ids.locs;
    return ErrorInfo::OkStatus();
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::EmbeddingInput>&    input,
                                                        const std::vector<rtp_llm::MultimodalInput>& mm_inputs) {
    CHECK_AND_RETURN_REF(padded_inputs, setMMPaddingSize(input->token_ids, mm_inputs));
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(padded_inputs, ""));
    MultimodalFeature mm_features;
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(mm_features.features, input->token_ids, padded_inputs, input->token_type_ids));
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
    CHECK_AND_RETURN_REF(padded_inputs, setMMPaddingSize(input_ids, mm_inputs));
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(padded_inputs));
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(expanded_ids, expandTokenIds(mm_features.features, input_ids, padded_inputs));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    return mm_features;
}

}  // namespace rtp_llm
