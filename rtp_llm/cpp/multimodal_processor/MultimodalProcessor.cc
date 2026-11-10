
#include <functional>
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTokenUtils.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

#if USING_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

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

// Token-plan media may split one GPU output into many frames. Queue their
// host copies together and finish once per device before TP serialization.
ErrorInfo pinTokenPlanFeatures(std::vector<torch::Tensor>& tensors) {
#if USING_CUDA
    std::map<c10::DeviceIndex, c10::cuda::CUDAStream> streams;
    try {
        std::vector<torch::Tensor> pinned;
        pinned.reserve(tensors.size());
        // Keep the original GPU tensors alive until every queued copy finishes,
        // including when a later allocation/copy throws on another frame.
        for (const auto& tensor : tensors) {
            if (!tensor.defined() || tensor.is_pinned()) {
                pinned.push_back(tensor);
            } else if (tensor.is_cuda()) {
                streams.emplace(tensor.get_device(), c10::cuda::getCurrentCUDAStream(tensor.get_device()));
                auto options =
                    torch::TensorOptions().dtype(tensor.scalar_type()).device(torch::kCPU).pinned_memory(true);
                pinned.push_back(tensor.to(options, /*non_blocking=*/true));
            } else {
                pinned.push_back(tensor.pin_memory());
            }
        }
        for (const auto& [device, stream] : streams) {
            stream.synchronize();
        }
        tensors = std::move(pinned);
    } catch (const std::exception& error) {
        // The input vector still owns every GPU source. Drain already queued
        // copies before returning an error and releasing those owners.
        for (const auto& [device, stream] : streams) {
            try {
                stream.synchronize();
            } catch (const std::exception&) {
            }
        }
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                         std::string("failed to move token-plan embeddings to pinned CPU: ") + error.what());
    }
    return ErrorInfo::OkStatus();
#else
    return pinMultimodalTensors(tensors, "embedding");
#endif
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

ErrorResult<ExpandedOutput>
MultimodalProcessor::expandTokenPlans(const std::vector<torch::Tensor>&                features,
                                      const torch::Tensor&                             token_ids,
                                      const std::vector<MultimodalTokenPlan>&          plans,
                                      const std::optional<std::vector<torch::Tensor>>& feature_hashes) {
    CHECK_AND_RETURN_REF(markers, getMultimodalTags(token_ids));
    if (!include_sep_tokens_ || markers.size() != plans.size()
        || (feature_hashes && feature_hashes->size() != features.size())) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "token plans require complete, matching media markers");
    }
    int64_t size          = token_ids.numel();
    size_t  feature_index = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        const auto& plan = plans[i];
        size += static_cast<int64_t>(plan.token_ids.size()) - (markers[i].second - markers[i].first);
        int64_t previous_end = 0;
        for (const auto& [offset, length] : plan.spans) {
            if (offset < previous_end || length <= 0 || int64_t(offset) + length > plan.token_ids.size()
                || feature_index >= features.size() || features[feature_index].dim() != 2
                || features[feature_index].size(0) != length) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "token plan spans do not match embedding segments");
            }
            previous_end = int64_t(offset) + length;
            ++feature_index;
        }
        if (plan.spans.empty()
            || std::any_of(plan.token_ids.begin(), plan.token_ids.end(), [](int32_t id) { return id < 0; })) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                             "token plan requires vocabulary IDs and embedding spans");
        }
    }
    if (feature_index != features.size() || size <= 0 || size >= max_seq_len_) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid token plan feature count or expanded length");
    }
    auto    ids    = torch::empty({size}, torch::kInt32);
    auto    mask   = torch::ones({size}, torch::kInt32);
    auto    locs   = torch::empty({static_cast<int64_t>(features.size())}, torch::kInt32);
    int64_t source = 0, destination = 0;
    feature_index = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        const auto&   plan        = plans[i];
        const int64_t text_length = markers[i].first - source;
        std::copy_n(token_ids.data_ptr<int32_t>() + source, text_length, ids.data_ptr<int32_t>() + destination);
        destination += text_length;
        std::copy(plan.token_ids.begin(), plan.token_ids.end(), ids.data_ptr<int32_t>() + destination);
        for (const auto& [offset, length] : plan.spans) {
            auto* target = ids.data_ptr<int32_t>() + destination + offset;
            if (feature_hashes) {
                const auto& hashes = feature_hashes->at(feature_index);
                if (!hashes.device().is_cpu() || hashes.scalar_type() != torch::kInt32 || hashes.dim() != 1
                    || hashes.numel() != length) {
                    return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid token plan feature hashes");
                }
                auto contiguous = hashes.contiguous();
                std::copy_n(contiguous.data_ptr<int32_t>(), length, target);
            } else {
                RETURN_IF_STATUS_ERROR(getFeatureHash(target, features[feature_index]));
            }
            std::fill_n(mask.data_ptr<int32_t>() + destination + offset, length, 0);
            locs.data_ptr<int32_t>()[feature_index++] = destination + offset;
        }
        destination += plan.token_ids.size();
        source = markers[i].second;
    }
    std::copy_n(
        token_ids.data_ptr<int32_t>() + source, token_ids.numel() - source, ids.data_ptr<int32_t>() + destination);
    return ExpandedOutput(std::move(ids), {}, std::move(mask), std::move(locs));
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

ErrorResult<MultimodalOutput> MultimodalProcessor::V41MultimodalEmbedding(
    const V41RequestInputs& inputs, const std::string&, int64_t timeout_ms, int64_t, grpc::ServerContext*) {
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
        auto             result = mm_process_engine_.attr("submit_v41")(images, timeout_ms);
        MultimodalOutput output;
        for (auto item : result.attr("embeddings")) {
            output.mm_features.push_back(py::cast<torch::Tensor>(item));
        }
        return output;
    } catch (const py::error_already_set& error) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
    }
}

ErrorResult<ExpandedOutput>
MultimodalProcessor::useExpandedTokenLayout(const GenerateInput&                             input,
                                            const std::vector<torch::Tensor>&                features,
                                            const std::optional<std::vector<torch::Tensor>>& feature_hashes) {
    const auto& layout = input.multimodal_token_layout.value();
    const auto& ids    = input.input_ids;
    if (layout.spans.empty() || layout.spans.size() != features.size()
        || (feature_hashes.has_value() && feature_hashes->size() != features.size())) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid frontend multimodal token layout");
    }
    if (!ids.defined() || !ids.device().is_cpu() || ids.scalar_type() != torch::kInt32 || ids.dim() != 1
        || !ids.is_contiguous()) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid expanded multimodal token ids");
    }
    RETURN_IF_STATUS_ERROR(checkExpandLength(ExpandedOutput(ids)));
    int64_t                    previous_end   = 0;
    bool                       hashes_changed = false;
    std::vector<torch::Tensor> verified_hashes;
    verified_hashes.reserve(features.size());
    for (size_t i = 0; i < layout.spans.size(); ++i) {
        const auto [offset, length] = layout.spans[i];
        const int64_t end           = int64_t(offset) + length;
        if (offset < previous_end || length <= 0 || end > ids.numel() || !features[i].defined() || features[i].dim() < 1
            || features[i].size(0) != length) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "frontend multimodal spans do not match ViT output");
        }
        previous_end = end;
        // Metadata can outlive an embedding. Check the fetched result before
        // using frontend hashes as KV-cache keys; never publish mismatched data.
        torch::Tensor hashes;
        try {
            hashes = feature_hashes.has_value() ? feature_hashes->at(i) : getMultimodalFeatureHash(features[i]);
        } catch (const std::exception& error) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
        }
        if (!hashes.defined() || !hashes.device().is_cpu() || hashes.scalar_type() != torch::kInt32 || hashes.dim() != 1
            || hashes.numel() != length) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid ViT feature hashes for expanded input");
        }
        hashes = hashes.contiguous();
        if (memcmp(ids.data_ptr<int32_t>() + offset, hashes.data_ptr<int32_t>(), hashes.nbytes()) != 0) {
            hashes_changed = true;
        }
        verified_hashes.push_back(std::move(hashes));
    }
    // Recomputed embeddings can differ after cache eviction. Preserve all text
    // and offsets, but make actual KV-cache keys describe the fetched values.
    // Validate every segment before making changes. Matching hashes reuse the
    // input storage; changed hashes copy it without rescanning or splicing.
    auto expanded_ids = hashes_changed ? ids.clone() : ids;
    auto mask         = torch::ones({ids.numel()}, torch::kInt32);
    auto locs         = torch::empty({int64_t(layout.spans.size())}, torch::kInt32);
    for (size_t i = 0; i < layout.spans.size(); ++i) {
        const auto [offset, length] = layout.spans[i];
        locs.data_ptr<int32_t>()[i] = offset;
        std::fill(mask.data_ptr<int32_t>() + offset, mask.data_ptr<int32_t>() + offset + length, 0);
        if (hashes_changed) {
            memcpy(expanded_ids.data_ptr<int32_t>() + offset,
                   verified_hashes[i].data_ptr<int32_t>(),
                   verified_hashes[i].nbytes());
        }
    }
    return ExpandedOutput(std::move(expanded_ids), {}, std::move(mask), std::move(locs));
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
            int64_t timeout_ms = 0;
            if (input->generate_config && input->generate_config->timeout_ms > 0) {
                const auto elapsed_ms = input->begin_time_us > 0 ? (currentTimeUs() - input->begin_time_us) / 1000 : 0;
                timeout_ms            = input->generate_config->timeout_ms - elapsed_ms;
                if (timeout_ms <= 0) {
                    return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "V4.1 ViT request deadline exceeded");
                }
            }
            CHECK_AND_RETURN_REF(
                result, V41MultimodalEmbedding(prepared, ip_port, timeout_ms, input->request_id, server_context));
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
    CHECK_AND_RETURN_REF(
        mm_embedding_res,
        MultimodalEmbedding(input->multimodal_inputs.value(), ip_port, input->request_id, server_context));
    if (!mm_embedding_res.token_plans.empty()
        && mm_embedding_res.token_plans.size() != input->multimodal_inputs->size()) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "token plan count must match media inputs");
    }
    auto                       mm_features = std::move(mm_embedding_res.mm_features);
    std::vector<torch::Tensor> mm_extra_input;
    if (mm_embedding_res.mm_extra_input.has_value()) {
        mm_extra_input = std::move(mm_embedding_res.mm_extra_input.value());
    }
    CHECK_AND_RETURN_REF(
        expanded_ids,
        !mm_embedding_res.token_plans.empty() ?
            expandTokenPlans(
                mm_features, input->input_ids, mm_embedding_res.token_plans, mm_embedding_res.mm_feature_hashes) :
        input->multimodal_token_layout.has_value() ?
            useExpandedTokenLayout(*input, mm_features, mm_embedding_res.mm_feature_hashes) :
            expandTokenIds(mm_features,
                           input->input_ids,
                           input->multimodal_inputs.value(),
                           {},
                           mm_embedding_res.mm_feature_hashes));
    RETURN_IF_STATUS_ERROR(checkExpandLength(expanded_ids));
    RETURN_IF_STATUS_ERROR(mm_embedding_res.token_plans.empty() ? pinMultimodalTensors(mm_features, "embedding") :
                                                                   pinTokenPlanFeatures(mm_features));
    RETURN_IF_STATUS_ERROR(pinMultimodalTensors(mm_extra_input, "extra input"));
    for (size_t i = 0; i < mm_embedding_res.token_plans.size(); ++i) {
        for (const auto& span : mm_embedding_res.token_plans[i].spans) {
            input->multimodal_feature_lengths[input->multimodal_inputs->at(i).mm_type] += span.second;
        }
    }
    input->multimodal_features = std::move(mm_features);
    input->mm_position_ids     = std::move(mm_embedding_res.mm_position_ids);
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
