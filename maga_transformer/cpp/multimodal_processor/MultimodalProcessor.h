#pragma once

#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace ft = fastertransformer;
namespace py = pybind11;

namespace rtp_llm {

struct ExpandedOutput {
    ft::BufferPtr expanded_ids;
    ft::BufferPtr text_tokens_mask;
    ft::BufferPtr locs;
    ExpandedOutput(ft::BufferPtr expanded_ids, ft::BufferPtr text_tokens_mask = nullptr, ft::BufferPtr locs = nullptr):
        expanded_ids(expanded_ids), text_tokens_mask(text_tokens_mask), locs(locs) {}
};

struct MMEmbeddingRes {
    std::vector<torch::Tensor> mm_features = {};
    std::optional<std::vector<ft::BufferPtr>> mm_position_ids = std::nullopt;
};

class MultimodalProcessor {
public:
        MultimodalProcessor(py::object mm_proces_engine, std::vector<std::vector<int64_t>> sep_token_ids, bool include_sep_tokens, int64_t max_seq_len):
        mm_process_engine_(mm_proces_engine), sep_token_ids_(sep_token_ids), include_sep_tokens_(include_sep_tokens), max_seq_len_(max_seq_len)
        {}

private:
    py::object mm_process_engine_;
    std::vector<std::vector<int64_t>> sep_token_ids_;
    bool include_sep_tokens_;
    int64_t max_seq_len_;

    absl::Status get_str_hash(int32_t* token_ids, std::string& url, int mm_emb_len) {
        int url_len = url.length(), data_size_scale = std::max(int(sizeof(size_t) / sizeof(int32_t)), 1);
        if (mm_emb_len / data_size_scale <= 0) {
            return absl::InternalError("length of multimodal input is too short");
        }
        int substr_len = (url_len - 1) / (mm_emb_len / data_size_scale) + 1;
        int now_idx = 0;
        std::hash<std::string> hasher;
        while (now_idx * substr_len < url_len && now_idx * data_size_scale < mm_emb_len) {
            size_t hash_res = hasher(url.substr(now_idx * substr_len, std::min(url_len - now_idx * substr_len, substr_len)));
            memcpy(token_ids + now_idx * data_size_scale, &hash_res, sizeof(size_t));
            now_idx++;
        }
        return absl::OkStatus();
    }

    virtual absl::StatusOr<MMEmbeddingRes> mm_embedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs) {
        if (mm_inputs.size() == 0) {
            return MMEmbeddingRes();
        } else if (!mm_process_engine_.is_none()) {
            std::vector<std::string> urls;
            std::vector<int32_t> types;
            for (auto& mm_input: mm_inputs) {
                urls.push_back(mm_input.url);
                types.push_back(mm_input.mm_type);
            }
            try {
                py::gil_scoped_acquire acquire;
                auto res = mm_process_engine_.attr("submit")(urls, types);
                auto mm_embedding_vec = ft::convertPyObjectToVec(res.attr("embeddings"));

                MMEmbeddingRes mm_embedding_res;
                std::vector<torch::Tensor> mm_features;
                for (auto& emb: mm_embedding_vec) {
                    mm_features.emplace_back(ft::convertPyObjectToTensor(emb));
                }
                mm_embedding_res.mm_features = mm_features;
                auto position_id_vec = res.attr("position_ids");
                std::vector<ft::BufferPtr> position_ids;
                if (!position_id_vec.is_none()) {
                    for (auto& position_id: ft::convertPyObjectToVec(position_id_vec)) {
                        auto pos = ft::torchTensor2Buffer(ft::convertPyObjectToTensor(position_id));
                        position_ids.emplace_back(pos);
                    }
                    mm_embedding_res.mm_position_ids = position_ids;
                }
                return mm_embedding_res;
            } catch (py::error_already_set &e) {
                return absl::InternalError(std::string(e.what()));
            }
        } else {
            return absl::InternalError("no mm process engine!");
        }
    }

    absl::StatusOr<ExpandedOutput> expand_token_ids(const std::vector<torch::Tensor>& mm_embedding, ft::BufferPtr token_ids, const std::vector<rtp_llm::MultimodalInput> mm_inputs) {
        if (mm_embedding.size() == 0) {
            return ExpandedOutput(token_ids);
        }
        std::vector<std::string> urls;
        for (auto& mm_input: mm_inputs) {
            urls.push_back(mm_input.url);
        }

        assert(token_ids->shape().size() == 1);
        int expanded_len = token_ids->shape()[0];
        std::vector<int> embed_len = {};

        CHECK_AND_RETURN_REF(locs, get_mm_tags(token_ids));

        int mm_num = mm_embedding.size();
        if (locs.size() != mm_num) {
            std::stringstream exception_str;
            exception_str << "number of multimodal tags and multimodal input not matched, expect " << locs.size() << ", get " << mm_num;
            return absl::InternalError(exception_str.str());
        }
        for (int i = 0;i < mm_num;i++) {
            // mm embedding is supposed to be a tensor of [expand_len, hidden_dim]
            expanded_len += mm_embedding[i].sizes()[0] - locs[i].second + locs[i].first;
        }

        if (expanded_len >= max_seq_len_) {
            std::stringstream exception_str;
            exception_str << "input after multimodal process is " << expanded_len << " > max_seq_len(" << max_seq_len_ << ")";
            return absl::InternalError(exception_str.str());
        }

        auto device = ft::DeviceFactory::getDefaultDevice();
        ft::BufferPtr expanded_ids = device->allocateBuffer(
            {token_ids->type(), {(size_t)expanded_len}, ft::AllocationType::HOST}, {});
        ft::BufferPtr token_masks = device->allocateBuffer(
            {token_ids->type(), {(size_t)expanded_len}, ft::AllocationType::HOST}, {});
        ft::BufferPtr new_locs = device->allocateBuffer(
            {ft::DataType::TYPE_INT32, {(size_t)mm_num}, ft::AllocationType::HOST}, {});
        memset(expanded_ids->data(), 0, expanded_ids->sizeBytes());
        std::fill(token_masks->data<int32_t>(), token_masks->dataWithOffset<int32_t>(token_masks->size()), 1);

        int new_loc_idx = 0, old_loc_idx = 0;
        // TODO: repeat urls embedding length times to make sure a url -> multi mmembedding correct
        bool hash_urls = urls.size() == mm_num;
        for (int i = 0;i < mm_num;i++) {
            auto& loc = locs[i];
            int copy_len = loc.first - old_loc_idx;
            memcpy(expanded_ids->dataWithOffset<int32_t>(new_loc_idx), token_ids->dataWithOffset<int32_t>(old_loc_idx), token_ids->typeSize() * copy_len);
            memset(token_masks->dataWithOffset<int32_t>(new_loc_idx + copy_len), 0, mm_embedding[i].sizes()[0] * token_masks->typeSize());
            *(new_locs->dataWithOffset<int32_t>(i)) = copy_len + new_loc_idx;

            if (hash_urls) {
                THROW_IF_STATUS_ERROR(get_str_hash(expanded_ids->dataWithOffset<int32_t>(new_loc_idx + copy_len), urls[i], mm_embedding[i].sizes()[0]));
            }

            new_loc_idx += copy_len + mm_embedding[i].sizes()[0];
            old_loc_idx = loc.second;
        }

        if (expanded_ids->shape()[0] - new_loc_idx != token_ids->shape()[0] - old_loc_idx) {
            throw std::runtime_error("expanded length calculate error");
        }
        memcpy(expanded_ids->dataWithOffset<int32_t>(new_loc_idx), token_ids->dataWithOffset<int32_t>(old_loc_idx), token_ids->typeSize() * (expanded_ids->shape()[0] - new_loc_idx));

        return ExpandedOutput(std::move(expanded_ids), std::move(token_masks), std::move(new_locs));
    }

    absl::StatusOr<std::vector<std::pair<int32_t, int32_t>>> get_mm_tags(ft::BufferPtr token_ids) {
        // sep_tokens will split input tokens to text part and multimodal part
        int32_t* data = token_ids->data<int32_t>();
        std::vector<std::pair<int32_t, int32_t>> locs;
        for (const auto& sep_token_id: sep_token_ids_) {
            // when sep token size == 1, just simply split
            if (sep_token_id.size() == 1) {
                for (int i = 0;i < token_ids->size();i++) {
                    auto now_id = *(data + i);
                    if (now_id == sep_token_id[0]) {
                        locs.emplace_back(i, i + 1);
                    }
                }
            }
            // when sep token size == 2, multimodal part is between but not include them
            else if (sep_token_id.size() == 2) {
                std::vector<int32_t> left, right;

                for (int i = 0;i < token_ids->size();i++) {
                    auto now_id = *(data + i);
                    if (now_id == sep_token_id[0]) {
                        if (right.size() != left.size()) {
                            return absl::InternalError("unmatched multimodal tag pairs");
                        }
                        if (!include_sep_tokens_){
                            left.emplace_back(i + 1);
                        } else {
                            left.emplace_back(i);
                        }
                    } else if (now_id == sep_token_id[1]) {
                        if (!include_sep_tokens_){
                            right.emplace_back(i);
                        } else {
                            right.emplace_back(i + 1);
                        }
                        if (right.size() != left.size()) {
                            return absl::InternalError("unmatched multimodal tag pairs");
                        }
                    }
                }
                if (left.size() != right.size()) {
                    return absl::InternalError("unclosed multimodal tag pairs");
                }
                for (int i = 0;i < left.size();i++) {
                    locs.emplace_back(left[i], right[i]);
                }
            } else {
                return absl::InternalError("more than 2 sep tokens or no sep tokens for multimodal model is not supported");
            }
        }
        std::sort(locs.begin(), locs.end());
        return locs;
    }

public:
    absl::Status update_mm_features(std::shared_ptr<rtp_llm::GenerateInput>& input) {
        CHECK_AND_RETURN_REF(mm_embedding_res, mm_embedding(input->multimodal_inputs.value()));
        input->multimodal_features = std::move(mm_embedding_res.mm_features);
        input->mm_position_ids = std::move(mm_embedding_res.mm_position_ids);
        CHECK_AND_RETURN_REF(expanded_ids, expand_token_ids(input->multimodal_features.value(), input->input_ids, input->multimodal_inputs.value()));
        input->input_ids = expanded_ids.expanded_ids;
        input->text_tokens_mask = expanded_ids.text_tokens_mask;
        input->mm_locs = expanded_ids.locs;
        return absl::OkStatus();
    }


    absl::StatusOr<MultimodalFeature> get_mm_features(const ft::BufferPtr& input_ids, const std::vector<MultimodalInput> &mm_inputs) {
        MultimodalFeature mm_features;
        CHECK_AND_RETURN_REF(features, mm_embedding(mm_inputs));
        mm_features.features = std::move(features);
        CHECK_AND_RETURN_REF(expanded_ids, expand_token_ids(mm_features.features, input_ids, mm_inputs));
        mm_features.expanded_ids = expanded_ids.expanded_ids;
        mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
        mm_features.locs = expanded_ids.locs;
        return mm_features;
    }
};

}
