#pragma once

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

struct ExpandedOutput {
    ft::BufferPtr expanded_ids;
    ft::BufferPtr text_tokens_mask;
    ft::BufferPtr locs;
    ExpandedOutput(ft::BufferPtr expanded_ids, ft::BufferPtr text_tokens_mask = nullptr, ft::BufferPtr locs = nullptr):
        expanded_ids(expanded_ids), text_tokens_mask(text_tokens_mask), locs(locs) {}
};

class MultimodalProcessor {
public:
    MultimodalProcessor(py::object mm_proces_engine, std::vector<int64_t> sep_token_ids, bool include_sep_tokens):
        mm_process_engine_(mm_proces_engine), sep_token_ids_(sep_token_ids), include_sep_tokens_(include_sep_tokens)
        {}

private:
    py::object mm_process_engine_;
    std::vector<int64_t> sep_token_ids_;
    bool include_sep_tokens_;

    absl::StatusOr<std::vector<torch::Tensor>> mm_embedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs) {
        if (mm_inputs.size() == 0) {
            return std::vector<torch::Tensor>();
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
                auto mm_embedding_vec = ft::convertPyObjectToVec(res);
                std::vector<torch::Tensor> embedding_res;
                for (auto& emb: mm_embedding_vec) {
                    embedding_res.emplace_back(ft::convertPyObjectToTensor(emb));
                }
                return embedding_res;
            } catch (py::error_already_set &e) {
                return absl::InternalError(std::string(e.what()));
            }
        } else {
            return absl::InternalError("no mm process engine!");
        }
    }

    absl::StatusOr<ExpandedOutput> expand_token_ids(const std::vector<torch::Tensor>& mm_embedding, ft::BufferPtr token_ids) {
        if (mm_embedding.size() == 0) {
            return ExpandedOutput(token_ids);
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
        for (int i = 0;i < mm_num;i++) {
            auto& loc = locs[i];
            int copy_len = loc.first - old_loc_idx;
            memcpy(expanded_ids->dataWithOffset<int32_t>(new_loc_idx), token_ids->dataWithOffset<int32_t>(old_loc_idx), token_ids->typeSize() * copy_len);
            memset(token_masks->dataWithOffset<int32_t>(loc.first), 0, (loc.second - loc.first) * token_masks->typeSize());
            *(new_locs->dataWithOffset<int32_t>(i)) = copy_len + new_loc_idx;
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
        
        // when sep token size == 1, just simply split
        if (sep_token_ids_.size() == 1) {
            for (int i = 0;i < token_ids->size();i++) {
                auto now_id = *(data + i);
                if (now_id == sep_token_ids_[0]) {
                    locs.emplace_back(i, i + 1);
                }
            }
        }
        // when sep token size == 2, multimodal part is between but not include them
        else if (sep_token_ids_.size() == 2) {
            std::vector<int32_t> left, right;

            for (int i = 0;i < token_ids->size();i++) {
                auto now_id = *(data + i);
                if (now_id == sep_token_ids_[0]) {
                    if (right.size() != left.size()) {
                        return absl::InternalError("unmatched multimodal tag pairs");
                    }
                    if (!include_sep_tokens_){
                        left.emplace_back(i + 1);
                    } else {
                        left.emplace_back(i);
                    }
                } else if (now_id == sep_token_ids_[1]) {
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
        return locs;
    }

public:
    absl::Status update_mm_features(std::shared_ptr<rtp_llm::GenerateInput>& input) {
        CHECK_AND_RETURN_REF(mm_features, mm_embedding(input->multimodal_inputs.value()));
        input->multimodal_features = std::move(mm_features);
        CHECK_AND_RETURN_REF(expanded_ids, expand_token_ids(input->multimodal_features.value(), input->input_ids));
        input->input_ids = expanded_ids.expanded_ids;
        input->text_tokens_mask = expanded_ids.text_tokens_mask;
        input->mm_locs = expanded_ids.locs;
        return absl::OkStatus();
    }
};