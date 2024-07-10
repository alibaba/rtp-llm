#pragma once

#include <string>
#include <optional>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <torch/python.h>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace ft = fastertransformer;
namespace py = pybind11;

struct ExpandedOutput {
    ft::BufferPtr expanded_ids;
    ft::BufferPtr text_tokens_mask;
    ft::BufferPtr locs;
};

class MultimodalProcessor {
public:
    MultimodalProcessor(py::object mm_proces_engine, std::vector<int64_t> sep_token_ids):
        mm_process_engine_(mm_proces_engine), sep_token_ids_(sep_token_ids)
        {}

    std::vector<torch::Tensor> mm_embedding(const std::vector<std::string>& urls) {
        if (urls.size() == 0) {
            return std::vector<torch::Tensor>();
        } else if (!mm_process_engine_.is_none()) {
            py::gil_scoped_acquire acquire;
            auto futures = mm_process_engine_.attr("submit")(urls);
            auto coro = mm_process_engine_.attr("get")(futures);

            auto loop = py::module::import("asyncio").attr("get_event_loop")();
            auto future = loop.attr("create_task")(coro);
            loop.attr("run_until_complete")(future);
            py::handle res = future.attr("result")();

            auto mm_embedding_vec = ft::convertPyObjectToVec(res);
            std::vector<torch::Tensor> embedding_res;
            for (auto& emb: mm_embedding_vec) {
                embedding_res.emplace_back(ft::convertPyObjectToTensor(emb));
            }
            return std::move(embedding_res);
        } else {
            throw std::runtime_error("Multimodal model but multimodal process engine is None");
        }
    }

    ExpandedOutput expand_token_ids(const std::vector<torch::Tensor>& mm_embedding, ft::BufferPtr token_ids) {
        if (mm_embedding.size() == 0) {
            return {token_ids, nullptr};
        }

        assert(token_ids->shape().size() == 1);
        int expanded_len = token_ids->shape()[0];
        std::vector<int> embed_len = {};

        auto locs = get_mm_tags(token_ids);
        int mm_num = mm_embedding.size();
        if (locs.size() != mm_num) {
            throw std::runtime_error("number of multimodal tags and multimodal input not matched");
        }

        for (int i = 0;i < mm_num;i++) {
            auto& emb = mm_embedding[i];
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
            throw std::runtime_error("number of multimodal tags and multimodal input not matched");
        }
        memcpy(expanded_ids->dataWithOffset<int32_t>(new_loc_idx), token_ids->dataWithOffset<int32_t>(old_loc_idx), token_ids->typeSize() * (expanded_ids->shape()[0] - new_loc_idx));

        return {std::move(expanded_ids), std::move(token_masks), std::move(new_locs)};
    }
    
private:
    py::object mm_process_engine_;
    std::vector<int64_t> sep_token_ids_;

    std::vector<std::pair<int32_t, int32_t>> get_mm_tags(ft::BufferPtr token_ids) {
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
                        throw std::runtime_error("unmatched multimodal tag pairs");
                    }
                    left.emplace_back(i + 1);
                } else if (now_id == sep_token_ids_[1]) {
                    right.emplace_back(i);
                    if (right.size() != left.size()) {
                        throw std::runtime_error("unmatched multimodal tag pairs");
                    }
                }
            }
            if (left.size() != right.size()) {
                throw std::runtime_error("unclosed multimodal tag pairs");
            }
            for (int i = 0;i < left.size();i++) {
                locs.emplace_back(left[i], right[i]);
            }
        } else {
            throw std::runtime_error("more than 2 sep tokens or no sep tokens for multimodal model is not supported");
        }
        return std::move(locs);
    }
};