#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

namespace fastertransformer {

#define LANGUAGE_TOKEN_TYPE 0
#define VISION_TOKEN_TYPE 1

struct SpanItem {
public:
    SpanItem(size_t origin_index, size_t offset):
        origin_index(origin_index), offset(offset), target_index(origin_index) {}

    size_t origin_index       = 0;
    size_t target_index       = 0;
    size_t split_index        = 0;
    size_t intermediate_index = 0;
    size_t offset             = 0;
};

/*
 *  Utility class for transforming the input token_type_ids to a vector of SpanItem.
 *  Assuming the input token_type_ids is a vector of [T, V, V, V, T, T], it will be transforming into
 *  two expert attention span, [1, 2] for vision tokens, [0, 1] and [3, 3] for text tokens,
 *  see https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B/blob/main/modeling_cogvlm.py#L92
 */
class ExpertAttentionSpan {
public:
    ExpertAttentionSpan(const int32_t* token_type_ids, const size_t length) {
        init_spans(token_type_ids, length);
    }

    void init_spans(const int32_t* token_type_ids, const size_t length) {
        assert(length >= 0);
        size_t span_item_left = 0;
        bool   text_span_flag = token_type_ids[0] == LANGUAGE_TOKEN_TYPE;

        for (size_t i = 0; i < length - 1; i++) {
            if (token_type_ids[i] != token_type_ids[i + 1]) {
                if (text_span_flag) {
                    text_span_.push_back({span_item_left, i - span_item_left + 1});
                    span_item_left = i + 1;
                } else {
                    vision_span_.push_back({span_item_left, i - span_item_left});
                    span_item_left = i;
                }
                text_span_flag = token_type_ids[i + 1] == LANGUAGE_TOKEN_TYPE;
            }
        }

        if (text_span_flag) {
            text_span_.push_back({span_item_left, length - span_item_left});
        } else {
            vision_span_.push_back({span_item_left, length - span_item_left - 1});
            text_span_.push_back({length - 1, 1});
        }

        assign_span_indices();

        vision_token_length_ =
            std::accumulate(vision_span_.begin(), vision_span_.end(), 0, [](size_t acc, const auto& item) {
                return acc + item.offset;
            });
    }

    void assign_span_indices() {
        int intermediate_mat_index = 0;
        int vision_span_index      = 0;
        int text_span_index        = 0;
        for (auto& item : vision_span_) {
            item.split_index        = vision_span_index;
            item.intermediate_index = intermediate_mat_index;
            vision_span_index += item.offset;
            intermediate_mat_index += item.offset;
        }

        for (auto& item : text_span_) {
            item.split_index        = text_span_index;
            item.intermediate_index = intermediate_mat_index;
            text_span_index += item.offset;
            intermediate_mat_index += item.offset;
        }
    }

    size_t vision_token_length() {
        return vision_token_length_;
    }

    const std::vector<SpanItem>& vision_span() {
        return vision_span_;
    }

    const std::vector<SpanItem>& text_span() {
        return text_span_;
    }

private:
    size_t                vision_token_length_;
    std::vector<SpanItem> text_span_;
    std::vector<SpanItem> vision_span_;
};

/**
 * Utility Class for Multimodal Processing in CogVLM2
 *
 * This utility class implements critical split and merge operations for the
 * expertAttention and expertFFN modules within the CogVLM2 framework. Referenced
 * in the paper https://arxiv.org/abs/2311.03079, these operations enable the
 * processing of vision and text modalities separately,
 *
 * A visual guide to the data processing pipeline is depicted below:
 *            input_buf_
 *                |
 *                v
 *     +--------------------+
 *     |                    |
 *     v                    v
 * vision_split_buf    text_split_buf
 *     | vision Gemm        | text Gemm
 *     v                    v
 *     |                    |
 *     +--------------------+
 *                |
 *                v
 *        intermediate_buf_
 *                |
 *               (reorganize memory)
 *                v
 *           output_buf_
 *
 * 'input_buf' is firstly splited into two separate streams, 'vision_split_buf' and 'text_split_buf',
 * each represented for a specific modality. Then two separate Gemm operations are performed on 'vision_split_buf'
 * and 'text_split_buf', results are saved in 'intermediate_buf'. Finally, the results are reorganized in
 * 'intermediate_buf' and saved for the 'output_buf'.
 */
template<typename T>
class ExpertAttentionUtil {
public:
    ExpertAttentionUtil(cudaStream_t*  stream,
                        IAllocator*    allocator,
                        const int32_t* token_type_ids,
                        size_t         token_length,
                        const T*       input_buf  = nullptr,
                        T*       output_buf = nullptr):
        stream_(stream),
        allocator_(allocator),
        spans_(token_type_ids, token_length),
        input_buf_(input_buf),
        output_buf_(output_buf) {}

    ~ExpertAttentionUtil() {
        freeBuffer();
    }

    void updateBufferShape(size_t m, size_t k, size_t n) {
        m_ = m;
        k_ = k;
        n_ = n;
    }

    void allocateBuffer();

    void freeBuffer();

    void split(const T* input);

    void reorganize(T* output);

    void reorganize() {
        assert(output_buf_);
        reorganize(output_buf_);
    }

    void split() {
        assert(input_buf_);
        split(input_buf_);
    }

    size_t vision_token_length() {
        return spans_.vision_token_length();
    }

    size_t text_token_length() {
        return m_ - spans_.vision_token_length();
    }

    size_t token_length() {
        return m_;
    }

    T* vision_split_buf() {
        return vision_split_buf_;
    }

    T* text_split_buf() {
        return text_split_buf_;
    }

    T* intermediate_buf() {
        return intermediate_buf_;
    }

    T* vision_intermediate_buf() {
        return intermediate_buf_;
    }

    T* text_intermediate_buf() {
        return intermediate_buf_ + vision_token_length() * n_;
    }

    void print_vision_split_buf(size_t layer_id) {
        print_bsd(layer_id, "vision split buf", vision_split_buf(), 1, vision_token_length(), k_);
    }

    void print_text_split_buf(size_t layer_id) {
        print_bsd(layer_id, "text split buf", text_split_buf(), 1, text_token_length(), k_);
    }

    void print_intermediate_buf(size_t layer_id) {
        print_bsd(layer_id, "intermediate buf", intermediate_buf(), 1, token_length(), n_);
    }

    void print_vision_intermediate_buf(size_t layer_id) {
        print_bsd(layer_id, "vision intermediate buf", vision_intermediate_buf(), 1, vision_token_length(), n_);
    }

    void print_text_intermediate_buf(size_t layer_id) {
        print_bsd(layer_id, "text intermediate buf", text_intermediate_buf(), 1, text_token_length(), n_);
    }


private:
    IAllocator*         allocator_;
    size_t              m_                = 0;
    size_t              k_                = 0;
    size_t              n_                = 0;
    T*                  intermediate_buf_ = nullptr;
    T*                  vision_split_buf_ = nullptr;
    T*                  text_split_buf_   = nullptr;
    const T*                  input_buf_        = nullptr;
    T*                  output_buf_       = nullptr;
    ExpertAttentionSpan spans_;
    cudaStream_t*       stream_;
};

template class ExpertAttentionUtil<float>;
template class ExpertAttentionUtil<half>;
#ifdef ENABLE_BF16
template class ExpertAttentionUtil<__nv_bfloat16>;
#endif

}  // namespace fastertransformer