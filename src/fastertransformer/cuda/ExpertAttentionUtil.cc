#include "src/fastertransformer/cuda/ExpertAttentionUtil.h"

namespace fastertransformer {

template<typename T>
void ExpertAttentionUtil<T>::allocateBuffer() {
    intermediate_buf_          = (T*)allocator_->reMalloc(intermediate_buf_, sizeof(T) * m_ * n_);
    size_t vision_token_length = this->vision_token_length();
    size_t text_input_length   = m_ - vision_token_length;
    if (vision_token_length > 0) {
        vision_split_buf_ = (T*)allocator_->reMalloc(vision_split_buf_, sizeof(T) * vision_token_length * k_);
    }
    if (text_input_length > 0) {
        text_split_buf_ = (T*)allocator_->reMalloc(text_split_buf_, sizeof(T) * text_input_length * k_);
    }
}

template<typename T>
void ExpertAttentionUtil<T>::freeBuffer() {
    if (intermediate_buf_) {
        allocator_->free((void**)(&intermediate_buf_));
    }
    if (vision_split_buf_) {
        allocator_->free((void**)(&vision_split_buf_));
    }
    if (text_split_buf_) {
        allocator_->free((void**)(&text_split_buf_));
    }
    intermediate_buf_ = nullptr;
    vision_split_buf_ = nullptr;
    text_split_buf_   = nullptr;
}

template<typename T>
void ExpertAttentionUtil<T>::split(const T* input) {
    if (vision_token_length() > 0) {
        for (const auto& vision_item : spans_.vision_span()) {
            cudaMemcpyAsync(vision_split_buf() + vision_item.split_index * k_,
                            input + vision_item.origin_index * k_,
                            sizeof(T) * vision_item.offset * k_,
                            cudaMemcpyDeviceToDevice,
                            *stream_);
        }
    }

    for (const auto& text_item : spans_.text_span()) {
        cudaMemcpyAsync(text_split_buf() + text_item.split_index * k_,
                        input + text_item.origin_index * k_,
                        sizeof(T) * text_item.offset * k_,
                        cudaMemcpyDeviceToDevice,
                        *stream_);
    }
}

template<typename T>
void ExpertAttentionUtil<T>::reorganize(T* output) {
    if (vision_token_length() > 0) {
        for (const auto& vision_item : spans_.vision_span()) {
            cudaMemcpyAsync(output + vision_item.target_index * n_,
                            intermediate_buf_ + vision_item.intermediate_index * n_,
                            sizeof(T) * vision_item.offset * n_,
                            cudaMemcpyDeviceToDevice,
                            *stream_);
        }
    }

    for (const auto& text_item : spans_.text_span()) {
        cudaMemcpyAsync(output + text_item.target_index * n_,
                        intermediate_buf_ + text_item.intermediate_index * n_,
                        sizeof(T) * text_item.offset * n_,
                        cudaMemcpyDeviceToDevice,
                        *stream_);
    }
}

}  // namespace fastertransformer