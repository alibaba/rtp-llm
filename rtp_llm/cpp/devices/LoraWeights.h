#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/models/models_weight/W.h"

#include <optional>
#include <memory>
#include <unordered_map>
#include <thread>
#include <shared_mutex>
#include <numeric>

namespace rtp_llm {

namespace lora {

namespace target {
static const std::array<std::string, 5> target_modules = {W::attn_qkv_w, W::attn_o_w, W::ffn_w1, W::ffn_w2, W::ffn_w3};
};

struct LoraWeights {
    ConstBufferPtr lora_a_;
    ConstBufferPtr lora_b_;

    LoraWeights(ConstBufferPtr lora_a, ConstBufferPtr lora_b) {
        RTP_LLM_CHECK_WITH_INFO((lora_a != nullptr && lora_b != nullptr), "lora lora_a and lora b cannot be empty.");
        RTP_LLM_CHECK_WITH_INFO((lora_a->dim() == lora_b->dim()), "lora lora_a and lora b need have same dim.");
        RTP_LLM_CHECK_WITH_INFO((lora_a->dim() >= 2), "lora dim must be greater than 2.");
        RTP_LLM_CHECK_WITH_INFO((lora_a->shape()[lora_a->dim() - 1] == lora_b->shape()[lora_b->dim() - 2]),
                                "lora lora_a[%ld] and lora lora_b[%ld] need has same rank.",
                                lora_a->shape()[lora_a->dim() - 1],
                                lora_b->shape()[lora_b->dim() - 2]);
        RTP_LLM_CHECK_WITH_INFO((lora_a->shape()[lora_a->dim() - 1] <= lora_a->shape()[lora_a->dim() - 2]),
                                "lora lora_a rank[%ld] must less than dim0[%ld].",
                                lora_a->shape()[lora_a->dim() - 1],
                                lora_a->shape()[lora_a->dim() - 2]);
        RTP_LLM_CHECK_WITH_INFO((lora_b->shape()[lora_b->dim() - 2] <= lora_b->shape()[lora_b->dim() - 1]),
                                "lora lora_b rank[%ld] must less than dim0[%ld].",
                                lora_b->shape()[lora_b->dim() - 2],
                                lora_b->shape()[lora_b->dim() - 1]);
        RTP_LLM_CHECK_WITH_INFO((lora_a->type() == lora_b->type()), "lora lora_a and lora b need have same type.");
        RTP_LLM_CHECK_WITH_INFO((lora_a->where() == lora_b->where()), "lora lora_a and lora b need have same memory.");
        lora_a_ = lora_a;
        lora_b_ = lora_b;
    }
};

typedef std::shared_ptr<const LoraWeights> LoraWeightsPtr;

struct LoraModelImpl {
    std::unordered_map<std::string, LoraWeightsPtr> lora_model_;

    void setLoraWeigths(const std::string& target_module, ConstBufferPtr lora_a, ConstBufferPtr lora_b) {
        LoraWeightsPtr lora_weights = nullptr;
        if (lora_a != nullptr && lora_b != nullptr) {
            lora_weights = std::make_shared<const LoraWeights>(lora_a, lora_b);
        }
        lora_model_[target_module] = lora_weights;
    };

    LoraWeightsPtr getLoraWeights(const std::string& target_module) const {
        auto it = lora_model_.find(target_module);
        if (it != lora_model_.end()) {
            return it->second;
        } else {
            return nullptr;
        }
    };
};

using loraLayerWeightsMap = std::vector<std::unordered_map<std::string, ConstBufferPtr>>;

struct LoraModel {
    std::vector<LoraModelImpl> lora_model_;

    LoraModel(loraLayerWeightsMap lora_a, loraLayerWeightsMap lora_b) {
        RTP_LLM_CHECK_WITH_INFO((lora_a.size() == lora_b.size()), "lora lora_a and lora b need has same size.");
        size_t layer_num = lora_a.size();
        lora_model_.resize(layer_num);

        for (size_t i = 0; i < layer_num; i++) {
            auto lora_model_impl = LoraModelImpl();
            for (auto target_module : target::target_modules) {
                lora_model_impl.setLoraWeigths(target_module, lora_a[i][target_module], lora_b[i][target_module]);
            }
            lora_model_[i] = lora_model_impl;
        }
    }

    LoraWeightsPtr getLoraWeights(const size_t layer_num, const std::string& target_module) const {
        RTP_LLM_CHECK_WITH_INFO((layer_num < lora_model_.size()),
                                "layer index[%d] is greate than layer num[%d]",
                                layer_num,
                                lora_model_.size());
        return lora_model_[layer_num].getLoraWeights(target_module);
    }
};

using LoraModelPtr = std::shared_ptr<const LoraModel>;

struct LoraOpInput {
    BufferPtr                   lora_input_lengths_;
    std::vector<ConstBufferPtr> lora_a_;
    std::vector<ConstBufferPtr> lora_b_;
    bool                        use_same_lora_;

    LoraOpInput(BufferPtr                   lora_input_lengths,
                std::vector<ConstBufferPtr> lora_a,
                std::vector<ConstBufferPtr> lora_b,
                bool                        use_same_lora = false):
        lora_input_lengths_(lora_input_lengths), lora_a_(lora_a), lora_b_(lora_b), use_same_lora_(use_same_lora) {}

    LoraOpInput(BufferPtr lora_input_length, std::vector<LoraWeightsPtr>& lora_weights, bool use_same_lora = false) {
        RTP_LLM_CHECK_WITH_INFO(
            (lora_input_length->dim() == 1), "lora_input_length[%d] dim must be 1.", lora_input_length->dim());

        size_t batch_size = lora_input_length->shape()[0];

        RTP_LLM_CHECK_WITH_INFO(((lora_weights.size() == batch_size)),
                                "lora_weights[%d] size must be equal to batch size[%d].",
                                lora_weights.size(),
                                batch_size);
        lora_input_lengths_ = lora_input_length;
        lora_a_.resize(batch_size);
        lora_b_.resize(batch_size);

        for (int i = 0; i < batch_size; i++) {
            if (lora_weights[i] == nullptr) {
                lora_a_[i] = nullptr;
                lora_b_[i] = nullptr;
            } else {
                lora_a_[i] = lora_weights[i]->lora_a_;
                lora_b_[i] = lora_weights[i]->lora_b_;
            }
        }
        use_same_lora_ = use_same_lora;
    }

    bool isEmpty() const {
        return std::accumulate(
                   lora_a_.begin(), lora_a_.end(), true, [](bool a, ConstBufferPtr b) { return a && (b == nullptr); })
               && std::accumulate(
                   lora_b_.begin(), lora_b_.end(), true, [](bool a, ConstBufferPtr b) { return a && (b == nullptr); });
    }
};

using LoraOpInputPtr = std::shared_ptr<LoraOpInput>;

struct AttentionLayerLoraInput {
    LoraOpInputPtr qkv_lora_input = nullptr;
    LoraOpInputPtr out_lora_input = nullptr;
};

struct FfnLayerLoraInput {
    LoraOpInputPtr gate_lora_input = nullptr;
    LoraOpInputPtr up_lora_input   = nullptr;
    LoraOpInputPtr down_lora_input = nullptr;

    bool isEmpty() const {
        return (gate_lora_input == nullptr || gate_lora_input->isEmpty())
               && (up_lora_input == nullptr || up_lora_input->isEmpty())
               && (down_lora_input == nullptr || down_lora_input->isEmpty());
    }
};

struct LoraModelInput {
    BufferPtr                 lora_input_lengths_;
    std::vector<LoraModelPtr> lora_model_input_;
    bool                      use_same_lora_;

    LoraModelInput(BufferPtr                 lora_input_lengths,
                   std::vector<LoraModelPtr> lora_model_input,
                   bool                      use_same_lora = false) {
        RTP_LLM_CHECK_WITH_INFO((lora_input_lengths != nullptr), "lora_input_lengths can not be empty");

        RTP_LLM_CHECK_WITH_INFO(
            (lora_input_lengths->dim() == 1), "lora_input_lengths[%d] dim must be 1", lora_input_lengths->dim());

        RTP_LLM_CHECK_WITH_INFO((lora_input_lengths->shape()[0] == lora_model_input.size()),
                                "lora input lengths batch_size [%d] must be equalt to lora_model_input size [%d]",
                                lora_input_lengths->shape()[0],
                                lora_model_input.size());

        lora_input_lengths_ = lora_input_lengths;
        lora_model_input_   = lora_model_input;
        use_same_lora_      = use_same_lora;
    }

    LoraOpInputPtr getOpInput(const size_t layer_num, const std::string& target_module) {
        std::vector<LoraWeightsPtr> lora_weights(lora_model_input_.size());
        for (int i = 0; i < lora_weights.size(); i++) {
            if (lora_model_input_[i] == nullptr) {
                lora_weights[i] = nullptr;
            } else {
                lora_weights[i] = lora_model_input_[i]->getLoraWeights(layer_num, target_module);
            }
        }
        return std::make_shared<LoraOpInput>(lora_input_lengths_, lora_weights, use_same_lora_);
    }

    AttentionLayerLoraInput getAttentionLayerLoraInput(const size_t layer_num) {
        auto result           = AttentionLayerLoraInput();
        result.qkv_lora_input = getOpInput(layer_num, W::attn_qkv_w);
        result.out_lora_input = getOpInput(layer_num, W::attn_o_w);
        return result;
    }

    FfnLayerLoraInput getFfnLayerLoraInput(const size_t layer_num) {
        auto result            = FfnLayerLoraInput();
        result.gate_lora_input = getOpInput(layer_num, W::ffn_w1);
        result.down_lora_input = getOpInput(layer_num, W::ffn_w2);
        result.up_lora_input   = getOpInput(layer_num, W::ffn_w3);
        return result;
    }
};

using LoraModelInputPtr = std::shared_ptr<LoraModelInput>;

}  // namespace lora

}  // namespace rtp_llm
