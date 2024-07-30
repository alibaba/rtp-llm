#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/models/W.h"

#include <optional>
#include <memory>
#include <unordered_map>
#include <thread>
#include <shared_mutex>

namespace fastertransformer {

namespace lora {


struct LoraWeights {
    ConstBufferPtr lora_a_;
    ConstBufferPtr lora_b_;

    LoraWeights(ConstBufferPtr lora_a, ConstBufferPtr lora_b) {
        FT_CHECK_WITH_INFO((lora_a != nullptr && lora_b != nullptr),
            "lora lora_a and lora b cannot be empty.");
        FT_CHECK_WITH_INFO((lora_a->dim() == lora_b->dim()),
            "lora lora_a and lora b need have same dim.");
        FT_CHECK_WITH_INFO((lora_a->dim() >= 2),
            "lora dim must be greater than 2.");
        FT_CHECK_WITH_INFO((lora_a->shape()[lora_a->dim() - 1] == lora_b->shape()[lora_b->dim() - 2]),
            "lora lora_a[%ld] and lora lora_b[%ld] need has same rank.",
            lora_a->shape()[lora_a->dim() - 1], lora_b->shape()[lora_b->dim() - 2]);
        FT_CHECK_WITH_INFO((lora_a->shape()[lora_a->dim() - 1] <= lora_a->shape()[lora_a->dim() - 2]),
            "lora lora_a rank[%ld] must less than dim0[%ld].",
            lora_a->shape()[lora_a->dim() - 1], lora_a->shape()[lora_a->dim() - 2]);
        FT_CHECK_WITH_INFO((lora_b->shape()[lora_b->dim() - 2] <= lora_b->shape()[lora_b->dim() - 1]),
            "lora lora_b rank[%ld] must less than dim0[%ld].",
            lora_b->shape()[lora_b->dim() - 2], lora_b->shape()[lora_b->dim() - 1]);
        FT_CHECK_WITH_INFO((lora_a->type() == lora_b->type()),
            "lora lora_a and lora b need have same type.");
        FT_CHECK_WITH_INFO((lora_a->where() == lora_b->where()),
            "lora lora_a and lora b need have same memory.");
        lora_a_ = lora_a;
        lora_b_ = lora_b;
    }
};

typedef std::shared_ptr<const LoraWeights>  LoraWeightsPtr;

struct LoraModelImpl {
    LoraWeightsPtr attn_qkv_lora_weights_ = nullptr;
    LoraWeightsPtr attn_out_lora_weights_ = nullptr;
    LoraWeightsPtr ffn_gate_lora_weights_ = nullptr;
    LoraWeightsPtr ffn_up_lora_weights_   = nullptr;
    LoraWeightsPtr ffn_down_lora_weights_ = nullptr;

    void setLoraWeigths(const std::string& target_module, ConstBufferPtr lora_a, ConstBufferPtr lora_b) {
        LoraWeightsPtr lora_weights = nullptr;
        if (lora_a != nullptr && lora_b != nullptr) {
            lora_weights = std::make_shared<const LoraWeights>(lora_a, lora_b);
        }
        if (target_module == W::attn_qkv_w) {
            attn_qkv_lora_weights_ = lora_weights;
        } else if (target_module == W::attn_o_w) {
            attn_out_lora_weights_ = lora_weights;
        } else if (target_module == W::ffn_w1) {
            ffn_gate_lora_weights_ = lora_weights;
        } else if (target_module == W::ffn_w2) {
            ffn_down_lora_weights_ = lora_weights;
        } else if (target_module == W::ffn_w3) {
            ffn_up_lora_weights_ = lora_weights;
        } else {
            FT_CHECK_WITH_INFO(false, "lora model do not support %s.", target_module.c_str());
        }
    };

    LoraWeightsPtr getLoraWeights(const std::string& target_module) const {
        if (target_module == W::attn_qkv_w) {
            return attn_qkv_lora_weights_;
        } else if (target_module == W::attn_o_w) {
            return attn_out_lora_weights_;
        } else if (target_module == W::ffn_w1) {
            return ffn_gate_lora_weights_;
        } else if (target_module == W::ffn_w2) {
            return ffn_down_lora_weights_;
        } else if (target_module == W::ffn_w3) {
            return ffn_up_lora_weights_;
        } else {
            FT_CHECK_WITH_INFO(false, "lora model do not support %s.", target_module.c_str());
        }
        return nullptr;
    };
};

using loraLayerWeightsMap = std::vector<std::unordered_map<std::string, ConstBufferPtr>>;

struct LoraModel {
    std::vector<LoraModelImpl> lora_model_;

    LoraModel(loraLayerWeightsMap lora_a, loraLayerWeightsMap lora_b) {
        FT_CHECK_WITH_INFO((lora_a.size() == lora_b.size()),
            "lora lora_a and lora b need has same size.");
        size_t layer_num = lora_a.size();
        lora_model_.resize(layer_num);
        std::vector<std::string> target_modules = {W::attn_qkv_w,
                                                   W::attn_o_w,
                                                   W::ffn_w1,
                                                   W::ffn_w2,
                                                   W::ffn_w3};
        for (size_t i = 0; i < layer_num; i++) {
            auto lora_model_impl = LoraModelImpl();
            for (auto target_module : target_modules) {
                lora_model_impl.setLoraWeigths(target_module,
                                               lora_a[i][target_module],
                                               lora_b[i][target_module]);
            }
            lora_model_[i] = lora_model_impl;
        }
    }

    LoraWeightsPtr getLoraWeights(const size_t layer_num, const std::string& target_module) {
        FT_CHECK_WITH_INFO((layer_num < lora_model_.size()),
            "layer index[%d] is greate than layer num[%d]", layer_num, lora_model_.size());
        return lora_model_[layer_num].getLoraWeights(target_module);
    }

};

using LoraModelPtr = std::shared_ptr<const LoraModel>;

struct LoraModelInput {
    BufferPtr lora_input_lengths_;
    std::vector<LoraModelPtr> lora_model_input_;

    LoraModelInput(BufferPtr lora_input_lengths, std::vector<LoraModelPtr> lora_model_input) {
        FT_CHECK_WITH_INFO((lora_input_lengths != nullptr),
            "lora_input_lengths can not be empty");

        FT_CHECK_WITH_INFO((lora_input_lengths->dim() == 1),
            "lora_input_lengths[%d] dim must be 1", lora_input_lengths->dim());

        FT_CHECK_WITH_INFO((lora_input_lengths->shape()[0] == lora_model_input.size()),
            "lora input lengths batch_size [%d] must be equalt to lora_model_input size [%d]",
            lora_input_lengths->shape()[0], lora_model_input.size());

        lora_input_lengths_ = lora_input_lengths;
        lora_model_input_   = lora_model_input;
    }
};

}  // namespace lora

}  // namespace fastertransformer
