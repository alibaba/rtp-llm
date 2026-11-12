#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include <cstdint>
#include <memory>
#include <cstdlib>
#include <functional>
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {

struct PreparedPrefillInput {
    GptModelInputs input;
    TensorHolder   holder;
};

class Executor {
public:
    Executor() {};
    virtual absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us = 0) = 0;

    virtual absl::StatusOr<PreparedPrefillInput> preparePrefillInput(const std::list<GenerateStreamPtr>&) {
        return absl::UnimplementedError("executor does not support Prefill preparation");
    }
    void setPreparedPrefillInput(PreparedPrefillInput input) {
        prepared_prefill_input_ = std::move(input);
    }
    void setNextPrefillPrepare(std::function<void()> prepare) {
        next_prefill_prepare_ = std::move(prepare);
    }

    static GptModelDescription genModelDescription(const ModelConfig&       model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   const EPLBConfig&        eplb_config,
                                                   const MoeConfig&         moe_config) {
        AttentionConfigs attention_config = model_config.getAttentionConfigs(parallelism_config.get_attn_tp_size());
        // TP在init的时候处理，认为每个MOE Plugin只看到一个TP rank；EP在MOE Plugin中处理；
        auto moe_configs =
            model_config.moe_style ?
                (std::optional<rtp_llm::MoeConfigs>)rtp_llm::MoeConfigs(
                    {(size_t)model_config.expert_num,
                     (size_t)(eplb_config.phy_exp_num(model_config.expert_num) - model_config.expert_num),
                     (size_t)model_config.moe_k,
                     model_config.moe_normalize_expert_scale,
                     model_config.has_moe_norm,
                     moe_config.use_all_gather,
                     (size_t)parallelism_config.ep_rank,
                     (size_t)parallelism_config.ep_size,
                     (size_t)parallelism_config.tp_rank,
                     (size_t)parallelism_config.tp_size,
                     (size_t)parallelism_config.dp_rank,
                     (size_t)parallelism_config.dp_size,
                     (int)model_config.scoring_func,
                     (int)model_config.moe_topk_group,
                     (int)model_config.moe_n_group,
                     model_config.routed_scaling_factor,
                     eplb_config.enable_eplb()}) :
                std::nullopt;
        rtp_llm::FfnConfigs ffn_config{
            model_config.activation_type,
            std::move(moe_configs),
        };
        rtp_llm::QScheme act_qscheme = rtp_llm::QScheme::NoQuantize;
        if (model_config.quant_algo.isPerTensorQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerTensor;
        } else if (model_config.quant_algo.isSmoothQuant() || model_config.quant_algo.isOmniQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerToken;
        } else if (model_config.quant_algo.isFp8() && !model_config.quant_algo.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTensor;
        } else if (model_config.quant_algo.isFp8() && model_config.quant_algo.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTokenBlock;
        } else if (model_config.quant_algo.isFp8PTPC()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerToken;
        } else if (model_config.quant_algo.isW4a8Int4PTPC()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerToken;
        } else if (model_config.quant_algo.isW8a8Int8PTPC()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerToken;
        }

        return {attention_config,
                ffn_config,
                model_config.norm_type,
                model_config.data_type,
                act_qscheme,
                model_config.data_type,
                model_config.layernorm_eps,
                (size_t)model_config.vocab_size,
                model_config.output_vocab_ids.size(),
                (size_t)model_config.output_vocab_padded_size,
                model_config.layernorm_type == rtp_llm::LayerNormType::post_layernorm,
                model_config.input_embedding_scalar,
                model_config.residual_scalar,
                model_config.reverse_e_h_norm,
                moe_config};
    }

    virtual ~Executor() {};

protected:
    std::optional<PreparedPrefillInput> prepared_prefill_input_;
    std::function<void()>               next_prefill_prepare_;

    GptModelInputs consumePreparedPrefillInput(TensorHolder& holder, const std::list<GenerateStreamPtr>& streams) {
        auto prepared = std::move(*prepared_prefill_input_);
        prepared_prefill_input_.reset();
        for (auto& tensor : prepared.holder.tensors) {
            holder.hold(tensor);
        }
        for (const auto& stream : streams) {
            stream->step();
        }
        return std::move(prepared.input);
    }

    void startNextPrefillPrepare() {
        if (next_prefill_prepare_) {
            auto prepare          = std::move(next_prefill_prepare_);
            next_prefill_prepare_ = nullptr;
            prepare();
        }
    }
};

}  // namespace rtp_llm
