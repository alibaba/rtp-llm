#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include <cstdint>
#include <memory>
#include <cstdlib>

namespace rtp_llm {

class Executor {
public:
    Executor() {};
    virtual absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us = 0) = 0;

    // Batch-recording helpers shared by executors. They cannot live in
    // rtp_llm/cpp/observability: engine_base/stream already depends on that
    // library (RecordedBatch in StreamGroups, identity publishing in
    // GenerateStream), so a back-dependency on stream types would close a
    // library cycle. The executor base is the common seam of both consumers.
protected:
    // Build the RecordedBatch from the gatherer-frozen CPU length snapshot.
    // Returns nullptr unless the recorder is enabled for this batch
    // (inputs.record_execution_id). Throws when the snapshot is missing or
    // its slot counts disagree with the stream groups; callers contain it
    // (recording must never fail the serving step).
    std::shared_ptr<RecordedBatch> recordBatchInputs(const GptModelInputs& inputs, const StreamGroups& groups) const;
    // Structural record-id gate: owner (recorder enabled, non-warmup,
    // non-special worker, tp_rank 0) plus a real, non-empty batch. Decode-step
    // policy (sampling) is executor-side and composed by the caller.
    static bool shouldRecordBatch(bool owner, bool has_request_stream, size_t total_model_batch_size);

    // Batch recording decode sampling ratio, clamped from
    // ProfilingDebugLoggingConfig by each executor ctor.
    double record_decode_sample_rate_ = 0.1;
    // Deterministic decode-step sampling credit (Bresenham-style): each
    // decode step adds record_decode_sample_rate_; crossing 1.0 records the
    // step and spends one unit, keeping the long-run recorded fraction exact
    // without an RNG. process()/decodeStep() run only on the engine loop
    // thread, so this plain state needs no lock; warmup/propose/non-tp0
    // executors never advance it (short-circuit && in the caller).
    double record_decode_sample_credit_ = 0.0;
    // Decode-step recording sampling; advances the credit only when called
    // (decode gate), so non-decode steps never consume samples.
    bool sampleDecodeStep();

public:
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

    virtual bool updateEplbConfig(const EPLBConfig& config) {
        return false;
    }

public:
};

}  // namespace rtp_llm
