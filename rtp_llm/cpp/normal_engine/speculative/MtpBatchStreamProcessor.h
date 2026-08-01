#pragma once

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

class MtpBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    MtpBatchStreamProcessor(const ModelConfig&                 model_config,
                            const PDSepConfig&                 pd_sep_config,
                            const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                            const CacheConfig&                 cache_config,
                            const SpeculativeExecutionConfig&  sp_config,
                            bool                               warm_up):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
        propose_step_(sp_config.gen_num_per_cycle),
        is_dspark_(sp_config.type == SP_TYPE_DSPARK),
        dspark_mask_token_id_(static_cast<int32_t>(sp_config.sp_dspark_mask_token_id)),
        dspark_sample_from_anchor_(sp_config.sp_dspark_sample_from_anchor),
        dspark_use_gumbel_(sp_config.draft_sample_method == "gumbel"),
        dspark_use_fp64_gumbel_(sp_config.use_fp64_gumbel) {}

    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups,
                                                    TensorHolder&       host_holder) const override;

    absl::Status dispatchPrefill(const StreamGroups& stream_groups,
                                 const MergedOutput& prefill_output,
                                 const MergedOutput& propose_output) const;

    absl::Status dispatchPrefill(const StreamGroups&  stream_groups,
                                 const MergedOutput&  prefill_output,
                                 const MergedOutput&  propose_output,
                                 const torch::Tensor& draft_last_hidden_states) const;

    absl::Status dispatchDecode(const StreamGroups&                          stream_groups,
                                const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                const MergedOutput&                          draft_prefill_output) const;

    absl::StatusOr<GptModelInputs> gatherDecodeModelInput(const StreamGroups& stream_groups,
                                                          TensorHolder&       host_holder) const;

    absl::StatusOr<SamplerInputs>
    gatherSpecSamplerInput(const StreamGroups&                         stream_groups,
                           const GptModelInputs&                       model_inputs,
                           const GptModelOutputs&                      model_output,
                           const SpecLogitsVerifyRunner::LaunchResult& spec_logits_result = {}) const;

    void prepareDecodeDraftModelInput(const StreamGroups& stream_groups,
                                      GptModelInputs&     model_input,
                                      TensorHolder&       host_holder);

    void prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                            GptModelInputs&     model_input,
                                            TensorHolder&       host_holder);

    // Device-state target-verify gather. Returns true only when every stream
    // has CUDA accept_len/tokens/next_seq_len/propose_tokens; otherwise leaves
    // model_input untouched so the caller can use the legacy mixed path.
    bool gatherMtpDecodeModelInputFromDeviceState(const StreamGroups& stream_groups,
                                                  GptModelInputs&     model_input,
                                                  TensorHolder&       host_holder) const;

    void updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                     const GptModelOutputs& model_output,
                                     const torch::Tensor&   draft_token_ids,
                                     TensorHolder&          host_holder);

    void updatePrefillPostDraftModelInput(GptModelInputs&        model_input,
                                          const GptModelOutputs& model_output,
                                          const SamplerOutput&   sampler_output,
                                          TensorHolder&          host_holder);

    // ---- DSpark/DFlash block-diffusion variants -------------------------
    // One non-causal draft block forward per round replaces the MTP
    // shift/decode-chain. Official DSV4 uses [anchor + (k-1)*noise], while
    // DFlash/speculators checkpoints use [anchor + k*mask].

    // Prefill seeding: anchor = target-sampled token, feature window = the
    // computed prompt suffix (prefix-cache reuse keeps its injected KV).
    void updatePrefillPostDSparkDraftModelInput(GptModelInputs&        model_input,
                                                const GptModelOutputs& model_output,
                                                const SamplerOutput&   sampler_output,
                                                TensorHolder&          host_holder);

    // Decode verify input: [anchor, k propose tokens] rectangle [B, k+1].
    void prepareDSparkVerifyModelInput(const StreamGroups& stream_groups,
                                       GptModelInputs&     model_input,
                                       TensorHolder&       host_holder);

    // Token-only draft output for coupled verification, from the per-stream
    // state stored last round: token_ids [B, k] int32; no draft probabilities.
    void updateDSparkDraftSamplerOutput(const StreamGroups& stream_groups,
                                        SamplerOutput&      draft_sampler_output,
                                        torch::Tensor&      draft_token_probs_d_t,
                                        TensorHolder&       host_holder);

    // Decode tail seeding: anchor = last accepted token, feature window =
    // this round's accepted rows of the verify aux export (ctx_lengths =
    // accept_len; overwrites last round's rejected-slot KV by position).
    void updateDecodePostDSparkDraftModelInput(GptModelInputs&                              model_input,
                                               const GptModelOutputs&                       model_output,
                                               const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                               const size_t                                 batch_size,
                                               torch::Tensor&                               hidden_states_d_t,
                                               TensorHolder&                                host_holder);

    // Non-root TP ranks do not run rejection sampling, but NCCL broadcast
    // requires their destination tensors to already have the same logical
    // shapes as rank 0's post-rejection draft input.
    void prepareDSparkDraftReceiverMetadata(GptModelInputs& model_input, int64_t batch_size);

    void updateDecodePostDraftModelInput(GptModelInputs&                              model_input,
                                         const GptModelOutputs&                       model_output,
                                         const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                         const size_t                                 batch_size,
                                         torch::Tensor&                               hidden_states_d_t,
                                         TensorHolder&                                host_holder);

    void updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                         SamplerOutput&      draft_sampler_output,
                                         torch::Tensor&      draft_token_probs_d_t,
                                         TensorHolder&       host_holder);

    void updateMultiStepDraftSamplerOutput(const StreamGroups&         stream_groups,
                                           SamplerOutput&              draft_sampler_output,
                                           torch::Tensor&              draft_token_ids_d_t,
                                           torch::Tensor&              spec_token_ids_d_t,
                                           torch::Tensor&              draft_token_probs_d_t,
                                           std::vector<torch::Tensor>& draft_token_probs_list);

protected:
    void updateProposeTokens(const StreamGroups&                stream_groups,
                             const MergedOutput&                draft_prefill_output,
                             std::vector<StreamSpecUpdateInfo>& spec_update_infos) const;

    void preparePrefillSpecUpdateInfo(const StreamGroups&                stream_groups,
                                      const MergedOutput&                prefill_output,
                                      const MergedOutput&                propose_output,
                                      const torch::Tensor&               draft_last_hidden_states,
                                      const torch::Tensor&               new_tokens_all,
                                      std::vector<StreamSpecUpdateInfo>& spec_update_infos) const;

    void prepareDecodeSpecUpdateInfo(const StreamGroups&                          stream_groups,
                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                     const MergedOutput&                          draft_prefill_output,
                                     std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const;

    void gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const;
    void prepareDSparkSamplingMetadata(const StreamGroups& stream_groups,
                                       GptModelInputs&     model_input,
                                       TensorHolder&       host_holder) const;

protected:
    // Grow-only caches for the dspark per-round constants: their contents
    // depend only on (batch_size, width), so rebuilding them every decode
    // round is pure allocator + launch overhead on the main thread.
    int64_t       dsparkQueryWidth() const;
    torch::Tensor dsparkComboTokens(int64_t batch_size, const torch::Tensor& anchors);
    torch::Tensor dsparkQueryLengths(int64_t batch_size);
    torch::Tensor dsparkDenseCtxLengths(int64_t batch_size);
    torch::Tensor dsparkLmIndexes(int64_t batch_size);

public:
    // Greedy spec-sampler fast path (dspark decode).  When every stream is
    // plain greedy with no logit shaping (penalties, ngram bans, logits
    // processors, beams/tiling) and no probs/logits/loss returns, the
    // coupled verifier's accept decision reads only proposal and target argmax
    // IDs, so no probability buffer exists on either side.
    // The whole sampler-input gather (host loops + O(seq_len) token-history
    // copy) and Sampler::forward (penalty kernels + [B*(k+1), V] softmax)
    // then collapse to one argmax over the verify logits.
    bool          canUseGreedySpecSamplerFastPath(const std::list<GenerateStreamPtr>& streams) const;
    bool          needsDSparkCoupledTargetProbs(const std::list<GenerateStreamPtr>& streams) const;
    SamplerOutput buildGreedySpecSamplerOutput(const torch::Tensor& logits, int64_t batch_size);
    SamplerOutput buildDSparkDraftSamplerOutput(const GptModelOutputs& model_output);

protected:

    int propose_step_;
    // DSpark/DFlash block-diffusion draft (see the DSpark variants above).
    bool    is_dspark_            = false;
    int32_t dspark_mask_token_id_ = -1;
    bool    dspark_sample_from_anchor_ = false;
    bool    dspark_use_gumbel_          = false;
    bool    dspark_use_fp64_gumbel_     = false;

    torch::Tensor dspark_combo_cache_;              // [cap, query_width] int32
    torch::Tensor dspark_query_lengths_cache_;       // [cap] = query_width
    torch::Tensor dspark_dense_ctx_lengths_cache_;   // [cap] = target width k+1
    torch::Tensor dspark_lm_indexes_cache_;           // [cap], query row bases
};
}  // namespace rtp_llm
