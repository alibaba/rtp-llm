#pragma once

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
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
        dspark_vocab_size_(model_config.vocab_size) {}

    absl::Status dispatchPrefill(const StreamGroups& stream_groups,
                                 const MergedOutput& prefill_output,
                                 const MergedOutput& propose_output) const;

    absl::Status dispatchDecode(const StreamGroups&                          stream_groups,
                                const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                const MergedOutput&                          draft_prefill_output) const;

    absl::StatusOr<GptModelInputs> gatherDecodeModelInput(const StreamGroups& stream_groups,
                                                          TensorHolder&       host_holder) const;

    absl::StatusOr<SamplerInputs> gatherSpecSamplerInput(const StreamGroups&    stream_groups,
                                                         const GptModelInputs&  model_inputs,
                                                         const GptModelOutputs& model_output) const;

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
    // shift/decode-chain: the draft input is always [anchor + k*mask] per
    // stream, plus the target aux features to inject as draft context KV.

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

    // Draft sampler output for rejection sampling, from the per-stream state
    // stored last round: token_ids [B, k] int32, all_probs [B, k, vocab].
    void updateDSparkDraftSamplerOutput(const StreamGroups& stream_groups,
                                        SamplerOutput&      draft_sampler_output,
                                        torch::Tensor&      draft_token_probs_d_t,
                                        TensorHolder&       host_holder);

    // Decode tail seeding (dense): anchor = last accepted token; feature
    // window is a fixed k+1-wide slice of the verify aux export based at
    // dspark_ctx_starts = old prefix (rows past accept_len carry garbage by
    // design — see the write-before-read note in the .cc). Overwrites last
    // round's rejected-slot KV by position.
    void updateDecodePostDSparkDraftModelInput(GptModelInputs&                              model_input,
                                               const GptModelOutputs&                       model_output,
                                               const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                               const size_t                                 batch_size,
                                               torch::Tensor&                               hidden_states_d_t,
                                               TensorHolder&                                host_holder);

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
                                      const torch::Tensor&               new_tokens_all,
                                      std::vector<StreamSpecUpdateInfo>& spec_update_infos) const;

    void prepareDecodeSpecUpdateInfo(const StreamGroups&                          stream_groups,
                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                     const MergedOutput&                          draft_prefill_output,
                                     std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const;

    void gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const;

protected:
    // Grow-only caches for the dspark per-round constants: their contents
    // depend only on (batch_size, width), so rebuilding them every decode
    // round is pure allocator + launch overhead on the main thread.
    torch::Tensor dsparkComboTokens(int64_t batch_size, const torch::Tensor& anchors);
    torch::Tensor dsparkCtxLengths(int64_t batch_size);
    torch::Tensor dsparkLmIndexes(int64_t batch_size);
    // Zero-filled stand-in consumed by the rejection kernel when a greedy-only
    // stream's draft probs were dropped on the PD wire: the kernel still
    // dereferences the [B, k, V] buffer, but a greedy row's accept decision
    // never depends on its contents (same_token short-circuit).
    torch::Tensor dsparkDummyProbs(int64_t batch_size);

    int propose_step_;
    // DSpark/DFlash block-diffusion draft (see the DSpark variants above).
    bool    is_dspark_            = false;
    int32_t dspark_mask_token_id_ = -1;

    int64_t dspark_vocab_size_ = 0;

    torch::Tensor dspark_combo_cache_;        // [cap, width] int32, mask-filled
    torch::Tensor dspark_ctx_lengths_cache_;  // [cap] int32, filled with width
    torch::Tensor dspark_lm_indexes_cache_;   // [cap] int32, arange(cap) * width
    torch::Tensor dspark_dummy_probs_;        // [cap, k, vocab] fp32 zeros
};
}  // namespace rtp_llm
