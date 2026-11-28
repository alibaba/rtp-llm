#pragma once
#include <climits>
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/models_py/bindings/LinearReplay.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <memory>
#include <torch/extension.h>
#include <torch/python.h>
#include <ATen/Generator.h>
#include <type_traits>

namespace rtp_llm {

enum class ParallelMode {
    TP = 0,
    DP = 1,
    /** Spans all ranks, including every PP stage. */
    WORLD  = 2,
    FFN_TP = 3,
    EP     = 4,
    EPLB   = 5,
    /** Spans all TP/DP ranks in the current PP stage; aliases WORLD when pp_size is one. */
    STAGE = 6,
    /** Spans PP stages with the same DP/TP coordinates. */
    PP = 7,
};

// A batch includes two parts: context batch and decoder batch.
// context batch is request for initial word, decoder batch is request for incremental word.
// ids and lengths are int32_t
struct GptModelInputs {
    // Input-owner-only snapshot captured while assembling this exact batch.
    // Not broadcast to TP peers and never read back from CUDA for recording.
    struct RecordedLengths {
        std::vector<int32_t> q_tokens;
        std::vector<int32_t> kv_tokens_before;
    };
    std::optional<RecordedLengths> record_lengths;
    int64_t                        record_execution_id = 0;
    torch::Tensor                  engram_token_windows;
    torch::Tensor                  mm_features_spans;
    // input_lengths holds original input length for requests,
    // shape [decoder_batch_size + context_batch_size], int32
    // sequence_lengths holds current sequence length for incremental decoding requests,
    // shape [decoder_batch_size], int32
    mutable torch::Tensor combo_tokens;       // [cumulated_seq_len]
    torch::Tensor         input_lengths;      // [batch_size]
    torch::Tensor         sequence_lengths;   // [decoder_batch_size]
    torch::Tensor         lm_output_indexes;  // selected output rows
    // Compact int64 token rows for context scoring; cached/unselected rows are omitted.
    torch::Tensor custom_output_indexes;
    // Kept for ModelInputsLogger/legacy micro-batch consumers; the async
    // scheduling redesign no longer populates it (stays undefined).
    torch::Tensor lm_output_lengths;  // [total_batch_size]
    torch::Tensor prefix_lengths;     // [context_batch_size]
    // First token of this request's FULL cache-store write; zero on its first chunk.
    torch::Tensor p2p_final_prefill;
    torch::Tensor cache_store_publish_start_tokens;  // [context_batch_size], optional
    torch::Tensor sequence_lengths_plus_1;  // optional CUDA mirror for target-verify linear attention
    std::optional<LinearReplayInputs> linear_replay;

    torch::Tensor combo_tokens_type_ids;  // [cumulated_seq_len]
    torch::Tensor combo_position_ids;     // [cumulated_seq_len]

    // for mtp model
    torch::Tensor last_hidden_states;

    torch::Tensor attention_mask;  // [batch_size, seq_len, seq_len]

    // - single-type cache: [batch_size, block_nums]
    // - hybrid cache: [group_nums, batch_size, block_nums]
    torch::Tensor kv_cache_block_id;
    torch::Tensor kv_cache_kernel_block_id;  // [group, batch, kernel_blocks], int32

    std::vector<std::string> kv_cache_group_tags;      // Local payload row -> group identity; not transmitted by TP.
    torch::Tensor            kv_cache_group_types;     // [group_num], int32, Convention: 0 -> LINEAR, 1 -> FULL.
    torch::Tensor            kv_cache_update_mapping;  // [block_copy_num, 3]: payload row, src/dst physical block IDs
    torch::Tensor            kv_cache_copy_token_lengths;  // CPU int32 [copy count]; zero means full copy.

    std::optional<std::vector<torch::Tensor>> multimodal_features;  // all features in gathered stream stored here
    torch::Tensor text_tokens_mask;  // text part in multimodal input tokens [cumulated_seq_len]
    torch::Tensor mm_features_locs;  // features index
    std::optional<std::vector<torch::Tensor>>
        mm_extra_input;  // model-specific extra input (opaque flat 1-D, e.g. deepstack)

    std::optional<std::vector<torch::Tensor>> input_embeddings;  // all input embeddings in gathered stream stored here
    torch::Tensor                             input_embeddings_locs;  // input embeddings index

    torch::Tensor request_id;             // int64, [context_batch_size]
    torch::Tensor request_deadline_ms;    // int64, [context_batch_size], absolute deadline in milliseconds
    // Receiver-local clock offset for P2P layer retention. The tensor remains source identity.
    int64_t       p2p_deadline_clock_offset_ms = 0;
    torch::Tensor request_pd_separation;  // bool, [context_batch_size]
    torch::Tensor cache_keys;             // [context_batch_size]
    // Bytes/physical block/layer for a single group; zero for multiple groups.
    // Per-group addressing must use the corresponding cache layout, not these scalars.
    size_t kv_block_stride_bytes     = 0;
    size_t kv_scale_stride_bytes     = 0;
    size_t seq_size_per_block        = 0;  // tokens/base cache-key block
    size_t kernel_seq_size_per_block = 0;  // tokens/kernel page for single group; 0 for multi-group topology
    bool   pd_separation             = false;
    bool   use_opaque_kv_cache_store = false;

    bool need_all_logits = false;
    // Set when any stream requests return_all_hidden_states. Gates whether the
    // CP prefill exit must materialize the full [seq, hidden] all_hidden_states
    // (true) or may gather only the last-token rows lm_head needs (false).
    bool need_all_hidden_states = false;
    bool skip_lm_head           = false;
    // PP batch contains only middle chunks; all TP ranks follow the same sampling/draft path.
    bool is_middle_chunk_only = false;
    bool capture_hidden_states  = false;
    bool need_moe_gating        = false;
    bool warmup                 = false;
    bool skip_run               = false;
    bool shutdown               = false;
    bool is_fake_stream         = false;

    // Linear attention target verify should write draft tokens mamba states
    // to extra kv_cache blocks when normal inference only write last token mamba state.
    // So, the model has different inference logic for target verify and normal inference.
    // To select correct inference mode, we need to set this flag manually.
    bool is_target_verify = false;
    // V4.1 bounded verification uses packed query rows with ragged request
    // lengths. Independent of is_target_verify, which also classifies proposal.
    bool is_packed_target_verify = false;

    /** Upstream stage activations; excluded from TP input synchronization. */
    std::map<std::string, torch::Tensor> pp_intermediates;
    // MTP draft model state extension after target verification. The outer
    // input remains prefill-shaped (input_lengths + prefix_lengths), while
    // MLA/Indexer must execute as flattened multi-token paged decode.
    bool is_draft_extend = false;

    // not sync to other tp rank
    std::vector<std::string> trace_ids;
    // Logging-only full prefill sequences, including cached prefixes; aligned with request_id.
    std::optional<std::vector<torch::Tensor>> all_input_token_ids;
    std::optional<std::vector<bool>>          request_is_multimodal;

public:
    // Metadata stays with the round; only target verification consumes it.
    const LinearReplayInputs* targetLinearReplay() const {
        return is_target_verify && linear_replay ? &*linear_replay : nullptr;
    }

    std::string debugString(bool force = false) const;
};

struct GptModelOutputs {
    torch::Tensor logits;
    // Same selected LM output rows as logits, independent of need_all_logits.
    torch::Tensor hidden_states;
    torch::Tensor all_hidden_states;
    torch::Tensor all_logits;
    torch::Tensor softmax_result;
    // Request-ordered GLM-5.2 MTP indexer seed. It stays on the draft
    // device and is carried by GenerateStream between decode cycles.
    torch::Tensor mtp_indexer_topk;

    std::vector<torch::Tensor> moe_gating;

    // Explicit model-forward output for speculative target features.  Keeping
    // this separate from all_hidden_states lets the executor distinguish a
    // real graph-instance output from the ordinary decoder hidden tensor.
    torch::Tensor mtp_target_hidden_states;

    // Optional V4.1 confidence-head hidden contribution, CUDA FP32 [B*gamma, 1].
    // Owned by this forward result; the sampler adds the previous-token Markov
    // contribution. Undefined for disabled confidence and non-proposal roles.
    torch::Tensor dspark_confidence_hidden_logits;

    // A single request-visible status. `REPLAYED` means this forward actually
    // executed a captured generation-prefill CUDA Graph; every other non-default value is
    // the readable fallback reason returned through AuxInfo.
    GenerationPrefillCudaGraphStatus generation_prefill_cuda_graph_status{
        GenerationPrefillCudaGraphStatus::NOT_REQUESTED};
    // Selected context rows; undefined when no handler ran on this step.
    torch::Tensor custom_output;
    // The dispatcher turns handler failures into per-stream execution errors.
    std::string custom_output_error;

    /** Downstream activations from a non-last target stage. */
    std::map<std::string, torch::Tensor> pp_intermediates;
};

struct CopyParams {
    const torch::Tensor& dst;
    const torch::Tensor& src;

    void check() const {
        RTP_LLM_CHECK_WITH_INFO(src.scalar_type() == dst.scalar_type(), "copy dst and src need has same type.");
        RTP_LLM_CHECK_WITH_INFO(
            src.nbytes() == dst.nbytes(), "src and dst copy size mismatch: %zu vs %zu", src.nbytes(), dst.nbytes());
    }
};

struct MultiMergeCopyParams {
    void*               dst_ptr;
    std::vector<void*>  src_ptrs;
    std::vector<size_t> copy_size;
    std::vector<size_t> dst_offsets;
};

struct BatchCopyParams {
    enum CopyType : uint32_t {
        D2H = 0,
        H2D = 1,
        D2D = 2,
        H2H = 3,

        // dummy enum indicating number of copy types
        TYPE_SIZE
    };

    struct BatchCopyBuffers {
        std::vector<void*>       dst_ptr;
        std::vector<const void*> src_ptr;
        std::vector<uint64_t>    sizes;
    };

    BatchCopyBuffers copy_buffers[TYPE_SIZE];

    static CopyType  get_copy_type(MemoryType dst_type, MemoryType src_type);
    BatchCopyParams& reserve(CopyType copy_type, size_t size);
    BatchCopyParams& add(void* dst, const void* src, size_t size, CopyType copy_type);
};

struct KvCacheInfo {
    int           layer_num;
    torch::Tensor kv_cache_block_id;  // [batch_size, block_nums], kv cache block offset
    // only meaningful for hybrid cache, per-group block tables, each is [batch_size, block_nums]
    std::vector<torch::Tensor> kv_cache_block_ids_by_group;
    // Base buffer for kv cache blocks. For current cache layout, this represents the base (K) address of kv blocks.
    // V address can be derived by offset/stride when needed.
    torch::Tensor kv_cache_buffer;
    // Optional scale buffer for kv cache quantization (int8/fp8). If set, it should match kv_cache_buffer layout.
    torch::Tensor kv_scale_buffer;
};

struct AttentionCommonInputs {
    // see detailed comments at GptModelInputs
    torch::Tensor input_lengths;     // int32_t, [decoder_batch_size + context_batch_size]
    torch::Tensor sequence_lengths;  // int32_t, [decoder_batch_size]

    std::optional<KvCacheInfo> kv_cache;

    torch::Tensor cu_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor kv_seqlens;
    torch::Tensor padding_offset;

    size_t context_batch_size      = 0;
    size_t decoder_batch_size      = 0;
    size_t context_max_seq_len     = 0;
    size_t decoder_max_seq_len     = 0;
    size_t context_token_num       = 0;
    size_t context_total_kv_length = 0;

    torch::Tensor position_ids;
    torch::Tensor attention_mask;
    torch::Tensor linear_bias_slopes;
    torch::Tensor prefix_prompt_lengths;
    int32_t       max_prefix_length = 0;

    ParamsPtr prefill_flash_infer_attn;
    ParamsPtr decode_flash_infer_attn;
};

struct AttentionModuleParams {
    const int32_t layer_id;
    // qkv shape[h_token_num, (head_num + 2 * kv_head_num) * size_per_head]
    const torch::Tensor& input;
    torch::Tensor&       output;  // shape [token_num, size_per_head]

    AttentionCommonInputs&       common;
    const AttentionLayerWeights& weights;
    const AttentionConfigs&      configs;
    const QScheme                qscheme;
    const DataType               compute_type = DataType::TYPE_INVALID;
};

struct MoeConfigs {
    size_t expert_num;
    size_t extra_expert_num = 0;
    size_t top_k;

    bool   normalize_expert_scale = false;
    bool   has_moe_norm           = false;
    bool   use_all_gather         = false;
    size_t ep_rank                = 0;
    size_t ep_size                = 1;
    size_t tp_rank                = 0;
    size_t tp_size                = 1;
    size_t dp_rank                = 0;
    size_t dp_size                = 1;

    int    scoring_func          = 0;  // 0: softmax, 1: sigmoid
    int    topk_group            = 1;
    int    n_group               = 1;
    double routed_scaling_factor = 1.0;  // used in deepseek v2 and glm4 moe

    bool enable_eplb = false;
};

struct FfnConfigs {
    ActivationType            activation_type;
    std::optional<MoeConfigs> moe_configs = std::nullopt;
};

// Host dispatch decisions for a batch whose row parameters already live on GPU.
// The caller must compute these over exactly the represented request set.
struct SamplingMode {
    bool all_top_k_one = true, all_top_k_unlimited = true;
    bool all_top_p_one = true, all_top_p_exact_one = true, any_top_p_one = false;
    bool temperature_active = false, repetition_active = false, ngram_active = false;
    bool need_do_sample = false, has_not_do_sample = false;
    int  min_top_k = INT_MAX, max_top_k = INT_MIN;
};

struct GreedyParams {
    torch::Tensor logits;            // [batch_size, vocab_size_padded], mutable for in-place penalty
    torch::Tensor input_lengths;     // [batch_size]
    torch::Tensor sequence_lengths;  // [batch_size]
    torch::Tensor token_ids;         // [batch_size, max_input_length + 1]
    size_t        step;

    torch::Tensor top_k;
    torch::Tensor top_p;
    torch::Tensor temperature;

    std::optional<torch::Tensor> repetition_penalty;
    std::optional<torch::Tensor> no_repeat_ngram_size;

    std::optional<torch::Tensor> cum_log_probs;
    std::optional<torch::Tensor> output_log_probs;

    bool return_original_all_probs = false;

    std::optional<torch::Tensor> output_all_probs;
    std::optional<torch::Tensor> presence_penalty;
    std::optional<torch::Tensor> frequency_penalty;
    std::optional<torch::Tensor> do_sample;

    std::vector<at::Generator> generator;

    // Caller declares params.logits private to this sampling round (discarded
    // afterwards): sampleGreedy may skip restoring softmax probabilities into
    // logits and take the fused greedy top-1 path (CUDA). Default false keeps
    // the legacy "logits become probs" in-place contract for every caller.
    bool skip_logits_probs_restore = false;

    // Caller declares that every consumer of output_all_probs treats the
    // renormalized distribution as a point mass at the sampled token (e.g.
    // the speculative rejection kernel's target_probs_point_mass synthesis):
    // the fused greedy top-1 path may then skip materializing the one-hot
    // [batch, vocab] matrix and only write the token. Default false keeps
    // the legacy fully-materialized all_probs contract for every caller.
    bool                         all_probs_point_mass = false;
    int                          force_topk           = 0;
    std::optional<torch::Tensor> output_all_probs_indices;
    // Also return the normalized distribution when output_all_probs requests
    // pre-filter probabilities. Otherwise reuse output_all_probs storage.
    bool return_sampling_probs = false;

    // Sampling execution consumes named tensors on the logits device.
    // sequence_lengths contains effective lengths for all rows; only the first
    // decoder_batch_size rows participate in the no-repeat-ngram kernel.
    SamplingMode  mode;
    torch::Tensor seed, offset, inverse_temperature, sequence_last_indexes, output_ids_ptrs;
    int64_t       decoder_batch_size = 0;
};

// Select the CUDA sampling implementation; ROCm and generic use their own backend.
bool enableCudaSampling();

int64_t samplingCandidateWidth(const GreedyParams& params, const SamplingMode& mode);

struct GreedyOutput {
    torch::Tensor success;
    // True only when the sampler actually ran the token-only point-mass mode
    // and output_all_probs was left unmaterialized; consumers must then treat
    // the distribution as a one-hot at the sampled token. Callers must never
    // read all_probs values when this is set.
    bool          all_probs_point_mass = false;
    torch::Tensor sampling_probs;
};

struct BeamSearchParams {
    // logits is modified inplace to save memory — callers must not reuse it after the call.
    torch::Tensor logits;            // [batch_size, num_beams_in, vocab_size]
    torch::Tensor token_ids;         // [batch_size, num_beams_in, max_seq_len]
    torch::Tensor input_lengths;     // [batch_size, num_beams_in]
    torch::Tensor sequence_lengths;  // [batch_size, num_beams_in]
    torch::Tensor cum_log_probs;     // [batch_size, num_beams_in]
    size_t        num_beams_out = 0;
};

struct BeamSearchOutput {
    torch::Tensor token_ids;         // [batch_size, num_beams_out, max_seq_len]
    torch::Tensor input_lengths;     // [batch_size, num_beams_out]
    torch::Tensor sequence_lengths;  // [batch_size, num_beams_out]
    torch::Tensor cum_log_probs;     // [batch_size, num_beams_out]
    torch::Tensor beam_indices;      // [batch_size, num_beams_out]
};

struct BroadcastParams {
    const std::vector<torch::Tensor>& buffers;
    const int64_t                     root;
    ParallelMode                      mode = ParallelMode::TP;
};

enum class ReduceOp {
    Sum  = 0,
    Prod = 1,
    Max  = 2,
    Min  = 3,
    Avg  = 4,
};

struct AllReduceParams {
    torch::Tensor  buffer;
    const ReduceOp op;
    ParallelMode   mode = ParallelMode::TP;
    torch::Tensor  dest;  // undefined = no separate dest
};

struct AllReduceOutput {
    torch::Tensor buffer;
};

struct AllGatherParams {
    const std::vector<torch::Tensor>& recv_buffers;
    ParallelMode                      mode = ParallelMode::TP;
    std::vector<torch::Tensor>        send_buffers;
    bool                              inplace = true;
};

struct RejectionSamplingParams {
    torch::Tensor draft_probs_d;
    torch::Tensor draft_token_ids_d;
    torch::Tensor uniform_samples_d;
    torch::Tensor target_probs_d;
    torch::Tensor target_token_ids_d;
    torch::Tensor output_token_ids_d;
    torch::Tensor output_accepted_token_num_d;
    torch::Tensor do_sample_d;
    /**
     * True selects an implicit point-mass proposal distribution: q is 1 at
     * each draft token and 0 elsewhere. The kernel does not read draft_probs_d,
     * so that tensor may be undefined.
     * PP uses this for greedy MTP, DSpARK handoff padding and fake streams.
     * Normal DSpARK proposals carry sampled tokens and dense q; keep this
     * false so verification uses the actual proposal distribution.
     */
    bool draft_probs_point_mass = false;
    // True when target_probs_d is a degenerate point mass on the last column
    // of target_token_ids_d (greedy top_k==1 verification may skip
    // materializing the one-hot matrix). The kernel then synthesizes q as
    // one-hot(target_token) instead of reading target_probs_d, bit-identical
    // to the materialized matrix. target_probs_d must still be defined with
    // the [batch, num_speculative_tokens + 1, vocab] shape.
    bool target_probs_point_mass = false;
    // Optional [batch] bool CUDA mask. A true row receives the force-accept
    // override: all draft tokens + the target bonus token (last column of the
    // row's final target_token_ids_d slot) with accept count
    // num_speculative_tokens + 1 — the exact values the legacy eager
    // torch::where chain wrote. Only honored when finalize_outputs is true;
    // leave undefined when no stream forces acceptance.
    torch::Tensor force_mask_d;
    // True folds the caller's post-rejection finalization into the kernel:
    // the per-row force override above plus the -1 -> 0 padding reset, in the
    // legacy order. With this flag the kernel writes every element of
    // output_token_ids_d and output_accepted_token_num_d on every path, so
    // callers may pass uninitialized (torch::empty) buffers. CUDA backend
    // only; query execRejectionSamplingSupportsFinalize() before setting —
    // ROCm/generic reject the flag loudly and keep the legacy eager chain.
    bool finalize_outputs = false;
    // Optional contiguous CUDA int32 [batch], each in [0, num_speculative_tokens].
    // Only this prefix participates in rejection; all accepted uses target row
    // L as bonus (including L=0). Undefined preserves the full-width path.
    // CUDA only: other backends must reject a defined tensor explicitly.
    torch::Tensor valid_draft_lengths_d;
    // Optional DSpARK compact probabilities. Target indices are original
    // vocabulary IDs [B, steps+1, K]; draft indices are [B, steps, K].
    // When draft indices are absent, draft_probs_d is dense [B, steps, V].
    torch::Tensor draft_probs_indices_d;
    torch::Tensor target_probs_indices_d;
    // Optional CUDA int32 [batch], written by the same rejection kernel.
    // Counts target-verified candidates L (or full gamma), before output clipping.
    torch::Tensor output_verified_draft_lengths_d;
};

struct RejectionSamplingLaunchConfig {
    int batch_size;
    int num_speculative_tokens;
    int target_vocab_size;
    int target_token_stride;
};

// One DSpARK draft step's fused distribution build:
//   q = softmax((logits + markov_bias) / temperature)
// written to both the strided all-probabilities slice and a contiguous
// sampling scratch. Replaces the eager cast/add/div_/softmax/copy_ chain.
struct MarkovSoftmaxStepParams {
    torch::Tensor logits;       // FP32 [B, V] row-strided view (proposal_logits.select(1, step))
    torch::Tensor markov_bias;  // [B, V] contiguous FP32/BF16/FP16 torch::mm output (native weight dtype)
    torch::Tensor temperature;  // FP32 [B] contiguous CUDA
    torch::Tensor q_all_probs;  // FP32 [B, V] row-strided view (all_probabilities.select(1, step))
    torch::Tensor q_contig;     // FP32 [B, V] contiguous sampling scratch
};

// One DSpARK draft step's fused softmax-free token for an all-greedy batch:
//   token = first argmax of (logits + markov_bias) / temperature
// Greedy verification accepts on exact target-token matches, so the caller
// can publish deterministic proposals as a point mass (token_ids_are_point_mass).
struct MarkovArgmaxStepParams {
    torch::Tensor logits;       // FP32 [B, V] row-strided view (proposal_logits.select(1, step))
    torch::Tensor markov_bias;  // [B, V] contiguous FP32/BF16/FP16 torch::mm output (native weight dtype)
    torch::Tensor temperature;  // FP32 [B] contiguous CUDA
};

struct MappingDraft2TargetParams {
    torch::Tensor tokens;
    torch::Tensor d2t_map;
    int           batch_size;
    int           token_offset;
    int           token_stride;
};

}  // namespace rtp_llm
