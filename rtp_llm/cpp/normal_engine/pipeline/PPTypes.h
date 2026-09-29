#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

/** Internal numeric request identity, exposed by GenerateStream::streamId(). */
using RequestId = int64_t;

struct RequestLogitsProcessorConfig {
    std::string grammar_type;   // scalar per stream
    std::string grammar_value;  // scalar per stream

    int                           combo_token_size = 0;                   // scalar per stream
    std::vector<std::vector<int>> banned_combo_token_ids;                 // [banned_combo_count, combo_token_size]
    std::vector<int>              end_think_token_ids;                    // [end_think_token_count]
    bool                          enable_cross_sequence_ban     = false;  // scalar per stream
    int                           cross_seq_diverge_start_combo = 0;      // scalar per stream
};

struct PPSamplingPlan {
    std::vector<std::optional<int>>           random_seeds;              // [stream_count]
    std::vector<RequestLogitsProcessorConfig> logits_processor_configs;  // [stream_count]
    std::vector<int32_t>                      num_return_sequences;      // [stream_count]

    torch::Tensor request_ids;  // [stream_count]

    /** CPU int32 packed suffixes per sequence; the first sampling plan carries the complete history. */
    torch::Tensor token_ids;
    /** CPU int64 [total_batch_size + 1], boundaries into token_ids. Undefined when sampling is skipped. */
    torch::Tensor token_offsets;
    torch::Tensor input_lengths;     // [total_batch_size]
    torch::Tensor sequence_lengths;  // [total_batch_size]
    torch::Tensor max_tokens;        // [total_batch_size], total sequence limit from stream->maxTokenNum()

    torch::Tensor top_k;                 // [total_batch_size]
    torch::Tensor top_p;                 // [total_batch_size]
    torch::Tensor temperature;           // [total_batch_size]
    torch::Tensor repetition_penalty;    // [total_batch_size]
    torch::Tensor presence_penalty;      // [total_batch_size]
    torch::Tensor frequency_penalty;     // [total_batch_size]
    torch::Tensor no_repeat_ngram_size;  // [total_batch_size]
    torch::Tensor do_sample;             // [total_batch_size]
    torch::Tensor finished_mask;         // [total_batch_size]

    /** CPU bool [stream_count]: stochastic() controls both rejection sampling and DSpARK temperature. */
    torch::Tensor spec_do_sample;
    torch::Tensor force_sp_accept;  // [stream_count]
};

struct PPPromptLogitsRequest {
    bool enabled               = false;
    int  top_k                 = 64;
    int  start                 = -1;
    int  end                   = -1;
    bool return_target_logprob = true;
};

struct PPRequestOutputConfig {
    bool return_logits            = false;
    bool return_softmax_probs     = false;
    bool return_cum_log_probs     = false;
    bool calculate_loss           = false;
    bool return_hidden_states     = false;
    bool return_all_hidden_states = false;
    bool return_all_probs         = false;

    PPPromptLogitsRequest prompt_logits;
};

/**
 * The first stage aggregates requirements for batch computation and D2H copies.
 * Per-request flags determine which output slices are returned.
 */
struct PPOutputConfig {
    bool               return_logits            = false;
    bool               return_softmax_probs     = false;
    bool               return_cum_log_probs     = false;
    bool               calculate_loss           = false;
    bool               return_hidden_states     = false;
    bool               return_all_hidden_states = false;
    ReturnAllProbsMode return_all_probs         = ReturnAllProbsMode::NONE;

    /** Aligned with PPSamplingPlan::request_ids. */
    std::vector<PPRequestOutputConfig> requests;
};

struct PPExecutionPlan {
    GptModelInputs       model_input;
    PPSamplingPlan       sampling_plan;
    PPOutputConfig       output_config;
    std::vector<RequestId> finished_request_ids;

    bool          is_decode = false;        // Request phase; verify may use a prefill input shape
    torch::Tensor draft_next_position_ids;  // [stream_count * position_id_len_factor], MTP prefill
};

/** CPU outputs for one request; unrequested fields remain undefined. */
struct PPRequestOutput {
    torch::Tensor logits;            /** [request_batch_size, vocab_size] */
    torch::Tensor softmax_probs;     /** [request_batch_size, new_token_count] */
    torch::Tensor cum_log_probs;     /** [request_batch_size] */
    torch::Tensor all_probs;         /** [request_batch_size, vocab_size] */
    torch::Tensor loss;              /** [loss_token_count], shared by return sequences */
    torch::Tensor hidden_states;     /** [request_batch_size, hidden_size] */
    torch::Tensor all_hidden_states; /** [request_executed_token_count, hidden_size] */

    std::optional<PromptLogitsOutput> prompt_logits;
};

/** Final outputs produced by the lm-head stage TP root. */
struct PPExecutionResult {
    /** Last-stage ordinary sampler input preparation and forward call, excluding result packing. */
    int64_t sample_input_us = 0;

    torch::Tensor request_ids;  // [stream_count]
    /** New tokens to append: CPU int32 [B, 1], or [B, K+1] for verify. */
    torch::Tensor new_token_ids;
    /** Committed prefix length per row in new_token_ids, before applying the request length limit: CPU int32 [B].*/
    torch::Tensor new_token_lengths;

    /** One slot per request_id, including requests with no optional output or with errors. */
    std::vector<PPRequestOutput> request_outputs;

    /** First error per request; failed-row placeholders are not committed. */
    std::vector<ErrorInfo> request_errors;

    /**
     * Draft candidates for the next verification, excluding the committed anchor.
     * PD PREFILL returns one candidate for MTP/EAGLE and none for DSpARK.
     * MTP/EAGLE PREFILL also includes Q and final draft hidden because D may run without PP:
     * Q verifies the carried candidate; hidden continues multi-step drafting.
     * Other roles leave Q and hidden undefined.
     */
    torch::Tensor propose_token_ids;     /** CPU int32 [B, K], target vocabulary. */
    torch::Tensor propose_all_probs;     /** Optional CPU FP32 [B, 1, draft_vocab_size]. */
    torch::Tensor propose_hidden_states; /** Optional CPU [B, H]. */
};

}  // namespace rtp_llm
