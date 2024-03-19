#pragma once

#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>

#include <mutex>
#include <condition_variable>

namespace th = torch;

namespace rtp_llm {

// TODO: complete params.
// TODO: implement hash function to bind with sampler.

// NOTE: The params in generate config should be splitted into two parts:
//       1. The params that can be different for a single sampler.
//       e.g. top_k, top_p, temperature, repetition_penalty, etc.
//       2. The params that must be the same for a single sampler.
//       e.g. beam_size, max_seq_len, etc.
//       For the second part, different samplers should be created for different params.
//       So they can not be batched together for now.

class GenerateConfig : public th::jit::CustomClassHolder {
public:
    int64_t max_seq_len = 8192;
    int64_t max_new_tokens = 8192;
    int64_t num_validate_token = 0; // for speculative decoding validation.

    int64_t beam_size = 1;
    th::optional<int64_t> top_k;
    th::optional<double> top_p;
    th::optional<double> temperature;
    th::optional<double> repetition_penalty;
    th::optional<double> presence_penalty;
    th::optional<int64_t> min_length;
    th::optional<double> length_penalty;
    th::optional<double> beam_search_diversity_rate;
    th::optional<int64_t> random_seed;
    th::optional<double> top_p_decay;
    th::optional<double> top_p_min;
    th::optional<int64_t> top_p_reset_ids;
};

// TODO: add error code.
class ErrorInfo : public th::jit::CustomClassHolder {
public:
    bool has_error = false;
    std::string error_message;
};

class QueryRequest : public th::jit::CustomClassHolder {
public:
    // NOTE: Every query must be assigned with a unique request_id.
    //       This is related to the cache mechanism.
    //       This id can be passed from outside, or generated during construction.
    std::string request_id;
    th::intrusive_ptr<GenerateConfig> generate_config;
    th::Tensor input_ids;
    // For multi-modality models, embedding might be calculated outside
    th::optional<th::Tensor> input_embeddings;
};

class GenerateResponse : public th::jit::CustomClassHolder {
public:
    th::Tensor output_token_ids;
    th::Tensor finished;
    bool all_finished;
    th::intrusive_ptr<ErrorInfo> error_info;

    th::optional<th::Tensor> log_probs;
    th::optional<th::Tensor> hidden_states;
    th::optional<th::Tensor> attentions;
    th::optional<th::Tensor> logits;
    th::optional<th::Tensor> loss;
};

class MagaQuery : public th::jit::CustomClassHolder {
public:
    MagaQuery(const th::intrusive_ptr<QueryRequest> &query);
    ~MagaQuery();

    // Exported to python world.
    th::intrusive_ptr<GenerateResponse> next_response();
    void cancel();

    // Only used in C++ world.
    void push_response(const th::intrusive_ptr<GenerateResponse> &response);
    bool is_done();

    // TODO: justify whether cache blocks should be managed here.
    void add_cache_block(const std::vector<int64_t> &kv_cache_block);

public:
    th::intrusive_ptr<QueryRequest> query;
    std::vector<std::vector<int64_t>> kv_cache_blocks;

private:
    size_t batch_size_;

    bool done_;
    bool cancelled_;
    th::intrusive_ptr<GenerateResponse> current_response_;
    std::mutex response_mutex_;
    std::condition_variable update_cv_;
};

} // namespace rtp_llm
