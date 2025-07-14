#include <torch/torch.h>

#include "rtp_llm/cpp/devices/OpData.h"

#include <tuple>

namespace rtp_llm {

torch::Tensor create_context_mask(const std::vector<int32_t>& input_lengths, bool is_causal = true) {
    int32_t       batch_size       = input_lengths.size();
    int32_t       max_input_length = *std::max_element(input_lengths.begin(), input_lengths.end());
    torch::Tensor attention_mask   = torch::ones({max_input_length, max_input_length}, torch::dtype(torch::kBool));
    if (is_causal) {
        attention_mask = attention_mask.tril();
    }
    attention_mask = attention_mask.unsqueeze(0).repeat({batch_size, 1, 1});
    for (int32_t b = 0; b < batch_size; ++b) {
        int32_t input_length                                       = input_lengths[b];
        attention_mask[b].slice(0, input_length, max_input_length) = 0;
        if (!is_causal) {
            attention_mask[b].slice(1, 0, input_length) = 0;
        }
    }
    return attention_mask;
}

torch::Tensor create_position_ids(const std::vector<int32_t>& input_lengths) {
    std::vector<torch::Tensor> tensors;
    for (int32_t i = 0; i < input_lengths.size(); ++i) {
        int32_t       input_length = input_lengths[i];
        torch::Tensor position_ids = torch::arange(input_length, torch::kInt32);
        tensors.push_back(position_ids);
    }
    return torch::concat(tensors, 0);
}

torch::Tensor rotate_half(const torch::Tensor& x) {
    torch::Tensor x1 = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/x.size(-1) / 2);
    torch::Tensor x2 = x.slice(/*dim=*/-1, /*start=*/x.size(-1) / 2, /*end=*/x.size(-1));
    return torch::cat({-x2, x1}, /*dim=*/-1);
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(const torch::Tensor& q,
                                                              const torch::Tensor& k,
                                                              const torch::Tensor& cos,
                                                              const torch::Tensor& sin,
                                                              const torch::Tensor& position_ids,
                                                              int64_t              unsqueeze_dim = 1) {
    auto cos_pos = (position_ids.defined() ? cos.index_select(/*dim=*/0, position_ids) : cos)
                       .unsqueeze(0)
                       .unsqueeze(unsqueeze_dim);
    auto sin_pos = (position_ids.defined() ? sin.index_select(/*dim=*/0, position_ids) : sin)
                       .unsqueeze(0)
                       .unsqueeze(unsqueeze_dim);

    auto q_rot_half = rotate_half(q);
    auto k_rot_half = rotate_half(k);
    auto q_embed    = (q * cos_pos) + (q_rot_half * sin_pos);
    auto k_embed    = (k * cos_pos) + (k_rot_half * sin_pos);

    return std::make_tuple(q_embed, k_embed);
}

class RotaryEmbedding: public torch::nn::Module {
public:
    RotaryEmbedding() = default;
    RotaryEmbedding(int64_t       dim,
                    int64_t       max_position_embeddings = 2048,
                    int64_t       base                    = 10000,
                    torch::Device device                  = torch::kCPU):
        dim_(dim), max_position_embeddings_(max_position_embeddings), base_(base) {
        inv_freq_ = 1.0 / torch::pow(base, torch::arange(0, dim, 2, torch::kInt64).to(torch::kFloat32) / dim);
        printf("rope dim=%ld, max_position_embeddings=%ld, base=%ld\n", dim_, max_position_embeddings_, base_);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, int64_t seq_len) {
        torch::Tensor t       = torch::arange(seq_len, x.device());
        torch::Tensor freqs   = torch::outer(t, inv_freq_);
        torch::Tensor emb     = torch::cat({freqs, freqs}, -1);
        torch::Tensor cos_emb = emb.cos().to(x.dtype());
        torch::Tensor sin_emb = emb.sin().to(x.dtype());
        return std::make_tuple(cos_emb, sin_emb);
    }

private:
    int64_t       dim_;
    int64_t       max_position_embeddings_;
    int64_t       base_;
    torch::Tensor inv_freq_;
};

torch::Tensor repeat_kv(torch::Tensor hidden_states, int n_rep) {
    auto batch               = hidden_states.size(0);
    auto num_key_value_heads = hidden_states.size(1);
    auto slen                = hidden_states.size(2);
    auto head_dim            = hidden_states.size(3);

    if (n_rep == 1) {
        return hidden_states;
    }

    hidden_states = hidden_states.unsqueeze(2).expand({batch, num_key_value_heads, n_rep, slen, head_dim});
    return hidden_states.reshape({batch, num_key_value_heads * n_rep, slen, head_dim});
}

class GptAttentionImpl: public torch::nn::Module {
public:
    GptAttentionImpl(const AttentionConfigs& config):
        hidden_size(config.head_num * config.size_per_head),
        num_heads(config.head_num),
        head_dim(config.size_per_head),
        num_key_value_heads(config.kv_head_num),
        num_key_value_groups(num_heads / num_key_value_heads),
        max_position_embeddings(config.rope_config.max_pos),
        rope_theta(config.rope_config.base),
        q_proj(torch::nn::Linear(hidden_size, num_heads * head_dim)),
        k_proj(torch::nn::Linear(hidden_size, num_key_value_heads * head_dim)),
        v_proj(torch::nn::Linear(hidden_size, num_key_value_heads * head_dim)),
        o_proj(torch::nn::Linear(num_heads * head_dim, hidden_size)),
        rotary_emb(RotaryEmbedding(head_dim, max_position_embeddings, rope_theta)) {}

    torch::Tensor forward(torch::Tensor hidden_states,
                          torch::Tensor attention_mask = torch::Tensor(),
                          torch::Tensor position_ids   = torch::Tensor()) {
        auto bsz   = hidden_states.size(0);
        auto q_len = hidden_states.size(1);
        std::cout << "hidden: " << hidden_states << std::endl;

        auto query_states = q_proj->forward(hidden_states);
        auto key_states   = k_proj->forward(hidden_states);
        auto value_states = v_proj->forward(hidden_states);

        query_states = query_states.view({bsz, q_len, num_heads, head_dim}).transpose(1, 2);
        key_states   = key_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);
        value_states = value_states.view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);

        auto [cos, sin] = rotary_emb.forward(value_states, q_len);

        std::tie(query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids);

        key_states   = repeat_kv(key_states, num_key_value_groups);
        value_states = repeat_kv(value_states, num_key_value_groups);

        auto attn_weights = torch::matmul(query_states, key_states.transpose(2, 3)) / sqrtf(head_dim * 1.0f);
        if (attention_mask.defined()) {
            // NOTE: the definition of mask is different in transformers and rtp_llm
            // in transformers, attention_mask is bias added to the attention weights
            // in rtp_llm, attention_mask is a binary value with 0s in the positions to be masked
            // this line of code transforms the rtp_llm mask to the transformers mask
            attention_mask = (1 - attention_mask) * -10000.0f;
            std::cout << "use attention mask: " << attention_mask << std::endl;
            attn_weights = attn_weights + attention_mask;
        }

        attn_weights = torch::softmax(attn_weights, -1, torch::kFloat32).to(query_states.dtype());

        auto attn_output = torch::matmul(attn_weights, value_states);
        attn_output      = attn_output.transpose(1, 2).contiguous();
        attn_output      = attn_output.view({bsz, q_len, hidden_size});
        attn_output      = o_proj->forward(attn_output);

        return attn_output;
    }

public:
    int    hidden_size;
    int    num_heads;
    int    head_dim;
    int    num_key_value_heads;
    int    num_key_value_groups;
    int    max_position_embeddings;
    double rope_theta;
    bool   is_causal;
    double attention_dropout;

    torch::nn::Linear q_proj, k_proj, v_proj, o_proj;
    RotaryEmbedding   rotary_emb;
};

TORCH_MODULE(GptAttention);
};  // namespace rtp_llm
