#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;

namespace rtp_llm {

extern size_t get_rhs_packed_size(int n, int k);

void softmax2(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float max_val = *std::max_element(input + i * cols, input + (i + 1) * cols);
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            output[i * cols + j] = std::exp(input[i * cols + j] - max_val);
            sum += output[i * cols + j];
        }
        for (int j = 0; j < cols; ++j) {
            output[i * cols + j] /= sum;
        }
    }
}

void sigmoid(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i * cols + j] = 1.0f / (1.0f + std::exp(-input[i * cols + j]));
        }
    }
}

void topKSelection(const float* input, int* output, float* values, int rows, int num_expert, int k) {
    for (int i = 0; i < rows; ++i) {
        std::vector<std::pair<float, int>> scores;
        for (int j = 0; j < num_expert; ++j) {
            scores.emplace_back(input[i * num_expert + j], j);
        }
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first;
                          });
        for (int j = 0; j < k; ++j) {
            output[i * k + j] = scores[j].second;
            values[i * k + j] = scores[j].first;
        }
    }
}

void topKSelectionNoauxTc(const float* input, int* index, float* scores, const float* bias, int rows,
                          int num_expert, int k, int n_group, int topk_group) {
    int group_size = num_expert / n_group;
    for (int i = 0; i < rows; ++i) {
        std::vector<std::pair<float, int>> scores_for_choice;
        for (int j = 0; j < num_expert; ++j) {
            float score = input[i * num_expert + j] + bias[j];
            scores_for_choice.emplace_back(score, j);
        }

        if (n_group > 1) {
            // compute group score (sum of top 2)
            std::vector<std::pair<float, int>> group_scores;
            for (int j = 0; j < n_group; ++j) {
                float max1 = -std::numeric_limits<float>::infinity();
                float max2 = -std::numeric_limits<float>::infinity();
                for (int g = 0; g < group_size; ++g) {
                    float s = scores_for_choice[j * group_size + g].first;
                    if (s > max1) {
                        max2 = max1;
                        max1 = s;
                    } else if (s > max2) {
                        max2 = s;
                    }
                }
                group_scores.emplace_back(max1 + max2, j);
            }
            // find topk_group groups
            std::partial_sort(group_scores.begin(), group_scores.begin() + topk_group, group_scores.end(),
                            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                                return a.first > b.first;
                            });
            // mask out scores_for_choice for non topk groups
            for (int j = topk_group; j < n_group; ++j) {
                int group_idx = group_scores[j].second;
                for (int g = 0; g < group_size; ++g) {
                    int idx = group_idx * group_size + g;
                    scores_for_choice[idx].first = -std::numeric_limits<float>::infinity();
                }
            }
        }

        // Sort the scores_for_choice to get the top k experts and their scores
        std::partial_sort(scores_for_choice.begin(), scores_for_choice.begin() + k, scores_for_choice.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first;
                          });
        for (int j = 0; j < k; ++j) {
            int expert_idx = scores_for_choice[j].second;
            index[i * k + j] = expert_idx;
            scores[i * k + j] = input[i * num_expert + expert_idx];
        }
    }
}

void accumulate_output(size_t hidden_dim, float16_t* output, const float16_t* down_output, float16_t weight) {
    int h;
    for (h = 0; h <= hidden_dim - 8; h += 8) {
        float16x8_t out_vec = vld1q_f16(output + h);
        float16x8_t down_vec = vld1q_f16(down_output + h);
        out_vec = vfmaq_n_f16(out_vec, down_vec, weight);
        vst1q_f16(output + h, out_vec);
    }
    for (; h < hidden_dim; h++) {
        output[h] += weight * down_output[h];
    }
}

void accumulate_output(size_t hidden_dim, float* output, const float* down_output, float weight) {
    int h;
    for (h = 0; h <= hidden_dim - 4; h += 4) {
        float32x4_t out_vec = vld1q_f32(output + h);
        float32x4_t down_vec = vld1q_f32(down_output + h);
        out_vec = vmlaq_n_f32(out_vec, down_vec, weight);
        vst1q_f32(output + h, out_vec);
    }
    for (; h < hidden_dim; h++) {
        output[h] += weight * down_output[h];
    }
}

extern size_t get_lhs_packed_size_kai_a8w4(int* m_array, int k, int bs, size_t* offsets);
extern void batch_pack_lhs_kai_a8w4(const float16_t* input, const size_t* input_offsets, uint8_t* output, const size_t* output_offsets, int* m_array, int k, int bs);
extern void batch_matmul_kai_a8w4(const uint8_t* input, const size_t* input_offsets,
    const uint8_t* weight, const size_t* weight_offsets,
    float16_t* output, const size_t* output_offsets,
    int* m_array, int k, int n, size_t output_stride,
    int bs);

FfnLayerOutput ArmCpuDevice::moe_ffn_a8w4(const BufferPtr expert_indices, const BufferPtr expert_weights, const BufferPtr output, const FfnLayerParams& params) {
    const auto& hidden = params.input;
    const auto token_num = hidden.shape()[0];
    const auto hidden_dim = hidden.shape()[1];
    const auto num_expert = params.weights.moe_gating_weight->kernel->shape()[1];

    const auto& moe_conf = params.configs.moe_configs.value();
    const auto top_k = moe_conf.top_k;
    const size_t moe_inter_size = moe_conf.moe_inter_padding_size;

    std::vector<int> m_array;
    std::vector<int> activated_experts;
    std::vector<std::vector<int>> expert_to_tokens(num_expert);
    std::vector<std::vector<float16_t>> expert_to_weights(num_expert);
    std::vector<std::vector<int>> token_to_indices(token_num);
    std::vector<std::vector<float16_t>> token_to_weights(token_num);

    // Allocate input buffer for the experts
    auto input_buffer = allocateBuffer({DataType::TYPE_FP16, {token_num + token_num * top_k, hidden_dim}});
    std::vector<size_t> input_offsets;

    for (int i = 0; i < token_num; ++i) {
        for (int j = 0; j < top_k; ++j) {
            int expert_idx = *(int*)(expert_indices->dataWithOffset(i * top_k + j));
            float weight = *(float*)(expert_weights->dataWithOffset(i * top_k + j));
            expert_to_tokens[expert_idx].push_back(i);
            expert_to_weights[expert_idx].push_back((float16_t)weight);
        }
    }

    // hack: shared expert
    m_array.push_back(token_num);
    input_offsets.push_back(0);

    int idx = token_num;
    for (int i = 0; i < num_expert; ++i) {
        if (expert_to_tokens[i].empty()) {
            continue;
        }
        activated_experts.push_back(i);
        m_array.push_back(expert_to_tokens[i].size());
        input_offsets.push_back(idx * hidden_dim);
        for (int j = 0; j < expert_to_tokens[i].size(); ++j) {
            int token_idx = expert_to_tokens[i][j];
            float16_t weight = expert_to_weights[i][j];
            token_to_indices[token_idx].push_back(idx);
            token_to_weights[token_idx].push_back(weight);
            idx++;
        }
    }

    #pragma omp parallel for if (token_num > 1)
    for (int i = 0; i < token_num; ++i) {
        // shared expert
        memcpy(input_buffer->dataWithOffset(i * hidden_dim), hidden.dataWithOffset(i * hidden_dim), hidden_dim * sizeof(float16_t));

        for (int j = 0; j < top_k; ++j) {
            int idx = token_to_indices[i][j];
            memcpy(input_buffer->dataWithOffset(idx * hidden_dim), hidden.dataWithOffset(i * hidden_dim), hidden_dim * sizeof(float16_t));
        }
    }

    int bs = activated_experts.size();

    // Calculate offsets for weights and outputs
    auto up_gate_weight_packed_size = get_rhs_packed_size(moe_inter_size * 2, hidden_dim);
    auto down_weight_packed_size = get_rhs_packed_size(hidden_dim, moe_inter_size);
    std::vector<size_t> up_weight_offsets(1 + bs);
    std::vector<size_t> gate_weight_offsets(1 + bs);
    std::vector<size_t> down_weight_offsets(1 + bs);
    std::vector<size_t> up_gate_output_offsets(1 + bs);

    // hack: shared expert
    up_weight_offsets[0] = (uint8_t*)params.weights.shared_expert->up_weight->kernel->data() - (uint8_t*)params.weights.moe_gate_weight->kernel->data();
    gate_weight_offsets[0] = (uint8_t*)params.weights.shared_expert->gate_weight->kernel->data() - (uint8_t*)params.weights.moe_gate_weight->kernel->data();
    down_weight_offsets[0] = (uint8_t*)params.weights.shared_expert->down_weight->kernel->data() - (uint8_t*)params.weights.moe_down_weight->kernel->data();
    up_gate_output_offsets[0] = 0;
    size_t offset = m_array[0];

    for (int i = 0; i < bs; ++i) {
        up_weight_offsets[1 + i] = activated_experts[i] * up_gate_weight_packed_size;
        gate_weight_offsets[1 + i] = activated_experts[i] * up_gate_weight_packed_size + up_gate_weight_packed_size / 2;
        down_weight_offsets[1 + i] = activated_experts[i] * down_weight_packed_size;
        up_gate_output_offsets[1 + i] = offset * moe_inter_size;
        offset += m_array[1 + i];
    }

    // up gate projection
    std::vector<size_t> packed_input_offsets(1 + bs);
    size_t packed_size = get_lhs_packed_size_kai_a8w4(m_array.data(), hidden_dim, 1 + bs, packed_input_offsets.data());
    auto packed_input = allocateBuffer({DataType::TYPE_UINT8, {packed_size}});
    batch_pack_lhs_kai_a8w4((const float16_t*)input_buffer->data(), input_offsets.data(), (uint8_t*)packed_input->data(), packed_input_offsets.data(), m_array.data(), hidden_dim, 1 + bs);
    auto up_output = allocateBuffer({DataType::TYPE_FP16, {token_num + token_num * top_k, moe_inter_size}});
    auto gate_output = allocateBuffer({DataType::TYPE_FP16, {token_num + token_num * top_k, moe_inter_size}});
    batch_matmul_kai_a8w4((uint8_t*)packed_input->data(), packed_input_offsets.data(),
        (uint8_t*)params.weights.moe_gate_weight->kernel->data(), up_weight_offsets.data(),
        (float16_t*)up_output->data(), up_gate_output_offsets.data(),
        m_array.data(), hidden_dim, moe_inter_size, moe_inter_size * sizeof(float16_t), 1 + bs);

    batch_matmul_kai_a8w4((uint8_t*)packed_input->data(), packed_input_offsets.data(),
        (uint8_t*)params.weights.moe_gate_weight->kernel->data(), gate_weight_offsets.data(),
        (float16_t*)gate_output->data(), up_gate_output_offsets.data(),
        m_array.data(), hidden_dim, moe_inter_size, moe_inter_size * sizeof(float16_t), 1 + bs);

    // Activation
    activation({params.configs.activation_type,
        up_output,
        mayGetRef(params.weights.moe_gate_weight->bias),
        *gate_output, std::nullopt, mayGetRef(params.weights.act_scale)});

    // Down projection
    packed_size = get_lhs_packed_size_kai_a8w4(m_array.data(), moe_inter_size, 1 + bs, packed_input_offsets.data());
    packed_input = allocateBuffer({DataType::TYPE_UINT8, {packed_size}});
    auto down_output = input_buffer; // reuse buffer
    batch_pack_lhs_kai_a8w4((float16_t*)up_output->data(), up_gate_output_offsets.data(), (uint8_t*)packed_input->data(), packed_input_offsets.data(), m_array.data(), moe_inter_size, bs + 1);
    batch_matmul_kai_a8w4((uint8_t*)packed_input->data(), packed_input_offsets.data(),
        (uint8_t*)params.weights.moe_down_weight->kernel->data(), down_weight_offsets.data(),
        (float16_t*)down_output->data(), input_offsets.data(),
        m_array.data(), moe_inter_size, hidden_dim, hidden_dim * sizeof(float16_t), 1 + bs);

    // Accumulate output
    #pragma omp parallel for if (token_num > 1)
    for (int i = 0; i < token_num; ++i) {
        float16_t* output_ptr = (float16_t*)output->dataWithOffset(i * hidden_dim);

        // shared expert
        accumulate_output(hidden_dim, output_ptr, (float16_t*)down_output->dataWithOffset(i * hidden_dim), 1.0f);

        for (int j = 0; j < top_k; ++j) {
            int idx = token_to_indices[i][j];
            float16_t w = token_to_weights[i][j];
            accumulate_output(hidden_dim, output_ptr, (float16_t*)down_output->dataWithOffset(idx * hidden_dim), w);
        }
    }

    return FfnLayerOutput({move(output)});
}

FfnLayerOutput ArmCpuDevice::moeFfnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

    const auto& moe_conf = params.configs.moe_configs.value();
    const auto& hidden = params.input;
    const auto type = hidden.type();
    const auto weight_type = params.weights.moe_down_weight->kernel->type();

    const auto token_num = hidden.shape()[0];
    const auto hidden_dim = hidden.shape()[1];
    const size_t moe_inter_size = moe_conf.moe_inter_padding_size;
    const auto num_expert = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto top_k = moe_conf.top_k;
    // const auto normalize_expert_scale = moe_conf.normalize_expert_scale;

    BufferPtr output = nullptr;
    if (params.output) {
        output = params.output;
    } else {
        output = allocateBuffer({type, {token_num, hidden_dim}});
    }
    memset(output->data(), 0, output->sizeBytes());

    printBufferData(*(params.weights.moe_gating_weight->kernel), "moe_gating_weight");
    auto gate_logits = gemm(GemmParams(hidden, *(params.weights.moe_gating_weight->kernel), nullopt, nullptr, DataType::TYPE_FP32));
    printBufferData(*gate_logits, "gate_logits");

    auto expert_weights = allocateBuffer({DataType::TYPE_FP32, {token_num, top_k}, AllocationType::HOST});
    auto expert_indices = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}, AllocationType::HOST});

    std::vector<float> gate_probs(token_num * num_expert);

    //scoring_func 0: softmax, 1: sigmoid
    if (moe_conf.scoring_func == 0) {
        softmax2(gate_logits->data<float>(), gate_probs.data(), token_num, num_expert);
    } else if (moe_conf.scoring_func == 1) {
        sigmoid(gate_logits->data<float>(), gate_probs.data(), token_num, num_expert);
    } else {
        throw std::runtime_error("Unsupported scoring function for moe");
    }

    if (params.weights.e_score_correction_bias) {
        float* bias = (float*)params.weights.e_score_correction_bias->data();
        topKSelectionNoauxTc((const float *)gate_probs.data(), (int *)expert_indices->data(), (float *)expert_weights->data(),
                             bias, token_num, num_expert, top_k, moe_conf.n_group, moe_conf.topk_group);
    } else {
        topKSelection((const float *)gate_probs.data(), (int *)expert_indices->data(), (float *)expert_weights->data(), token_num, num_expert, top_k);
    }

    if (moe_conf.has_moe_norm) {
        for (int s = 0; s < token_num; ++s) {
            float sum = 0.0f;
            float* weight_ptr = (float *)expert_weights->dataWithOffset(s * top_k);
            for (int k = 0; k < top_k; ++k) {
                sum += weight_ptr[k];
            }
            for (int k = 0; k < top_k; ++k) {
                weight_ptr[k] /= sum;
            }
        }
    }

    printBufferData(*expert_weights, "expert_weights");
    printBufferData(*expert_indices, "expert_indices");

    if (type == DataType::TYPE_FP16 && weight_type == DataType::TYPE_QFP8_E4M3) {
        return moe_ffn_a8w4(expert_indices, expert_weights, output, params);
    }

    std::vector<std::vector<int>> expert_to_tokens(num_expert);
    std::vector<std::vector<float>> expert_to_weights(num_expert);
    // Build mapping
    for (int s = 0; s < token_num; ++s) {
        for (int k = 0; k < top_k; ++k) {
            int expert_idx = *(int*)(expert_indices->dataWithOffset(s * top_k + k));
            float weight = *(float*)(expert_weights->dataWithOffset(s * top_k + k));
            expert_to_tokens[expert_idx].push_back(s);
            expert_to_weights[expert_idx].push_back(weight);
        }
    }

    // Process each expert
    for (int expert_idx = 0; expert_idx < num_expert; ++expert_idx) {
        const auto& tokens = expert_to_tokens[expert_idx];
        const auto& weights = expert_to_weights[expert_idx];

        if (tokens.empty()) continue;

        // Gather hidden for this expert
        auto input_buffer = allocateBuffer({type, {tokens.size(), hidden_dim}}, {"input"});
        for (int i = 0; i < tokens.size(); ++i) {
            // copy hidden[tokens[i]] to input_buffer[i]
            memcpy(input_buffer->dataWithOffset(i * hidden_dim), hidden.dataWithOffset(tokens[i] * hidden_dim), hidden_dim * hidden.typeSize());
        }

        // Up projection
        auto expert_up_gate_size = params.weights.moe_gate_weight->kernel->size() / num_expert;

        std::vector<size_t> expert_up_shape = {hidden_dim, moe_inter_size};
        auto up_data = params.weights.moe_gate_weight->kernel->dataWithOffset(expert_idx * expert_up_gate_size);
        if (weight_type == DataType::TYPE_QFP8_E4M3) {
            // TYPE_QFP8_E4M3 dataWithOffset will return data with no offset
            // Manually handle the offset
            auto rhs_packed_size = get_rhs_packed_size(moe_inter_size * 2, hidden_dim);
            up_data = (char*)params.weights.moe_gate_weight->kernel->data() + expert_idx * rhs_packed_size;
        }
        auto moe_up_buffer = BufferPtr(new Buffer(MemoryType::MEMORY_CPU,
            weight_type,
            expert_up_shape,
            up_data));

        auto up_output = gemm(GemmParams(*input_buffer, *moe_up_buffer));

        // Gate projection
        auto expert_up_size = params.weights.moe_gate_weight->kernel->size() / num_expert / 2 ;
        auto gate_data = params.weights.moe_gate_weight->kernel->dataWithOffset(expert_idx * expert_up_gate_size + expert_up_size);
        if (weight_type == DataType::TYPE_QFP8_E4M3) {
            // TYPE_QFP8_E4M3 dataWithOffset will return data with no offset
            // Manually handle the offset
            auto rhs_packed_size = get_rhs_packed_size(moe_inter_size * 2, hidden_dim);
            gate_data = (char*)params.weights.moe_gate_weight->kernel->data() + expert_idx * rhs_packed_size + rhs_packed_size / 2;
        }
        auto moe_gate_buffer = BufferPtr(new Buffer(MemoryType::MEMORY_CPU,
            weight_type,
            expert_up_shape,
            gate_data));

        auto gate_output = gemm(GemmParams(*input_buffer, *moe_gate_buffer));

        // Activation
        activation({params.configs.activation_type,
            up_output,
            mayGetRef(params.weights.moe_gate_weight->bias),
            *gate_output, std::nullopt, mayGetRef(params.weights.act_scale)});

        // Down projection
        auto expert_down_size = params.weights.moe_down_weight->kernel->size() / num_expert;
        std::vector<size_t> expert_down_shape = {moe_inter_size, hidden_dim};
        auto down_data = params.weights.moe_down_weight->kernel->dataWithOffset(expert_idx * expert_down_size);
        if (weight_type == DataType::TYPE_QFP8_E4M3) {
            // TYPE_QFP8_E4M3 dataWithOffset will return data with no offset
            // Manually handle the offset
            auto rhs_packed_size = get_rhs_packed_size(hidden_dim, moe_inter_size);
            down_data = (char*)params.weights.moe_down_weight->kernel->data() + expert_idx * rhs_packed_size;
        }
        auto moe_down_buffer = BufferPtr(new Buffer(MemoryType::MEMORY_CPU,
            weight_type,
            expert_down_shape,
            down_data));

        auto down_output = gemm(GemmParams(*up_output, *moe_down_buffer));

        // Scatter back with weight
        for (int i = 0; i < tokens.size(); ++i) {
            int token_idx = tokens[i];
            float w = weights[i];
            // accumulate output[token_idx] += w * down_output[i]
            if (type == DataType::TYPE_FP32) {
                accumulate_output(hidden_dim, (float*)output->dataWithOffset(token_idx * hidden_dim), (float*)down_output->dataWithOffset(i * hidden_dim), w);
            } else if (type == DataType::TYPE_FP16) {
                accumulate_output(hidden_dim, (float16_t*)output->dataWithOffset(token_idx * hidden_dim), (float16_t*)down_output->dataWithOffset(i * hidden_dim), (float16_t)w);
            } else {
                throw std::runtime_error("Unsupported data type");
            }
        }
    }
    printBufferData(*output, "moe_ffn_out");

    return FfnLayerOutput({move(output)});
}

} // namespace rtp_llm
