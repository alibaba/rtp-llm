#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/kernels/moe_kernels.h"

// aiter kernels
#include "aiter_enum.h"
#include "moe_op.h"
#include "quant.h"
#include "moe_sorting.h"
#include "moe_ck.h"

using namespace std;

namespace rtp_llm {

MoeDispatchOutput ROCmDevice::epDispatch(const MoeDispatchParams& params) {
    DevicePerfWrapper wrapper(this, "epDispatch");
#ifdef ENABLE_DEEP_EP
    if (init_params_.use_deepep_moe) {
        if (init_params_.use_deepep_low_latency) {
            return deepEpLLDispatch(params);
        } else {
            return deepEpDispatch(params);
        }
    }
#else
    RTP_LLM_LOG_INFO("[ROCm epDispatch]: deep_ep is not enabled");
#endif
    const auto& moe_conf = params.moe_configs;

    auto const expert_num = moe_conf.expert_num + moe_conf.extra_expert_num;
    auto const top_k      = moe_conf.top_k;
    auto const ep_size    = moe_conf.ep_size;
    auto const tp_size    = moe_conf.tp_size;
    size_t     token_num  = params.expert_ids.shape()[0];

    assert(moe_conf.ep_size == tp_size * moe_conf.dp_size);
    size_t    tp_token_size    = (token_num + tp_size - 1) / tp_size;
    size_t    slice_begin      = std::min(tp_token_size * moe_conf.tp_rank, token_num);
    size_t    slice_size       = std::min(token_num - slice_begin, tp_token_size);
    BufferPtr hidden           = params.input.slice(slice_begin, slice_size);
    auto      expert_ids       = params.expert_ids.slice(slice_begin, slice_size);
    auto      expert_scales    = params.expert_scales.slice(slice_begin, slice_size);
    token_num                  = hidden->shape()[0];
    BufferPtr experts_ids_host = clone({*expert_ids, AllocationType::HOST, BufferHints(), false, false});
    BufferPtr token_nums_per_rank =
        allocateBuffer({DataType::TYPE_INT32, {ep_size}, AllocationType::HOST}, {"token_nums_per_rank"});
    bufMemset(*token_nums_per_rank, 0);
    int32_t*                          token_nums_per_rank_ptr = token_nums_per_rank->data<int32_t>();
    std::vector<std::vector<int32_t>> token_idx_per_rank;
    token_idx_per_rank.resize(ep_size);
    for (int i = 0; i < ep_size; ++i) {
        token_idx_per_rank[i].reserve(token_num);
    }
    int const num_experts_per_node = expert_num / ep_size;
    for (int i = 0; i < token_num; ++i) {
        for (int j = 0; j < top_k; ++j) {
            size_t expert_id = *(experts_ids_host->dataWithOffset<int32_t>(i * top_k + j));
            assert(expert_id < expert_num);
            const auto ep_rank = expert_id / num_experts_per_node;
            if (token_idx_per_rank[ep_rank].empty() || *token_idx_per_rank[ep_rank].rbegin() != i) {
                token_nums_per_rank_ptr[ep_rank] += 1;
                token_idx_per_rank[ep_rank].push_back(i);
            }
        }
    }
    printBufferData(*token_nums_per_rank, "token_nums_per_rank");
    auto token_nums_per_rank_gpu = clone({*token_nums_per_rank, AllocationType::DEVICE, BufferHints(), false, false});
    // all_token_nums_per_rank_gpu[i]: current rank receive token num from other rank
    auto all_token_nums_per_rank_gpu = allToAll({{token_nums_per_rank_gpu}}).outputs[0];
    printBufferData(*all_token_nums_per_rank_gpu, "all_token_nums_per_rank_gpu");
    size_t    total_size = std::accumulate(token_nums_per_rank_ptr, token_nums_per_rank_ptr + ep_size, 0);
    BufferPtr all_token_indices_cpu =
        allocateBuffer({DataType::TYPE_INT32, {total_size}, AllocationType::HOST}, {"token_nums_per_rank"});

    size_t offset = 0;
    for (int i = 0; i < ep_size; ++i) {
        memcpy(all_token_indices_cpu->dataWithOffset<int32_t>(offset),
               token_idx_per_rank[i].data(),
               token_idx_per_rank[i].size() * sizeof(int32_t));
        offset += token_idx_per_rank[i].size();
    }

    printBufferData(*all_token_indices_cpu, "all_token_indices_cpu");
    // each rank token idx
    BufferPtr all_token_indices = clone({*all_token_indices_cpu, AllocationType::DEVICE, BufferHints(), false, false});
    // sync allToAll all_token_nums_per_rank_gpu
    cudaStreamSynchronize(stream_);
    syncCommunication(false);

    auto all_token_nums_per_rank =
        clone({*all_token_nums_per_rank_gpu, AllocationType::HOST, BufferHints(), false, false});
    std::vector<size_t> input_split_sizes;
    std::vector<size_t> output_split_sizes;
    input_split_sizes.resize(ep_size);
    output_split_sizes.resize(ep_size);
    for (int i = 0; i < ep_size; ++i) {
        input_split_sizes[i]  = token_nums_per_rank_ptr[i];
        output_split_sizes[i] = *(all_token_nums_per_rank->dataWithOffset<int32_t>(i));
    }
    vector<BufferPtr> selected_buffers;
    QBufferPtr        q_hidden;
    BufferPtr         select_hidden = select({*hidden, *all_token_indices});
    if (params.qscheme != QScheme::NoQuantize) {
        // ignore groupSize when using per_token quantization
        q_hidden = std::dynamic_pointer_cast<QBuffer>(
            quantize(QuantizeParams(*select_hidden, DataType::TYPE_QFP8_E4M3, 1, params.qscheme, 128, 0)));
        selected_buffers.emplace_back(q_hidden->kernelPtr());
        selected_buffers.emplace_back(q_hidden->scalesPtr());
    } else {
        selected_buffers.emplace_back(select_hidden);
    }
    selected_buffers.emplace_back(select({*expert_ids, *all_token_indices}));
    selected_buffers.emplace_back(select({*expert_scales, *all_token_indices}));

    auto all2all_output = allToAll({selected_buffers, input_split_sizes, output_split_sizes, params.overlapped});
    // syncCommunication(false);  // Debug only
    const auto& global_buffers = all2all_output.outputs;
    if (params.qscheme != QScheme::NoQuantize) {
        BufferPtr hidden_fp8 = BufferPtr(new QBuffer(
            std::move(const_cast<BufferPtr&>(global_buffers[0])),
            std::move(const_cast<BufferPtr&>(global_buffers[1])),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));

        updateExpertGpuLoads(moe_conf, params.expert_stats, global_buffers[2]);

        return {hidden_fp8,
                global_buffers[2],
                global_buffers[3],
                all_token_indices,
                input_split_sizes,
                output_split_sizes,
                selected_buffers,
                all2all_output.concated_input,
                all2all_output.output_to_split,
                move(all2all_output.comm_barrier_hook)};
    } else {
        updateExpertGpuLoads(moe_conf, params.expert_stats, global_buffers[2]);

        return {global_buffers[0],
                global_buffers[1],
                global_buffers[2],
                all_token_indices,
                input_split_sizes,
                output_split_sizes,
                selected_buffers,
                all2all_output.concated_input,
                all2all_output.output_to_split,
                move(all2all_output.comm_barrier_hook)};
    }
}

MoeCombineOutput ROCmDevice::epCombine(const MoeCombineParams& params) {
    DevicePerfWrapper wrapper(this, "epCombine");
#ifdef ENABLE_DEEP_EP
    if (init_params_.use_deepep_moe) {
        if (init_params_.use_deepep_low_latency) {
            return deepEpLLCombine(params);
        } else {
            return deepEpCombine(params);
        }
    }
#else
    RTP_LLM_LOG_INFO("[ROCm epCombine]: deep_ep is not enabled");
#endif
    // 当前卡接受计算完moe的token
    bool overlapped = false;

    auto all2all_ret = allToAll({{params.input}, params.output_split_sizes, params.input_split_sizes, overlapped});

    return MoeCombineOutput({all2all_ret.outputs[0], nullptr, params, std::move(all2all_ret.comm_barrier_hook)});
}

FfnLayerOutput ROCmDevice::gatherCombineOutput(const MoeCombineOutput& combine_outputs) {
    auto&       all_output     = combine_outputs.all_output;
    auto        scatter_output = combine_outputs.scatter_output;
    const auto& params         = combine_outputs.params;

    if (params.moe_configs.tp_size > 1) {
        // TODO: can use torch all gather unequal size to avoid copy
        const size_t tp_token_size =
            (params.origin_token_num + params.moe_configs.tp_size - 1) / params.moe_configs.tp_size;
        size_t dim1_size = all_output->shape()[1];
        size_t current_token_num =
            std::max(0,
                     std::min((int)params.origin_token_num - int(tp_token_size * params.moe_configs.tp_rank),
                              (int)tp_token_size));

        if (scatter_output == nullptr) {
            scatter_output = allocateBuffer({all_output->type(), {current_token_num, dim1_size}});
            // TODO: why this assertion?
            // assert(all_output->shape()[0] == current_token_num);
            if (scatter_output->shape()[0] > 0) {
                cudaMemsetAsync(scatter_output->data(), 0, scatter_output->sizeBytes(), stream_);
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(scatter_output->type(),
                                                 invokeScatterAdd,
                                                 all_output->data(),
                                                 all_output->shape()[0],
                                                 all_output->shape()[1],
                                                 params.indices->data<int32_t>(),
                                                 scatter_output->data(),
                                                 /*this->use_stable_scatter_add*/ false,
                                                 stream_);
            }
        }
        BufferPtr padding_output;
        if (params.origin_token_num == tp_token_size * params.moe_configs.tp_size && params.output) {
            padding_output = params.output;
        } else {
            padding_output =
                allocateBuffer({all_output->type(), {tp_token_size * params.moe_configs.tp_size, dim1_size}});
        }
        if (scatter_output->shape()[0] > 0) {
            copy({padding_output->view(tp_token_size * params.moe_configs.tp_rank, scatter_output->shape()[0]),
                  *scatter_output});
        }
        allGather({{padding_output}, ParallelMode::TP, {}, true});
        if (params.origin_token_num == tp_token_size * params.moe_configs.tp_size) {
            return {padding_output};
        } else {
            BufferPtr output =
                params.output ? params.output : allocateBufferLike(padding_output->view(0, params.origin_token_num));
            copy({*output, padding_output->view(0, params.origin_token_num)});
            return {output};
        }
    } else {
        if (scatter_output == nullptr) {
            vector<size_t> new_shape = all_output->shape();
            new_shape[0]             = params.origin_token_num;
            BufferPtr output         = params.output ? params.output : allocateBuffer({all_output->type(), new_shape});

            printBufferData(*output, "scatter_add_input");
            if (output->shape()[0] > 0 && params.origin_token_num > 0) {
                cudaMemsetAsync(output->data(), 0, output->sizeBytes(), stream_);
                DISPATCH_CUDA_FUNCTION_DATA_TYPE(output->type(),
                                                 invokeScatterAdd,
                                                 all_output->data(),
                                                 all_output->shape()[0],
                                                 all_output->shape()[1],
                                                 params.indices->data<int32_t>(),
                                                 output->data(),
                                                 /*this->use_stable_scatter_add*/ false,
                                                 stream_);
            }
            printBufferData(*output, "scatter_add_output");

            return {output};
        } else {
            return {all_output};
        }
    }
}

MoeGateSelectOutput ROCmDevice::moeGateSelect(const FfnLayerParams& params) {
    const MoeConfigs& moe_conf = params.configs.moe_configs.value();

    const Buffer& hidden     = params.input;
    const size_t  num_token  = hidden.shape()[0];
    const size_t  model_dim  = hidden.shape()[1];
    const size_t  num_expert = moe_conf.expert_num;
    const size_t  topk       = moe_conf.top_k;
    const int     n_group    = moe_conf.n_group;
    const int     topk_group = moe_conf.topk_group;
    const bool has_moe_norm  = moe_conf.has_moe_norm;  // FIXME(liyangcheng.lyc): normalize_expert_scale? has_moe_norm?

    // step 1. calculate gating logits
    BufferPtr logits = allocateBuffer({DataType::TYPE_FP32, {num_token, num_expert}}, {"rocm_logits"});
    gemm({hidden, *(params.weights.moe_gating_weight->kernel), nullopt, logits, DataType::TYPE_FP32});

    BufferPtr moe_gating;

    // step 2. calculate topk function to get topk_ids and topk_weights
    torch::Tensor logits_tensor = Buffer2torchTensor(*logits, false);

    BufferPtr     topk_weights        = allocateBuffer({DataType::TYPE_FP32, {num_token, topk}}, {"rocm_topk_weights"});
    BufferPtr     topk_ids            = allocateBuffer({DataType::TYPE_INT32, {num_token, topk}}, {"rocm_topk_ids"});
    torch::Tensor topk_weights_tensor = Buffer2torchTensor(*topk_weights, false);
    torch::Tensor topk_ids_tensor     = Buffer2torchTensor(*topk_ids, false);

    // use grouped topk
    if (n_group > 1) {
        // use biased_grouped_topk, in aiter will invoke function `biased_grouped_topk`
        // act must be `sigmoid` when using bias
        if (params.weights.e_score_correction_bias) {
            torch::Tensor e_score_correction_bias_tensor =
                Buffer2torchTensor(*(params.weights.e_score_correction_bias), false).to(torch::kFloat32);

            // invoke aiter kernel
            biased_grouped_topk(
                logits_tensor,
                e_score_correction_bias_tensor,
                topk_weights_tensor,
                topk_ids_tensor,
                n_group,
                topk_group,
                has_moe_norm);  // FIXME(liyangcheng.lyc): not set routed_scaling_factor, no such config now

            if (params.need_moe_gating) {
                // TODO(zhangjianning.zjn): would be better to get the corrected moe gating from the kernel above
                // directly
                moe_gating                   = clone({*logits});
                torch::Tensor moe_gating_tsr = Buffer2torchTensor(*moe_gating, false);
                moe_gating_tsr.add_(e_score_correction_bias_tensor);
            }
        } else {  // use grouped_topk, in aiter will invoke function `grouped_topk`
            // TODO(zhangjianning.zjn): should support returning moe_gating once the branch is supported
            // FIXME(liyangcheng.lyc): not implemented yet
            RTP_LLM_FAIL("[ROCm moeGateSelect]: n_group > 1 and e_score_correction_bias is null not implemented yet");
        }
    } else {  // use normal topk softmax, in aiter will invoke function `topk_softmax`
        // NOTE(liyangcheng.lyc): this buffer not used, but maybe used in the future
        BufferPtr token_expert_indicies =
            allocateBuffer({DataType::TYPE_INT32, {num_token, topk}}, {"rocm_token_expert_indicies"});
        torch::Tensor token_expert_indicies_tensor = Buffer2torchTensor(*token_expert_indicies, false);

        // invoke aiter kernel
        aiter::topk_softmax(
            topk_weights_tensor, topk_ids_tensor, token_expert_indicies_tensor, logits_tensor, has_moe_norm);

        moe_gating = std::move(logits);
    }

    balanceExperts(topk_ids, params.expert_stats, params.configs.moe_configs.value(), params.weights);
    if (init_params_.moe_config.fake_balance_expert) {
        if (topk_ids->type() == DataType::TYPE_INT64) {
            fake_balance_expert(topk_ids->data<int64_t>(),
                                topk_weights->data<float>(),
                                moe_conf.dp_rank,
                                moe_conf.dp_size,
                                moe_conf.ep_size,
                                num_expert,
                                num_token * topk,
                                stream_);
        } else {
            fake_balance_expert(topk_ids->data<int>(),
                                topk_weights->data<float>(),
                                moe_conf.dp_rank,
                                moe_conf.dp_size,
                                moe_conf.ep_size,
                                num_expert,
                                num_token * topk,
                                stream_);
        }
    }
    return {topk_ids, topk_weights, moe_gating};
}

FfnLayerOutput ROCmDevice::moeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    // deepseek deepep low latency only
#ifdef ENABLE_DEEP_EP
    if (init_params_.ep_size > 1 && init_params_.use_deepep_moe && init_params_.use_deepep_low_latency) {
        return deepEpLLMoeFfn(params, gate_outputs);
    }
#else
    RTP_LLM_LOG_INFO("[ROCm moeFfn]: deep_ep is not enabled");
#endif
    const MoeConfigs& moe_conf = params.configs.moe_configs.value();

    const Buffer& hidden = params.input;

    const size_t num_token           = hidden.shape()[0];
    const size_t model_dim           = hidden.shape()[1];
    const int    inter_dim           = static_cast<int>(params.weights.moe_down_weight->kernel->shape()[2]);
    const size_t num_expert          = moe_conf.expert_num + moe_conf.extra_expert_num;
    const size_t num_expert_per_rank = num_expert / moe_conf.ep_size;
    const size_t topk                = moe_conf.top_k;
    DataType     dtype;
    if (params.qscheme == QScheme::NoQuantize) {
        dtype = hidden.type();
    } else {
        dtype = DataType::TYPE_BF16;
    }

    BufferPtr moe_out_final = allocateBuffer({dtype, {num_token, model_dim}}, {"rocm_moe_final_out"});
    if (num_token == 0) {
        return {moe_out_final};
    }

    torch::Tensor topk_ids_tensor     = Buffer2torchTensor(*(gate_outputs.expert_ids), false).to(torch::kInt32);
    torch::Tensor topk_weights_tensor = Buffer2torchTensor(*(gate_outputs.expert_scales), false);

    // get input
    torch::Tensor                hidden_tensor;
    std::optional<torch::Tensor> hidden_scale_tensor;
    QBufferPtr                   q_hidden;
    if (params.qscheme == QScheme::NoQuantize) {
        hidden_tensor = Buffer2torchTensor(hidden, false);
    } else if (params.input.isQBuffer()) {
        const QBuffer& qhidden = reinterpret_cast<const QBuffer&>(hidden);
        hidden_tensor          = Buffer2torchTensor(qhidden.kernel(), false).view({(int)num_token, (int)model_dim});
        hidden_scale_tensor    = Buffer2torchTensor(qhidden.scales(), false);
    } else {
        // ignore groupSize when using per_token quantization
        q_hidden = std::dynamic_pointer_cast<QBuffer>(
            quantize(QuantizeParams(hidden, DataType::TYPE_QFP8_E4M3, 1, params.qscheme, 128, 0)));
        hidden_tensor       = Buffer2torchTensor(q_hidden->kernel(), false);
        hidden_scale_tensor = Buffer2torchTensor(q_hidden->scales(), false);
    }

    // get w1 and w2
    torch::Tensor                w1_tensor, w2_tensor;
    std::optional<torch::Tensor> w1_scale_tensor, w2_scale_tensor;
    if (params.qscheme == QScheme::NoQuantize) {
        w1_tensor = Buffer2torchTensor(*(params.weights.moe_gate_weight->kernel), false);
        w2_tensor = Buffer2torchTensor(*(params.weights.moe_down_weight->kernel), false);
    } else {
        const QBuffer& qmoe_gate_weight = reinterpret_cast<const QBuffer&>(*(params.weights.moe_gate_weight->kernel));
        const QBuffer& qmoe_down_weight = reinterpret_cast<const QBuffer&>(*(params.weights.moe_down_weight->kernel));
        w1_tensor                       = Buffer2torchTensor(qmoe_gate_weight.kernel(), false);
        w1_scale_tensor                 = Buffer2torchTensor(qmoe_gate_weight.scales(), false);
        w2_tensor                       = Buffer2torchTensor(qmoe_down_weight.kernel(), false);
        w2_scale_tensor                 = Buffer2torchTensor(qmoe_down_weight.scales(), false);
    }

    // get expert mask when ep_size > 1
    BufferPtr local_expert_mask =
        allocateBuffer({DataType::TYPE_INT32, {(size_t)num_expert}}, {"rocm_moe_local_expert_mask"});
    torch::Tensor local_expert_mask_tensor = Buffer2torchTensor(*local_expert_mask, false);
    local_expert_mask_tensor.zero_();
    if (init_params_.use_deepep_moe) {
        // deepep has already offset the topk_ids and set the masked expert to num_expert_per_rank
        local_expert_mask_tensor.index_put_({torch::indexing::Slice(0, num_expert_per_rank)},
                                            torch::ones(num_expert_per_rank, torch::device(torch::kCUDA)));
    } else {
        local_expert_mask_tensor.index_put_({torch::indexing::Slice(moe_conf.ep_rank * num_expert_per_rank,
                                                                    (moe_conf.ep_rank + 1) * num_expert_per_rank)},
                                            torch::ones(num_expert_per_rank, torch::device(torch::kCUDA)));
    }

    // moe sorting
    int unit_size = 32;
    if (params.qscheme != QScheme::Qfp8PerTokenBlock) {
        // get block_size_m, i.e. unit_size
        const int              cu_num                 = rocmDevProp.multiProcessorCount;
        const int              tile_n                 = 128;
        const int              tg_n                   = (inter_dim + tile_n - 1) / tile_n;
        int                    min_rnd                = std::numeric_limits<int>::max();
        int                    min_empty              = std::numeric_limits<int>::max();
        const std::vector<int> unit_size_support_list = {32, 64, 128};

        for (const int& el : unit_size_support_list) {
            int max_num_tokens = num_token * topk + num_expert_per_rank * el - topk;
            int tg_num         = tg_n * (max_num_tokens + el - 1) / el;
            int rnd            = (tg_num + cu_num - 1) / cu_num;
            int empty          = cu_num - tg_num % cu_num;

            if (rnd < min_rnd) {
                min_rnd   = rnd;
                min_empty = empty;
                unit_size = el;
            } else if (rnd == min_rnd) {
                if (empty < min_empty) {
                    min_empty = empty;
                    unit_size = el;
                }
            }
        }
    }

    const int max_num_token_padded = topk_ids_tensor.numel() + num_expert * unit_size - topk;
    const int max_num_m_block      = (max_num_token_padded + unit_size - 1) / unit_size;

    BufferPtr sorted_ids =
        allocateBuffer({DataType::TYPE_INT32, {(size_t)max_num_token_padded}}, {"rocm_moe_sorted_ids"});
    BufferPtr sorted_weights =
        allocateBuffer({DataType::TYPE_FP32, {(size_t)max_num_token_padded}}, {"rocm_moe_sorted_weights"});
    BufferPtr sorted_expert_ids =
        allocateBuffer({DataType::TYPE_INT32, {(size_t)max_num_m_block}}, {"rocm_moe_sorted_expert_ids"});
    BufferPtr num_valid_ids = allocateBuffer({DataType::TYPE_INT32, {1}}, {"rocm_moe_num_valid_ids"});

    torch::Tensor sorted_ids_tensor        = Buffer2torchTensor(*sorted_ids, false);
    torch::Tensor sorted_weights_tensor    = Buffer2torchTensor(*sorted_weights, false);
    torch::Tensor sorted_expert_ids_tensor = Buffer2torchTensor(*sorted_expert_ids, false);
    torch::Tensor num_valid_ids_tensor     = Buffer2torchTensor(*num_valid_ids, false);

    torch::Tensor moe_out_tensor = Buffer2torchTensor(*moe_out_final, false);

    // invoke aiter moe_sorting kernel
    moe_sorting_fwd(
        /*topk_ids=*/topk_ids_tensor,
        /*topk_weights=*/topk_weights_tensor,
        /*sorted_token_ids=*/sorted_ids_tensor,
        /*sorted_weights=*/sorted_weights_tensor,
        /*sorted_expert_ids=*/sorted_expert_ids_tensor,
        /*num_valid_ids=*/num_valid_ids_tensor,
        /*moe_buf=*/moe_out_tensor,
        /*num_experts=*/num_expert,
        /*unit_size=*/unit_size,
        /*local_expert_mask=*/local_expert_mask_tensor,
        /*num_local_tokens*/ std::nullopt,
        /*dispatch_policy*/ 0);

    if (params.qscheme == QScheme::Qfp8PerTokenBlock) {
        RTP_LLM_CHECK_WITH_INFO(dtype == DataType::TYPE_BF16,
                                "input hidden datatype should be bf16 when using Qfp8PerTokenBlock");
        const int block_scale_n                    = 128;
        const int block_scale_k                    = 128;
        hidden_scale_tensor                        = hidden_scale_tensor.value().t().contiguous();
        w1_scale_tensor                            = w1_scale_tensor.value().view({(int)num_expert_per_rank, -1});
        w2_scale_tensor                            = w2_scale_tensor.value().view({(int)num_expert_per_rank, -1});
        std::string fmoe_fp8_block_scale_g1u1_name = "";

        // invoke aiter moe kernel
        fmoe_fp8_blockscale_g1u1(
            /*out=*/moe_out_tensor,
            /*input=*/hidden_tensor,
            /*gate=*/w1_tensor,
            /*down=*/w2_tensor,
            /*sorted_token_ids=*/sorted_ids_tensor,
            /*sorted_weight_buf=*/sorted_weights_tensor,
            /*sorted_expert_ids=*/sorted_expert_ids_tensor,
            /*num_valid_ids=*/num_valid_ids_tensor,
            /*topk=*/topk,
            /*input_scale=*/hidden_scale_tensor.value(),
            /*fc1_scale=*/w1_scale_tensor.value(),
            /*fc2_scale=*/w2_scale_tensor.value(),
            /*kernel_name*/ fmoe_fp8_block_scale_g1u1_name,
            /*fc_scale_blkn=*/block_scale_n,
            /*fc_scale_blkk*/ block_scale_k,
            /*fc2_smooth_scale=*/nullopt,
            /*activation*/ ::ActivationType::Silu);
    } else {
        BufferPtr     a2        = allocateBuffer({dtype, {num_token, topk, (size_t)inter_dim}}, {"rocm_a2"});
        torch::Tensor a2_tensor = Buffer2torchTensor(*a2, false);
        // FIXME(liyangcheng.lyc): workaround for two stage moe accuracy issue, see
        // https://github.com/ROCm/aiter/issues/566
        a2_tensor.zero_();
        std::optional<torch::Tensor> a2_scale_tensor;
        QBufferPtr                   a2_q;
        std::string                  ck_moe_stage1_kernel_name  = "";
        std::string                  asm_moe_stage1_kernel_name = "";
        std::string                  ck_moe_stage2_kernel_name  = "";

        auto aiterQscheme = [qscheme = params.qscheme]() {
            switch (qscheme) {
                case QScheme::NoQuantize:
                    return ::QuantType::No;
                case QScheme::Qfp8PerToken:
                    return ::QuantType::per_Token;
                default:
                    RTP_LLM_FAIL("[ROCm moeFfn]: quant type %d not implemented yet", (int)qscheme);
            }
        }();

        // invoke aiter two stage moe kernels
        if (params.qscheme == QScheme::NoQuantize) {
            ck_moe_stage1(
                /*hidden_states*/ hidden_tensor,
                /*w1*/ w1_tensor,
                /*w2*/ w2_tensor,
                /*sorted_token_ids*/ sorted_ids_tensor,
                /*sorted_expert_ids*/ sorted_expert_ids_tensor,
                /*num_valid_ids*/ num_valid_ids_tensor,
                /*out*/ a2_tensor,
                /*topk*/ topk,
                /*kernelName*/ ck_moe_stage1_kernel_name,
                /*w1_scale*/ w1_scale_tensor,
                /*a1_scale*/ hidden_scale_tensor,
                /*block_m*/ unit_size,
                /*sorted_weights*/ std::nullopt,
                /*quant_type*/ static_cast<int>(aiterQscheme),
                /*activation*/ static_cast<int>(::ActivationType::Silu),
                /*splitk*/ 1,
                /*dst_type*/ std::nullopt);
        } else {
            moe_stage1_g1u1(
                /*input*/ hidden_tensor,
                /*w1*/ w1_tensor,
                /*w2*/ w2_tensor,
                /*sorted_token_ids*/ sorted_ids_tensor,
                /*sorted_expert_ids*/ sorted_expert_ids_tensor,
                /*num_valid_ids*/ num_valid_ids_tensor,
                /*out*/ a2_tensor,
                /*inter_dim*/ inter_dim,
                /*kernelName*/ asm_moe_stage1_kernel_name,
                /*block_m*/ unit_size,
                /*ksplit*/ 0,
                /*activation*/ ::ActivationType::Silu,
                /*quant_type*/ aiterQscheme,
                /*a1_scale*/ hidden_scale_tensor,
                /*w1_scale*/ w1_scale_tensor,
                /*sorted_weights*/ std::nullopt);
        }

        if (params.qscheme != QScheme::NoQuantize) {
            a2_q = std::dynamic_pointer_cast<QBuffer>(
                quantize(QuantizeParams(*a2, DataType::TYPE_QFP8_E4M3, 1, params.qscheme, 128, 0)));
            a2_tensor       = Buffer2torchTensor(a2_q->kernel(), false).view({(int)num_token, (int)topk, inter_dim});
            a2_scale_tensor = Buffer2torchTensor(a2_q->scales(), false);
        }
        ck_moe_stage2(
            /*inter_states*/ a2_tensor,
            /*w1*/ w1_tensor,
            /*w2*/ w2_tensor,
            /*sorted_token_ids*/ sorted_ids_tensor,
            /*sorted_expert_ids*/ sorted_expert_ids_tensor,
            /*num_valid_ids*/ num_valid_ids_tensor,
            /*out*/ moe_out_tensor,
            /*topk*/ topk,
            /*kernelName*/ ck_moe_stage2_kernel_name,
            /*w2_scale*/ w2_scale_tensor,
            /*a2_scale*/ a2_scale_tensor,
            /*block_m*/ unit_size,
            /*sorted_weights*/ sorted_weights_tensor,
            /*quant_type*/ static_cast<int>(aiterQscheme),
            /*activation*/ static_cast<int>(::ActivationType::Silu),
            /*splitk*/ 1,
            /*dst_type*/ std::nullopt);
    }
    return {moe_out_final};
}

FfnLayerOutput ROCmDevice::ffnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(!params.residual, "default FFN implementation does not support residual!");
    BufferPtr output;
    if (params.weights.moe_gating_weight) {
        RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
        auto moe_output = moeFfnLayer(params);
        output          = moe_output.hidden_states;

        auto shared_expert_output = moeSharedExpert(params).hidden_states;

        // for deep ep ll, the gather should be defered afater shared expert.
        if (moe_output.moe_combine_output) {
            moe_output.comm_barrier_hook->hook_sync();
            moe_output = gatherCombineOutput(moe_output.moe_combine_output.value());
            output     = moe_output.hidden_states;
        }

        printBufferData(*output, "moe_out_after_barrier");
        if (shared_expert_output) {
            // just add bias to output
            layernorm({output, nullptr, nullopt, mayGetRef(shared_expert_output)}).output;
        }
    } else {
        BufferPtr up_output;
        bool      fuse_gate_up_weight = (params.weights.gate_up_weight != nullptr);
        if (isGatedActivation(params.configs.activation_type)) {
            BufferPtr ffn_input_ptr = nullptr;
            RTP_LLM_LOG_DEBUG("enable_sp %d ffn_tp_size %d", params.enable_sp, init_params_.ffn_tp_size);
            if (params.enable_sp && init_params_.ffn_tp_size > 1) {
                BufferPtr ag_recv_buffer = nullptr;
                size_t    pad_token_num  = params.input.shape()[0] * init_params_.ffn_tp_size;
                if (params.qscheme == NoQuantize) {
                    ffn_input_ptr  = params.input.slice(0, params.input.shape()[0]);
                    ag_recv_buffer = allocateBuffer({ffn_input_ptr->type(), {pad_token_num, ffn_input_ptr->shape()[1]}},
                                                    {"ag_recv_buffer"});
                } else if (params.qscheme == Qint8PerToken) {
                    ffn_input_ptr = reinterpret_cast<const QBuffer&>(params.input).qslice(0, params.input.shape()[0]);
                    BufferPtr kernel = allocateBuffer(
                        {ffn_input_ptr->type(), {pad_token_num, ffn_input_ptr->shape()[1]}}, {"ag_recv_buffer"});
                    BufferPtr scales = allocateBuffer({DataType::TYPE_FP32, {pad_token_num}, AllocationType::DEVICE},
                                                      {"ag_recv_buffer_scale"});
                    ag_recv_buffer =
                        BufferPtr(new QBuffer(std::move(kernel),
                                              std::move(scales),
                                              std::move(BufferPtr(new Buffer(
                                                  MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
                } else if (params.qscheme == Qfp8PerTensor) {
                    ffn_input_ptr =
                        reinterpret_cast<const QBuffer&>(params.input).qslicePerTensor(0, params.input.shape()[0]);
                    BufferPtr kernel = allocateBuffer(
                        {ffn_input_ptr->type(), {pad_token_num, ffn_input_ptr->shape()[1]}}, {"ag_recv_buffer"});
                    BufferPtr scales = reinterpret_cast<const QBuffer&>(params.input).scalesPtr();
                    ag_recv_buffer =
                        BufferPtr(new QBuffer(std::move(kernel),
                                              std::move(scales),
                                              std::move(BufferPtr(new Buffer(
                                                  MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
                } else {
                    throw OpException(
                        {OpErrorType::ERROR_UNIMPLEMENTED, "allGatherloraLinear qscheme type not supported"});
                }
                printBufferData(*ffn_input_ptr, "ffn_ag_input");

                GemmParams up_gemm_params = fuse_gate_up_weight ?
                                                GemmParams(*ag_recv_buffer, *(params.weights.gate_up_weight->kernel)) :
                                                GemmParams(*ag_recv_buffer, *(params.weights.up_weight->kernel));

                AllGatherLoraLinearOutput all_gather_output =
                    allGatherloraLinear({LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input),
                                         ffn_input_ptr,
                                         ag_recv_buffer,
                                         params.qscheme,
                                         params.output->type(),
                                         ParallelMode::FFN_TP});
                // syncAndCheck();
                ffn_input_ptr = all_gather_output.all_gather_recv_buffer;
                up_output     = all_gather_output.output;
                printBufferData(*ffn_input_ptr, "ffn_ag_inter_output");
                printBufferData(*up_output, "ffn_ag_final_output");
            } else {
                printBufferData(params.input, "input");
                auto up_gemm_weight =
                    fuse_gate_up_weight ? params.weights.gate_up_weight->kernel : params.weights.up_weight->kernel;
                GemmParams up_gemm_params(
                    params.input, *up_gemm_weight, std::nullopt, nullptr, DataType::TYPE_INVALID, params.compute_type);
                up_output = loraLinear(LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input)).output;
                printBufferData(*up_output, "ffn_up");
            }
            if (!fuse_gate_up_weight) {
                BufferPtr gate_output = nullptr;
                if (params.enable_sp && init_params_.ffn_tp_size > 1) {
                    GemmParams gate_gemm_params = GemmParams(*ffn_input_ptr, *(params.weights.gate_weight->kernel));
                    gate_output =
                        loraLinear(LoraLinearParams(gate_gemm_params, params.lora_input.gate_lora_input)).output;
                } else {
                    GemmParams gate_gemm_params = GemmParams(params.input, *(params.weights.gate_weight->kernel));
                    gate_output =
                        loraLinear(LoraLinearParams(gate_gemm_params, params.lora_input.gate_lora_input)).output;
                }
                printBufferData(*gate_output, "ffn_gate");
                activation({params.configs.activation_type,
                            up_output,
                            mayGetRef(params.weights.up_weight->bias),
                            *gate_output,
                            std::nullopt,
                            mayGetRef(params.weights.act_scale)});
            } else {
                printBufferData(*up_output, "ffn_up_gate");
                if (params.configs.activation_type == ActivationType::Swiglu
                    || params.configs.activation_type == ActivationType::Silu
                    || params.configs.activation_type == ActivationType::Gelu) {
                    auto act_output = allocateBuffer({up_output->type(),
                                                      {up_output->shape()[0], up_output->shape()[1] / 2},
                                                      AllocationType::DEVICE});
                    up_output       = activation({params.configs.activation_type,
                                                  up_output,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  act_output,
                                                  true,
                                                  params.qscheme});
                } else {
                    printBufferData(*up_output, "gate_up_output buffer");
                    torch::Tensor              gate_up_output_torch_tensor = Buffer2torchTensor(up_output, false);
                    std::vector<torch::Tensor> split_tensors = torch::chunk(gate_up_output_torch_tensor, 2, -1);
                    torch::Tensor              first_half    = split_tensors[0].clone();
                    torch::Tensor              second_half   = split_tensors[1].clone();
                    BufferPtr                  gate_output   = torchTensor2Buffer(first_half);
                    BufferPtr                  up_out        = torchTensor2Buffer(second_half);

                    activation({params.configs.activation_type,
                                up_out,
                                std::nullopt,
                                *gate_output,
                                std::nullopt,
                                mayGetRef(params.weights.act_scale)});

                    up_output = std::move(up_out);
                }
            }
        } else if (params.qscheme == QScheme::Qint8PerTensor) {
            BufferPtr D = allocateBuffer(
                {DataType::TYPE_FP16, {params.input.shape()[0], params.weights.up_weight->kernel->shape()[1]}});
            auto      up_gemm_params     = GemmParams(params.input,
                                             *(params.weights.up_weight->kernel),
                                             std::nullopt,
                                             D,
                                             DataType::TYPE_FP16,
                                             DataType::TYPE_FP16,
                                             TransposeOperation::NONE,
                                             TransposeOperation::NONE);
            auto      lora_linear_params = LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input);
            auto      activation_params  = ActivationParams(params.configs.activation_type,
                                                      nullptr,
                                                      mayGetRef(params.weights.up_weight->bias),
                                                      std::nullopt,
                                                      std::nullopt,
                                                      mayGetRef(params.weights.act_scale));
            BufferPtr up_gemm_output     = loraLinear(lora_linear_params).output;
            activation_params.states     = up_gemm_output;
            up_output                    = activation(activation_params);
        } else if (params.qscheme == QScheme::Qfp8PerToken) {
            auto up_gemm_params     = GemmParams(params.input,
                                             *(params.weights.up_weight->kernel),
                                             std::nullopt,
                                             nullptr,
                                             DataType::TYPE_INVALID,
                                             params.output->type());
            auto lora_linear_params = LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input);
            auto activation_params  = ActivationParams(params.configs.activation_type,
                                                      nullptr,
                                                      mayGetRef(params.weights.up_weight->bias),
                                                      std::nullopt,
                                                      std::nullopt,
                                                      mayGetRef(params.weights.act_scale));
            up_output               = loraLinearWithActivation({lora_linear_params, activation_params});
        } else {
            RTP_LLM_CHECK_WITH_INFO(!params.enable_sp, "enable_sp is not supported for non-gated activation");
            auto up_gemm_params     = GemmParams(params.input, *(params.weights.up_weight->kernel));
            auto lora_linear_params = LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input);
            auto activation_params  = ActivationParams(params.configs.activation_type,
                                                      nullptr,
                                                      mayGetRef(params.weights.up_weight->bias),
                                                      std::nullopt,
                                                      std::nullopt,
                                                      mayGetRef(params.weights.act_scale),
                                                      nullptr,
                                                      false,
                                                      params.qscheme);
            up_output               = loraLinearWithActivation({lora_linear_params, activation_params});
        }

        if (params.qscheme != QScheme::NoQuantize && params.qscheme != QScheme::Qfp8PerTokenBlock
            && params.qscheme != QScheme::Qfp8PerToken) {
            DataType quant_out_data_type =
                params.qscheme == QScheme::Qfp8PerTensor || params.qscheme == QScheme::Qfp8PerTokenBlock ?
                    DataType::TYPE_FP8_E4M3 :
                    DataType::TYPE_INT8;
            auto quant_params = QuantizeParams(
                *up_output,
                quant_out_data_type,
                1,
                params.qscheme,
                params.weights.smoother_weight ? (OptionalConstBufferRef) * (params.weights.smoother_weight->kernel) :
                                                 std::nullopt,
                std::nullopt,
                params.weights.intermediate_weight2_static_scale_weight ?
                    (OptionalConstBufferRef) * (params.weights.intermediate_weight2_static_scale_weight->kernel) :
                    std::nullopt,
                params.weights.intermediate_weight2_static_scale_reciprocal_weight ?
                    (OptionalConstBufferRef)
                        * (params.weights.intermediate_weight2_static_scale_reciprocal_weight->kernel) :
                    std::nullopt);
            up_output = quantize(quant_params);
        }

        printBufferData(*up_output, "ffn_act");
        if (params.enable_sp && init_params_.ffn_tp_size > 1) {
            BufferPtr gemm_output = allocateBuffer(
                {params.output->type(), {up_output->shape()[0], params.weights.down_weight->kernel->shape()[1]}},
                {"ffn_rs_input"});
            GemmParams down_gemm_params =
                GemmParams(*(up_output), *(params.weights.down_weight->kernel), nullopt, gemm_output);
            ReduceScatterLoraLinearOutput reduce_scatter_output =
                loraLinearReduceScatter({LoraLinearParams(down_gemm_params, params.lora_input.down_lora_input),
                                         params.output,
                                         params.qscheme,
                                         params.output->type(),
                                         ParallelMode::FFN_TP});
            // syncAndCheck();
            gemm_output = reduce_scatter_output.reduce_scatter_recv_buffer;
            output      = reduce_scatter_output.output;
            printBufferData(*gemm_output, "ffn_rs_inter_output");
            printBufferData(*output, "ffn_rs_final_output");
        } else {
            auto down_gemm_params =
                GemmParams(*(up_output), *(params.weights.down_weight->kernel), nullopt, params.output);
            down_gemm_params.qscheme = params.qscheme;
            output = loraLinear(LoraLinearParams(down_gemm_params, params.lora_input.down_lora_input)).output;
        }
    }

    printBufferData(*output, "ffn_out");
    return FfnLayerOutput({std::move(output)});
}

}  // namespace rtp_llm
