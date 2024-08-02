#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/cuda/Dispatch.h"

// kernels
#include "src/fastertransformer/kernels/moe_topKSoftmax_kernels.h"

using namespace std;

namespace fastertransformer {

FfnLayerOutput ROCmDevice::moeFfnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

    const auto& moe_conf     = params.configs.moe_configs.value();
    const auto& hidden       = params.input;
    const auto& weights      = params.weights;
    const auto  compute_type = hidden.type();
    const auto  weights_type = weights.moe_up_weight->kernel->type();
    const auto  num_token    = hidden.shape()[0];
    const auto  model_dim    = hidden.shape()[1];
    const auto  num_expert   = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto  inter_dim    = params.weights.moe_gate_weight->kernel->shape()[2];
    const auto  top_k        = moe_conf.top_k;
    // TODO group_size
    auto group_size = 0;
    if (params.weights.moe_gate_weight->kernel->isQBuffer()) {
        if (dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zerosData() != nullptr) {
            group_size =
                params.weights.moe_gate_weight->kernel->shape()[1]
                / dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zeros().shape()[1];
        }
    }

    const auto output          = allocateBuffer({compute_type, {num_token, model_dim}});
    const auto output_gate     = allocateBuffer({compute_type, {2, top_k * num_token, inter_dim}});
    const auto hidden_permuted = allocateBuffer({compute_type, {top_k * num_token, model_dim}});

    const auto gate = gemm({hidden, *params.weights.moe_gating_weight->kernel, nullopt, nullptr, DataType::TYPE_FP32});

    MOEParallelismConfig parallelism_config;
    // TODO: cuda version also not init this
    const size_t num_experts_per_node = num_expert / parallelism_config.ep_size;
    const int    start_expert         = num_experts_per_node * parallelism_config.ep_rank;
    const int    end_expert           = start_expert + num_experts_per_node;

    // const size_t num_topkTokens = pad_to_multiple_of_16(num_token * top_k);
    const size_t num_topkTokens = num_token * top_k;
    const auto   topk_scales    = allocateBuffer({DataType::TYPE_FP32, {num_token, top_k}});
    // top_k will be 1, 2, 4, 8, ..., 2**n
    // topk_rowColID >> log2(top_k)   = rowID (i.e. tokenID)
    // topk_rowColID &&     (top_k-1) = colID (i.e. k_ID)
    const auto topk_rowColID                 = allocateBuffer({DataType::TYPE_INT32, {num_token, top_k}});
    const auto topk_rowColID_sorted          = allocateBuffer({DataType::TYPE_INT32, {num_token, top_k}});
    const auto topk_expertID                 = allocateBuffer({DataType::TYPE_INT32, {num_token, top_k}});
    const auto topk_expertID_sorted          = allocateBuffer({DataType::TYPE_INT32, {num_token, top_k}});
    const auto total_rows_before_expert      = allocateBuffer({DataType::TYPE_INT32, {num_experts_per_node}});
    const auto total_rows_before_expert_host =
        allocateBuffer({DataType::TYPE_INT32, {num_experts_per_node}, AllocationType::HOST});

    const auto expanded_source_row_to_expanded_dest_row = allocateBuffer({DataType::TYPE_INT32, {num_token, top_k}});

    topkGatingSoftmax_KL(gate->data<float>(),
                         nullptr,  // finished
                         nullptr,  // softmax_out
                         topk_scales->data<float>(),
                         topk_expertID->data<int>(),
                         topk_rowColID->data<int>(),
                         num_token,
                         num_experts_per_node,
                         top_k,
                         start_expert,
                         end_expert,
                         stream_);
    printBufferData(*topk_scales, "topk_scales");
    printBufferData(*topk_expertID, "topk_expertID");
    printBufferData(*topk_rowColID, "topk_rowColID");

    size_t num_bits = (int)log2(num_experts_per_node) + 1;
    sort_KL(topk_expertID->data<int>(),
            topk_rowColID->data<int>(),
            topk_expertID_sorted->data<int>(),
            topk_rowColID_sorted->data<int>(),
            num_topkTokens,
            num_bits,
            stream_);
    printBufferData(*topk_expertID_sorted, "topk_expertID_sorted");
    printBufferData(*topk_rowColID_sorted, "topk_rowColID_sorted");

    // Upper bound on number of expanded rows
    computeTotalRowsBeforeExpert_KL(topk_expertID_sorted->data<int>(),
                                    num_token * top_k,
                                    num_experts_per_node,
                                    total_rows_before_expert->data<int>(),
                                    stream_);
    printBufferData(*total_rows_before_expert, "total_rows_before_expert");

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(compute_type,
                                     permutInputRows_KL,
                                     hidden.data(),
                                     hidden_permuted->data(),
                                     topk_rowColID_sorted->data<int>(),
                                     expanded_source_row_to_expanded_dest_row->data<int>(),
                                     num_token,
                                     model_dim,
                                     nullptr,  // num_valid_tokens_ptr
                                     top_k,
                                     stream_);
    printBufferData(*expanded_source_row_to_expanded_dest_row, "expanded_source_row_to_expanded_dest_row");
    // std::cout << "hidden" << hidden.debugStringMeta() << std::endl;
    // printBufferData(hidden, "hidden");
    // std::cout << "hidden_permuted" << hidden_permuted->debugStringMeta() << std::endl;
    // printBufferData(*hidden_permuted, "hidden_permuted");

    copy({*total_rows_before_expert_host, *total_rows_before_expert});
    syncAndCheck();

    auto moeParams = rocmMoeParams({hidden_permuted->data(),
                                    params.configs.activation_type,

                                    weights.moe_gate_weight->kernel->data(),
                                    BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
                                    BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
                                    OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),

                                    weights.moe_up_weight->kernel->data(),
                                    BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_up_weight->kernel),
                                    BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_up_weight->kernel),
                                    OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_up_weight->bias),

                                    weights.moe_down_weight->kernel->data(),
                                    BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_down_weight->kernel),
                                    BUFFER_GET_ZERO_IF_Q_BUFFER(weights.moe_down_weight->kernel),
                                    OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_down_weight->bias),

                                    output->data(),
                                    output_gate->data(),
                                    num_experts_per_node,
                                    total_rows_before_expert_host->data<int>(),
                                    inter_dim,
                                    model_dim,
                                    stream_});
    moe_runner_->runCKMoe(moeParams, compute_type, weights_type);

    // std::cout << "1111gate" << weights.moe_gate_weight->kernel->debugStringMeta() << std::endl;
    // printBufferData(*((weights.moe_gate_weight->kernel->index(0))->slice(0, 6)), "moe_gate_weight");
    std::cout << "output_gate" << output_gate->index(0)->debugStringMeta() << std::endl;
    printBufferData(*(output_gate->index(0)), "output_gate");

    // std::cout << "1111up" << weights.moe_up_weight->kernel->debugStringMeta() << std::endl;
    // printBufferData(*((weights.moe_up_weight->kernel->index(0))->slice(0, 6)), "moe_up_weight");
    std::cout << "output_up" << output_gate->index(1)->debugStringMeta() << std::endl;
    printBufferData(*(output_gate->index(1)), "output_up");

    // std::cout << "1111down" << weights.moe_down_weight->kernel->debugStringMeta() << std::endl;
    // printBufferData(*((weights.moe_down_weight->kernel->index(0))->slice(0, 6)), "moe_down_weight");
    std::cout << "output_all1" << hidden_permuted->debugStringMeta() << std::endl;
    printBufferData(*(hidden_permuted), "output_all");

    // todo: moe_norm not implemented, this dispatch need be fused into runCKMoe
    assert(moe_conf.has_moe_norm == false);
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(compute_type,
                                     finalizeMoeRoutingKernelLauncher,
                                     hidden_permuted->data(),
                                     output->data(),
                                     nullptr,  // skip_1
                                     nullptr,  // skip_2
                                     nullptr,  // bias
                                     topk_scales->data<float>(),
                                     expanded_source_row_to_expanded_dest_row->data<int>(),
                                     topk_expertID->data<int>(),
                                     num_token,
                                     model_dim,
                                     top_k,
                                     nullptr,  // num_valid_ptr
                                     parallelism_config.tp_rank,
                                     1,  // renormalization_mode, 0: no norm, 1: * topk_scale, 2: renorm_scale
                                     stream_);
    std::cout << "output_all2" << output->debugStringMeta() << std::endl;
    printBufferData(*(output), "output_all2");

    return FfnLayerOutput({move(output)});
}

}  // namespace fastertransformer
