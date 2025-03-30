#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/cutlass/cutlass_kernels/moe_gemm/moe_fp8_kernels.h"
#include "src/fastertransformer/deep_gemm/DeepGemmPlugin.h"

using namespace std;
namespace trt = tensorrt_llm::kernels;
namespace fastertransformer {

FfnLayerOutput CudaDevice::moeFfnFp8(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
    using T = __nv_bfloat16;
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    RUNTIME_ASSERT_OP_ARG(params.input.isQBuffer(), "fp8 moe ffn must be qbuffer");
    const auto& moe_conf            = params.configs.moe_configs.value();
    bool        is_gated_activation = isGatedActivation(params.configs.activation_type);
    const auto& hidden              = reinterpret_cast<const QBuffer&>(params.input).kernel();
    const auto& hidden_scales       = reinterpret_cast<const QBuffer&>(params.input).scales();
    const auto&  weights              = params.weights;
    const auto   token_num            = hidden.shape()[0];
    const auto   hidden_size          = hidden.shape()[1];
    BufferPtr output = nullptr;
    if (params.output) {
        output = params.output;
    } else {
        output = allocateBuffer({DataType::TYPE_BF16, {token_num, hidden_size}});
    }
    if (token_num == 0) {
        return {output};
    }
    const auto   num_experts          = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto   top_k                = moe_conf.top_k;
    const auto   moe_inter_size       = moe_conf.moe_inter_padding_size;
    const size_t num_experts_per_node = num_experts / moe_conf.ep_size;
    const auto   src_row_to_dst       = allocateBuffer({DataType::TYPE_INT32, {top_k, token_num}}, {"moe_src_to_dst"});
    cudaMemsetAsync(src_row_to_dst->data(), -1, src_row_to_dst->sizeBytes(), stream_);

    const auto source_rows      = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"source_rows"});
    const auto permuted_experts = allocateBuffer({DataType::TYPE_INT32, {top_k, token_num}}, {"permuted_experts"});
    const auto permuted_rows    = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"permuted_rows"});
    const auto expert_first_token_offset =
        allocateBuffer({DataType::TYPE_INT64, {num_experts_per_node + 1}}, {"expert_first_token_offset"});

    trt::CubKeyValueSorter sorter;
    const size_t           sorter_size           = trt::CubKeyValueSorter::getWorkspaceSize(token_num, num_experts);
    const auto             sorter_ws             = allocateBuffer({DataType::TYPE_BYTES, {sorter_size}}, {"sorter_ws"});
    int const              start_expert          = num_experts_per_node * moe_conf.ep_rank;
    int const              end_expert            = start_expert + num_experts_per_node;
    auto                   expert_for_source_row = gate_outputs.expert_ids;
    auto                   expert_scales         = gate_outputs.expert_scales;
    trt::genSourceRow(expert_for_source_row->data<int>(),
                      source_rows->data<int>(),
                      token_num,
                      top_k,
                      num_experts,
                      start_expert,
                      end_expert,
                      stream_);
    trt::sortAndScanSoftmaxOutput(expert_for_source_row->data<int>(),
                                  source_rows->data<int>(),
                                  permuted_experts->data<int>(),
                                  permuted_rows->data<int>(),
                                  expert_first_token_offset->data<int64_t>(),
                                  token_num,
                                  num_experts,
                                  num_experts_per_node,
                                  top_k,
                                  sorter,
                                  static_cast<void*>(sorter_ws->data()),
                                  stream_);
    sync_check_cuda_error();

    const auto expert_first_token_offset_host     = clone({*expert_first_token_offset, AllocationType::HOST});
    int64_t*   expert_first_token_offset_host_ptr = expert_first_token_offset_host->data<int64_t>();
    size_t     total_padding_num                  = 0;
    const auto permuted_src_row_to_dst =
        allocateBuffer({DataType::TYPE_INT32, {token_num * top_k}, AllocationType::HOST}, {"permuted_rows"});
    int* permuted_src_row_to_dst_ptr = permuted_src_row_to_dst->data<int>();
    BufferPtr padding_group_index = allocateBuffer(
        {DataType::TYPE_INT32, {pad_to_multiple_of_128(token_num) * num_experts_per_node}, AllocationType::HOST},
        {"padding_group_index"});
    int* padding_group_index_ptr = padding_group_index->data<int>();
    for (int i = 0; i < num_experts_per_node; ++i) {
        size_t src_row_offset = expert_first_token_offset_host_ptr[i];
        size_t num_row_now    = expert_first_token_offset_host_ptr[i + 1] - expert_first_token_offset_host_ptr[i];
        for (int j = 0; j < num_row_now; ++j) {
            permuted_src_row_to_dst_ptr[src_row_offset + j] = total_padding_num + j;
        }
        size_t padding_size = pad_to_multiple_of_128(num_row_now);
        for (int j = 0; j < padding_size; ++j) {
            padding_group_index_ptr[total_padding_num + j] = i;
        }
        total_padding_num += padding_size;
    }
    BufferPtr permuted_src_row_to_dst_device = clone({*permuted_src_row_to_dst});
    int64_t dest_num_rows = expert_first_token_offset_host_ptr[num_experts_per_node];
    BufferPtr permuted_padding_input =
        allocateBuffer({DataType::TYPE_FP8_E4M3, {total_padding_num, hidden_size}}, {"permuted_padding_input"});
    BufferPtr permuted_padding_input_fp8_scales =
        allocateBuffer({DataType::TYPE_FP32, {total_padding_num, hidden_size / 128}}, {"permuted_padding_input_fp8_scales"});
    BufferPtr permuted_padding_scales =
        allocateBuffer({DataType::TYPE_FP32, {total_padding_num}}, {"permuted_padding_scales"});
    expandInputRowsKernelLauncherContiguous<__nv_fp8_e4m3>(hidden.data<__nv_fp8_e4m3>(),
                                                           hidden_scales.data<float>(),
                                                           permuted_padding_input->data<__nv_fp8_e4m3>(),
                                                           permuted_padding_input_fp8_scales->data<float>(),
                                                           expert_scales->data<float>(),
                                                           permuted_padding_scales->data<float>(),
                                                           permuted_rows->data<int>(),
                                                           permuted_src_row_to_dst_device->data<int>(),
                                                           src_row_to_dst->data<int>(),
                                                           token_num,
                                                           dest_num_rows,
                                                           hidden_size,
                                                           top_k,
                                                           stream_);
    sync_check_cuda_error();

    BufferPtr fc1_result;
    if (is_gated_activation) {
        fc1_result =
            allocateBuffer({DataType::TYPE_BF16, {total_padding_num, (size_t)moe_inter_size * 2}}, {"fc1_result"});
    } else {
        fc1_result = allocateBuffer({DataType::TYPE_BF16, {total_padding_num, (size_t)moe_inter_size}}, {"fc1_result"});
    }
    BufferPtr permuted_padding_input_fp8(new QBuffer(std::move(permuted_padding_input),
                                                     std::move(permuted_padding_input_fp8_scales),
                                                     std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                                                                    DataType::TYPE_INVALID,
                                                                                    {0},
                                                                                    nullptr)))));
    
    DeepGemmPlugin::groupedGemmFp8Contiguous(*permuted_padding_input_fp8,
                                             *weights.moe_gate_weight->kernel,
                                             *fc1_result,
                                             padding_group_index->view(0, total_padding_num),
                                             stream_);

    sync_check_cuda_error();
    using GemmOutputType = __nv_bfloat16;
    using ScaleBiasType  = __nv_bfloat16;
    BufferPtr fc1_activation =
        allocateBuffer({DataType::TYPE_BF16, {total_padding_num, (size_t)moe_inter_size}}, {"fc1_result"});

    doActivationContiguous<T, GemmOutputType, ScaleBiasType>(
        fc1_activation->data<T>(),
        static_cast<GemmOutputType const*>(fc1_result->data<T>()),
        nullptr,
        (ScaleBiasType*)OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),
        true,
        permuted_src_row_to_dst_device->data<int>(),
        dest_num_rows,
        moe_inter_size,
        params.configs.activation_type,
        permuted_experts->data<int>(),
        stream_);
    fc1_result.reset();

    sync_check_cuda_error();
    const auto fc2_result = allocateBuffer({DataType::TYPE_BF16, {total_padding_num, hidden_size}}, {"fc2_result"});
    BufferPtr fc1_activation_fp8 = quantize({*fc1_activation, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerTokenBlock});
    
    DeepGemmPlugin::groupedGemmFp8Contiguous(
        *fc1_activation_fp8, *weights.moe_down_weight->kernel, *fc2_result, padding_group_index->view(0, total_padding_num), stream_);

    sync_check_cuda_error();
    using OutputType = __nv_bfloat16;

    trt::MOEParallelismConfig parallel_config(1, 0, moe_conf.ep_size, moe_conf.ep_rank);

    trt::finalizeMoeRoutingKernelLauncher<OutputType, OutputType, GemmOutputType, ScaleBiasType>(
        fc2_result->data<GemmOutputType>(),
        output->data<OutputType>(),
        (ScaleBiasType*)OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_down_weight->bias),
        expert_scales->data<float>(),
        src_row_to_dst->data<int>(),
        expert_for_source_row->data<int>(),
        token_num,
        hidden_size,
        top_k,
        nullptr,
        parallel_config,
        trt::MOEExpertScaleNormalizationMode::NONE,
        stream_);

    sync_check_cuda_error();
    return {output};
}

}  // namespace fastertransformer
