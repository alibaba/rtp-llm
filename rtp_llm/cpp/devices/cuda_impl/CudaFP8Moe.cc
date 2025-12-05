#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/kernels/moe_kernels.h"
#ifdef ENABLE_FP8
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_fp8_kernels.h"
#include "rtp_llm/cpp/cuda/deep_gemm/DeepGemmPlugin.h"

using namespace std;
namespace trt = tensorrt_llm::kernels;
#endif

namespace rtp_llm {

FfnLayerOutput CudaDevice::moeFfnFp8(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
#ifdef ENABLE_FP8
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

    const auto token_num = params.input.shape()[0];
    if (token_num <= static_cast<size_t>(init_params_.moe_config.max_moe_normal_masked_token_num)) {
        return moeFfnFp8Masked(params, gate_outputs);
    } else {
        return moeFfnFp8Contiguous(params, gate_outputs);
    }
#else
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    return {nullptr};
#endif
}

FfnLayerOutput CudaDevice::moeFfnFp8Contiguous(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
#ifdef ENABLE_FP8
    using T = __nv_bfloat16;
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    BufferPtr hidden_fp8;
    BufferPtr hidden_fp8_scales;
    BufferPtr quantize_buffer;

    printBufferData(params.input, "moeFfnFp8_input");
    if (params.input.isQBuffer()) {
        hidden_fp8        = reinterpret_cast<const QBuffer&>(params.input).kernelPtr();
        hidden_fp8_scales = reinterpret_cast<const QBuffer&>(params.input).scalesPtr();
    } else {
        quantize_buffer   = quantize({params.input, DataType::TYPE_QFP8_E4M3, 1, params.qscheme});
        hidden_fp8        = std::dynamic_pointer_cast<QBuffer>(quantize_buffer)->kernelPtr();
        hidden_fp8_scales = std::dynamic_pointer_cast<QBuffer>(quantize_buffer)->scalesPtr();
    }

    const auto& moe_conf            = params.configs.moe_configs.value();
    bool        is_gated_activation = isGatedActivation(params.configs.activation_type);
    const auto& weights             = params.weights;
    const auto  token_num           = hidden_fp8->shape()[0];
    const auto  hidden_size         = hidden_fp8->shape()[1];
    BufferPtr   output              = nullptr;
    if (params.output) {
        output = params.output;
    } else {
        output = allocateBuffer({DataType::TYPE_BF16, {token_num, hidden_size}});
    }
    if (token_num == 0) {
        return {output};
    }
    const auto   num_experts          = moe_conf.expert_num + moe_conf.extra_expert_num;
    const auto   top_k                = moe_conf.top_k;
    const auto   moe_inter_size       = is_gated_activation ? weights.moe_gate_weight->kernel->shape()[1] / 2 : weights.moe_gate_weight->kernel->shape()[1];
    const size_t num_experts_per_node = num_experts / moe_conf.ep_size;
    const auto   src_row_to_dst       = allocateBuffer({DataType::TYPE_INT32, {top_k, token_num}}, {"moe_src_to_dst"});
    check_cuda_value(cudaMemsetAsync(src_row_to_dst->data(), -1, src_row_to_dst->sizeBytes(), stream_));

    const auto source_rows      = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"source_rows"});
    const auto permuted_experts = allocateBuffer({DataType::TYPE_INT32, {top_k, token_num}}, {"permuted_experts"});
    const auto permuted_rows    = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"permuted_rows"});
    const auto expert_first_token_offset =
        allocateBuffer({DataType::TYPE_INT64, {num_experts_per_node + 1}}, {"expert_first_token_offset"});

    trt::CubKeyValueSorter sorter(num_experts);
    const size_t           sorter_size =
        pad_to_multiple_of_128(trt::CubKeyValueSorter::getWorkspaceSize(token_num * top_k, num_experts));
    const auto sorter_ws             = allocateBuffer({DataType::TYPE_BYTES, {sorter_size}}, {"sorter_ws"});
    int const  start_expert          = num_experts_per_node * moe_conf.ep_rank;
    int const  end_expert            = start_expert + num_experts_per_node;
    auto       expert_for_source_row = gate_outputs.expert_ids;
    auto       expert_scales         = gate_outputs.expert_scales;
    printBufferData(*expert_for_source_row, "expert_for_source_row");

    // these logics are from DeepEPDispatch, might could be fused.
    if (expert_for_source_row->type() != DataType::TYPE_INT32) {
        expert_for_source_row = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"moe_expert_ids_int32"});

        genSourceRowRevert(gate_outputs.expert_ids->data<int64_t>(),
                           expert_for_source_row->data<int>(),
                           token_num,
                           top_k,
                           start_expert,
                           stream_);
        check_cuda_error();
    }

    trt::genSourceRow(expert_for_source_row->data<int>(),
                      source_rows->data<int>(),
                      token_num,
                      top_k,
                      num_experts,
                      start_expert,
                      end_expert,
                      stream_);
    printBufferData(*source_rows, "source_rows");
    check_cuda_error();
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
    printBufferData(*permuted_experts, "permuted_experts");
    printBufferData(*permuted_rows, "permuted_rows");
    printBufferData(*expert_first_token_offset, "expert_first_token_offset");
    check_cuda_error();

    const auto expert_first_token_offset_host     = clone({*expert_first_token_offset, AllocationType::HOST});
    int64_t*   expert_first_token_offset_host_ptr = expert_first_token_offset_host->data<int64_t>();
    size_t     total_padding_num                  = 0;
    const auto permuted_src_row_to_dst =
        allocateBuffer({DataType::TYPE_INT32, {token_num * top_k}, AllocationType::HOST}, {"permuted_rows"});
    int* permuted_src_row_to_dst_ptr = permuted_src_row_to_dst->data<int>();
    // First calculate padding with 128 alignment to check sparsity
    size_t total_padding_num_128 = 0;
    for (int i = 0; i < num_experts_per_node; ++i) {
        size_t num_row_now  = expert_first_token_offset_host_ptr[i + 1] - expert_first_token_offset_host_ptr[i];
        size_t padding_size = pad_to_multiple_of_128(num_row_now);
        total_padding_num_128 += padding_size;
    }
    // for use_all_gather=1 path, we need to check padding num = 0 after get selected-expert token num
    if (total_padding_num_128 == 0) {
        // for all-reduce, we need to set output to 0
        bufMemset(*output, 0, DeviceStream::DEFAULT);
        return {output};
    }
    // Decide padding strategy based on sparsity
    bool use_64_padding = (float(total_padding_num_128) / (token_num * top_k)) > 1.5;
    // Allocate buffer based on chosen strategy
    BufferPtr padding_group_index;
    if (use_64_padding) {
        padding_group_index = allocateBuffer(
            {DataType::TYPE_INT32, {pad_to_multiple_of_64(token_num) * num_experts_per_node}, AllocationType::HOST},
            {"padding_group_index"});
    } else {
        padding_group_index = allocateBuffer(
            {DataType::TYPE_INT32, {pad_to_multiple_of_128(token_num) * num_experts_per_node}, AllocationType::HOST},
            {"padding_group_index"});
    }
    int* padding_group_index_ptr = padding_group_index->data<int>();
    // Fill data with chosen padding strategy
    total_padding_num = 0;
    for (int i = 0; i < num_experts_per_node; ++i) {
        size_t src_row_offset = expert_first_token_offset_host_ptr[i];
        size_t num_row_now    = expert_first_token_offset_host_ptr[i + 1] - expert_first_token_offset_host_ptr[i];
        for (int j = 0; j < num_row_now; ++j) {
            permuted_src_row_to_dst_ptr[src_row_offset + j] = total_padding_num + j;
        }
        size_t padding_size = use_64_padding ? pad_to_multiple_of_64(num_row_now) : pad_to_multiple_of_128(num_row_now);
        for (int j = 0; j < padding_size; ++j) {
            padding_group_index_ptr[total_padding_num + j] = i;
        }
        total_padding_num += padding_size;
    }
    BufferPtr permuted_src_row_to_dst_device = clone({*permuted_src_row_to_dst});
    BufferPtr padding_group_index_device     = clone({*padding_group_index});
    check_cuda_value(cudaStreamSynchronize(stream_));
    int64_t   dest_num_rows = expert_first_token_offset_host_ptr[num_experts_per_node];
    BufferPtr permuted_padding_input =
        allocateBuffer({DataType::TYPE_FP8_E4M3, {total_padding_num, hidden_size}}, {"permuted_padding_input"});
    BufferPtr permuted_padding_input_fp8_scales = allocateBuffer(
        {DataType::TYPE_FP32, {total_padding_num, hidden_size / 128}}, {"permuted_padding_input_fp8_scales"});
    BufferPtr permuted_padding_scales =
        allocateBuffer({DataType::TYPE_FP32, {total_padding_num}}, {"permuted_padding_scales"});
    printBufferData(*hidden_fp8, "moe_hidden_fp8");
    printBufferData(*hidden_fp8_scales, "moe_hidden_fp8_scales");
    expandInputRowsKernelLauncherContiguous<__nv_fp8_e4m3>(hidden_fp8->data<__nv_fp8_e4m3>(),
                                                           hidden_fp8_scales->data<float>(),
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
    check_cuda_error();

    BufferPtr fc1_activation =
        allocateBuffer({DataType::TYPE_FP8_E4M3, {total_padding_num, (size_t)moe_inter_size}}, {"fc1_activation"});
    BufferPtr fc1_activation_fp8_scales = allocateBuffer(
        {DataType::TYPE_FP32, {total_padding_num, (size_t)moe_inter_size / 128}}, {"fc1_activation_fp8_scales"});

    BufferPtr fc1_result;
    if (is_gated_activation) {
        fc1_result =
            allocateBuffer({DataType::TYPE_BF16, {total_padding_num, (size_t)moe_inter_size * 2}}, {"fc1_result"});
    } else {
        fc1_result = allocateBuffer({DataType::TYPE_BF16, {total_padding_num, (size_t)moe_inter_size}}, {"fc1_result"});
    }
    BufferPtr permuted_padding_input_fp8(
        new QBuffer(std::move(permuted_padding_input),
                    std::move(permuted_padding_input_fp8_scales),
                    std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    printBufferData(*permuted_padding_input_fp8, "fc1_input_fp8");
    printBufferData(*weights.moe_gate_weight->kernel, "moe_gate_weight");
    printBufferData(*padding_group_index_device, "padding_group_index_device");
    DeepGemmPlugin::groupedGemmFp8Contiguous(*permuted_padding_input_fp8,
                                             *weights.moe_gate_weight->kernel,
                                             *fc1_result,
                                             padding_group_index_device->view(0, total_padding_num),
                                             init_params_.user_deep_gemm_num_sm,
                                             use_64_padding,
                                             stream_);
    printBufferData(*fc1_result, "fc1_result");
    check_cuda_error();
    using GemmOutputType = __nv_bfloat16;
    using ScaleBiasType  = __nv_bfloat16;

    doActivationContiguous<GemmOutputType, ScaleBiasType>(
        fc1_activation->data<__nv_fp8_e4m3>(),
        fc1_activation_fp8_scales->data<float>(),
        static_cast<GemmOutputType const*>(fc1_result->data<T>()),
        (ScaleBiasType*)OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),
        true,
        permuted_src_row_to_dst_device->data<int>(),
        dest_num_rows,
        moe_inter_size,
        params.configs.activation_type,
        permuted_experts->data<int>(),
        stream_);
    fc1_result.reset();

    check_cuda_error();
    const auto fc2_result = allocateBuffer({DataType::TYPE_BF16, {total_padding_num, hidden_size}}, {"fc2_result"});
    BufferPtr  fc1_activation_fp8(
        new QBuffer(std::move(fc1_activation),
                    std::move(fc1_activation_fp8_scales),
                    std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    printBufferData(*fc1_activation_fp8, "fc1_activation_fp8");
    DeepGemmPlugin::groupedGemmFp8Contiguous(*fc1_activation_fp8,
                                             *weights.moe_down_weight->kernel,
                                             *fc2_result,
                                             padding_group_index_device->view(0, total_padding_num),
                                             init_params_.user_deep_gemm_num_sm,
                                             use_64_padding,
                                             stream_);
    printBufferData(*fc2_result, "fc2_result");
    check_cuda_error();
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

    printBufferData(*output, "moe_ffn_out");

    check_cuda_error();
    return {output};

#else
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    return {nullptr};
#endif
}

FfnLayerOutput CudaDevice::moeFfnFp8Masked(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
#ifdef ENABLE_FP8
    using T = __nv_bfloat16;
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    BufferPtr hidden_fp8;
    BufferPtr hidden_fp8_scales;
    BufferPtr quantize_buffer;

    printBufferData(params.input, "moeFfnFp8_input");
    if (params.input.isQBuffer()) {
        hidden_fp8        = reinterpret_cast<const QBuffer&>(params.input).kernelPtr();
        hidden_fp8_scales = reinterpret_cast<const QBuffer&>(params.input).scalesPtr();
    } else {
        quantize_buffer   = quantize({params.input, DataType::TYPE_QFP8_E4M3, 1, params.qscheme});
        hidden_fp8        = std::dynamic_pointer_cast<QBuffer>(quantize_buffer)->kernelPtr();
        hidden_fp8_scales = std::dynamic_pointer_cast<QBuffer>(quantize_buffer)->scalesPtr();
    }

    const auto& moe_conf            = params.configs.moe_configs.value();
    bool        is_gated_activation = isGatedActivation(params.configs.activation_type);
    const auto& weights             = params.weights;
    const auto  token_num           = hidden_fp8->shape()[0];
    const auto  hidden_size         = hidden_fp8->shape()[1];
    BufferPtr   output              = nullptr;
    if (params.output) {
        output = params.output;
    } else {
        output = allocateBuffer({DataType::TYPE_BF16, {token_num, hidden_size}});
    }
    if (token_num == 0) {
        return {output};
    }
    const auto   num_experts          = moe_conf.expert_num + moe_conf.extra_expert_num;
    const auto   top_k                = moe_conf.top_k;
    const auto   moe_inter_size       = is_gated_activation ? weights.moe_gate_weight->kernel->shape()[1] / 2 : weights.moe_gate_weight->kernel->shape()[1];
    const size_t num_experts_per_node = num_experts / moe_conf.ep_size;
    const auto   src_row_to_dst       = allocateBuffer({DataType::TYPE_INT32, {top_k, token_num}}, {"moe_src_to_dst"});
    check_cuda_value(cudaMemsetAsync(src_row_to_dst->data(), -1, src_row_to_dst->sizeBytes(), stream_));

    const auto source_rows      = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"source_rows"});
    const auto permuted_experts = allocateBuffer({DataType::TYPE_INT32, {top_k, token_num}}, {"permuted_experts"});
    const auto permuted_rows    = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"permuted_rows"});
    const auto expert_first_token_offset =
        allocateBuffer({DataType::TYPE_INT64, {num_experts_per_node + 1}}, {"expert_first_token_offset"});

    trt::CubKeyValueSorter sorter(num_experts);
    const size_t           sorter_size =
        pad_to_multiple_of_128(trt::CubKeyValueSorter::getWorkspaceSize(token_num * top_k, num_experts));
    const auto sorter_ws             = allocateBuffer({DataType::TYPE_BYTES, {sorter_size}}, {"sorter_ws"});
    int const  start_expert          = num_experts_per_node * moe_conf.ep_rank;
    int const  end_expert            = start_expert + num_experts_per_node;
    auto       expert_for_source_row = gate_outputs.expert_ids;
    auto       expert_scales         = gate_outputs.expert_scales;
    printBufferData(*expert_for_source_row, "expert_for_source_row");

    // these logics are from DeepEPDispatch, might could be fused.
    if (expert_for_source_row->type() != DataType::TYPE_INT32) {
        expert_for_source_row = allocateBuffer({DataType::TYPE_INT32, {token_num, top_k}}, {"moe_expert_ids_int32"});

        genSourceRowRevert(gate_outputs.expert_ids->data<int64_t>(),
                           expert_for_source_row->data<int>(),
                           token_num,
                           top_k,
                           start_expert,
                           stream_);
        check_cuda_error();
    }

    trt::genSourceRow(expert_for_source_row->data<int>(),
                      source_rows->data<int>(),
                      token_num,
                      top_k,
                      num_experts,
                      start_expert,
                      end_expert,
                      stream_);
    printBufferData(*source_rows, "source_rows");
    check_cuda_error();
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
    printBufferData(*permuted_experts, "permuted_experts");
    printBufferData(*permuted_rows, "permuted_rows");
    printBufferData(*expert_first_token_offset, "expert_first_token_offset");
    check_cuda_error();

    size_t padding_size            = DeepGemmPlugin::paddingMasked(token_num);
    size_t max_num_rows            = token_num * top_k;
    size_t expected_m              = std::min(padding_size, ceil_div<size_t>(max_num_rows, num_experts));
    auto   permuted_src_row_to_dst = allocateBuffer({DataType::TYPE_INT32, {max_num_rows}}, {"permuted_rows"});
    auto   masked_m                = allocateBuffer({DataType::TYPE_INT32, {num_experts_per_node}}, {"masked_m"});
    computeSrc2Dst(expert_first_token_offset->data<int64_t>(),
                   permuted_src_row_to_dst->data<int>(),
                   masked_m->data<int>(),
                   num_experts_per_node,
                   padding_size,
                   stream_);
    check_cuda_error();

    BufferPtr permuted_padding_input = allocateBuffer(
        {DataType::TYPE_FP8_E4M3, {num_experts_per_node, padding_size, hidden_size}}, {"permuted_padding_input"});
    BufferPtr permuted_padding_input_fp8_scales =
        allocateBuffer({DataType::TYPE_FP32, {num_experts_per_node, padding_size, hidden_size / 128}},
                       {"permuted_padding_input_fp8_scales"});
    BufferPtr permuted_padding_scales =
        allocateBuffer({DataType::TYPE_FP32, {num_experts_per_node, padding_size}}, {"permuted_padding_scales"});
    printBufferData(*hidden_fp8, "moe_hidden_fp8");
    printBufferData(*hidden_fp8_scales, "moe_hidden_fp8_scales");
    expandInputRowsKernelLauncherContiguous_V2<__nv_fp8_e4m3>(hidden_fp8->data<__nv_fp8_e4m3>(),
                                                              hidden_fp8_scales->data<float>(),
                                                              permuted_padding_input->data<__nv_fp8_e4m3>(),
                                                              permuted_padding_input_fp8_scales->data<float>(),
                                                              expert_scales->data<float>(),
                                                              permuted_padding_scales->data<float>(),
                                                              permuted_rows->data<int>(),
                                                              permuted_src_row_to_dst->data<int>(),
                                                              src_row_to_dst->data<int>(),
                                                              expert_first_token_offset->data<int64_t>(),
                                                              num_experts_per_node,
                                                              token_num,
                                                              max_num_rows,
                                                              hidden_size,
                                                              top_k,
                                                              stream_);
    check_cuda_error();

    BufferPtr fc1_result;
    if (is_gated_activation) {
        fc1_result = allocateBuffer(
            {DataType::TYPE_BF16, {num_experts_per_node, padding_size, (size_t)moe_inter_size * 2}}, {"fc1_result"});
    } else {
        fc1_result = allocateBuffer({DataType::TYPE_BF16, {num_experts_per_node, padding_size, (size_t)moe_inter_size}},
                                    {"fc1_result"});
    }
    BufferPtr permuted_padding_input_fp8(
        new QBuffer(std::move(permuted_padding_input),
                    std::move(permuted_padding_input_fp8_scales),
                    std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    printBufferData(*permuted_padding_input_fp8, "fc1_input_fp8");
    printBufferData(*weights.moe_gate_weight->kernel, "moe_gate_weight");
    printBufferData(*masked_m, "masked_m");
    DeepGemmPlugin::groupedGemmFp8Masked_V2(*permuted_padding_input_fp8,
                                            *weights.moe_gate_weight->kernel,
                                            *fc1_result,
                                            *masked_m,
                                            expected_m,
                                            init_params_.user_deep_gemm_num_sm,
                                            stream_);
    printBufferData(*fc1_result, "fc1_result");
    check_cuda_error();
    using GemmOutputType     = __nv_bfloat16;
    using ScaleBiasType      = __nv_bfloat16;
    BufferPtr fc1_activation = allocateBuffer(
        {DataType::TYPE_FP8_E4M3, {num_experts_per_node, padding_size, (size_t)moe_inter_size}}, {"fc1_activation"});
    BufferPtr fc1_activation_fp8_scales =
        allocateBuffer({DataType::TYPE_FP32, {num_experts_per_node, (size_t)moe_inter_size / 128, padding_size}},
                       {"fc1_activation_fp8_scales"});

    doActivationMasked<GemmOutputType, ScaleBiasType>(
        fc1_activation->data<__nv_fp8_e4m3>(),
        fc1_activation_fp8_scales->data<float>(),
        static_cast<GemmOutputType const*>(fc1_result->data<T>()),
        (ScaleBiasType*)OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),
        true,
        num_experts_per_node,
        padding_size,
        moe_inter_size,
        params.configs.activation_type,
        masked_m->data<int>(),
        stream_);
    fc1_result.reset();

    check_cuda_error();
    auto fc2_result =
        allocateBuffer({DataType::TYPE_BF16, {num_experts_per_node, padding_size, hidden_size}}, {"fc2_result"});
    BufferPtr fc1_activation_fp8(
        new QBuffer(std::move(fc1_activation),
                    std::move(fc1_activation_fp8_scales),
                    std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    printBufferData(*fc1_activation_fp8, "fc1_activation_fp8");
    DeepGemmPlugin::groupedGemmFp8Masked(*fc1_activation_fp8,
                                         *weights.moe_down_weight->kernel,
                                         *fc2_result,
                                         *masked_m,
                                         expected_m,
                                         init_params_.user_deep_gemm_num_sm,
                                         stream_);
    printBufferData(*fc2_result, "fc2_result");
    check_cuda_error();
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

    printBufferData(*output, "moe_ffn_out");

    check_cuda_error();
    return {output};

#else
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    return {nullptr};
#endif
}

FfnLayerOutput CudaDevice::deepEpLLMoeFfn(const FfnLayerParams& params, const MoeGateSelectOutput& gate_outputs) {
#ifdef ENABLE_FP8
#ifdef ENABLE_DEEP_EP
    using T = __nv_bfloat16;
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");
    const auto&  moe_conf             = params.configs.moe_configs.value();
    bool         is_gated_activation  = isGatedActivation(params.configs.activation_type);
    const auto&  weights              = params.weights;
    const auto   num_experts          = moe_conf.expert_num + moe_conf.extra_expert_num;
    const auto   moe_inter_size       = is_gated_activation ? weights.moe_gate_weight->kernel->shape()[1] / 2 : weights.moe_gate_weight->kernel->shape()[1];
    const size_t num_experts_per_node = num_experts / moe_conf.ep_size;

    BufferPtr      quantize_hidden;
    BufferPtr      quantize_hidden_holder;
    auto           deep_ep_ll_output = gate_outputs.deep_ep_ll_output;
    torch::Tensor  hidden            = deep_ep_ll_output->packed_recv_x;
    vector<size_t> hidden_shape;
    if (deep_ep_ll_output->packed_recv_x_scales.has_value()) {
        BufferPtr hidden_fp8        = torchTensor2Buffer(deep_ep_ll_output->packed_recv_x);
        hidden_shape                = hidden_fp8->shape();
        BufferPtr hidden_fp8_scales = torchTensor2Buffer(deep_ep_ll_output->packed_recv_x_scales.value());
        quantize_hidden.reset(new QBuffer(
            std::move(hidden_fp8),
            std::move(hidden_fp8_scales),
            std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    } else {
        RUNTIME_ASSERT_OP_ARG(false, "not support deepepffn bf16 input now");
        // BufferPtr hidden       = torchTensor2Buffer(deep_ep_ll_output->packed_recv_x);
        // hidden_shape           = hidden->shape();
        // // quantize should be 3 dims, not support now
        // quantize_hidden_holder = quantize({hidden->reshape({hidden_shape[0] * hidden_shape[1], hidden_shape[2]}),
        //                                    DataType::TYPE_QFP8_E4M3,
        //                                    1,
        //                                    params.qscheme});
        // auto hidden_fp8        = std::dynamic_pointer_cast<QBuffer>(quantize_hidden_holder)->kernelPtr();
        // auto hidden_fp8_scales = std::dynamic_pointer_cast<QBuffer>(quantize_hidden_holder)->scalesPtr();
        // hidden_fp8->updateShape(hidden_shape);
        // hidden_fp8_scales->updateShape({hidden_shape[0], hidden_shape[1], hidden_shape[2] / 128});
        // quantize_hidden.reset(new QBuffer(
        //     std::move(hidden_fp8),
        //     std::move(hidden_fp8_scales),
        //     std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    }
    const auto token_num   = hidden_shape[1];
    const auto hidden_size = hidden_shape[2];
    RUNTIME_ASSERT_OP_ARG(hidden_shape.size() == 3 && hidden_shape[0] == num_experts_per_node,
                          "hidden_shape dims should be 3 and dim 0 should be num_experts_per_node");
    if (token_num == 0) {
        BufferPtr output = allocateBuffer({DataType::TYPE_BF16, {num_experts_per_node, token_num, hidden_size}});
        return {output};
    }

    BufferPtr fc1_result;
    if (is_gated_activation) {
        fc1_result = allocateBuffer(
            {DataType::TYPE_BF16, {num_experts_per_node, token_num, (size_t)moe_inter_size * 2}}, {"fc1_result"});
    } else {
        fc1_result = allocateBuffer({DataType::TYPE_BF16, {num_experts_per_node, token_num, (size_t)moe_inter_size}},
                                    {"fc1_result"});
    }
    BufferPtr masked_m = torchTensor2Buffer(deep_ep_ll_output->packed_recv_count);
    DeepGemmPlugin::groupedGemmFp8Masked(*quantize_hidden,
                                         *weights.moe_gate_weight->kernel,
                                         *fc1_result,
                                         *masked_m,
                                         token_num / moe_conf.ep_size,
                                         init_params_.user_deep_gemm_num_sm,
                                         stream_);

    check_cuda_error();
    using GemmOutputType     = __nv_bfloat16;
    using ScaleBiasType      = __nv_bfloat16;
    BufferPtr fc1_activation = allocateBuffer(
        {DataType::TYPE_FP8_E4M3, {num_experts_per_node, token_num, (size_t)moe_inter_size}}, {"fc1_activation"});
    BufferPtr fc1_activation_fp8_scales =
        allocateBuffer({DataType::TYPE_FP32, {num_experts_per_node, token_num, (size_t)moe_inter_size / 128}},
                       {"fc1_activation_fp8_scales"});

    doActivationMasked<GemmOutputType, ScaleBiasType>(
        fc1_activation->data<__nv_fp8_e4m3>(),
        fc1_activation_fp8_scales->data<float>(),
        static_cast<GemmOutputType const*>(fc1_result->data<T>()),
        (ScaleBiasType*)OPTIONAL_BUFFER_GET_DATA_OR_NULLPTR(weights.moe_gate_weight->bias),
        true,
        num_experts_per_node,
        token_num,
        moe_inter_size,
        params.configs.activation_type,
        masked_m->data<int>(),
        stream_);
    fc1_result.reset();

    check_cuda_error();
    const auto fc2_result =
        allocateBuffer({DataType::TYPE_BF16, {num_experts_per_node, token_num, hidden_size}}, {"fc2_result"});
    BufferPtr fc1_activation_fp8(
        new QBuffer(std::move(fc1_activation),
                    std::move(fc1_activation_fp8_scales),
                    std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr)))));
    DeepGemmPlugin::groupedGemmFp8Masked(*fc1_activation_fp8,
                                         *weights.moe_down_weight->kernel,
                                         *fc2_result,
                                         *masked_m,
                                         token_num / moe_conf.ep_size,
                                         init_params_.user_deep_gemm_num_sm,
                                         stream_);

    check_cuda_error();
    return {fc2_result};
#else
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    return {nullptr};
#endif
#else
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    return {nullptr};
#endif
}

}  // namespace rtp_llm
