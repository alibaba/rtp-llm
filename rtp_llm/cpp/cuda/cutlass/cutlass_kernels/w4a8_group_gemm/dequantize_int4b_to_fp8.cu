#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/w4a8_group_gemm.h"

using namespace cute;

template<class QuantizedElement,
         class DequantizedElement,
         class OperandLayout,
         class ElementScale,
         class ElementZero,
         class ScaleBroadCastLayout,
         class ThrLayout>
__global__ void dequantize_kernel(DequantizedElement*        dq_buffer,
                                  QuantizedElement const*    q_buffer,
                                  OperandLayout const        operand_layout,
                                  ElementScale const*        scale_buffer,
                                  ElementZero const*         zero_buffer,
                                  ScaleBroadCastLayout const broadcasted_scale_layout,
                                  ThrLayout                  thr_layout) {
    // Represent the full tensors to gmem elements.
    // These are expected to have shape [MN, K, L]
    cute::Tensor gmem_op_dq              = cute::make_tensor(cute::make_gmem_ptr(dq_buffer), operand_layout);
    auto         init_quantized_iterator = [&]() {
        if constexpr (cute::sizeof_bits_v<QuantizedElement> >= 8) {
            return cute::make_gmem_ptr(q_buffer);
        } else {
            return cute::subbyte_iterator<const QuantizedElement>(q_buffer);
        }
    };
    cute::Tensor gmem_op_q = cute::make_tensor(init_quantized_iterator(), operand_layout);
    // While the scales are expected to have shape [MN, G, L] but with a stride to allow broadcasting
    // It is expected that K % G == 0
    cute::Tensor gmem_scale_broadcasted = cute::make_tensor(make_gmem_ptr(scale_buffer), broadcasted_scale_layout);
    cute::Tensor gmem_zero_broadcasted  = cute::make_tensor(make_gmem_ptr(zero_buffer), broadcasted_scale_layout);

    // Assign 1 thread per element in the thread block
    auto blk_shape = cute::make_shape(size<0>(thr_layout), _1{}, _1{});  //
    auto blk_coord = cute::make_coord(_, blockIdx.x, blockIdx.y);        // (MN, K, L)

    // Tile across the block
    auto gOp_dq = cute::local_tile(gmem_op_dq, blk_shape, blk_coord);
    auto gScale = cute::local_tile(gmem_scale_broadcasted, blk_shape, blk_coord);
    auto gZero  = cute::local_tile(gmem_zero_broadcasted, blk_shape, blk_coord);
    auto gOp_q  = cute::local_tile(gmem_op_q, blk_shape, blk_coord);

    auto tOpDq_gOpDq   = cute::local_partition(gOp_dq, thr_layout, threadIdx.x);
    auto tScale_gScale = cute::local_partition(gScale, thr_layout, threadIdx.x);
    auto tZero_gZero   = cute::local_partition(gZero, thr_layout, threadIdx.x);
    auto tOpQ_gOpQ     = cute::local_partition(gOp_q, thr_layout, threadIdx.x);

    // Make a fragment of registers to hold gmem loads
    cute::Tensor rmem_op_q      = cute::make_fragment_like(tOpQ_gOpQ(_, _, _, 0));
    cute::Tensor rmem_scale     = cute::make_fragment_like(tScale_gScale(_, _, _, 0));
    cute::Tensor rmem_zero      = cute::make_fragment_like(tZero_gZero(_, _, _, 0));
    cute::Tensor rmem_op_dq     = cute::make_fragment_like(tOpDq_gOpDq(_, _, _, 0));
    cute::Tensor rmem_op_scaled = cute::make_fragment_like<ElementScale>(rmem_op_dq);
    cute::Tensor rmem_zero_buf  = cute::make_fragment_like<ElementScale>(rmem_zero);

    cute::Tensor pred_id            = cute::make_identity_tensor(shape(operand_layout));
    auto         pred_blk_tile      = cute::local_tile(pred_id, blk_shape, blk_coord);
    auto         pred_thr_partition = cute::local_partition(pred_blk_tile, thr_layout, threadIdx.x);

    const auto num_iters = cute::size<3>(tOpDq_gOpDq);

    for (int ii = 0; ii < num_iters; ++ii) {
        const auto thread_offset = cute::get<0>(pred_thr_partition(0, 0, 0, ii));
        if (thread_offset < cute::size<0>(operand_layout)) {
            cute::copy(tOpQ_gOpQ(_, _, _, ii), rmem_op_q);
            cute::copy(tScale_gScale(_, _, _, ii), rmem_scale);
            cute::copy(tZero_gZero(_, _, _, ii), rmem_zero);
            cute::transform(rmem_op_q, rmem_op_scaled, [](const QuantizedElement& elt) { return ElementScale(elt); });
            cute::transform(rmem_zero, rmem_zero_buf, [](const ElementZero& elt) { return ElementScale(elt); });
            cute::transform(rmem_op_scaled, rmem_scale, rmem_op_scaled, cute::multiplies{});
            cute::transform(rmem_op_scaled, rmem_zero_buf, rmem_op_scaled, cute::plus{});
            cute::transform(
                rmem_op_scaled, rmem_op_dq, [](const ElementScale& elt) { return DequantizedElement(elt); });
            cute::copy(rmem_op_dq, tOpDq_gOpDq(_, _, _, ii));
        }
    }
}

template<class QuantizedElement,
         class DequantizedElement,
         class OperandLayout,
         class ElementScale,
         class ElementZero,
         class ScaleLayout>
static void dequantize(DequantizedElement*     dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const     operand_layout,
                       ElementScale const*     scale_buffer,
                       ElementZero const*      zero_buffer,
                       ScaleLayout const       scale_layout,
                       int const               group_size,
                       cudaStream_t&           stream) {
    constexpr int tpb        = 128;
    auto          thr_layout = make_layout(make_shape(Int<tpb>{}));

    const auto num_rows = get<0>(shape(operand_layout));
    const auto gemm_k   = get<1>(shape(operand_layout));  // [MN, K, L]
    const auto batches  = get<2>(shape(operand_layout));  // [MN, K, L]
    const auto scale_k  = get<1>(shape(scale_layout));    // [MN, Scale_K, L]

    if (num_rows != size<0>(scale_layout)) {
        std::cerr << "Invalid first dimension for scales. Must match first dim for weights."
                  << " But got shapes " << shape(operand_layout) << " " << shape(scale_layout) << std::endl;
        exit(-1);
    }

    const auto scale_stride0 = get<0>(stride(scale_layout));
    const auto scale_stride1 = get<1>(stride(scale_layout));
    const auto scale_stride2 = get<2>(stride(scale_layout));

    auto scale_shape_bcast  = make_shape(num_rows, make_shape(group_size, scale_k), batches);
    auto scale_stride_bcast = make_stride(scale_stride0, make_stride(0, scale_stride1), scale_stride2);
    auto scale_layout_bcast = make_layout(scale_shape_bcast, scale_stride_bcast);

    const auto blocks_x = gemm_k;
    const auto blocks_y = batches;

    dim3 blocks(blocks_x, blocks_y, 1);
    dequantize_kernel<<<blocks, tpb, 0, stream>>>(
        dq_buffer, q_buffer, operand_layout, scale_buffer, zero_buffer, scale_layout_bcast, thr_layout);
}

torch::Tensor rtp_llm::run_dequantize_int4b_to_fp8(const torch::Tensor& input,
                                                   const torch::Tensor& scale,
                                                   const torch::Tensor& zero,
                                                   const int            group_size) {
    TORCH_CHECK(input.dtype() == torch::kInt8, "Input must be of type int8.");
    TORCH_CHECK(scale.dtype() == torch::kFloat8_e4m3fn, "Scale must be of type float8_e4m3fn.");
    TORCH_CHECK(zero.dtype() == torch::kFloat8_e4m3fn, "Zero must be of type float8_e4m3fn.");
    TORCH_CHECK(group_size > 0, "Group size must be positive.");

    auto stream  = at::cuda::getCurrentCUDAStream().stream();
    auto M       = input.sizes()[0];
    auto N       = input.sizes()[1] * 8 / cutlass::sizeof_bits<cutlass::int4b_t>::value;
    auto scale_n = cutlass::ceil_div(N, group_size);

    auto input_layout = make_layout(cute::make_shape(M, N, Int<1>{}), cute::make_stride(N, Int<1>{}, Int<0>{}));
    auto scale_zero_layout =
        make_layout(cute::make_shape(M, scale_n, Int<1>{}), cute::make_stride(Int<1>{}, M, Int<0>{}));

    auto output = torch::empty({M, N}, zero.options());

    dequantize(static_cast<cutlass::float_e4m3_t*>(output.data_ptr()),
               static_cast<const cutlass::int4b_t*>(input.data_ptr()),
               input_layout,
               static_cast<const cutlass::float_e4m3_t*>(scale.data_ptr()),
               static_cast<const cutlass::float_e4m3_t*>(zero.data_ptr()),
               scale_zero_layout,
               group_size,
               stream);

    return output;
}
