#include <sstream>
#include <iostream>
#include "fused_multihead_attention_common.h"

namespace tensorrt_llm {
namespace kernels {
// Debug string for Fused_multihead_attention_params_v2
inline std::string debugStr(const Fused_multihead_attention_params_v2& params) {
    std::ostringstream oss;
    oss << "Fused_multihead_attention_params_v2 Debug Info:\n";
    oss << "  qkv_ptr addr: " << params.qkv_ptr << "\n";
    oss << "  q_ptr addr: " << params.q_ptr << "\n";
    oss << "  k_ptr addr: " << params.k_ptr << "\n";
    oss << "  v_ptr addr: " << params.v_ptr << "\n";
    oss << "  kv_ptr addr: " << params.kv_ptr << "\n";
    oss << "  packed_mask_ptr addr: " << params.packed_mask_ptr << "\n";
    oss << "  attention_sinks_ptr addr: " << params.attention_sinks_ptr << "\n";
    oss << "  o_ptr addr: " << params.o_ptr << "\n";
    oss << "  softmax_stats_ptr addr: " << params.softmax_stats_ptr << "\n";
    oss << "  q_stride_in_bytes: " << params.q_stride_in_bytes << "\n";
    oss << "  k_stride_in_bytes: " << params.k_stride_in_bytes << "\n";
    oss << "  v_stride_in_bytes: " << params.v_stride_in_bytes << "\n";
    oss << "  packed_mask_stride_in_bytes: " << params.packed_mask_stride_in_bytes << "\n";
    oss << "  o_stride_in_bytes: " << params.o_stride_in_bytes << "\n";
    oss << "  softmax_stats_stride_in_bytes: " << params.softmax_stats_stride_in_bytes << "\n";
    oss << "  blocks_per_tma_load: " << params.blocks_per_tma_load << "\n";
    oss << "  blocks_per_tma_load_log2: " << params.blocks_per_tma_load_log2 << "\n";
    oss << "  b: " << params.b << ", h: " << params.h << ", h_kv: " << params.h_kv
        << ", h_q_per_kv: " << params.h_q_per_kv << "\n";
    oss << "  s: " << params.s << ", d: " << params.d << ", dv: " << params.dv << "\n";
    oss << "  num_grouped_heads: " << params.num_grouped_heads << "\n";
    oss << "  sliding_window_size: " << params.sliding_window_size << "\n";
    oss << "  log2_chunked_attention_size: " << params.log2_chunked_attention_size << "\n";
    oss << "  scale_bmm1: " << params.scale_bmm1 << ", softcapping_scale_bmm1: " << params.softcapping_scale_bmm1
        << "\n";
    oss << "  scale_softmax: " << params.scale_softmax << ", scale_bmm2: " << params.scale_bmm2 << "\n";
    oss << "  cu_q_seqlens: " << params.cu_q_seqlens << ", cu_kv_seqlens: " << params.cu_kv_seqlens
        << ", cu_mask_rows: " << params.cu_mask_rows << "\n";
    oss << "  has_alibi: " << params.has_alibi << "\n";
    oss << "  tile_id_counter_ptr addr: " << params.tile_id_counter_ptr << "\n";
    oss << "  num_tiles: " << params.num_tiles << ", num_tiles_per_head: " << params.num_tiles_per_head << "\n";
    oss << "  use_balanced_scheduling: " << params.use_balanced_scheduling << "\n";
    oss << "  is_s_padded: " << params.is_s_padded << "\n";
    oss << "  SageAttention.q.max_nblock: " << params.sage.q.max_nblock << "\n";
    oss << "  SageAttention.k.max_nblock: " << params.sage.k.max_nblock << "\n";
    oss << "  SageAttention.v.max_nblock: " << params.sage.v.max_nblock << "\n";
    return oss.str();
}

// Debug string for Launch_params
inline std::string debugStr(const Launch_params& params) {
    std::ostringstream oss;
    oss << "Launch_params Debug Info:\n";
    oss << "  kernel_s: " << params.kernel_s << "\n";
    oss << "  total_q_seqlen: " << params.total_q_seqlen << "\n";
    oss << "  total_kv_seqlen: " << params.total_kv_seqlen << "\n";
    oss << "  padded_d: " << params.padded_d << "\n";
    oss << "  ignore_b1opt: " << params.ignore_b1opt << "\n";
    oss << "  force_unroll: " << params.force_unroll << "\n";
    oss << "  force_fp32_acc: " << params.force_fp32_acc << "\n";
    oss << "  interleaved: " << params.interleaved << "\n";
    oss << "  use_tma: " << params.use_tma << "\n";
    oss << "  flash_attention: " << params.flash_attention << "\n";
    oss << "  warp_specialization: " << params.warp_specialization << "\n";
    oss << "  granular_tiling: " << params.granular_tiling << "\n";
    oss << "  dynamic_scheduler: " << params.dynamic_scheduler << "\n";
    oss << "  attention_mask_type: " << static_cast<int>(params.attention_mask_type) << "\n";
    oss << "  attention_input_layout: " << static_cast<int>(params.attention_input_layout) << "\n";
    oss << "  useKernelWithoutAlibi: " << params.useKernelWithoutAlibi << "\n";
    oss << "  useBase2ExpTrick: " << params.useBase2ExpTrick << "\n";
    oss << "  enableAttnLogitSoftcapping: " << params.enableAttnLogitSoftcapping << "\n";
    oss << "  multi_processor_count: " << params.multi_processor_count << "\n";
    oss << "  device_l2_cache_size: " << params.device_l2_cache_size << "\n";
    oss << "  total_device_memory: " << params.total_device_memory << "\n";
    oss << "  sage_block_size_q: " << params.sage_block_size_q << "\n";
    oss << "  sage_block_size_k: " << params.sage_block_size_k << "\n";
    oss << "  sage_block_size_v: " << params.sage_block_size_v << "\n";
    oss << "  supportReturnSoftmaxStats: " << params.supportReturnSoftmaxStats << "\n";
    return oss.str();
}
}  // namespace kernels
}  // namespace tensorrt_llm