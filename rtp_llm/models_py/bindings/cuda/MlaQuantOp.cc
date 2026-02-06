#include "rtp_llm/models_py/bindings/cuda/MlaQuantOp.h"
#include "rtp_llm/cpp/kernels/mla_quant_kernel.h"

using namespace rtp_llm;

namespace torch_ext {

void indexer_k_quant_and_cache(at::Tensor&        k,
                               at::Tensor&        kv_cache,
                               at::Tensor&        slot_mapping,
                               int64_t            quant_block_size,
                               const std::string& scale_fmt) {

    rtp_llm::indexer_k_quant_and_cache(k, kv_cache, slot_mapping, quant_block_size, scale_fmt);
}

void cp_gather_indexer_k_quant_cache(const at::Tensor& kv_cache,
                                     at::Tensor&       dst_k,
                                     at::Tensor&       dst_scale,
                                     const at::Tensor& block_table,
                                     const at::Tensor& cu_seq_lens) {

    rtp_llm::cp_gather_indexer_k_quant_cache(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens);
}

void cp_gather_and_upconvert_fp8_kv_cache(const at::Tensor& src_cache,
                                          at::Tensor&       dst,
                                          const at::Tensor& block_table,
                                          const at::Tensor& seq_lens,
                                          const at::Tensor& workspace_starts,
                                          int64_t           batch_size) {

    rtp_llm::cp_gather_and_upconvert_fp8_kv_cache(src_cache, dst, block_table, seq_lens, workspace_starts, batch_size);
}

void concat_and_cache_mla(at::Tensor&        kv_c,
                          at::Tensor&        k_pe,
                          at::Tensor&        kv_cache,
                          at::Tensor&        slot_mapping,
                          const std::string& kv_cache_dtype,
                          at::Tensor&        scale) {

    rtp_llm::concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale);
}

}  // namespace torch_ext
