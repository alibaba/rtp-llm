#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/IndexerKQuantOp.h"
#include "rtp_llm/cpp/kernels/indexer_k_quant_kernel.h"

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

}  // namespace torch_ext
