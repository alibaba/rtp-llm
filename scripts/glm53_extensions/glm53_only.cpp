#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY(flash_mla_glm53_22, m) {
 m.def("sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, Tensor(a!)? tile_scheduler_metadata, Tensor(b!)? num_splits, Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, int d_v, float sm_scale, bool enable_batch_invariant=False) -> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");
}
