#include <torch/csrc/stable/library.h>

#include "sparse_fp8_decode.h"

STABLE_TORCH_LIBRARY(rtp_fp8_sparse_decode, m) {
    m.def(
        "forward(Tensor q, Tensor kv, Tensor indices, Tensor(a!)? scheduler, Tensor(b!)? splits, float scale, bool enable_batch_invariant=False) -> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");
}

STABLE_TORCH_LIBRARY_IMPL(rtp_fp8_sparse_decode, CUDA, m) {
    m.impl("forward", TORCH_BOX(&rtp_kernel::blackwell::sparse_fp8_decode));
}
