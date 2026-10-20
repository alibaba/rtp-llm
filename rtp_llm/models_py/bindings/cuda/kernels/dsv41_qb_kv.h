#pragma once
#include <torch/all.h>

namespace torch_ext {
bool dsv41_qb_kv_available();
// variant: 0=swap BM128, 1=swap BM192, 2=non-swap BN224,
// 3=non-swap BN256 (one accumulator), 4=swap BM256.
void dsv41_qb_out(torch::Tensor x, torch::Tensor xs, torch::Tensor w,
                  torch::Tensor ws, torch::Tensor out, int64_t variant,
                  int64_t clusters, bool wait_pdl);
void dsv41_qb_kv_out(torch::Tensor x, torch::Tensor xs, torch::Tensor w,
                     torch::Tensor ws, torch::Tensor out, torch::Tensor kv,
                     torch::Tensor gamma, torch::Tensor positions,
                     torch::Tensor frequencies, torch::Tensor pool,
                     torch::Tensor slots, torch::Tensor kv_out, double eps,
                     int64_t variant, int64_t clusters, bool wait_pdl,
                     bool trap_invalid);
void dsv41_qb_kv_index_out(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
                     torch::Tensor out, torch::Tensor kv, torch::Tensor gamma,
                     torch::Tensor positions, torch::Tensor frequencies, torch::Tensor pool,
                     torch::Tensor slots, torch::Tensor kv_out, double eps, int64_t variant,
                     int64_t clusters, bool wait_pdl, bool trap_invalid,
                     torch::Tensor req, torch::Tensor table, torch::Tensor local,
                     torch::Tensor global, int64_t logical_entries, int64_t index_entries,
                     bool trigger_pdl);
}  // namespace torch_ext
