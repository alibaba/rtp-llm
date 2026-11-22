#pragma once

#include <torch/all.h>

namespace torch_ext {

// Host descriptors are packed as [tile_end, first_request, request_end,
// row_start, rows, key_span, key_width, tile_start]. Output contains three
// int32 planes per group: key starts, key ends and request-local visibility.
void dsv41_grouped_score_bounds(const torch::Tensor& descriptors,
                                const torch::Tensor& positions,
                                const torch::Tensor& requests,
                                const torch::Tensor& key_counts,
                                torch::Tensor&       output,
                                int64_t              ratio);

}  // namespace torch_ext
