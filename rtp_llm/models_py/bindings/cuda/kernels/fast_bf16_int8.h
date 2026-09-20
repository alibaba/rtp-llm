#pragma once

#include <ATen/ATen.h>
#include <cstdint>

namespace rtp_llm {

// Quantize finite BF16 x[M,K] into INT8 q[M,K] and BF16 scales[M,K/group_size] on the current stream.
// group_size is 16, 32, 64 or 128; outputs must not overlap inputs. Requires SM90+.
void fastBf16Int8Quantize(at::Tensor x, at::Tensor q, at::Tensor scales, int64_t group_size);

// Decode q[R,M,K] using finite nonnegative scales[R,M,K/group_size] and reduce in source order to BF16 out[M,K].
// Products and additions round to BF16. Sources are individually contiguous; out must not overlap inputs.
void fastBf16Int8DequantizeReduce(at::Tensor q, at::Tensor scales, at::Tensor out, int64_t group_size);

}  // namespace rtp_llm
