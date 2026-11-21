// Coefficient generation follows Pillow 11.2.1 src/libImaging/Resample.c.
// This is a C++ adaptation of RTP-LLM's Python reference.
// Pillow 11.2.1 LICENSE (MIT-CMU), retained for these derived portions:
//
// The Python Imaging Library (PIL) is
//
//     Copyright © 1997-2011 by Secret Labs AB
//     Copyright © 1995-2011 by Fredrik Lundh and contributors
//
// Pillow is the friendly PIL fork. It is
//
//     Copyright © 2010 by Jeffrey A. Clark and contributors
//
// Like PIL, Pillow is licensed under the open source MIT-CMU License:
//
// By obtaining, using, and/or copying this software and/or its associated
// documentation, you agree that you have read, understood, and will comply
// with the following terms and conditions:
//
// Permission to use, copy, modify and distribute this software and its
// documentation for any purpose and without fee is hereby granted,
// provided that the above copyright notice appears in all copies, and that
// both that copyright notice and this permission notice appear in supporting
// documentation, and that the name of Secret Labs AB or the author not be
// used in advertising or publicity pertaining to distribution of the software
// without specific, written prior permission.
//
// SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
// SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
// IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
// INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
// LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
// OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// PERFORMANCE OF THIS SOFTWARE.

#include "rtp_llm/cpp/pybind/VitResizeCoefficients.h"

#include <ATen/ops/empty.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace rtp_llm {
namespace {

constexpr int64_t max_dimension    = 65536;
constexpr int64_t max_taps         = 1025;
constexpr int64_t max_coefficients = 1 << 20;
constexpr int64_t precision_bits   = 22;

// Same scalar evaluation order as the Pillow-derived CPU reference.
double cubic(double distance) {
    const double x = std::abs(distance);
    if (x < 1.0) {
        return (1.5 * x - 2.5) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5;
    }
    return 0.0;
}

}  // namespace

std::pair<at::Tensor, at::Tensor> vitResizeCoefficients(int64_t input, int64_t output) {
    TORCH_CHECK(input > 0 && output > 0 && input <= max_dimension && output <= max_dimension,
                "unsupported V4.1 resize axis");
    const double scale            = static_cast<double>(input) / output;
    const double filter_scale     = std::max(scale, 1.0);
    const double support          = 2.0 * filter_scale;
    const double inv_filter_scale = 1.0 / filter_scale;
    const auto   taps             = static_cast<int64_t>(std::ceil(support)) * 2 + 1;
    TORCH_CHECK(taps <= max_taps && output * taps <= max_coefficients, "unsupported V4.1 resize coefficients");

    const auto          options      = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    auto                bounds       = at::empty({output, 2}, options);
    auto                coefficients = at::empty({output, taps}, options);
    auto*               bound        = bounds.data_ptr<int32_t>();
    auto*               coeff        = coefficients.data_ptr<int32_t>();
    std::vector<double> weights(taps);
    for (int64_t dest = 0; dest < output; ++dest) {
        const double center = (dest + 0.5) * scale;
        const auto   first  = std::max<int64_t>(static_cast<int64_t>(center - support + 0.5), 0);
        const auto   last   = std::min(static_cast<int64_t>(center + support + 0.5), input);
        const auto   count  = last - first;
        double       total  = 0;
        for (int64_t source = first; source < last; ++source) {
            const double value      = cubic((source - center + 0.5) * inv_filter_scale);
            weights[source - first] = value;
            total += value;
        }
        bound[dest * 2]     = first;
        bound[dest * 2 + 1] = count;
        auto* row           = coeff + dest * taps;
        for (int64_t index = 0; index < count; ++index) {
            const double value = total != 0.0 ? weights[index] / total : weights[index];
            const double fixed = value * (1 << precision_bits);
            row[index]         = static_cast<int32_t>(fixed + (value < 0.0 ? -0.5 : 0.5));
        }
        std::fill(row + count, row + taps, 0);
    }
    return {bounds, coefficients};
}

}  // namespace rtp_llm
