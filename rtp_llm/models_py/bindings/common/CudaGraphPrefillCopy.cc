#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/common/CudaGraphPrefillCopy.h"
#include "rtp_llm/models_py/bindings/core/Dispatch.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/common/kernels/cuda_graph_copy_kernel.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
using namespace std;
using namespace rtp_llm;
namespace torch_ext {

namespace {

int64_t checkedMul(int64_t lhs, int64_t rhs, const char* expression) {
    TORCH_CHECK(lhs >= 0 && rhs >= 0, expression, " requires non-negative dimensions, got ", lhs, " and ", rhs);
    TORCH_CHECK(rhs == 0 || lhs <= std::numeric_limits<int64_t>::max() / rhs,
                expression,
                " overflows int64: ",
                lhs,
                " * ",
                rhs);
    return lhs * rhs;
}

void validateCudaGraphCopyArguments(const at::Tensor& input_tensor,
                                    const at::Tensor& output_tensor,
                                    const at::Tensor& batch_size,
                                    int64_t           max_batch_size,
                                    int64_t           max_seq_len,
                                    const at::Tensor& input_lengths,
                                    int64_t           hidden_size,
                                    const at::Tensor& cu_seq_len,
                                    bool              small_to_large) {
    TORCH_CHECK(max_batch_size > 0, "max_batch_size must be positive, got ", max_batch_size);
    TORCH_CHECK(max_seq_len > 0, "max_seq_len must be positive, got ", max_seq_len);
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive, got ", hidden_size);
    TORCH_CHECK(max_batch_size < std::numeric_limits<int64_t>::max(),
                "max_batch_size is too large to represent its cumulative-length metadata: ",
                max_batch_size);

    const int64_t aligned_rows     = checkedMul(max_batch_size, max_seq_len, "max_batch_size * max_seq_len");
    const auto&   compact_tensor   = small_to_large ? input_tensor : output_tensor;
    const auto&   aligned_tensor   = small_to_large ? output_tensor : input_tensor;
    const int64_t compact_elements = checkedMul(compact_tensor.size(0), hidden_size, "compact rows * hidden_size");
    const int64_t aligned_elements = checkedMul(aligned_rows, hidden_size, "aligned rows * hidden_size");

    TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type(),
                "input and output dtype must match, got ",
                input_tensor.scalar_type(),
                " and ",
                output_tensor.scalar_type());
    TORCH_CHECK(input_tensor.device() == output_tensor.device() && input_tensor.device() == batch_size.device()
                    && input_tensor.device() == input_lengths.device() && input_tensor.device() == cu_seq_len.device(),
                "all CUDA graph copy tensors must be on the same device");
    TORCH_CHECK(batch_size.numel() == 1, "batch_size must contain exactly one element, got ", batch_size.numel());
    TORCH_CHECK(input_lengths.numel() >= max_batch_size,
                "input_lengths capacity must be at least max_batch_size, got ",
                input_lengths.numel(),
                " < ",
                max_batch_size);
    TORCH_CHECK(cu_seq_len.numel() >= max_batch_size + 1,
                "cu_seq_len capacity must be at least max_batch_size + 1, got ",
                cu_seq_len.numel(),
                " < ",
                max_batch_size + 1);
    TORCH_CHECK(input_tensor.size(1) == hidden_size && output_tensor.size(1) == hidden_size,
                "hidden_size must match both tensor row widths, got hidden_size=",
                hidden_size,
                ", input width=",
                input_tensor.size(1),
                ", output width=",
                output_tensor.size(1));
    TORCH_CHECK(aligned_tensor.size(0) == aligned_rows,
                "aligned tensor row count must equal max_batch_size * max_seq_len, got ",
                aligned_tensor.size(0),
                " != ",
                aligned_rows);
    TORCH_CHECK(compact_tensor.size(0) <= aligned_rows,
                "compact tensor row count exceeds aligned capacity: ",
                compact_tensor.size(0),
                " > ",
                aligned_rows);
    TORCH_CHECK(compact_tensor.numel() == compact_elements,
                "compact tensor element count is inconsistent with its shape");
    TORCH_CHECK(aligned_tensor.numel() == aligned_elements,
                "aligned tensor element count is inconsistent with its shape");
}

}  // namespace

void cuda_graph_copy_small2large(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len) {
    CHECK_INPUT(input_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_INPUT(batch_size);
    CHECK_INPUT(input_lengths);
    CHECK_INPUT(cu_seq_len);
    CHECK_DIM(2, input_tensor);   // input: (total_elements, hidden_size)
    CHECK_DIM(2, output_tensor);  // output: (max_batch_size * max_seq_len, hidden_size)
    CHECK_DIM(1, batch_size);     // batch_size: (1), fixed-address device memory
    CHECK_DIM(1, input_lengths);  // input_lengths: (batch_size)
    CHECK_DIM(1, cu_seq_len);     // cu_seq_len: (batch_size + 1)
    TORCH_CHECK(batch_size.scalar_type() == at::kInt, "batch_size must be int32");
    TORCH_CHECK(input_lengths.scalar_type() == at::kInt, "input_lengths must be int32");
    TORCH_CHECK(cu_seq_len.scalar_type() == at::kInt, "cu_seq_len must be int32");
    validateCudaGraphCopyArguments(input_tensor,
                                   output_tensor,
                                   batch_size,
                                   max_batch_size,
                                   max_seq_len,
                                   input_lengths,
                                   hidden_size,
                                   cu_seq_len,
                                   true);
    auto       input_ptr         = input_tensor.data_ptr();
    auto       output_ptr        = output_tensor.data_ptr();
    auto       batch_size_ptr    = batch_size.data_ptr<int>();
    auto       input_lengths_ptr = input_lengths.data_ptr<int>();
    auto       cu_seq_len_ptr    = cu_seq_len.data_ptr<int>();
    StreamType stream            = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_tensor.scalar_type(), c_type, [&] {
        rtp_llm::invokeCudaGraphCopySmall2Large<c_type>(static_cast<c_type*>(input_ptr),
                                                        static_cast<c_type*>(output_ptr),
                                                        batch_size_ptr,
                                                        max_batch_size,
                                                        max_seq_len,
                                                        input_lengths_ptr,
                                                        hidden_size,
                                                        cu_seq_len_ptr,
                                                        stream);
        return true;
    });
}

void cuda_graph_copy_large2small(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len) {
    CHECK_INPUT(input_tensor);
    CHECK_INPUT(output_tensor);
    CHECK_INPUT(batch_size);
    CHECK_INPUT(input_lengths);
    CHECK_INPUT(cu_seq_len);
    CHECK_DIM(2, input_tensor);   // input: (max_batch_size * max_seq_len, hidden_size)
    CHECK_DIM(2, output_tensor);  // output: (total_elements, hidden_size)
    CHECK_DIM(1, batch_size);     // batch_size: (1)
    CHECK_DIM(1, input_lengths);  // input_lengths: (batch_size)
    CHECK_DIM(1, cu_seq_len);     // cu_seq_len: (batch_size + 1)
    TORCH_CHECK(batch_size.scalar_type() == at::kInt, "batch_size must be int32");
    TORCH_CHECK(input_lengths.scalar_type() == at::kInt, "input_lengths must be int32");
    TORCH_CHECK(cu_seq_len.scalar_type() == at::kInt, "cu_seq_len must be int32");
    validateCudaGraphCopyArguments(input_tensor,
                                   output_tensor,
                                   batch_size,
                                   max_batch_size,
                                   max_seq_len,
                                   input_lengths,
                                   hidden_size,
                                   cu_seq_len,
                                   false);
    auto       input_ptr         = input_tensor.data_ptr();
    auto       output_ptr        = output_tensor.data_ptr();
    auto       batch_size_ptr    = batch_size.data_ptr<int>();
    auto       input_lengths_ptr = input_lengths.data_ptr<int>();
    auto       cu_seq_len_ptr    = cu_seq_len.data_ptr<int>();
    StreamType stream            = GET_CURRENT_STREAM();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_tensor.scalar_type(), c_type, [&] {
        rtp_llm::invokeCudaGraphCopyLarge2Small<c_type>(static_cast<c_type*>(input_ptr),
                                                        static_cast<c_type*>(output_ptr),
                                                        batch_size_ptr,
                                                        max_batch_size,
                                                        max_seq_len,
                                                        input_lengths_ptr,
                                                        hidden_size,
                                                        cu_seq_len_ptr,
                                                        stream);
        return true;
    });
}

}  // namespace torch_ext
