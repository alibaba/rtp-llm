#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/common/CudaGraphPrefillCopy.h"
#include "rtp_llm/cpp/kernels/cuda_graph_copy_kernel.h"
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace std;
using namespace rtp_llm;

namespace torch_ext {

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

    auto device = input_tensor.device();
    CHECK_EQ(output_tensor.device(), device);
    CHECK_EQ(batch_size.device(), device);
    CHECK_EQ(input_lengths.device(), device);
    CHECK_EQ(cu_seq_len.device(), device);

    // Validate tensor dimensions
    CHECK_DIM(1, input_tensor);   // input: (total_elements)
    CHECK_DIM(1, output_tensor);  // output: (max_batch_size * max_seq_len * hidden_size)
    CHECK_DIM(1, batch_size);     // batch_size: (1)
    CHECK_DIM(1, input_lengths);  // input_lengths: (batch_size)
    CHECK_DIM(1, cu_seq_len);     // cu_seq_len: (batch_size + 1)

    // Get tensor data pointers
    auto input_ptr         = input_tensor.data_ptr();
    auto output_ptr        = output_tensor.data_ptr();
    auto batch_size_ptr    = batch_size.data_ptr<int>();
    auto input_lengths_ptr = input_lengths.data_ptr<int>();
    auto cu_seq_len_ptr    = cu_seq_len.data_ptr<int>();

    StreamType stream = GET_CURRENT_STREAM();

    // Dispatch based on tensor dtype
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_tensor.scalar_type(), c_type, [&] {
        rtp_llm::invokeCudaGraphCopySmall2Large<c_type>(static_cast<c_type*>(input_ptr),
                                                        static_cast<c_type*>(output_ptr),
                                                        batch_size_ptr,
                                                        static_cast<int>(max_batch_size),
                                                        static_cast<int>(max_seq_len),
                                                        input_lengths_ptr,
                                                        static_cast<int>(hidden_size),
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

    auto device = input_tensor.device();
    CHECK_EQ(output_tensor.device(), device);
    CHECK_EQ(batch_size.device(), device);
    CHECK_EQ(input_lengths.device(), device);
    CHECK_EQ(cu_seq_len.device(), device);

    // Validate tensor dimensions
    CHECK_DIM(1, input_tensor);   // input: (max_batch_size * max_seq_len * hidden_size)
    CHECK_DIM(1, output_tensor);  // output: (total_elements)
    CHECK_DIM(1, batch_size);     // batch_size: (1)
    CHECK_DIM(1, input_lengths);  // input_lengths: (batch_size)
    CHECK_DIM(1, cu_seq_len);     // cu_seq_len: (batch_size + 1)

    // Get tensor data pointers
    auto input_ptr         = input_tensor.data_ptr();
    auto output_ptr        = output_tensor.data_ptr();
    auto batch_size_ptr    = batch_size.data_ptr<int>();
    auto input_lengths_ptr = input_lengths.data_ptr<int>();
    auto cu_seq_len_ptr    = cu_seq_len.data_ptr<int>();

    StreamType stream = GET_CURRENT_STREAM();

    // Dispatch based on tensor dtype
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_tensor.scalar_type(), c_type, [&] {
        rtp_llm::invokeCudaGraphCopyLarge2Small<c_type>(static_cast<c_type*>(input_ptr),
                                                        static_cast<c_type*>(output_ptr),
                                                        batch_size_ptr,
                                                        static_cast<int>(max_batch_size),
                                                        static_cast<int>(max_seq_len),
                                                        input_lengths_ptr,
                                                        static_cast<int>(hidden_size),
                                                        cu_seq_len_ptr,
                                                        stream);
        return true;
    });
}

}  // namespace torch_ext
