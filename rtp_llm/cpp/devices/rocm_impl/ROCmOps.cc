#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"

namespace rtp_llm {

MultiplyOutput ROCmDevice::multiply(const MultiplyParams& params) {
    const auto& A = params.A;
    const auto& B = params.B;

    int m, n;
    if (A.shape() == B.shape()) {
        m = A.size();
        n = 1;
    } else if (A.size() == 1) {
        m = 1;
        n = B.size();
    } else if (A.shape().size() == 1 && B.shape()[0] == A.shape()[0]) {
        m = A.shape()[0];
        n = B.size() / m;
    } else {
        RUNTIME_ASSERT_OP_ARG(
            false, "multiply can not be applied to A[%s] and B[%s]", A.debugString().c_str(), B.debugString().c_str());
    }

    RUNTIME_ASSERT_OP_ARG(A.type() == B.type(), "A and B must have same type, but got %d vs %d", A.type(), B.type());
    const auto data_type = A.type();

    BufferPtr output;
    if (params.output) {
        output = params.output;
        RUNTIME_ASSERT_OP_ARG(output->type() == data_type,
                              "Output type must be same as A and B, but got %d vs %d",
                              output->type(),
                              data_type);
        RUNTIME_ASSERT_OP_ARG(output->shape()[0] == m, "Output 0-d size must be %d, but got %d", m, output->shape()[0]);
        RUNTIME_ASSERT_OP_ARG(
            output->size() == B.size(), "Output size must be %d, but got %d", B.size(), output->size());
    } else {
        output = allocateBuffer({data_type, B.shape()});
    }

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeScaledDot, output->data(), B.data(), A.data(), m, n, stream_);

    printBufferData(*output, "multiply_output");

    return move(output);
}

SliceOutput ROCmDevice::slice(const SliceParams& params) {
    const auto& input        = params.input;
    const auto& starts       = params.start;
    const auto& step         = params.step;
    auto        input_t      = Buffer2torchTensor(params.input, false);
    auto        sliceTensor  = input_t.slice(params.dim, starts, params.end, step);
    auto        buffer_shape = torchShapeToBufferShape(sliceTensor.sizes());
    auto        out          = allocateBuffer({input.type(), buffer_shape});
    auto        out_t        = Buffer2torchTensor(out, false);
    out_t.copy_(sliceTensor, false);
    return out;
}

}  // namespace rtp_llm
