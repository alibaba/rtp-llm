#pragma once
#include "src/fastertransformer/devices/testing/TestBase.h"
#include <torch/torch.h>

class GemmOpTest: public DeviceTestBase {
public:

    struct GemmOpTestInput {
        torch::Tensor A;
        torch::Tensor B;
    };

    struct GemmOpTestOutput {
        torch::Tensor C;
    };

    // Basic
    GemmOpTestInput PrepareGemmOpInput(size_t m,
                                       size_t n,
                                       size_t k,
                                       DataType type)
    {
        auto dtype = dataTypeToTorchType(type);
        auto A = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(dtype);
        auto B = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(dtype);
        return GemmOpTestInput({A, B});
    }

    // Batch
    GemmOpTestInput PrepareGemmOpInput(size_t b,
                                       size_t m,
                                       size_t n,
                                       size_t k,
                                       DataType type)
    {
        auto dtype = dataTypeToTorchType(type);
        auto A = torch::rand({(int)b, (int)m, (int)k}, torch::Device(torch::kCPU)).to(dtype);
        auto B = torch::rand({(int)b, (int)k, (int)n}, torch::Device(torch::kCPU)).to(dtype);
        return GemmOpTestInput({A, B});
    }

    // Batch + Transpose
    GemmOpTestInput PrepareGemmOpInput(size_t b,
                                       size_t m1,
                                       size_t k1,
                                       size_t k2,
                                       size_t n2,
                                       DataType type)
    {
        auto dtype = dataTypeToTorchType(type);

        auto A = torch::rand({(int)b, (int)m1, (int)k1}, torch::Device(torch::kCPU)).to(dtype);
        auto B = torch::rand({(int)b, (int)k2, (int)n2}, torch::Device(torch::kCPU)).to(dtype);
        return GemmOpTestInput({A, B});
    }

    GemmOpTestOutput BasicGemmOpRun(GemmOpTestInput& input)
    {
        auto A = tensorToBuffer(input.A);
        auto B = tensorToBuffer(input.B);

        GemmParams params {*A, *B};
        auto C = device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*C)});
    }

    GemmOpTestOutput BatchGemmOpRun(GemmOpTestInput& input)
    {
        auto A = tensorToBuffer(input.A);
        auto B = tensorToBuffer(input.B);

        GemmParams params {*A, *B};
        auto C = device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*C)});
    }

    GemmOpTestOutput BatchTransposeGemmOpRun(GemmOpTestInput& input,
                                             TransposeOperation a_op,
                                             TransposeOperation b_op)
    {   
        auto A = tensorToBuffer(input.A);
        auto B = tensorToBuffer(input.B);
        
        GemmParams params {*A, *B, std::nullopt, DataType::TYPE_INVALID, a_op, b_op};
        auto C = device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*C)});
    }

    GemmOpTestOutput MixtureBatchTransposeGemmOpRun(GemmOpTestInput& input,
                                                    TransposeOperation a_op,
                                                    TransposeOperation b_op,
                                                    DataType type)
    {   
        auto A = tensorToBuffer(input.A);
        auto B = tensorToBuffer(input.B);
        
        GemmParams params {*A, *B, std::nullopt, type, a_op, b_op};
        auto C = device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*C)});
    }

    GemmOpTestOutput BasicGemmTorchRefRun(GemmOpTestInput& input)
    {
        return GemmOpTestOutput(
            {torch::matmul(input.A.to(torch::kFloat), input.B.to(torch::kFloat))}
        );
    }

    GemmOpTestOutput BatchGemmTorchRefRun(GemmOpTestInput& input)
    {
        return GemmOpTestOutput(
            {torch::matmul(input.A.to(torch::kFloat), input.B.to(torch::kFloat))}
        );
    }

    GemmOpTestOutput BatchTransposeGemmTorchRefRun(GemmOpTestInput& input,
                                                   TransposeOperation a_op,
                                                   TransposeOperation b_op)
    {   
        auto A = input.A;
        auto B = input.B;
        if (a_op == TransposeOperation::TRANSPOSE) {
            A = A.transpose(1, 2);
        }
        if (b_op == TransposeOperation::TRANSPOSE) {
            B = B.transpose(1, 2);
        }
        return GemmOpTestOutput(
            {torch::matmul(A.to(torch::kFloat), B.to(torch::kFloat))}
        );
    }


    void BasicGemmOpTest(size_t m,
                         size_t n,
                         size_t k,
                         DataType dtype) 
    {
        auto input = PrepareGemmOpInput(m, n, k, dtype);
        auto result = BasicGemmOpRun(input);
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result.C);

    }

    void BatchGemmOpTest(size_t b,
                         size_t m,
                         size_t n,
                         size_t k,
                         DataType dtype)
    {
        auto input = PrepareGemmOpInput(b, m, n, k, dtype);
        auto result = BatchGemmOpRun(input);
        auto result_ref = BatchGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result.C);
    }

    void BatchTransposeGemmOp(TransposeOperation op_a,
                              TransposeOperation op_b,
                              size_t b,
                              size_t m1,
                              size_t k1,
                              size_t k2,
                              size_t n2,
                              DataType dtype)
    {
        auto input = PrepareGemmOpInput(b, m1, k1, k2, n2, dtype);
        auto result = BatchTransposeGemmOpRun(input, op_a, op_b);
        auto result_ref = BatchTransposeGemmOpRun(input, op_a, op_b);
        assertTensorClose(result.C.to(result_ref.C.type()), result.C);

    }

    void MixtureBatchTransposeGemmOp(TransposeOperation op_a,
                                     TransposeOperation op_b,
                                     size_t b,
                                     size_t m1,
                                     size_t k1,
                                     size_t k2,
                                     size_t n2,
                                     DataType dtype,
                                     DataType type)
    {
        auto input = PrepareGemmOpInput(b, m1, k1, k2, n2, dtype);
        auto result = MixtureBatchTransposeGemmOpRun(input, op_a, op_b, type);
        auto result_ref = BatchTransposeGemmOpRun(input, op_a, op_b);
        assertTensorClose(result.C.to(result_ref.C.type()), result.C);

    }

};