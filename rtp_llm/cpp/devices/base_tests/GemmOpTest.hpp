#pragma once
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <torch/torch.h>

using namespace rtp_llm;

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
        if (b == 1) {
            auto A = torch::rand({(int)m1, (int)k1}, torch::Device(torch::kCPU)).to(dtype);
            auto B = torch::rand({(int)k2, (int)n2}, torch::Device(torch::kCPU)).to(dtype);
            return GemmOpTestInput({A, B});
        } else {
            auto A = torch::rand({(int)b, (int)m1, (int)k1}, torch::Device(torch::kCPU)).to(dtype);
            auto B = torch::rand({(int)b, (int)k2, (int)n2}, torch::Device(torch::kCPU)).to(dtype);
            return GemmOpTestInput({A, B});
        }
    }



    GemmOpTestOutput BasicGemmOpRun(GemmOpTestInput& input)
    {
        auto A = tensorToBuffer(input.A);
        auto B = tensorToBuffer(input.B);
        auto D = device_->allocateBuffer({A->type(), {A->shape()[0], B->shape()[1]}});
        GemmParams params {*A, *B, std::nullopt, D};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
    }
    
    GemmOpTestOutput BasicFP8GemmOpRun(GemmOpTestInput& input, float scaleA = 1.0f, float scaleB = 1.0f, DataType dDtype = DataType::TYPE_FP32)
    {
        auto A = tensorToBuffer(input.A);
        auto B = tensorToBuffer(input.B);
        auto D = device_->allocateBuffer({dDtype, {A->shape()[0], B->shape()[1]}});
        float alpha = scaleA * scaleB;
        GemmParams params {*A, *B, std::nullopt, D, DataType::TYPE_FP32,  TransposeOperation::NONE,  TransposeOperation::NONE, ActivationType::Identity, alpha, 0.0f};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
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

        GemmParams params {*A, *B, std::nullopt, nullptr, DataType::TYPE_INVALID, a_op, b_op};
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

        GemmParams params {*A, *B, std::nullopt, nullptr, type, a_op, b_op};
        auto C = device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*C)});
    }

    GemmOpTestOutput BasicGemmTorchRefRun(GemmOpTestInput& input, float alpha = 1.0f)
    {
        auto A = input.A;
        auto B = input.B;
        return GemmOpTestOutput(
            {torch::mul(torch::matmul(A.to(torch::kFloat), B.to(torch::kFloat)), alpha)}
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
            A = A.transpose(A.dim()-2, A.dim()-1);
        }
        if (b_op == TransposeOperation::TRANSPOSE) {
            B = B.transpose(B.dim()-2, B.dim()-1);
        }
        return GemmOpTestOutput(
            {torch::matmul(A.to(torch::kFloat), B.to(torch::kFloat))}
        );
    }


    void BasicFP8GemmOpTest(size_t m,
                            size_t n,
                            size_t k,
                            DataType dDtype)
    {
        auto input = PrepareGemmOpInput(m, n, k, DataType::TYPE_FP16);

        // quant
        const float FP8_E4M3_FNUZ_MAX = 240.0f; 
        const float min_scaling_factor = 1.0f / (FP8_E4M3_FNUZ_MAX * 512.0f);

        float maxA = input.A.abs().max().item<float>();
        float scaling_factorA = maxA / FP8_E4M3_FNUZ_MAX;
        scaling_factorA = std::max(min_scaling_factor, scaling_factorA);
        auto qA = torch::div(input.A, scaling_factorA).to(dataTypeToTorchType(DataType::TYPE_FP8_E4M3));
        
        float maxB = input.B.abs().max().item<float>();
        float scaling_factorB = maxB / FP8_E4M3_FNUZ_MAX;
        scaling_factorB = std::max(min_scaling_factor, scaling_factorB);
        auto qB = torch::div(input.B, scaling_factorB).to(dataTypeToTorchType(DataType::TYPE_FP8_E4M3));

        GemmOpTestInput qinput({qA, qB});

        auto result = BasicFP8GemmOpRun(qinput, scaling_factorA, scaling_factorB, dDtype);
        auto result_ref = BasicGemmTorchRefRun(qinput, scaling_factorA * scaling_factorB);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C);
    }

    void BasicGemmOpTest(size_t m,
                         size_t n,
                         size_t k,
                         DataType dtype)
    {
        auto input = PrepareGemmOpInput(m, n, k, dtype);
        auto result = BasicGemmOpRun(input);
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C);

    }

    void TransposeGemmOpTest(TransposeOperation op_a,
                             TransposeOperation op_b,
                             size_t m1,
                             size_t k1,
                             size_t k2,
                             size_t n2,
                             DataType dtype)
    {
        auto input = PrepareGemmOpInput(1, m1, k1, k2, n2, dtype);
        auto result = BatchTransposeGemmOpRun(input, op_a, op_b);
        auto result_ref = BatchTransposeGemmTorchRefRun(input, op_a, op_b);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C);

    }

    void BatchGemmOpTest(size_t b,
                         size_t m,
                         size_t n,
                         size_t k,
                         DataType dtype,
                         double rtol = 1e-3,
                         double atol = 1e-3)
    {
        auto input = PrepareGemmOpInput(b, m, n, k, dtype);
        auto result = BatchGemmOpRun(input);
        auto result_ref = BatchGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, rtol, atol);
    }


    void BatchTransposeGemmOp(TransposeOperation op_a,
                              TransposeOperation op_b,
                              size_t b,
                              size_t m1,
                              size_t k1,
                              size_t k2,
                              size_t n2,
                              DataType dtype,
                              double rtol = 1e-3,
                              double atol = 1e-3)
    {
        auto input = PrepareGemmOpInput(b, m1, k1, k2, n2, dtype);
        auto result = BatchTransposeGemmOpRun(input, op_a, op_b);
        auto result_ref = BatchTransposeGemmTorchRefRun(input, op_a, op_b);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, rtol, atol);

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
        auto result_ref = BatchTransposeGemmTorchRefRun(input, op_a, op_b);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C);

    }


    GemmOpTestOutput BasicCUDAQGemmOpRun(GemmOpTestInput& input) {
        auto A = tensorToBuffer(input.A);
        auto host_B = tensorToBuffer(input.B, AllocationType::HOST);
        auto QB = device_->quantize({*host_B,
                                    DataType::TYPE_QINT8,
                                    1,
                                    QScheme::Qint8WeightOnly});
        auto device_B  = device_->clone({*QB});
        auto D = device_->allocateBuffer({A->type(), {A->shape()[0], device_B->shape()[1]}});
        GemmParams params {*A, *device_B, std::nullopt, D};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
    }

    GemmOpTestOutput qInt8xQInt82DGemmOpRun(GemmOpTestInput& input) {
        auto A = tensorToBuffer(input.A);
        auto B = tensorToBuffer(input.B);
        auto QB = device_->quantize({*B,
                                    DataType::TYPE_QINT8,
                                    1});
        auto QA = device_->quantize({*A,
                                    DataType::TYPE_QINT8,
                                    1});
        auto D = device_->allocateBuffer({DataType::TYPE_FP16, {QA->shape()[0], QB->shape()[1]}});
        GemmParams params {*QA, *QB, std::nullopt, D};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
    }

    void BasicQGemmOpTest(size_t m,
                          size_t n,
                          size_t k,
                          DataType dtype)
    {
        auto input = PrepareGemmOpInput(m, n, k, dtype);
        auto result = BasicCUDAQGemmOpRun(input);
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, 1e-2, 1e-2);
    }

    void qInt8QInt82DGemmOpTest(size_t m,
                                size_t n,
                                size_t k)
    {
        auto input = PrepareGemmOpInput(m, n, k, DataType::TYPE_FP32);
        auto result = qInt8xQInt82DGemmOpRun(input);
        auto result_ref =GemmOpTestOutput(
            {torch::matmul(input.A.to(torch::kFloat), input.B.t().to(torch::kFloat))}
        );
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, 1e-2, 1e-2);
    }

};