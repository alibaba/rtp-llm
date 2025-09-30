#pragma once

#include <torch/torch.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;
using namespace rtp_llm;

class GeneralOpsTest: public DeviceTestBase {
public:
    void testCopyWithSlicing() {
        using TestT = int32_t;

        vector<TestT> input = {1, 2, 3, 4, 5, 6, 7, 8};
        auto          src   = createHostBuffer({4, 2}, input.data());
        auto          dst   = createBuffer<TestT>({2, 2}, {0, 0, 0, 0});

        device_->copy({*dst, src->view(1, 2)});

        assertBufferValueEqual<TestT>(*dst, {3, 4, 5, 6});

        device_->copy({dst->view(1, 1), src->view(3, 1)});
        assertBufferValueEqual<TestT>(*dst, {3, 4, 7, 8});
    }

    void testTranspose() {
        auto                 input    = createBuffer<int32_t>({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        std::vector<int32_t> expected = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
        auto                 output   = device_->transpose({*input});
        EXPECT_EQ(output->shape(), std::vector<size_t>({3, 4}));
        assertBufferValueEqual(*output, expected);
    }

    void testSplit() {
        auto input_t   = torch::rand({255, 2062}, torch::kFloat16);
        auto input_buf = tensorToBuffer(input_t);
        input_buf->updateTypeAndShape(rtp_llm::DataType::TYPE_BYTES, {255, 2062 * 2});
        auto outputs   = device_->split({*input_buf, {2048 * 2, 12 * 2, 2 * 2}, 1}).outputs;
        auto outputs_t = input_t.split_with_sizes({2048, 12, 2}, 1);
        outputs[0]->updateTypeAndShape(rtp_llm::DataType::TYPE_FP16, {255, 2048});
        outputs[1]->updateTypeAndShape(rtp_llm::DataType::TYPE_FP16, {255, 12});
        outputs[2]->updateTypeAndShape(rtp_llm::DataType::TYPE_FP16, {255, 2});
        assertTensorClose(bufferToTensor(*outputs[0]), outputs_t[0].contiguous());
        assertTensorClose(bufferToTensor(*outputs[1]), outputs_t[1].contiguous());
        assertTensorClose(bufferToTensor(*outputs[2]), outputs_t[2].contiguous());
    }

    void testConvert() {
        auto source    = createBuffer<float>({7}, {0, -10, -1234, 1, 100, 10000, 3456});
        auto tensor    = bufferToTensor(*source);
        auto testTypes = {
            DataType::TYPE_FP16, DataType::TYPE_BF16, DataType::TYPE_FP32, DataType::TYPE_INT32, DataType::TYPE_INT64};
        for (auto type1 : testTypes) {
            for (auto type2 : testTypes) {
                cout << "Testing " << type1 << " -> " << type2 << endl;
                auto src    = tensorToBuffer(tensor.to(dataTypeToTorchType(type1)));
                auto output = device_->convert({src, type2});
                assertTensorClose(tensor.to(torch::kFloat32), bufferToTensor(*output).to(torch::kFloat32), 1, 1e-3);
                device_->syncAndCheck();
            }
        }
    }

    void testQBufferCopy() {
        auto tensor     = torch::ones({5}, torch::kInt8);
        auto scales     = torch::ones({5}, torch::kFloat);
        auto zeros      = torch::ones({5}, torch::kFloat);
        auto src        = torchTensor2Buffer(tensor, scales, zeros);
        auto result_src = QBuffer2torchTensor(static_pointer_cast<const QBuffer>(src));
        EXPECT_TRUE(torch::equal(result_src[0], tensor));
        EXPECT_TRUE(torch::equal(result_src[1], scales));
        EXPECT_TRUE(torch::equal(result_src[2], zeros));
        auto dst_tensor = torch::zeros({5}, torch::kInt8);
        auto dst_scales = torch::zeros({5}, torch::kFloat);
        auto dst_zeros  = torch::zeros({5}, torch::kFloat);
        auto dst        = torchTensor2Buffer(dst_tensor, dst_scales, dst_zeros);
        auto result_dst = QBuffer2torchTensor(static_pointer_cast<const QBuffer>(dst));
        EXPECT_TRUE(torch::equal(result_dst[0], dst_tensor));
        EXPECT_TRUE(torch::equal(result_dst[1], dst_scales));
        EXPECT_TRUE(torch::equal(result_dst[2], dst_zeros));
        device_->copy({*dst, *src});
        result_dst = QBuffer2torchTensor(static_pointer_cast<const QBuffer>(dst));
        EXPECT_TRUE(torch::equal(result_dst[0], tensor));
        EXPECT_TRUE(torch::equal(result_dst[1], scales));
        EXPECT_TRUE(torch::equal(result_dst[2], zeros));
    }

    void testConcat() {
        auto src1   = createBuffer<float>({4, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        auto src2   = createBuffer<float>({1, 3}, {111, 222, 333});
        auto src3   = createBuffer<float>({2, 3},
                                          {
                                            1000,
                                            1001,
                                            1002,
                                            1003,
                                            1004,
                                            1005,
                                        });
        auto result = device_->concat({{src1, src2, src3}});
        device_->syncAndCheck();
        auto expected = torch::tensor(
            {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}, {111, 222, 333}, {1000, 1001, 1002}, {1003, 1004, 1005}},
            torch::kFloat32);
        assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);
    }

    void testSelect() {
        auto src   = createBuffer<float>({6, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
        auto index = createBuffer<int32_t>({3}, {0, 2, 3});

        auto result   = device_->select({*src, *index});
        auto expected = torch::tensor({{0, 1, 2, 3, 4}, {10, 11, 12, 13, 14}, {15, 16, 17, 18, 19}}, torch::kFloat32);
        assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);

        auto src2    = device_->clone({*src, AllocationType::HOST});
        auto index2  = device_->clone({*index, AllocationType::HOST});
        auto result2 = device_->select({*src2, *index2});
        assertTensorClose(bufferToTensor(*result2), expected, 1e-6, 1e-6);
    }

    void testSelect1d() {
        auto src   = createBuffer<float>({2, 6}, {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15});
        auto index = createBuffer<int32_t>({3}, {0, 4, 5}, AllocationType::HOST);

        auto result   = device_->select({*src, *index, 1});
        auto expected = torch::tensor({{0, 4, 5}, {10, 14, 15}}, torch::kFloat32);
        assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);

        src      = createBuffer<float>({2, 5, 3}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
        index    = createBuffer<int32_t>({4}, {0, 1, 3, 4}, AllocationType::HOST);
        result   = device_->select({*src, *index, 1});
        expected = torch::tensor(
            {{0, 1, 2}, {3, 4, 5}, {9, 10, 11}, {12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {24, 25, 26}, {27, 28, 29}},
            torch::kFloat32);
    }

    void testEmbeddingLookup() {
        const auto vocab_size  = 102400;
        const auto hidden_size = 1024;
        const auto seq_len     = 4;

        auto ids_vec       = vector<int32_t>{100, 20000, 2010, 1024};
        auto ids           = createBuffer<int32_t>({seq_len}, ids_vec);
        auto table_tensor  = torch::rand({vocab_size, hidden_size}, torch::Device(torch::kCPU)).to(torch::kHalf);
        auto table         = createDeviceBuffer<half>(table_tensor);
        auto output        = device_->embeddingLookup({*ids, *table});
        auto output_tensor = bufferToTensor(*output);

        auto ids_tensor      = bufferToTensor(*ids);
        auto expected_values = table_tensor.index_select(0, ids_tensor);

        ASSERT_TRUE(torch::allclose(expected_values, output_tensor, 1e-03, 1e-03));
    }

    void testMultiply() {
        const auto m = 16;
        auto       n = 8;

        auto A = torch::rand({m}, torch::kFloat16);
        auto B = torch::rand({m}, torch::kFloat16);

        auto A_buf = tensorToBuffer(A);
        auto B_buf = tensorToBuffer(B);

        auto ref    = A.to(torch::kFloat32) * (B.to(torch::kFloat32));
        auto result = device_->multiply({*A_buf, *B_buf});

        auto result_tensor = bufferToTensor(*result);
        assertTensorClose(result_tensor, ref);

        B     = torch::rand({m, n}, torch::kFloat16);
        B_buf = tensorToBuffer(B);

        ref    = A.to(torch::kFloat32) * (B.to(torch::kFloat32).t());
        result = device_->multiply({*A_buf, *B_buf});

        result_tensor = bufferToTensor(*result);
        assertTensorClose(result_tensor, ref.t());
    }

    void testLoss() {
        vector<float>   logits_v = {1, 2, 3, 4, 5, 6, 7, 8};
        vector<int32_t> labels_v = {2, 3};
        auto            logits   = createBuffer<float>({2, 4}, logits_v);
        auto            labels   = createBuffer<int32_t>({2}, labels_v);
        auto            res      = device_->loss({*logits, *labels});
        auto            expected = torch::tensor({1.4402, 0.4402}, torch::kFloat32);
        assertTensorClose(bufferToTensor(*res), expected);
    }

    void testSigmoid() {
        {
            vector<float> gate_v   = {1, 2, 3, 4, 5, 6, 7, 8};
            vector<float> expect_v = {0.7311, 0.8808, 0.9526, 0.9820, 0.9933, 0.9975, 0.9991, 0.9997};
            for (size_t i = 1; i <= gate_v.size(); ++i) {
                vector<float> sub_gate(gate_v.begin(), gate_v.begin() + i);
                vector<float> sub_expect(expect_v.begin(), expect_v.begin() + i);
                auto          gate     = createBuffer<float>({i}, sub_gate);
                auto          res      = device_->activation({ActivationType::Sigmoid, gate});
                auto          expected = torch::tensor(sub_expect, torch::kFloat32);
                assertTensorClose(bufferToTensor(*res), expected);
            }
        }
        {
            vector<half>  gate_v         = {__float2half(1.0f),
                                            __float2half(2.0f),
                                            __float2half(3.0f),
                                            __float2half(4.0f),
                                            __float2half(5.0f),
                                            __float2half(6.0f),
                                            __float2half(7.0f),
                                            __float2half(8.0f)};
            vector<float> expect_v_float = {0.7310, 0.8809, 0.9526, 0.9819, 0.9932, 0.9976, 0.9990, 0.9995};

            for (size_t i = 1; i <= gate_v.size(); ++i) {
                vector<half>  sub_gate(gate_v.begin(), gate_v.begin() + i);
                vector<float> sub_expect_float(expect_v_float.begin(), expect_v_float.begin() + i);

                auto gate = createBuffer<half>({i}, sub_gate);

                auto res = device_->activation({ActivationType::Sigmoid, gate});

                auto expected = torch::tensor(sub_expect_float, torch::kFloat32).to(torch::kFloat16);  // 显式转换为FP16

                assertTensorClose(bufferToTensor(*res), expected);
            }
        }
#ifdef ENABLE_BF16
        {
            // BF16测试用例修改
            vector<__nv_bfloat16> gate_v         = {__float2bfloat16(1.0f),
                                                    __float2bfloat16(2.0f),
                                                    __float2bfloat16(3.0f),
                                                    __float2bfloat16(4.0f),
                                                    __float2bfloat16(5.0f),
                                                    __float2bfloat16(6.0f),
                                                    __float2bfloat16(7.0f),
                                                    __float2bfloat16(8.0f)};
            vector<float>         expect_v_float = {0.7305, 0.8789, 0.9531, 0.9805, 0.9922, 0.9961, 1.0000, 1.0000};

            for (size_t i = 1; i <= gate_v.size(); ++i) {
                vector<__nv_bfloat16> sub_gate(gate_v.begin(), gate_v.begin() + i);
                vector<float>         sub_expect_float(expect_v_float.begin(), expect_v_float.begin() + i);

                auto gate = createBuffer<__nv_bfloat16>({i}, sub_gate);

                auto res = device_->activation({ActivationType::Sigmoid, gate});

                auto expected =
                    torch::tensor(sub_expect_float, torch::kFloat32).to(torch::kBFloat16);  // 显式转换为BF16

                assertTensorClose(bufferToTensor(*res), expected);
            }
        }
#endif
    }
};
