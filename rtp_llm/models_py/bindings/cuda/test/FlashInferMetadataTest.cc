#include <gtest/gtest.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"

namespace rtp_llm {
namespace {

class FlashInferMetadataTest: public DeviceTestBase {
protected:
    AttentionConfigs config() {
        AttentionConfigs result{};
        result.head_num                = 14;
        result.kv_head_num             = 2;
        result.size_per_head           = 64;
        result.tokens_per_block        = 128;
        result.kernel_tokens_per_block = 128;
        result.is_causal               = true;
        result.rope_config.style       = RopeStyle::No;
        return result;
    }

    torch::Tensor lengths(const std::vector<int>& values, bool device) {
        auto base = torch::tensor(values, torch::kInt32);
        if (device) {
            base = base.cuda();
        }
        // The planner must also normalize noncontiguous host/device views.
        return torch::stack({base, torch::full_like(base, -99)}, 1).select(1, 0);
    }

    torch_ext::PyAttentionInputs
    prefillInputs(const std::vector<int>& input_lengths, const std::vector<int>& prefix_lengths, bool device) {
        torch_ext::PyAttentionInputs input;
        input.is_prefill       = true;
        input.dtype            = torch::scalarTypeToTypeMeta(torch::kBFloat16);
        input.input_lengths    = lengths(input_lengths, device);
        input.prefix_lengths   = lengths(prefix_lengths, device);
        input.sequence_lengths = torch::empty({0}, torch::kInt32);
        if (device) {
            input.sequence_lengths = input.sequence_lengths.cuda();
        }
        return input;
    }

    torch::Tensor randomQkv(int tokens, const AttentionConfigs& cfg) {
        return torch::randn({tokens,
                             static_cast<int64_t>(cfg.head_num + 2 * cfg.kv_head_num),
                             static_cast<int64_t>(cfg.size_per_head)},
                            torch::TensorOptions(torch::kBFloat16).device(torch::kCUDA))
            .mul(0.25);
    }

    torch::Tensor denseOracle(const torch::Tensor& full_qkv, int prefix, const AttentionConfigs& cfg) {
        const auto q = full_qkv.slice(0, prefix).narrow(1, 0, cfg.head_num).to(torch::kFloat32).transpose(0, 1);
        const auto k = torch::repeat_interleave(
                           full_qkv.narrow(1, cfg.head_num, cfg.kv_head_num), cfg.head_num / cfg.kv_head_num, 1)
                           .to(torch::kFloat32)
                           .transpose(0, 1);
        const auto v = torch::repeat_interleave(full_qkv.narrow(1, cfg.head_num + cfg.kv_head_num, cfg.kv_head_num),
                                                cfg.head_num / cfg.kv_head_num,
                                                1)
                           .to(torch::kFloat32)
                           .transpose(0, 1);
        auto       scores        = torch::bmm(q, k.transpose(1, 2)) / std::sqrt(static_cast<double>(cfg.size_per_head));
        const auto key_positions = torch::arange(full_qkv.size(0), scores.options()).unsqueeze(0);
        const auto query_positions = torch::arange(prefix, full_qkv.size(0), scores.options()).unsqueeze(1);
        scores.masked_fill_(key_positions.gt(query_positions).unsqueeze(0), -std::numeric_limits<float>::infinity());
        return torch::bmm(torch::softmax(scores, -1), v)
            .transpose(0, 1)
            .contiguous()
            .reshape({full_qkv.size(0) - prefix, static_cast<int64_t>(cfg.head_num * cfg.size_per_head)});
    }

    torch_ext::LayerKVCache pageCache(const std::vector<torch::Tensor>& full_qkv, const AttentionConfigs& cfg) {
        const int page_size = cfg.kernel_tokens_per_block;
        int       pages     = 0;
        for (const auto& qkv : full_qkv) {
            pages += (qkv.size(0) + page_size - 1) / page_size;
        }
        torch_ext::LayerKVCache cache;
        cache.seq_size_per_block = page_size;
        cache.kv_cache_base      = torch::zeros(
            {pages, 2, static_cast<int64_t>(cfg.kv_head_num), page_size, static_cast<int64_t>(cfg.size_per_head)},
            full_qkv.front().options());
        int page = 0;
        for (const auto& qkv : full_qkv) {
            for (int offset = 0; offset < qkv.size(0); offset += page_size, ++page) {
                const int  count = std::min<int64_t>(page_size, qkv.size(0) - offset);
                const auto rows  = qkv.narrow(0, offset, count);
                cache.kv_cache_base[page][0]
                    .narrow(1, 0, count)
                    .copy_(rows.narrow(1, cfg.head_num, cfg.kv_head_num).transpose(0, 1));
                cache.kv_cache_base[page][1]
                    .narrow(1, 0, count)
                    .copy_(rows.narrow(1, cfg.head_num + cfg.kv_head_num, cfg.kv_head_num).transpose(0, 1));
            }
        }
        return cache;
    }

    void expectClose(const torch::Tensor& actual, const torch::Tensor& expected) {
        ASSERT_EQ(actual.sizes(), expected.sizes());
        EXPECT_TRUE(torch::allclose(actual.to(torch::kFloat32), expected, 1e-2, 1e-2))
            << "maximum absolute error: " << (actual.to(torch::kFloat32) - expected).abs().max().item<float>();
    }
};

TEST_F(FlashInferMetadataTest, CachelessHostAndDeviceMetadataRunsNativeRaggedPrefill) {
    const auto cfg = config();
    for (const auto& lens : {std::vector<int>{2047}, std::vector<int>{3, 129, 17}}) {
        for (bool device : {false, true}) {
            SCOPED_TRACE(device);
            auto                       input  = prefillInputs(lens, std::vector<int>(lens.size(), 0), device);
            const auto                 before = input.input_lengths.clone();
            std::vector<torch::Tensor> rows;
            std::vector<torch::Tensor> expected;
            std::vector<int>           indptr{0};
            for (int length : lens) {
                rows.push_back(randomQkv(length, cfg));
                expected.push_back(denseOracle(rows.back(), 0, cfg));
                indptr.push_back(indptr.back() + length);
            }
            FlashInferPrefillOp op(cfg);
            ASSERT_TRUE(op.support(input));
            const auto params = std::dynamic_pointer_cast<FlashInferAttnParams>(op.prepare(input));
            ASSERT_NE(params, nullptr);
            EXPECT_TRUE(params->ragged_kv);
            EXPECT_FALSE(params->decode_plan);
            EXPECT_TRUE(torch::equal(params->qo_indptr_h, torch::tensor(indptr, torch::kInt32)));
            EXPECT_TRUE(torch::equal(params->page_indptr_h, params->qo_indptr_h));
            EXPECT_TRUE(torch::equal(input.input_lengths, before));
            EXPECT_EQ(input.input_lengths.is_cuda(), device);
            const auto packed = torch::cat(rows, 0).flatten(1);
            expectClose(op.forward(packed, std::nullopt, params), torch::cat(expected, 0));
        }
    }
}

TEST_F(FlashInferMetadataTest, DeviceOnlyPagedTableAndPackedQkvPreservePrefixCausality) {
    const auto                       cfg = config();
    const std::vector<torch::Tensor> rows{randomQkv(4, cfg), randomQkv(257, cfg)};
    auto                             input = prefillInputs({3, 130}, {1, 127}, true);
    input.kv_cache_kernel_block_id_device  = torch::tensor({{0, -1, -1}, {1, 2, 3}}, torch::kInt32).cuda();
    const auto          cache              = pageCache(rows, cfg);
    FlashInferPrefillOp op(cfg);
    const auto          params = std::dynamic_pointer_cast<FlashInferAttnParams>(op.prepare(input));
    ASSERT_NE(params, nullptr);
    EXPECT_FALSE(params->ragged_kv);
    EXPECT_TRUE(torch::equal(params->qo_indptr_h, torch::tensor({0, 3, 133}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(params->page_indptr_h, torch::tensor({0, 1, 4}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(params->page_indice_h.narrow(0, 0, 4), torch::tensor({0, 1, 2, 3}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(params->paged_kv_last_page_len_h, torch::tensor({4, 1}, torch::kInt32)));
    const auto packed   = torch::cat({rows[0].slice(0, 1), rows[1].slice(0, 127)}, 0);
    const auto expected = torch::cat({denseOracle(rows[0], 1, cfg), denseOracle(rows[1], 127, cfg)}, 0);
    expectClose(op.forward(packed.flatten(1), cache, params), expected);
    expectClose(op.forward(packed.narrow(1, 0, cfg.head_num), cache, params), expected);
}

TEST_F(FlashInferMetadataTest, DeviceDecodeMetadataRefreshPreservesCapturedNativeGraph) {
    auto cfg     = config();
    cfg.head_num = 4;
    const std::vector<torch::Tensor> rows{randomQkv(4, cfg), randomQkv(4, cfg)};
    const auto                       cache = pageCache(rows, cfg);
    torch_ext::PyAttentionInputs     input;
    input.dtype                           = torch::scalarTypeToTypeMeta(torch::kBFloat16);
    input.input_lengths                   = lengths({1}, true);
    input.sequence_lengths                = lengths({2}, true);
    input.kv_cache_kernel_block_id_device = torch::tensor({{1}}, torch::kInt32).cuda();
    FlashInferDecodeOp op(cfg, MlaOpsType::AUTO, true);
    const auto         params = std::dynamic_pointer_cast<FlashInferAttnParams>(op.prepare(input));
    ASSERT_NE(params, nullptr);
    EXPECT_TRUE(params->decode_plan);
    EXPECT_FALSE(params->ragged_kv);
    auto query = rows[1].slice(0, 2, 3).narrow(1, 0, cfg.head_num).contiguous();
    expectClose(op.forward(query, cache, params), denseOracle(rows[1].slice(0, 0, 3), 2, cfg));

    auto                stream = c10::cuda::getStreamFromPool();
    at::cuda::CUDAGraph graph;
    torch::Tensor       output;
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        graph.capture_begin();
        output = op.forward(query, cache, params);
        graph.capture_end();
    }
    graph.replay();
    expectClose(output, denseOracle(rows[1].slice(0, 0, 3), 2, cfg));

    input.sequence_lengths                = lengths({3}, true);
    input.kv_cache_kernel_block_id_device = torch::tensor({{0}}, torch::kInt32).cuda();
    params->fillParams(input.sequence_lengths, input.input_lengths, input.kv_cache_kernel_block_id_device, 1, 128);
    EXPECT_TRUE(torch::equal(params->kvlen_h, torch::tensor({4}, torch::kInt32)));
    EXPECT_EQ(params->page_indice_h[0].item<int>(), 0);
    query.copy_(rows[0].slice(0, 3, 4).narrow(1, 0, cfg.head_num));
    graph.replay();
    expectClose(output, denseOracle(rows[0], 3, cfg));
}

TEST_F(FlashInferMetadataTest, RejectsMalformedMetadataBeforeNativePlanning) {
    const auto          cfg = config();
    FlashInferPrefillOp op(cfg);
    auto                input = prefillInputs({3}, {0}, true);
    input.input_lengths       = input.input_lengths.to(torch::kFloat32);
    EXPECT_THROW(op.prepare(input), std::exception);
    input               = prefillInputs({3}, {0}, true);
    input.input_lengths = input.input_lengths.unsqueeze(1);
    EXPECT_THROW(op.prepare(input), std::exception);
    input = prefillInputs({3}, {1}, true);
    EXPECT_THROW(op.prepare(input), std::exception);
    input = prefillInputs({3}, {0, 0}, true);
    EXPECT_THROW(op.prepare(input), std::exception);
    input                                 = prefillInputs({129}, {0}, true);
    input.kv_cache_kernel_block_id_device = torch::tensor({{0}}, torch::kInt32).cuda();
    EXPECT_THROW(op.prepare(input), std::exception);
    input                                 = prefillInputs({3}, {0}, true);
    input.kv_cache_kernel_block_id_device = torch::tensor({{-1}}, torch::kInt32).cuda();
    EXPECT_THROW(op.prepare(input), std::exception);
    FlashInferDecodeOp decode(cfg);
    input                  = prefillInputs({1}, {}, true);
    input.sequence_lengths = lengths({1}, true);
    EXPECT_THROW(decode.prepare(input), std::exception);
}

TEST_F(FlashInferMetadataTest, RaggedPlanCannotConsumeMissingKvRowsOrChangeStorageMode) {
    const auto          cfg = config();
    FlashInferPrefillOp op(cfg);
    auto                input  = prefillInputs({3}, {0}, true);
    const auto          params = std::dynamic_pointer_cast<FlashInferAttnParams>(op.prepare(input));
    ASSERT_NE(params, nullptr);
    const auto qkv = randomQkv(3, cfg);
    EXPECT_THROW(op.forward(qkv.slice(0, 0, 2), std::nullopt, params), std::exception);
    EXPECT_THROW(op.forward(qkv.narrow(1, 0, cfg.head_num), std::nullopt, params), std::exception);
    EXPECT_THROW(op.forward(qkv, pageCache({qkv}, cfg), params), std::exception);
    EXPECT_THROW(params->fillParams(input.sequence_lengths,
                                    input.input_lengths,
                                    torch::tensor({{0}}, torch::kInt32).cuda(),
                                    1,
                                    128,
                                    input.prefix_lengths),
                 std::exception);
}

}  // namespace
}  // namespace rtp_llm
