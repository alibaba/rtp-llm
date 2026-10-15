#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

#include <vector>

using namespace rtp_llm;

namespace {

torch::Tensor makeCudaBlockTable(const std::vector<int32_t>& values, int64_t groups, int64_t batch, int64_t blocks) {
    auto table = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32));
    return table.reshape({groups, batch, blocks}).to(torch::kCUDA);
}

std::vector<int32_t> flattenCuda(const torch::Tensor& tensor) {
    auto flat = tensor.reshape({-1}).to(torch::kCPU).contiguous();
    const auto* data = flat.data_ptr<int32_t>();
    return std::vector<int32_t>(data, data + flat.numel());
}

// Enough dependent matmuls to keep the GPU busy for several milliseconds, so a
// row-index upload enqueued right after is genuinely in flight (and its pinned
// staging tensor already destroyed) by the time the reorder result is read.
void keepStreamBusy() {
    auto lhs = torch::randn({4096, 4096}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto rhs = torch::randn({4096, 4096}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    for (int i = 0; i < 8; ++i) {
        lhs = torch::mm(lhs, rhs);
    }
}

}  // namespace

class ModelTypesReorderCacheRowsTest: public DeviceTestBase {};

TEST_F(ModelTypesReorderCacheRowsTest, ReordersCudaRows) {
    // Groups tagged {full, linear} sort to rows {1, 0}; the block table below
    // stores group 0 first, so the reorder must swap the group rows.
    const auto table = makeCudaBlockTable({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, /*groups=*/3, /*batch=*/1, /*blocks=*/4);
    const std::vector<int64_t> rows{2, 0, 1};

    const auto result = reorderCacheRows(table, rows, /*block_table=*/true);

    ASSERT_TRUE(result.is_cuda());
    ASSERT_EQ(result.size(0), 3);
    const auto flat = flattenCuda(result);
    // Row 2 first, then rows 0 and 1.
    const std::vector<int32_t> expected{9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8};
    ASSERT_VECTOR_EQUAL(expected, flat);
}

TEST_F(ModelTypesReorderCacheRowsTest, ReordersCudaRowsWhileStreamIsBusy) {
    const auto table = makeCudaBlockTable({2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11}, /*groups=*/3, /*batch=*/2, /*blocks=*/2);
    const std::vector<int64_t> rows{1, 2, 0};

    keepStreamBusy();
    const auto result = reorderCacheRows(table, rows, /*block_table=*/true);
    // Force a full device sync before reading back, so the async row-index copy
    // and the index_select kernel have completed.
    const auto flat = flattenCuda(result);

    // Group rows {1, 2, 0}: group1 = {10,12,1,3}, group2 = {5,7,9,11}, group0 = {2,4,6,8}.
    const std::vector<int32_t> expected{10, 12, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8};
    ASSERT_VECTOR_EQUAL(expected, flat);
}

TEST_F(ModelTypesReorderCacheRowsTest, CpuRowsKeepBlockingSemantics) {
    // Non-block-table payloads are 1-D [groups] (e.g. kv_cache_group_types).
    auto table = torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(torch::kInt32));
    const std::vector<int64_t> rows{2, 0, 1};

    const auto result = reorderCacheRows(table, rows, /*block_table=*/false);

    ASSERT_FALSE(result.is_cuda());
    const auto flat = flattenCuda(result);
    const std::vector<int32_t> expected{3, 1, 2};
    ASSERT_VECTOR_EQUAL(expected, flat);
}

TEST_F(ModelTypesReorderCacheRowsTest, TrivialInputsShortCircuit) {
    const std::vector<int64_t> rows{0};
    const auto table = makeCudaBlockTable({7, 8, 9}, /*groups=*/1, /*batch=*/1, /*blocks=*/3);

    // Single row: identity, no upload at all.
    const auto identity = reorderCacheRows(table, rows, /*block_table=*/true);
    ASSERT_TRUE(identity.is_same(table));

    // Undefined tensor and 2-D block tables pass through untouched.
    ASSERT_FALSE(reorderCacheRows(torch::Tensor(), rows, true).defined());
    auto table_2d = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    const auto passthrough = reorderCacheRows(table_2d, rows, /*block_table=*/true);
    ASSERT_TRUE(passthrough.is_same(table_2d));
}
