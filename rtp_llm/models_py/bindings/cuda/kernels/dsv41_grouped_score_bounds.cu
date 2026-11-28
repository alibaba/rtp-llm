#include "rtp_llm/models_py/bindings/cuda/kernels/dsv41_grouped_score_bounds.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>

#include <cstdint>
#include <cstring>

namespace torch_ext {
namespace {

constexpr int kBlockRows   = 128;
constexpr int kSmallGroups = 64;
constexpr int kMaxGroups   = 512;

struct Group {
    int32_t tile_end;
    int32_t first_request;
    int32_t request_end;
    int32_t row_start;
    int32_t rows;
    int32_t key_span;
    int32_t key_width;
    int32_t tile_start;
};
static_assert(sizeof(Group) == 8 * sizeof(int32_t));

template<int Capacity>
struct Layout {
    Group groups[Capacity];
};

template<typename Position, typename Request, int Capacity>
__global__ void write_bounds(const __grid_constant__ Layout<Capacity> layout,
                             const Position*                          positions,
                             const Request*                           requests,
                             const int32_t*                           key_counts,
                             int32_t*                                 output,
                             int                                      num_groups,
                             int                                      ratio) {
    const int tile  = blockIdx.x;
    int       lower = 0, upper = num_groups;
    while (lower < upper) {
        const int middle = (lower + upper) / 2;
        if (layout.groups[middle].tile_end <= tile) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    const Group group = layout.groups[lower];

    using Scan = cub::BlockScan<int64_t, kBlockRows>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int32_t                    counts[kBlockRows];
    __shared__ int64_t                    offsets[kBlockRows];
    const int                             lane   = threadIdx.x;
    const int                             index  = group.first_request + lane;
    const int32_t                         count  = index < group.request_end ? key_counts[index] : 0;
    const int64_t                         padded = (static_cast<int64_t>(max(count, 0)) + 255) & ~int64_t{255};
    int64_t                               offset;
    Scan(scan).ExclusiveSum(padded, offset);
    counts[lane]  = count;
    offsets[lane] = offset;
    __syncthreads();

    const int row = (tile - group.tile_start) * kBlockRows + lane;
    if (row >= group.rows) {
        return;
    }
    const int     packed_row = group.row_start + row;
    const int     request    = static_cast<int32_t>(requests[packed_row]);
    const bool    in_group   = request >= group.first_request && request < group.request_end;
    const int     slot       = in_group ? request - group.first_request : 0;
    const int32_t width      = counts[slot];
    const int64_t start      = offsets[slot];
    const bool    valid      = in_group && width >= 0 && width <= group.key_width && start + width <= group.key_span;
    const int64_t position   = static_cast<int64_t>(positions[packed_row]);
    // Match the Triton signed int64 add, including its wrap at INT64_MAX.
    const int64_t next                 = static_cast<int64_t>(static_cast<uint64_t>(position) + 1);
    const int32_t visible              = valid ? min(max(next, int64_t{0}) >> (ratio - 1), int64_t{width}) : 0;
    const int32_t key_start            = valid ? start : 0;
    int32_t*      group_output         = output + 3LL * group.row_start;
    group_output[row]                  = key_start;
    group_output[group.rows + row]     = key_start + visible;
    group_output[2 * group.rows + row] = visible;
}

template<typename Position, typename Request, int Capacity>
void launch(const Layout<Capacity>& layout,
            const torch::Tensor&    positions,
            const torch::Tensor&    requests,
            const torch::Tensor&    counts,
            torch::Tensor&          output,
            int                     num_groups,
            int                     tiles,
            int                     ratio,
            cudaStream_t            stream) {
    write_bounds<Position, Request><<<tiles, kBlockRows, 0, stream>>>(layout,
                                                                      positions.data_ptr<Position>(),
                                                                      requests.data_ptr<Request>(),
                                                                      counts.data_ptr<int32_t>(),
                                                                      output.data_ptr<int32_t>(),
                                                                      num_groups,
                                                                      ratio);
    const auto status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "DSV4.1 grouped score bounds launch failed: ", cudaGetErrorString(status));
}

template<int Capacity>
void dispatch(const torch::Tensor& descriptors,
              const torch::Tensor& positions,
              const torch::Tensor& requests,
              const torch::Tensor& counts,
              torch::Tensor&       output,
              int                  ratio,
              cudaStream_t         stream) {
    Layout<Capacity> layout{};
    const int        num_groups = descriptors.size(0);
    std::memcpy(layout.groups, descriptors.data_ptr<int32_t>(), num_groups * sizeof(Group));

    int rows = 0, tiles = 0;
    for (int index = 0; index < num_groups; ++index) {
        const auto& group = layout.groups[index];
        TORCH_CHECK(group.first_request >= 0 && group.request_end > group.first_request
                        && group.request_end <= counts.numel() && group.request_end - group.first_request <= kBlockRows,
                    "invalid grouped score request bounds");
        TORCH_CHECK(group.row_start == rows && group.rows > 0 && group.rows <= 4096 && group.rows % 4 == 0
                        && group.key_span >= 0 && group.key_width >= 0,
                    "invalid grouped score layout");
        TORCH_CHECK(group.tile_start == tiles && group.tile_end == tiles + (group.rows + kBlockRows - 1) / kBlockRows,
                    "invalid grouped score tile ranges");
        rows += group.rows;
        tiles = group.tile_end;
    }
    TORCH_CHECK(positions.numel() >= rows && output.numel() == 3LL * rows, "grouped score rows exceed tensor ranges");

    if (positions.scalar_type() == torch::kInt64) {
        if (requests.scalar_type() == torch::kInt64) {
            launch<int64_t, int64_t>(layout, positions, requests, counts, output, num_groups, tiles, ratio, stream);
        } else {
            launch<int64_t, int32_t>(layout, positions, requests, counts, output, num_groups, tiles, ratio, stream);
        }
    } else if (requests.scalar_type() == torch::kInt64) {
        launch<int32_t, int64_t>(layout, positions, requests, counts, output, num_groups, tiles, ratio, stream);
    } else {
        launch<int32_t, int32_t>(layout, positions, requests, counts, output, num_groups, tiles, ratio, stream);
    }
}

}  // namespace

void dsv41_grouped_score_bounds(const torch::Tensor& descriptors,
                                const torch::Tensor& positions,
                                const torch::Tensor& requests,
                                const torch::Tensor& key_counts,
                                torch::Tensor&       output,
                                int64_t              ratio) {
    TORCH_CHECK(descriptors.device().is_cpu() && descriptors.scalar_type() == torch::kInt32 && descriptors.dim() == 2
                    && descriptors.is_contiguous() && descriptors.size(1) == 8 && descriptors.size(0) > 0
                    && descriptors.size(0) <= kMaxGroups,
                "grouped score descriptors must be contiguous CPU int32[G,8]");
    TORCH_CHECK(positions.is_cuda() && (ratio == 1 || ratio == 2),
                "expected CUDA grouped score bounds with compression ratio 1 or 2");
    for (const auto& tensor : {positions, requests, key_counts, output}) {
        TORCH_CHECK(tensor.device() == positions.device() && tensor.dim() == 1 && tensor.is_contiguous(),
                    "grouped score bounds require 1D contiguous tensors on a common CUDA device");
    }
    const auto is_index = [](const torch::Tensor& tensor) {
        return tensor.scalar_type() == torch::kInt32 || tensor.scalar_type() == torch::kInt64;
    };
    TORCH_CHECK(is_index(positions) && is_index(requests) && positions.numel() == requests.numel()
                    && key_counts.scalar_type() == torch::kInt32 && output.scalar_type() == torch::kInt32,
                "invalid grouped score tensor shapes or dtypes");
    for (const auto& input : {positions, requests, key_counts}) {
        at::assert_no_overlap(output, input);
    }

    const c10::cuda::CUDAGuard guard(positions.device());
    const auto                 stream = at::cuda::getCurrentCUDAStream(positions.get_device());
    if (descriptors.size(0) <= kSmallGroups) {
        dispatch<kSmallGroups>(descriptors, positions, requests, key_counts, output, ratio, stream);
    } else {
        dispatch<kMaxGroups>(descriptors, positions, requests, key_counts, output, ratio, stream);
    }
}

}  // namespace torch_ext
