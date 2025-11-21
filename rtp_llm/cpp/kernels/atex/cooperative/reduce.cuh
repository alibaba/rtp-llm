#include "rtp_llm/cpp/kernels/atex/block/reduce.cuh"
#include <cooperative_groups.h>  // requires cuda 9

// The device-side reduction functions rely on support from Cooperative Groups,
// which is used here instead of manually writing atomic synchronization primitives.
// Utilizing Cooperative Groups is considered a more modern and potentially more
// performant approach for thread cooperation and synchronization.

// IMPORTANT: When your kernel utilizes functions from the device/reduce.cuh
// header that rely on Cooperative Groups for grid-wide synchronization (e.g.,
// for operations spanning multiple thread blocks), the kernel **must** be
// launched using the specific CUDA API: `cudaLaunchCooperativeKernel`.
// This is because standard kernel launches (`<<<...>>>`) do not guarantee that
// all thread blocks in the grid will be simultaneously resident on the device
// (which is required for block-to-block synchronization).

// Consult the CUDA Programming Guide for further details on Cooperative Groups
// and the requirements for cooperative grid launches.

namespace cg = cooperative_groups;

namespace atex {
namespace cooperative {

__device__ inline void sync() {
    cg::grid_group grid = cg::this_grid();
    grid.sync();
}

template<uint32_t TPB>
__device__ inline fp32_t reduce_max(fp32_t local_max, fp32_t* smem, fp32_t* gmem) {
    // 使用此函数时，smem 的大小至少应该为 TPB / warpSize
    // gmem 必须首先清零
    cg::grid_group grid = cg::this_grid();
    uint32_t       tid  = threadIdx.x;

    local_max = atex::block::reduce_max<TPB, false>(local_max, smem);

    if (tid == 0) {
        atomicMax((int32_t*)gmem, __float_as_int(local_max));
    }
    grid.sync();

    return *gmem;
}

template<uint32_t TPB>
__device__ inline fp32_t reduce_sum(fp32_t local_sum, fp32_t* smem, fp32_t* gmem) {
    // 使用此函数时，smem 的大小至少应该为 TPB / warpSize
    // 线程数必须是 warpSize 的整倍数
    // gmem 必须首先清零
    cg::grid_group grid = cg::this_grid();
    uint32_t       tid  = threadIdx.x;

    local_sum += atex::block::reduce_sum<TPB, false>(local_sum, smem);

    if (tid == 0) {
        atomicAdd(gmem, local_sum);
    }
    grid.sync();

    return *gmem;
}

}  // namespace cooperative
}  // namespace atex