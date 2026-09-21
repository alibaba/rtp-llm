// include/epilogue.cuh — 结果写回 (epilogue)
//
// 两条输出路径 (由 splitK 决定, 都在主循环之外, 对性能无影响):
//
//   [DIRECT]  n_split == 1
//     累加器就是最终结果, 就地转 bf16 写回 D。
//
//   [ATOMIC]  n_split > 1
//     每个 split 把自己的部分和**直接以 bf16 累加进 D**, 于是完全不需要
//     第二阶段的 reduce kernel, 也不需要 fp32 workspace。
//
//     用 red.global.add.noftz.bf16x2 而不是 atomicAdd:
//       * red 是 fire-and-forget, 不等回值 —— 而 atomicAdd 必须等
//         (它要返回旧值), 后者会在高争用下阻塞 SM;
//       * bf16x2 一次累加两个相邻元素, 指令数减半。
//     硬件从 sm_90 起支持 bf16 的原子加, 不需要 CAS 循环。
//
//     代价: bf16 只有 8 位尾数, 累加顺序不确定 => 结果不是位确定的
//     (float 累加也一样不确定, 只是这里的舍入误差更大)。详见 store_atomic 的说明。
#pragma once

#include "common.cuh"

namespace atex {
namespace eplogue {
namespace sm120 {

using namespace atex::sm120;

// ============================================================================
// mma.m16n8k16 的累加器元素坐标 (每 lane 4 个 f32):
//   c0 -> (row = g,     col = 2r)
//   c1 -> (row = g,     col = 2r + 1)
//   c2 -> (row = g + 8, col = 2r)
//   c3 -> (row = g + 8, col = 2r + 1)
//   其中 g = lane >> 2, r = lane & 3
// ============================================================================
struct AccCoord {
    int row_lo, row_hi, col;
};

__device__ __forceinline__ AccCoord acc_coord(int lane) {
    const int g = lane >> 2;
    return AccCoord{g, g + 8, 2 * (lane & 3)};
}

// ---------------------------------------------------------------------------
// 标准布局: 行 = M 维, 列 = N 维。同一行相邻两列合并成一次 4 字节写。
// ---------------------------------------------------------------------------
// ============================================================================
// 走 mma 累加器的输出 (每 lane 4 个 f32, 坐标见上面的 acc_coord)
//
// 两种路径的**累加器布局完全相同** (都是 mma 的 C fragment), 差别只在
// "这 4 个数落在 D 的哪两行、哪两列":
//
//   标准 (m 是行):  m 方向随 lane 变化 (2r, 2r+1), n 方向随块起点变化
//                      -> m_base = 块起点 + 行偏移, n_base = 列起点
//                      -> 同一行两列相邻, 可合成一次 4 字节写
//   swapAB (m 是列): 累加器是 D^T 的切片, 于是
//                      -> m_base = 恒为 0 (列坐标全由 lane 决定)
//                      -> n_base = 块起点 (行坐标才是 N 维)
//                      -> 每列只有 2 个连续字节, 无法合并访存
//
// 所以这里不做两套函数, 而是让调用方把已加好偏移的坐标传进来 —— 差别只是
// 传参, 不是逻辑。M/N 依然要传, 因为两者都要做**行/列越界掩码** (M 或 N
// 不是 tile 整数倍时, 越界的那些必须挡掉)。
// ============================================================================
__device__ __forceinline__ void
store_direct(bf16_t* __restrict__ D, int M, int N, int m_base, int n_base, int lane, const float* acc) {
    const AccCoord c  = acc_coord(lane);
    const int      r0 = m_base + c.row_lo, r1 = m_base + c.row_hi;
    const int      col = n_base + c.col;

    if (r0 < M) {
        if (col + 1 < N)
            *reinterpret_cast<uint32_t*>(&D[static_cast<size_t>(r0) * N + col]) = pack_bf16x2_u32(acc[0], acc[1]);
        else if (col < N)
            D[static_cast<size_t>(r0) * N + col] = __float2bfloat16(acc[0]);
    }
    if (r1 < M) {
        if (col + 1 < N)
            *reinterpret_cast<uint32_t*>(&D[static_cast<size_t>(r1) * N + col]) = pack_bf16x2_u32(acc[2], acc[3]);
        else if (col < N)
            D[static_cast<size_t>(r1) * N + col] = __float2bfloat16(acc[2]);
    }
}

// ---------------------------------------------------------------------------
// ============================================================================
// [ATOMIC] 把部分和以 bf16 直接累加进 D (splitK > 1)
//
// ============================== 为什么能用 bf16 累加 ==============================
// 硬件从 sm_90 起支持 bf16 / bf16x2 的原子加 (atom.add.noftz.bf16x2),
// 不需要 CAS 重试循环。所以可以直接把每个 split 的部分和加到 D 上,
// **彻底省掉第二阶段的 reduce kernel 和 fp32 workspace**。
//
// 用 red 而不是 atomicAdd:
//   atomicAdd 必须返回旧值 (C 语义), 于是 SM 要等这次访存回来;
//   red.global.add 没有返回值, 发出即走, 不占发射槽等待 —— 这正是这里要的。
//   nvcc 对 "丢弃返回值的 atomicAdd" 不一定能优化成 red, 所以直接写 asm。
//
// ============================== 精度上的两点提醒 ==============================
// 1) bf16 只有 8 位尾数。n_split 次累加意味着最多 n_split 次舍入, 有效精度
//    低于 "fp32 累加后一次性转 bf16"。半加法的最大相对误差约 2^-9 ≈ 0.2%,
//    splitK=4 时累计误差量级也在 1e-3 以下, 相对 int4 量化本身的 ~0.115
//    完全可忽略。
// 2) 累加顺序不确定 => 结果**不是位确定的**。这是 splitK 的固有性质
//    (fp32 累加同样如此), 只是 bf16 的方差更大。回归测试不要卡到 0 误差。
//
// ============================== 前置条件 ==============================
// D 必须在 kernel 启动前清零 —— 因为现在 D 既是输出也是累加目标。
// 这一步比原来的 fp32 私有切片方案多一次 M*N*2 字节的写 (bf16 比 fp32 减半)。
// ============================================================================

// 一个 bf16x2 的 fire-and-forget atomic add。p 必须 4 字节对齐。
__device__ __forceinline__ void red_bf16x2(void* p, uint32_t v) {
    asm volatile("red.global.add.noftz.bf16x2 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void
store_atomic_bf16(bf16_t* __restrict__ D, int M, int N, int m_base, int n_base, int lane, const float* acc) {
    const AccCoord c  = acc_coord(lane);
    const int      r0 = m_base + c.row_lo, r1 = m_base + c.row_hi;
    const int      col = n_base + c.col;

    // 相邻两列合并成一个 bf16x2 原子加 (一次指令累加两个元素)
    if (r0 < M) {
        if (col + 1 < N) {
            red_bf16x2(&D[static_cast<size_t>(r0) * N + col], pack_bf16x2_u32(acc[0], acc[1]));
        } else if (col < N) {
            atomicAdd(&D[static_cast<size_t>(r0) * N + col], __float2bfloat16(acc[0]));
        }
    }
    if (r1 < M) {
        if (col + 1 < N) {
            red_bf16x2(&D[static_cast<size_t>(r1) * N + col], pack_bf16x2_u32(acc[2], acc[3]));
        } else if (col < N) {
            atomicAdd(&D[static_cast<size_t>(r1) * N + col], __float2bfloat16(acc[2]));
        }
    }
}

// ---------------------------------------------------------------------------
// swapAB 布局: 累加器的"行"是 N 维, "列"是 M 维 (即 D^T 的切片)。
//   列坐标完全由 lane 决定 (row = 2r, row+1 = 2r+1, 共 8 列 = M 维),
//   所以这里**没有 M 方向的 base 偏移** —— 若再传一个 m_base 会重复计入。
//   写回 [M][N] 的 D 需要转置, 每列只有 2 个连续字节, 无法合并访存。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
store_transposed(bf16_t* __restrict__ D, int M, int N, int n_base, int lane, const float* acc) {
    const AccCoord c     = acc_coord(lane);
    const int      ncol0 = n_base + c.row_lo, ncol1 = n_base + c.row_hi;
    const int      row = c.col;  // M 维下标 (0, 2, 4, 6)

    if (ncol0 < N) {
        if (row < M)
            D[static_cast<size_t>(row) * N + ncol0] = __float2bfloat16(acc[0]);
        if (row + 1 < M)
            D[static_cast<size_t>(row + 1) * N + ncol0] = __float2bfloat16(acc[1]);
    }
    if (ncol1 < N) {
        if (row < M)
            D[static_cast<size_t>(row) * N + ncol1] = __float2bfloat16(acc[2]);
        if (row + 1 < M)
            D[static_cast<size_t>(row + 1) * N + ncol1] = __float2bfloat16(acc[3]);
    }
}

__device__ __forceinline__ void
store_atomic_bf16_transposed(bf16_t* __restrict__ D, int M, int N, int n_base, int lane, const float* acc) {
    const AccCoord c     = acc_coord(lane);
    const int      ncol0 = n_base + c.row_lo, ncol1 = n_base + c.row_hi;
    const int      row = c.col;

    // swapAB 的累加器是 D^T 的切片: 同一行的两个元素在 D 里按列相邻
    // (相隔 N 个元素), 无法合并成一个 bf16x2, 所以只能用标量 atomicAdd。
    if (ncol0 < N) {
        if (row < M)
            atomicAdd(&D[static_cast<size_t>(row) * N + ncol0], __float2bfloat16(acc[0]));
        if (row + 1 < M)
            atomicAdd(&D[static_cast<size_t>(row + 1) * N + ncol0], __float2bfloat16(acc[1]));
    }
    if (ncol1 < N) {
        if (row < M)
            atomicAdd(&D[static_cast<size_t>(row) * N + ncol1], __float2bfloat16(acc[2]));
        if (row + 1 < M)
            atomicAdd(&D[static_cast<size_t>(row + 1) * N + ncol1], __float2bfloat16(acc[3]));
    }
}

}  // namespace sm120
}  // namespace eplogue
}  // namespace atex
