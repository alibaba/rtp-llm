// include/mma/sm80.cuh — SM80 起的张量核 mma 指令
//
// 本文件只放 **mma 指令本身** 与"从 smem 组装 mma 操作数"的加载器;
// int4 反量化 (把权重的 nibble 变成 fragment) 在 include/quant.cuh。
//
// 编译目标必须用 sm_120a / sm_120f: mma.sync bf16 在 sm_120 上即可用,
// 但 ldmatrix + cp.async 的组合在 sm_120a 下才完整放开 (见 build.sh)。
#pragma once

#include "../common.cuh"

namespace atex {
namespace mma {
namespace sm80 {

using namespace atex::sm120;

// ============================================================================
// mma.m16n8k16 — bf16 输入 / f32 累加
//   D[4] = A[4] * B[2] + C[4]
//   A fragment: 16x16, 每 lane 4 个 b32
//   B fragment: 16x8,  每 lane 2 个 b32
// ============================================================================
__device__ __forceinline__ void mma_m16n8k16_bf16(float* d, const uint32_t* a, const uint32_t* b, const float* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ============================================================================
// swapAB 的 B 操作数 (激活) — 直接从 smem 的 [8][TileK] 布局拼出 2 个寄存器
//
//   激活以 [m][k] 行主序放在 smem (m 只有 8 行), 于是 (k, k+1) 两个相邻
//   bf16 正好是一个 4 字节对齐单元, 直接一次 32-bit 读就能拿到 bf16x2。
//   B fragment 需要 k = 2r/2r+1 与 2r+8/2r+9 两对, 即两次 4 字节读。
//
//   注: 这里不用 ldmatrix —— 整个 k16 只读 2 次, 且该 fragment 会被块内
//   所有 n16 tile 复用 (TileN/16 次), 摊薄后开销可忽略。
//
//   sAct 必须**已经指向本 k16 块的起始列** (即 sAct + i16*16),
//   函数内部再按 g * row_stride 取行。
// ============================================================================
__device__ __forceinline__ void
load_act_b_frag_swapab(const bf16_t* __restrict__ sAct, int row_stride, int lane, uint32_t* b_out) {
    const int     g   = lane >> 2;  // m 维下标 (0..7)
    const int     r   = lane & 3;   // k 偏移 / 2
    const bf16_t* row = sAct + g * row_stride;
    b_out[0]          = *reinterpret_cast<const uint32_t*>(row + 2 * r);      // k=2r, 2r+1
    b_out[1]          = *reinterpret_cast<const uint32_t*>(row + 2 * r + 8);  // k=2r+8,+9
}

}  // namespace sm80
}  // namespace mma
}  // namespace atex
