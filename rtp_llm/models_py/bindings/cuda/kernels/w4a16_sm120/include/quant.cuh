// include/quant.cuh — int4 权重的反量化: nibble -> mma fragment
//
// 核心思路 (借鉴 Marlin, IST-DASLab/marlin):
//   1. 权重在离线阶段被重排成 tensor-core fragment 顺序 (见 weight_transform.cu),
//      于是反量化可以**直接在寄存器里产出 mma 的操作数**, 完全省掉
//      "解压 -> 写 bf16 回 smem -> ldmatrix 读回" 这 4 倍 smem 流量。
//      小 M 场景下这是最有效的优化 —— 权重的复用次数只有 M 次。
//   2. 用 LOP3 把 nibble 塞进 bf16 的尾数位, 再用一次 SIMD 减法得到有符号整数:
//          X  = (w & 0x000F000F) | 0x3F803F80  -> bf16x2(1 + n/128)
//          Xb = bf16x2(1 + 8/128) = 0x3F883F88
//          X - Xb = bf16x2((n - 8) / 128)
//      (n - 8) 就是 [-8, 7] 的有符号 int4, 再乘 (128 * scale) 抵消 1/128 因子。
//   3. mma.m16n8k16 的 B fragment 天然分成 b0 (k=0..7) 与 b1 (k=8..15) 两半,
//      而 group size = 8 的 scale 边界正好落在中间 -> 两半各乘自己的 scale,
//      一个 mma 满负载完成, 无需拆解、无需额外累加器。
//
// 每个函数的输出都是**已经乘好 scale 的 bf16x2**, 可直接喂给
// include/mma/sm80.cuh 的 mma_m16n8k16_bf16。
//
// 文件内部自下而上分三层:
//   dequant_int4x2_to_bf16x2   —— 最底层原语: 一个 u32 里取 2 个 nibble 变 bf16x2
//   dequant_b_frag_e4m3        —— 标准布局: 产出 mma 的 B fragment (2 个 b32)
//   dequant_a_frag_swapab_e4m3 —— swapAB:   产出 mma 的 A fragment (4 个 b32)
#pragma once

#include "common.cuh"

namespace atex {
namespace quant {

using namespace atex::sm120;

// ============================================================================
// 最底层原语: int4 x2 -> bf16 x2
//
// 把 uint32 里的 2 个 nibble 变成 bf16x2, 值 = (nib - 8) / 128。
// 调用方再乘 (128 * scale) 即可得到真正的量化值 —— 乘 128 是为抵消这里的
// 1/128 因子, 这样整数部分完全由位运算完成, 不占浮点 ALU。
//
// 原理: bf16 的位型是 [s:1][E:8][m:7], 而 1 + n/128 恰好等于把 n 放进
// 尾数低 7 位、指数取 0 偏置 (0x3F80)。所以:
//     X  = (w & mask) | 0x3F803F80   -> bf16x2(1 + n/128)
//     Xb = 0x3F883F88 = bf16x2(1 + 8/128)
//     X - Xb                         -> bf16x2((n - 8) / 128)
// (n - 8) 就是 [-8, 7] 的有符号 int4。
//
// LOP3 的 LUT 0xEA == (a & b) | c, 一条指令同时完成"取 nibble"与"贴到尾数位",
// 比先移位再或省一条。
//
// mask 的选择对应不同的 nibble 位置 (由调用方的打包格式决定):
//   0x000F000F: bit[3:0] 与 bit[19:16] -> bf16x2 的两个 lane
//   0x00F000F0: bit[7:4] 与 bit[23:20]
// ============================================================================
__device__ __forceinline__ bf16x2_t dequant_int4x2_to_bf16x2(uint32_t w, uint32_t mask) {
    constexpr uint32_t ONE  = 0x3F803F80u;  // bf16x2(1.0, 1.0)
    constexpr uint32_t BIAS = 0x3F883F88u;  // bf16x2(1 + 8/128, 同上)
    uint32_t           x;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(x) : "r"(w), "r"(mask), "r"(ONE));
    const bf16x2_t v = *reinterpret_cast<const bf16x2_t*>(&x);
    const bf16x2_t b = *reinterpret_cast<const bf16x2_t*>(&BIAS);
    return __hsub2(v, b);  // (nib - 8) / 128
}

// ---------------------------------------------------------------------------
// 从 wpack 的一个 32-bit 字产出完整的 **B fragment** (b0, b1) —— scale 走 e4m3
//
//   w0   : wpack 里的原始字, 覆盖 n = 2*np 与 2*np+1 两组 nibble
//   n_odd: 该 lane 取偶数 n 还是奇数 n (奇数需要整体右移 8 位)
//   vsc  : smem 里**相邻两字节**的 e4m3 scale, 低字节 = k-group 0, 高字节 = k-group 1
//          这正是新布局的好处: 一个 mma 要的两个 scale 一次 2 字节 load 就拿全
//   b_out: 输出 2 个 b32 = 完整 B fragment
//
// 相比 bf16 存法 (两次 2 字节 load + 两次广播) 省一次 load 与一次广播。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dequant_b_frag_e4m3(uint32_t w0, bool n_odd, uint32_t vsc, uint32_t* b_out) {
    constexpr uint32_t MASK = 0x000F000Fu;             // bit[3:0] 与 bit[19:16]
    const uint32_t     wq   = n_odd ? (w0 >> 8) : w0;  // 统一到低 16 位
    uint32_t           s0, s1;
    scale_pair_e4m3_to_bf16x2(vsc, &s0, &s1);  // 一次解码出两个 scale
    // b0 -> k = 2r, 2r+1 ; b1 -> k = 8+2r, 9+2r (都右移 4 位后同构)
    const bf16x2_t m0 = *reinterpret_cast<const bf16x2_t*>(&s0);
    const bf16x2_t m1 = *reinterpret_cast<const bf16x2_t*>(&s1);
    const bf16x2_t r0 = __hmul2(dequant_int4x2_to_bf16x2(wq, MASK), m0);
    const bf16x2_t r1 = __hmul2(dequant_int4x2_to_bf16x2(wq >> 4, MASK), m1);
    b_out[0]          = *reinterpret_cast<const uint32_t*>(&r0);
    b_out[1]          = *reinterpret_cast<const uint32_t*>(&r1);
}

// ============================================================================
// swapAB 专用: 从**标准打包**的两个 word 直接构造 **A fragment** (4 个寄存器)
//
// 背景 —— 为什么 M=8 要 swapAB:
//   mma.m16n8k16 的 M 维下限就是 16。若直接用 M=8 当 M 维, 要凑满 16 行,
//   有一半张量核算力被浪费在补零上。
//   swapAB 把问题反过来算:  D^T[N, M] = W[N, K] @ A[M, K]^T
//   此时 mma 的 M 维 = W 的 N (很大), N 维 = 原来的 M (恰好 8, 正好填满 n8),
//   于是零浪费。
//
// ============================ 关键: 不需要另一套权重布局 ============================
// 权重的打包次序**从头到尾**都按标准 B fragment 布局 (pack_std_kernel 的
// [K/16][N/2][4])。差别只在**怎么读它**:
//
//   标准布局: 权重当 B 操作数, 每个 lane 读**一个** word (np = n8*4 + np_off),
//             从中取出 4 个 nibble 组成 B fragment。
//   swapAB:   权重当 A 操作数, 每个 lane 读**两个** word —— 因为 A fragment
//             需要 8 个元素, 而一个 word 只覆盖 2 行 n (邻接的 n=2np, 2np+1)。
//
// 这两个 word 的 np 下标:
//      wA: np = (g >> 1)           -> 覆盖行 n = g
//      wB: np = (g >> 1) + 4       -> 覆盖行 n = g + 8
// 之所以加 4 而不是加 1: np -> n 的映射是 n ∈ {2np, 2np+1}, 一次跳两行, 所以
// 要跨过 8 行必须跳 4 个 np。又因为 g 与 g+8 同奇偶, 两个 word 共享同一个
// parity 移位 —— 于是两条读路径完全对称。
//
// 代价仅仅是"多读一个 word", 而它**本来就在这一级流水里**: 32 个 lane 各取
// 两个 word, 每个 word 恰好被 2 个 lane 用到, smem 流量与单读一个完全一样。
// 于是省掉了整份 swapAB 布局的权重 (N*K/2 字节的额外显存与一次性打包时间),
// 也省掉了"用户要按 M 准备不同权重"这个对外约束。
//
// ============================ nibble -> a0..a3 的对应 ============================
// 标准 word 的位序 (见 weight_transform.cu 的 nib_k_off / nib_n_odd):
//     bit s -> n = 2np + ((s&2)?1:0),  k = 2r + ((s&1)?8:0) + ((s&4)?1:0)
// 代入后 wA (行 n=g) 的 bit0..3 恰好是:
//     bit0: k=2r      bit1: k=2r+8    bit2: k=2r+1   bit3: k=2r+9
// 而 A fragment 要的四对是 (2r, 2r+1) 与 (2r+8, 2r+9)。所以:
//     a0 = (wq & 0x000F000F)  -> k=2r, 2r+1        (bit0 与 bit2)
//     a2 = (wq >> 4)          -> k=2r+8, 2r+9      (bit1 与 bit3)
// 即 **不需要任何位序重排**, 直接复用与 B fragment 相同的两次解压即可 ——
// (n 奇偶的右移 8 位) 与 (k+8 的右移 4 位) 在这里含义相同, 只是换了名字。
// ============================================================================
__device__ __forceinline__ void
dequant_a_frag_swapab_e4m3(uint32_t w_lo, uint32_t w_hi, bool n_odd, uint32_t vsc_g, uint32_t vsc_g8, uint32_t* a_out) {
    constexpr uint32_t MASK  = 0x000F000Fu;                 // bit[3:0] 与 bit[19:16]
    const uint32_t     wq_lo = n_odd ? (w_lo >> 8) : w_lo;  // 统一到低 16 位
    const uint32_t     wq_hi = n_odd ? (w_hi >> 8) : w_hi;
    // scale 顺序: a0 用 (g,   group0)  a1 用 (g+8, group0)
    //             a2 用 (g,   group1)  a3 用 (g+8, group1)
    uint32_t s[4];
    scale_pair_e4m3_to_bf16x2(vsc_g, &s[0], &s[2]);
    scale_pair_e4m3_to_bf16x2(vsc_g8, &s[1], &s[3]);

    const uint32_t     wq[4]    = {wq_lo, wq_hi, wq_lo, wq_hi};
    constexpr uint32_t SHIFT[4] = {0, 0, 4, 4};  // a0/a1 取 k=2r, a2/a3 取 2r+8
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bf16x2_t m = *reinterpret_cast<const bf16x2_t*>(&s[i]);
        const bf16x2_t r = __hmul2(dequant_int4x2_to_bf16x2(wq[i] >> SHIFT[i], MASK), m);
        a_out[i]         = *reinterpret_cast<const uint32_t*>(&r);
    }
}

}  // namespace quant
}  // namespace atex
