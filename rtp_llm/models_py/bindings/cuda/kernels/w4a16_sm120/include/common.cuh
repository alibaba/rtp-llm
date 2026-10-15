// include/common.cuh — 公共宏、数据类型与设备端小工具
//
// 本文件被 mma/sm80.cuh / quant.cuh / copy/*.cuh / prelogue.cuh / epilogue.cuh
// 共同依赖, 必须最先包含。只放"无业务含义"的通用设施:
// 类型别名、bf16x2 打包、scale 的 e4m3 编解码。
//
// smem 地址转换 (__cvta_generic_to_shared) 与 LOP3 不做包装 —— 直接用
// intrinsic / asm, 少一层间接。
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace atex {
namespace sm120 {

// ---------------------------------------------------------------- 数据类型
using bf16_t   = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;

// ============================================================================
// bf16 x2 打包 (两个 f32 -> 一个 u32, 低半字是 lo)
// ============================================================================
__device__ __forceinline__ uint32_t pack_bf16x2_u32(float lo, float hi) {
    bf16x2_t h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// ============================================================================
// scale 的 e4m3 编解码 (pow2 专用)
//
// ============================ 为什么 e4m3 是无损的 ============================
// weight_transform 把量化 scale 钉成 **2 的幂**:
//     scale = 2^ceil(log2(amax / 7))
// 2 的幂在 IEEE 浮点里的尾数位恒为 0 —— 所以用 e4m3 (1-4-3) 存它**不丢任何
// 信息**, 只是把每个 scale 从 2 字节压到 1 字节。
//
// 这在 int4 权重下占比可观, 因为 scale 是每 8 个元素一个:
//     权重 4 bit/元素 + scale 2 bit/元素 (bf16) -> 1 bit/元素 (e4m3)
//     int4 那一包从 6 bit/元素 降到 5 bit/元素
// 实测 (N=12288, K=4096): 24MB(wpack) + 12MB(bf16 scale) = 36MB
//                      -> 24MB + 6MB = 30MB, 权重流量降 17%。
// 而 kernel 是纯带宽瓶颈 (实测各口径都压在 ~1100 GB/s = HBM 上限),
// 所以这部分流量是实打实的收益。
//
// ============================ 解码代价 ============================
// e4m3 的正规数位型: [s:1][E:4][m:3], 值 = 2^(E-7) * (1 + m/8)。
// 我们存的是 scale 本身 (不是 128*scale), 所以 m = 0, 只剩 E。
//
// GEMM 里要的是 **128*scale** 的 bf16x2 位型 (乘 128 是为抵消反量化里
// nib2bf 引入的 1/128 因子), 而 128*scale = 2^E, 于是:
//     bf16 位型 = (127 + E) << 7 = 0x3F80 + (E << 7)
// 而 E = (b >> 3) & 0xF, 即 E << 7 = (b & 0x78) << 4, 所以
//     one16 = 0x3F80 + ((b & 0x78) << 4)
// 再用一次 prmt 把 16 位铺满 32 位 (两个 lane 同一个 scale)。
//
// **pow2 契约是正确性的前提**: 若 scale 尾数非 0, 这些函数会静默算错。
// 另外 E 必须 ∈ [1, 15] (scale ∈ [2^-6, 2^8]); weight_transform 已 clamp。
// ============================================================================

// 相邻两字节 [b_lo, b_hi] -> 两个 bf16x2。
// 这是 smem 布局设计的关键: 一个 mma 要的 k-group 0 与 1 两个 scale 正好相邻,
// 所以一次 2 字节 load 就能拿到两个, 省掉一次 load。
//   低字节 (k-group 0) 在 bits 0-7, 高字节 (k-group 1) 在 bits 8-15
//
// 注意是 **加** 不是或: bf16 位型 = 0x3F80 + (E<<7), 而 E 最大 15 时
// (E<<7) = 0x780, 与 0x3F80 有重叠位, OR 会把结果搞错。
// 两个 lane 各自的和都 < 0x10000, 不会跨 lane 进位, 所以可以直接整数加。
__device__ __forceinline__ void scale_pair_e4m3_to_bf16x2(uint32_t v, uint32_t* x2_lo, uint32_t* x2_hi) {
    constexpr uint32_t BIAS2 = 0x3F803F80u;  // bf16x2(1.0, 1.0)
    // ... & 0x78 -> E<<3 ; <<4 -> E<<7 ; *0x10001 把 16 位铺到两个 lane
    *x2_lo = BIAS2 + (((v & 0x78u) << 4) * 0x00010001u);
    *x2_hi = BIAS2 + (((v & 0x7800u) >> 4) * 0x00010001u);
}

// e4m3 (pow2) -> scale 本身的 f32 值 (供 weight_transform 的逆变换用)
__device__ __forceinline__ float scale_e4m3_to_f32(uint32_t b) {
    const uint32_t E = (b >> 3) & 0xFu;  // value = 2^(E-7)
    return __uint_as_float((E + 120u) << 23);
}

// 2 的指数 e -> e4m3 字节 (value = 2^e, 要求 e ∈ [-6, 8])
// 直接拿量化时已经算出的指数, 不需要 log2f 反推 —— 那条路既有精度风险
// (log2f(2^k) 在 device 上不保证精确), 又多几条指令。
__device__ __forceinline__ uint32_t scale_exp_to_e4m3(int e) {
    return static_cast<uint32_t>(e + 7) << 3;  // bias 7, 尾数 0
}

}  // namespace sm120
}  // namespace atex
