// include/prelogue.cuh — 主循环的全局 -> smem 数据加载
//
// 每个 K-tile 往 smem 装三样东西 (标准布局):
//   sA : A 的 [TileK/16][TileM][16] bf16   —— 已排成 ldmatrix 需要的 16x16 块
//   sW : Wpack 的 [TileK/16][TileN/2][4] int32 —— 权重已是 fragment 顺序
//   sS : Scales 的 [TileK/32][TileN][4] uint8 —— e4m3 存 pow2 scale
//
// 布局设计要点:
//   - sA 排成 [k16][m][16] 而非普通 [m][k], 使每个 (i16, m16块) 的 32 字节
//     正好是 ldmatrix.x4 要的 16x16 块, 行跨距恒为 32 字节 (不依赖 TileK),
//     于是 ldmatrix 的地址计算是纯编译期常量。
//   - sW 的 16 字节 (= 4 个 int32) 正好覆盖一个 warp 在 B fragment 上所需的
//     全部数据, 一次 16B 拷贝直接喂给 32 个 lane, 无浪费。
//   - sS 把同一个 mma 要的两个 scale 放在相邻 2 字节, 于是只需一次 load。
#pragma once

#include "common.cuh"
#include "copy/sm75.cuh"
#include "copy/sm80.cuh"

namespace atex {
namespace prelogue {
namespace sm120 {

using namespace atex::sm120;

// ---------------------------------------------------------------------------
// 每级流水的 smem 字节数 (host 侧据此算动态 smem)
//
// Scales 用 **e4m3 (1 字节)** 而不是 bf16 (2 字节):
//   * scale 恒为 2 的幂, 尾数位为 0, 所以 e4m3 存储是无损的 (见 common.cuh)
//   * smem 占用减半, 对多级流水是实打实的余量
//   * 全局流量也从 2 bit/元素 降到 1 bit/元素
//
// smem_A_stage 对标准与 swapAB **通用**: 两者都是 TileM x TileK 个 bf16,
// 只是排列不同 (标准按 [TileK/16][TileM][16] 供 ldmatrix 用; swapAB 按
// [TileM][TileK] 行主序供 32-bit 直读)。字节数相同 = TileM*TileK*2。
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ constexpr int smem_A_stage(int TileM, int TileK) {
    return TileM * TileK * 2;
}
__host__ __device__ __forceinline__ constexpr int smem_W_stage(int TileN, int TileK) {
    return (TileK / 16) * (TileN / 2) * 16;
}
__host__ __device__ __forceinline__ constexpr int smem_S_stage(int TileN, int TileK) {
    return (TileK / 8) * TileN;  // e4m3: 1 字节/scale
}
__host__ __device__ __forceinline__ constexpr int smem_per_stage(int TileM, int TileN, int TileK) {
    return smem_A_stage(TileM, TileK) + smem_W_stage(TileN, TileK) + smem_S_stage(TileN, TileK);
}

// ============================================================================
// A 的加载 —— 两种布局
// ============================================================================
// 标准: 全局 [M][K] (行主序) -> smem [TileK/16][TileM][16]
//   一个工作项 = 一个 (i16, m) 对, 搬 32 字节, 用两次 cp.async.16
//
// 这样排的目的是让每个 (i16, m16块) 的 32 字节正好是 ldmatrix.x4 要的
// 16x16 块, 行跨距恒为 32 字节 (不依赖 TileK), 于是 ldmatrix 的地址计算
// 是纯编译期常量。
//
// swapAB: 全局 [M][K] -> smem [TileM][TileK] 行主序
//   激活在那里当 mma 的 B 操作数, (k, k+1) 相邻两个 bf16 就是一个 4 字节
//   单元, 一次 32-bit 读就能拼出 bf16x2, 不必过 ldmatrix。整个 k16 只读 2 次,
//   且会被块内所有 n16 tile 复用。
// ============================================================================
template<int TileM, int TileK, int NumThread>
__device__ __forceinline__ void
load_A_tile(char* a_dst, const bf16_t* __restrict__ A, int m_base, int m_hi, int k0, int K, int tid) {
    constexpr int KI = TileK / 16;
    constexpr int NA = KI * TileM;

#pragma unroll
    for (int i = tid; i < NA; i += NumThread) {
        const int i16 = i / TileM;
        const int m   = i % TileM;
        char*     d   = a_dst + (static_cast<size_t>(i16) * TileM + m) * 32;
        const int gm  = m_base + m;

        if (gm < m_hi && k0 + i16 * 16 < K) {
            const char* src = reinterpret_cast<const char*>(A) + static_cast<size_t>(gm) * K * 2
                              + static_cast<size_t>(k0 + i16 * 16) * 2;
            if (k0 + i16 * 16 + 16 <= K) {
                copy::sm80::cp_async<16>(d, src);
                copy::sm80::cp_async<16>(d + 16, src + 16);
            } else {
                uint4 v = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int t = 0; t < 16; ++t)
                    if (k0 + i16 * 16 + t < K)
                        reinterpret_cast<bf16_t*>(&v)[t] = reinterpret_cast<const bf16_t*>(src)[t];
                *reinterpret_cast<uint4*>(d)      = v;
                *reinterpret_cast<uint4*>(d + 16) = make_uint4(0, 0, 0, 0);
            }
        } else {
            // M 或 K 越界补 0, 保证 ldmatrix 读到确定值
            *reinterpret_cast<uint4*>(d)      = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(d + 16) = make_uint4(0, 0, 0, 0);
        }
    }
}

template<int TileM, int TileK, int NumThread>
__device__ __forceinline__ void
load_A_tile_swapab(char* a_dst, const bf16_t* __restrict__ A, int m_base, int m_hi, int k0, int K, int tid) {
    constexpr int NV = TileM * (TileK / 8);

#pragma unroll
    for (int i = tid; i < NV; i += NumThread) {
        const int m  = i / (TileK / 8);
        const int c  = (i % (TileK / 8)) * 8;
        char*     d  = a_dst + (static_cast<size_t>(m) * TileK + c) * 2;
        const int gm = m_base + m;

        if (gm < m_hi && k0 + c < K) {
            const char* src =
                reinterpret_cast<const char*>(A) + static_cast<size_t>(gm) * K * 2 + static_cast<size_t>(k0 + c) * 2;
            if (k0 + c + 8 <= K) {
                copy::sm80::cp_async<16>(d, src);
            } else {
                uint4 v = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int t = 0; t < 8; ++t)
                    if (k0 + c + t < K)
                        reinterpret_cast<bf16_t*>(&v)[t] = reinterpret_cast<const bf16_t*>(src)[t];
                *reinterpret_cast<uint4*>(d) = v;
            }
        } else {
            *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
        }
    }
}

// ============================================================================
// Wpack: 全局 [K/16][N/2][4] int32 -> smem 同布局, 每项 16 字节
// ============================================================================
template<int TileN, int TileK, int NumThread>
__device__ __forceinline__ void
load_W_tile(char* w_dst, const int32_t* __restrict__ Wp, int n0, int k16_0, int np_stride, int tid) {
    constexpr int KI  = TileK / 16;
    constexpr int NWP = KI * (TileN / 2);

#pragma unroll
    for (int i = tid; i < NWP; i += NumThread) {
        const int   i16 = i / (TileN / 2);
        const int   np  = i % (TileN / 2);
        const char* src =
            reinterpret_cast<const char*>(Wp) + (static_cast<size_t>(k16_0 + i16) * np_stride + (n0 / 2 + np)) * 16;
        copy::sm80::cp_async<16>(w_dst + (static_cast<size_t>(i16) * (TileN / 2) + np) * 16, src);
    }
}

// ============================================================================
// Scales: 全局 [K/32][N][4] uint8 -> smem [TileK/32][TileN][4] uint8
//
// ============================ 布局设计 ============================
// 4 字节 = 一个 **k32** 内 4 个 k8 组的 scale (每 8 个元素一个 scale):
//     字节 0-1 : 前半个 k16 (k8 组 0 与 1)
//     字节 2-3 : 后半个 k16 (k8 组 2 与 3)
// 全局侧按 n 连续 (每 n 一个 4 字节), 所以加载是纯 uint32 拷贝:
// 一个 warp 的 32 个 lane 覆盖 32 个连续 n, 一次 coalesced 128 字节事务。
//
// ============================ 为什么这样能省 load ============================
// mma.m16n8k16 的 B fragment 需要 k-group 0 与 1 两个 scale, 而它们**相邻**
// (字节 0 与 1), 所以一次 2 字节 load 就够 —— 旧的 bf16 布局里这两个 scale
// 隔着 TileN 个元素, 必须两次 load。
// swapAB (A fragment) 需要 (行 g, g+8) x (k-group 0,1) 共 4 个, 也用 2 次
// 2 字节 load (旧的 bf16 布局要 4 次)。
// ============================================================================
template<int TileN, int TileK, int NumThread>
__device__ __forceinline__ void
load_S_tile(char* s_dst, const uint8_t* __restrict__ Sc, int n0, int k32_0, int k32_stride, int tid) {
    constexpr int K32 = TileK / 32;   // 每 tile 的 k32 组数
    constexpr int NS  = K32 * TileN;  // 每项 4 字节
    static_assert(TileK % 32 == 0, "TileK 必须是 32 的倍数");

#pragma unroll
    for (int i = tid; i < NS; i += NumThread) {
        const int   i32 = i / TileN;
        const int   n   = i % TileN;
        const char* src =
            reinterpret_cast<const char*>(Sc) + (static_cast<size_t>(k32_0 + i32) * k32_stride + n0 + n) * 4;
        copy::sm80::cp_async<4>(s_dst + (static_cast<size_t>(i32) * TileN + n) * 4, src);
    }
}

// ---------------------------------------------------------------------------
// 装完整个 stage
//   np_stride : Wpack 的 n 方向步长, 以"每 n 半个元素"为单位 = N/2
//   sc_stride : Scales 的 n 方向步长, 以 K/16 的 4 字节组为单位 = K/16
//
// SWAP_AB 只切换 A 的排列方式; W 与 S 两种路径**完全共用** (因为 swapAB 吃
// 的是同一份 [K/16][N/2][4] 权重, 见 quant.cuh)。用模板参数而不是运行期
// 分支: 两者都是 __forceinline__ 的, 而排列方式是编译期已知的。
// ---------------------------------------------------------------------------
template<int TileM, int TileN, int TileK, int NumThread, bool SWAP_AB>
__device__ __forceinline__ void load_stage(char* sA,
                                           char* sW,
                                           char* sS,
                                           const bf16_t* __restrict__ A,
                                           const int32_t* __restrict__ Wp,
                                           const uint8_t* __restrict__ Sc,
                                           int m_base,
                                           int m_hi,
                                           int n0,
                                           int k0,
                                           int K,
                                           int np_stride,
                                           int sc_stride,
                                           int tid) {
    if constexpr (SWAP_AB) {
        load_A_tile_swapab<TileM, TileK, NumThread>(sA, A, m_base, m_hi, k0, K, tid);
    } else {
        load_A_tile<TileM, TileK, NumThread>(sA, A, m_base, m_hi, k0, K, tid);
    }
    load_W_tile<TileN, TileK, NumThread>(sW, Wp, n0, k0 / 16, np_stride, tid);
    load_S_tile<TileN, TileK, NumThread>(sS, Sc, n0, k0 / 32, sc_stride, tid);
}

}  // namespace sm120
}  // namespace prelogue
}  // namespace atex
