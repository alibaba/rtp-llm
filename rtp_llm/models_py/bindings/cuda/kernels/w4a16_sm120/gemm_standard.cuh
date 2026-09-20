// gemm_standard.cuh — 主 GEMM kernel 模板 (两种形态, 由 SWAP_AB 选择)
//
// ============================================================================
// SWAP_AB = false: 标准形态 (服务 M ∈ [9, 32])
//   D[M,N] = A[M,K] @ W[K,N]^T
//   mma 的 M 维 = 问题的 M (按 TileM 向上取整), 权重走 B fragment
//   (wpack [K/16][N/2][4] int32), 激活走 A fragment (ldmatrix 从 smem 载入)。
//
//   TileM 决定解压复用次数 MI = TileM/16, 于是 TileM=16/32 两份实例化的
//   行为差异全部体现在内层 `for (i < MI)` 循环上:
//       TileM = 16 -> MI = 1: 解压 1 份 B fragment 喂 1 个 mma
//       TileM = 32 -> MI = 2: 解压 1 份喂 2 个 mma
//   后者把纯 ALU 的解压开销摊到 2 倍的张量核工作量上, 这就是 M32 相对 M16 的
//   **全部**差别 —— 不需要另一份代码, 循环多跑一圈即可。
//
// ============================================================================
// SWAP_AB = true: swapAB 形态 (服务 M ∈ [1, 8])
//   mma.m16n8k16 的 M 维下限是 16。若让 M=8 当 M 维就要补 8 行零, 一半算力落空。
//   互换操作数后:
//       D^T[N, M] = W[N, K] @ A[M, K]^T
//   mma 的 M 维 = W 的 N (可以很大), N 维 = 原来的 M (恰好 8, 正好填满 n8 维),
//   零浪费。角色互换有三处后果:
//     1. 权重成为 mma 的 **A 操作数** —— 每 lane 要 8 个元素, 而标准打包的
//        一个 word 只覆盖 2 行 n, 所以读**两个** word。但依然是同一套
//        [K/16][N/2][4] 打包, **不需要另一份权重** (推导见 quant.cuh)。
//     2. 激活成为 mma 的 **B 操作数** —— 直接按 [TileM][TileK] 行主序放在
//        smem, (k, k+1) 相邻两个 bf16 是一个 4 字节单元, 一次 32-bit 读就
//        拼出 bf16x2, 不必过 ldmatrix (整个 k16 只读 2 次, 且被块内所有
//        n16 tile 复用)。
//     3. 累加器是 D^T 的切片, 写回 [M][N] 的 D 需要转置 (每列仅 2 连续字节,
//        无法合并访存)。
//
// ============================================================================
// 为什么两个形态能共用一份代码
//
// 把 N 方向的块宽参数化就能对齐所有索引 —— 这是合并的关键:
//
//     NB = SWAP_AB ? 16 : 8        每 warp 处理的 N 块宽度
//     MI = SWAP_AB ? 1  : TileM/16 复用次数 (swapAB 天然只有 1)
//
// 于是这些式子对**两者都成立** (曾逐个验证):
//     np    = ncol >> 1        权重 word 的 n 下标 (标准: n8*4+(g>>1);
//                              swapAB: n16*8+(g>>1) —— 二者都等于 ncol>>1)
//     boff  = ((i16>>1)*TileN + ncol)*4 + (i16&1)*2    scale 字节偏移
//     m_base= blockIdx.z * TileM   (swapAB 的 grid.z 恒为 1, 自然得 0)
//
// 真正随形态变化只剩三处, 都用 if constexpr 分开 (都是编译期, 无运行期开销):
//     1. A 操作数怎么装 (ldmatrix vs 32-bit 直读)
//     2. 权重 word 怎么解 (B fragment vs A fragment)
//     3. 输出坐标语义 (m 是行 vs m 是列)
//
// 合并前两个文件各有 250~300 行, 归一化后 **55% 的行完全重复** —— 骨架
// (流水线、smem 布局、W/S 加载、launch) 本来就是同一份。
//
// ============================== 关键设计 (两形态共用) ==============================
//   * 解压全程在寄存器内完成, 不经 smem。权重在离线阶段就被重排成 fragment
//     顺序 (见 weight_transform.cu), 于是省掉
//     "解压 -> 写 bf16 回 smem -> ldmatrix 读回" 这 4 倍 smem 流量。
//     小 M 场景权重的复用次数只有 M 次, 这是最有效的优化点。
//   * mma.m16n8k16 的 B fragment 天然分成 b0 (k=0..7) 与 b1 (k=8..15) 两半,
//     而 group size = 8 的 scale 边界正好落在中间 -> 两半各乘自己的 scale,
//     一个 mma 满负载完成, 无需拆解、无需额外累加器。
//   * scale 按 k32 分组存, 所以一个 mma 要的两个 k-group scale 落在**相邻
//     2 字节**, 取 scale 只需一次 2 字节 load (见 prelogue.cuh / common.cuh)。
#pragma once

#include "include/common.cuh"
#include "include/mma/sm80.cuh"
#include "include/quant.cuh"
#include "include/copy/sm75.cuh"
#include "include/copy/sm80.cuh"
#include "include/prelogue.cuh"
#include "include/epilogue.cuh"

namespace atex {
namespace gemm {
namespace sm120 {

using namespace atex::sm120;

// ---------------------------------------------------------------------------
// 调优参数 (RTX PRO 5000, 110 SM, 96 MiB L2, smem/block 99 KiB)
//
// ==================== 标准形态: TileM 16 / 32 ====================
// 两个实例化的取法不同, 原因是寄存器压力:
//   TileM=16 (MI=1): 累加器小, 可以把 TileN 切到 128 换取更多 block
//   TileM=32 (MI=2): 累加器翻倍, TileN 取 256 让每 warp 多分 n8 块
//
// ==================== 关于"加大 TileN 提升带宽"的尝试: 失败 ====================
// 曾经想按 N 大小分宽/窄两档 (大 N 用 TileN=512/Warps=8), 理由是"更大 TileN
// 让每个 block 搬更长的权重连续段"。在修正过的计时口径下复核, 结论是**不成立**:
//
//     宽blk (grid.x*n_split)   宽档/TileN=128 档
//          8 ~ 32                  0.37x ~ 0.68x   (宽档大幅更慢)
//          48 ~ 68                 0.73x ~ 0.92x   (仍更慢)
//         96 及以上                0.94x ~ 1.06x   (基本持平, 无收益)
//   注: 比值 = 窄档耗时 / 宽档耗时, >1 表示宽档更优。
//
// 根源: 这是个**带宽已饱和**的 kernel, 在目标形状上各配置都能贴到 HBM 上限,
// 所以 TileN 只影响能否喂满 SM (并行度), 不影响单 block 效率。既然宽档在
// 任何 block 数下都无法超出窄档, 就没有理由引入第二档 —— 于是**只有一组参数**。
//
// 同理 TileK=64/128 与 Stage=8/16 也只慢不快: smem 总量固定 (100KiB/SM),
// 这本质上还是个占用率 (occupancy) 问题 ——
//     Stage=4, TileN=128 -> per-stage 3.5KiB -> 整块 14KiB -> 7 block/SM
//     Stage=8, TileN=512 -> 每块 52KiB                  -> 1 block/SM
// 靠多 block 并发掩盖访存延迟, 比加深流水更有效。
//
// M > 32 由 TileM=32 的 grid.z 直接覆盖 (多个 z-block 各算 32 行), 不需要
// 再加更大的 TileM —— 那样累加器会随 MI 线性膨胀, 迫使 TileN 反复调小。
// 代价是每个 z-block 都完整读一遍权重, 所以 M 很大时访存会翻
// ceil(M/32) 倍; M ≤ 64 时最多 2 倍, 可接受 (本工程的目标场景是 M ≤ 32)。
//
// ==================== swapAB 形态: TileN 小比大好 (反直觉, 但是实测的) ====================
// swapAB 里 TileN 是 mma 的 **M 维** (权重作 A 操作数), 直觉上应该取大以提高
// 每 block 的计算密度。实测恰恰相反 —— 因为真正的瓶颈是**并行度**:
//
//     TileN=256 -> grid.x = N/256, N=4096 时仅 16 个 block
//                  GPU 有 110 个 SM, 只喂满 15%
//     TileN=64  -> grid.x = N/64,  同样 N 下 64 个 block (4 倍)
//
// 实测 (6 个目标形状 x M=1..8, 每个 (形状,M) 取最优 splitK;
//       rel = 相对该点最优配置的倍数, 越小越好):
//     TN=64  TK=64  -> 平均 37.0us   rel 1.044
//     TN=64  TK=128 -> 平均 37.5us   rel 1.022
//     TN=128 TK=64  -> 平均 38.8us   rel 1.128
//     TN=128 TK=32  -> 平均 40.1us   rel 1.164
//     TN=256 TK=32  -> 平均 43.3us   rel 1.296   <- 原值, 最差
//     TN=256 TK=64  -> 平均 43.6us   rel 1.319
//
// 小形状上的差距最大 (N=4096/M=1: TN=64 是 15.4us, TN=256 是 20.5us, 差 33%),
// 大形状上趋于一致 (N=27648: 75.8 vs 77.1)。这与"并行度不够"完全吻合:
// N 越大 grid.x 越大, TN=256 的饥饿问题自动缓解。
//
// TileK 取 64 (而非 32): 总时间最优。Warps 保持 4 —— TileN=64 时 NB=16,
// 即 NJ = 64/16 = 4 个 n16 块, 恰好每 warp 一个 (NI=1), 累加器只有 4 个 f32,
// 寄存器压力极低。
//
// 曾把 Warps 提到 8 (理由"更多 warp 分担 epilogue 的原子加"), 复核后不成立,
// 已回退 —— W=4 从不比 W=8 差 (实测比值 0.83~1.00)。
// ---------------------------------------------------------------------------
template<int TileM, bool SWAP_AB = false>
struct StandardTuned;

template<>
struct StandardTuned<16, false> {
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 32;
    static constexpr int kWarps = 4;
    static constexpr int kStage = 4;
};
template<>
struct StandardTuned<32, false> {
    static constexpr int kTileN = 256;
    static constexpr int kTileK = 32;
    static constexpr int kWarps = 4;
    static constexpr int kStage = 4;
};
// swapAB: TileM 固定 8 (物理行数), 参数按上面的实测定
template<>
struct StandardTuned<8, true> {
    static constexpr int kTileN = 64;
    static constexpr int kTileK = 64;
    static constexpr int kWarps = 4;
    static constexpr int kStage = 4;
};

// ============================================================================
// 主 GEMM kernel
//   TileM  : 物理行数。SWAP_AB=false 时须为 16 的倍数 (决定 MI = TileM/16);
//            SWAP_AB=true 时固定 8 (mma 的 n8 维)
//   TileN  : N 方向 tile (SWAP_AB=true 时它是 mma 的 M 维)
//   TileK  : K 方向 tile
//   Warps  : 每 block 的 warp 数 (全部沿 N 方向切分)
//   Stage  : cp.async 流水级数
//   ATOMIC : splitK > 1 时把部分和以 bf16 直接累加进 D (见 epilogue.cuh)
//   SWAP_AB: 见文件头。决定 A 的装载方式、权重的解码方式、输出的坐标语义
//
//   grid = (N/TileN, n_split, ceil(M/TileM))
//   grid.z 覆盖 M 方向, 所以 M > TileM 时也能算 (超出的行由 epilogue 的行
//   掩码挡掉)。SWAP_AB=true 时 grid.z 恒为 1 (TileM=8 且 M<=8)。
// ============================================================================
template<int TileM, int TileN, int TileK, int Warps, int Stage, bool ATOMIC, bool SWAP_AB = false>
__global__ void __launch_bounds__(Warps * 32) gemm_standard_kernel(const bf16_t* __restrict__ A,
                                                                   const int32_t* __restrict__ Wp,
                                                                   const uint8_t* __restrict__ Sc,
                                                                   bf16_t* __restrict__ D,
                                                                   int M,
                                                                   int N,
                                                                   int K,
                                                                   int Ks) {
    if constexpr (SWAP_AB) {
        static_assert(TileM == 8, "swapAB 的 TileM 固定为 8 (mma 的 n8 维)");
    } else {
        static_assert(TileM % 16 == 0, "TileM 必须是 16 的倍数");
        static_assert(TileM >= 16, "TileM 至少 16 (mma 的 M 维下限)");
    }
    static_assert(TileK % 16 == 0, "TileK 必须是 16 的倍数");

    // ---- 两形态的统一: N 块宽度与复用次数 ----
    // NB 是每 warp 处理的 N 方向块宽: 标准用 n8 (=8), swapAB 用 n16 (=16)。
    // MI 是"解压一次权重喂几个 mma": 标准 = TileM/16, swapAB 天然只有 1
    // (因为它的 M 维是 N, 与 TileM 无关)。
    constexpr int NB = SWAP_AB ? 16 : 8;
    constexpr int MI = SWAP_AB ? 1 : TileM / 16;

    constexpr int KI      = TileK / 16;   // 每 tile 的 k16 块数
    constexpr int NBN     = TileN / NB;   // N 方向的 NB 块数
    constexpr int NI      = NBN / Warps;  // 每 warp 负责的 NB 块数
    constexpr int Threads = Warps * 32;

    static_assert(NBN % Warps == 0, "TileN/NB 必须能被 Warps 整除");
    static_assert(NI >= 1, "warp 数过多, 每 warp 分不到 NB 块");

    constexpr int A_BYTES = prelogue::sm120::smem_A_stage(TileM, TileK);
    constexpr int W_BYTES = prelogue::sm120::smem_W_stage(TileN, TileK);
    constexpr int S_BYTES = prelogue::sm120::smem_S_stage(TileN, TileK);

    extern __shared__ char smem_raw[];
    char*                  sA = smem_raw;
    char*                  sW = sA + Stage * A_BYTES;
    char*                  sS = sW + Stage * W_BYTES;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int n0     = blockIdx.x * TileN;
    const int sp     = blockIdx.y;
    const int m_base = blockIdx.z * TileM;
    const int k_lo   = sp * Ks;
    const int k_hi   = (k_lo + Ks < K) ? (k_lo + Ks) : K;
    const int niter  = (k_hi - k_lo + TileK - 1) / TileK;

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.f;

    // lane 在 fragment 里的固定坐标。两形态的 g/r 含义相同 (都直接来自 lane):
    //   标准: g = n8 块内行偏移, r 决定列 (m 维)
    //   swapAB: g = n16 块内行 (N 维, 另一行是 g+8), r 决定列 (m 维)
    const int  g     = lane >> 2;
    const int  r     = lane & 3;
    const bool n_odd = (g & 1) != 0;  // 两形态都用它选 word 的奇偶半
    // 标准形态每个 lane 只取一个 word, 由 lane 的高位决定取偶数还是奇数 np
    const int np_off = lane >> 3;

    const int np_stride = N / 2;  // Wpack 的 n 方向步长 = N/2
    const int sc_stride = N;      // Scales 是 [K/32][N][4], 步长 = N

    // ---------------------------------------------------------------- 计算
    auto compute_stage = [&](int st) {
        const char* a = sA + static_cast<size_t>(st) * A_BYTES;
        const char* w = sW + static_cast<size_t>(st) * W_BYTES;
        const char* s = sS + static_cast<size_t>(st) * S_BYTES;

#pragma unroll
        for (int i16 = 0; i16 < KI; ++i16) {
            // ---- 两个操作数: 谁是 A、谁是 B 由形态决定 ----
            //   af : mma 的 A 操作数, 4 个 b32 (标准 = 激活经 ldmatrix;
            //                                       swapAB = 权重解压而来)
            //   bf : mma 的 B 操作数, 2 个 b32 (标准 = 权重解压;
            //                                       swapAB = 激活直读)
            uint32_t af[MI][4];
            uint32_t bf[2];
            if constexpr (SWAP_AB) {
                // 激活是 B 操作数: 直接 32-bit 读拼 bf16x2 (不必过 ldmatrix)。
                // helper 内部按 g*TileK 取行, 所以这里只偏移本 k16 的起始列。
                mma::sm80::load_act_b_frag_swapab(reinterpret_cast<const bf16_t*>(a) + i16 * 16, TileK, lane, bf);
            } else {
                // 激活是 A 操作数: 一次 ldmatrix.x4 拿全 16x16, 每 lane 4 个 b32
#pragma unroll
                for (int i = 0; i < MI; ++i)
                    copy::sm75::ldmatrix<4>(af[i], a + (static_cast<size_t>(i16) * TileM + i * 16) * 32, 32, lane);
            }

#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int nb   = warp * NI + j;  // 本 warp 的第几个 NB 块
                const int ncol = nb * NB + g;    // 该 lane 覆盖的两行 (g 与 g+8)

                // ---- scale: 两形态共用同一式子 ----
                // smem 条目 [i16>>1][n] 是个 4 字节组, 对应一个 k32 = 两个 k16:
                //   字节 0-1 = 本 k32 前半个 k16 的两个 k8 scale
                //   字节 2-3 = 后半个
                // (i16 & 1) * 2 选出属于当前 k16 的那 2 字节, 一次 2 字节 load 就拿全。
                // 行 g+8 在 n 方向 +8 个条目 = +32 字节。
                const uint8_t* sb   = reinterpret_cast<const uint8_t*>(s);
                const size_t   boff = ((static_cast<size_t>(i16) >> 1) * TileN + ncol) * 4 + ((i16 & 1) * 2);
                const uint32_t vsc0 = *reinterpret_cast<const uint16_t*>(sb + boff);

                // ---- 权重 word: 两形态共用索引 np = ncol >> 1 ----
                //   标准 : np = n8*4 + (g>>1)   = (n8*8  + g)>>1 = ncol>>1
                //   swapAB: np = n16*8 + (g>>1) = (n16*16 + g)>>1 = ncol>>1
                const uint32_t* wrow =
                    reinterpret_cast<const uint32_t*>(w + static_cast<size_t>(i16) * (TileN / 2) * 16);

                if constexpr (SWAP_AB) {
                    // 权重是 A 操作数: 每个 lane 要 8 个元素, 一个 word 只覆盖 2 行 n,
                    // 所以读**两个** word。行 n=g 与 n=g+8 同奇偶 (g 与 g+8 同奇偶),
                    // 共用一次 parity 移位; 跨 8 行要跳 4 个 np (np -> n 是 2np/2np+1)。
                    const int      np_lo  = ncol >> 1;
                    const uint32_t w_lo   = wrow[np_lo * 4 + r];
                    const uint32_t w_hi   = wrow[(np_lo + 4) * 4 + r];
                    const uint32_t vsc_g8 = *reinterpret_cast<const uint16_t*>(sb + boff + 32);
                    quant::dequant_a_frag_swapab_e4m3(w_lo, w_hi, n_odd, vsc0, vsc_g8, af[0]);
                    // MI 恒为 1, 解压一次喂一个 mma
                    mma::sm80::mma_m16n8k16_bf16(acc[0][j], af[0], bf, acc[0][j]);
                } else {
                    // 权重是 B 操作数: 每 lane 一个 word。lane 16..31 取奇数 np
                    // (np 的奇偶 = 它覆盖的那两个 n 的奇偶)。
                    const int      np = nb * 4 + np_off;
                    const uint32_t w0 = wrow[np * 4 + r];
                    quant::dequant_b_frag_e4m3(w0, n_odd, vsc0, bf);
                    // 解压一次, 复用 MI 次 —— MI = TileM/16 就是这里的关键
#pragma unroll
                    for (int i = 0; i < MI; ++i)
                        mma::sm80::mma_m16n8k16_bf16(acc[i][j], af[i], bf, acc[i][j]);
                }
            }
        }
    };

    // ---------------------------------------------------------------- 流水
    // 等待深度按公式算, 不能用随 it 递增的计数器:
    // 流水线未填满时 (niter < Stage) 固定用 Stage-1 会等待过松, 读到未就绪的
    // smem —— 表现为大 K / 大网格时的 NaN, 而小规模测试不会复现。
    const int npre = (niter < Stage) ? niter : Stage;
    for (int it = 0; it < npre; ++it) {
        prelogue::sm120::load_stage<TileM, TileN, TileK, Threads, SWAP_AB>(sA + static_cast<size_t>(it) * A_BYTES,
                                                                           sW + static_cast<size_t>(it) * W_BYTES,
                                                                           sS + static_cast<size_t>(it) * S_BYTES,
                                                                           A,
                                                                           Wp,
                                                                           Sc,
                                                                           m_base,
                                                                           M,
                                                                           n0,
                                                                           k_lo + it * TileK,
                                                                           K,
                                                                           np_stride,
                                                                           sc_stride,
                                                                           tid);
        copy::sm80::cp_commit();
    }

    for (int it = 0; it < niter; ++it) {
        copy::sm80::cp_wait_pending(copy::sm80::cp_wait_depth(npre, niter, it));
        __syncthreads();

        compute_stage(it % Stage);
        __syncthreads();  // 该 slot 可以复用了

        if (it + Stage < niter) {
            prelogue::sm120::load_stage<TileM, TileN, TileK, Threads, SWAP_AB>(
                sA + static_cast<size_t>(it % Stage) * A_BYTES,
                sW + static_cast<size_t>(it % Stage) * W_BYTES,
                sS + static_cast<size_t>(it % Stage) * S_BYTES,
                A,
                Wp,
                Sc,
                m_base,
                M,
                n0,
                k_lo + (it + Stage) * TileK,
                K,
                np_stride,
                sc_stride,
                tid);
            copy::sm80::cp_commit();
        }
    }

    // ---------------------------------------------------------------- 输出
    // 两形态的累加器布局相同, 但坐标语义相反:
    //   标准 : m 是行 -> 传 (m_base + 行块, 列起点)
    //   swapAB: m 是列 -> m 坐标全由 lane 决定, 所以 m_base 传 0 (即 m_base
    //           本身, 因为 SWAP_AB 时 grid.z 恒为 1 -> m_base 恒为 0);
    //           N 维才是行, 块起点 + lane 的 g/g+8
    // ATOMIC 为真时把部分和以 bf16 累加进 D (D 已在启动前清零); 否则覆盖写。
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const int nb0 = n0 + (warp * NI + j) * NB;
        if constexpr (SWAP_AB) {
            if (ATOMIC) {
                eplogue::sm120::store_atomic_bf16_transposed(D, M, N, nb0, lane, acc[0][j]);
            } else {
                eplogue::sm120::store_transposed(D, M, N, nb0, lane, acc[0][j]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int mrow = m_base + i * 16;
                if (ATOMIC) {
                    eplogue::sm120::store_atomic_bf16(D, M, N, mrow, nb0, lane, acc[i][j]);
                } else {
                    eplogue::sm120::store_direct(D, M, N, mrow, nb0, lane, acc[i][j]);
                }
            }
        }
    }
}

// ============================================================================
// Host 侧启动
// ============================================================================
template<int TileM, int TileN, int TileK, int Warps, int Stage, bool SWAP_AB = false>
static inline void launch_standard(const bf16_t*  A,
                                   const int32_t* Wp,
                                   const uint8_t* Sc,
                                   bf16_t*        D,
                                   int            M,
                                   int            N,
                                   int            K,
                                   int            Ks,
                                   bool           atomic_out,
                                   cudaStream_t   st) {
    constexpr int SMEM    = Stage * prelogue::sm120::smem_per_stage(TileM, TileN, TileK);
    constexpr int Threads = Warps * 32;

    if (SMEM > 48 * 1024) {
        cudaFuncSetAttribute(gemm_standard_kernel<TileM, TileN, TileK, Warps, Stage, true, SWAP_AB>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM);
        cudaFuncSetAttribute(gemm_standard_kernel<TileM, TileN, TileK, Warps, Stage, false, SWAP_AB>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM);
    }

    const int n_split = (K + Ks - 1) / Ks;
    // swapAB 的 TileM=8 覆盖全部 M<=8, grid.z 恒为 1
    const dim3 grid((N + TileN - 1) / TileN, n_split, SWAP_AB ? 1 : (M + TileM - 1) / TileM);
    if (atomic_out) {
        gemm_standard_kernel<TileM, TileN, TileK, Warps, Stage, true, SWAP_AB>
            <<<grid, Threads, SMEM, st>>>(A, Wp, Sc, D, M, N, K, Ks);
    } else {
        gemm_standard_kernel<TileM, TileN, TileK, Warps, Stage, false, SWAP_AB>
            <<<grid, Threads, SMEM, st>>>(A, Wp, Sc, D, M, N, K, Ks);
    }
}

}  // namespace sm120
}  // namespace gemm
}  // namespace atex
