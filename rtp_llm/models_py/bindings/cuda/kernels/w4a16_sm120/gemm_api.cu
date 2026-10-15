// gemm_api.cu — 统一 C 入口 (对外唯一接口)
//
// ==================== 按 M 分发 ====================
//     M ∈ [1, 8]  -> swapAB 路径 (零张量核算力浪费)
//     M ∈ [9, 16] -> 标准布局, TileM=16 (MI=1)
//     M ∈ [17,32] -> 标准布局, TileM=32 (MI=2, 解压复用 2 次)
//     M > 32      -> 同上走 TileM=32, 由 grid.z 分多个 block 覆盖 M 方向
//
// M > 32 没有专门调优 (没做更大的 TileM): 累加器会随 MI 线性膨胀, 迫使
// TileN 反复调小, 不划算。代价是每个 z-block 都要完整读一遍权重, M=64 时
// 权重访存翻倍。本工程的主场景是 M ≤ 32, 大 M 只保证能跑对。
//
// ==================== 关于调参 (TileN/TileK/Stage/Warps) ====================
// 结论是**不动默认值**。曾尝试过两件事, 在修正过的计时口径下复核都不成立,
// 已回退:
//
//   * 按 N 大小分"宽/窄"两档 (大 N 用 TileN=512/Warps=8, 期望更长的权重
//     连续段)。实测宽档**在任何 block 数下都不优于**默认档: block 少时
//     慢 2~3 倍 (0.37x~0.68x), block 多时持平 (0.94x~1.06x)。
//   * 把 swapAB 的 Warps 提到 8。实测 W=4 从不比 W=8 差 (0.83~1.00)。
//
// 两者的共同根源: 这个 kernel 在目标形状上已经**贴住 HBM 上限**, 瓶颈是
// 权重流量而非并行度或流水深度, 所以 TileN/Stage/TileK 怎么调都是"要么
// 持平、要么因为 block 数变少而变慢"。详见 gemm_standard.cuh 里调优参数
// 处的实测数据。
//
// 因此这里**没有任何运行期算法选择** —— 三个入口各只有一组编译期参数。
//
// 附: 早期那版"按 N 与 splitK 选宽窄档"的分派代码已删除, 它建立在错误的
// 测量上 —— 当时的 cold 计时把 L2 冲刷写进计时区间再减掉, 而减数 (828us)
// 比被测量 (十几 us) 大一个量级, 误差被放大到报出 2541 GB/s 这种超过 HBM
// 上限的值。
//
// ==================== splitK: 没有第二阶段的 reduce ====================
// 各 split 把部分和**以 bf16 直接累加进 D** (red.global.add.noftz.bf16x2),
// 所以这里既不需要 fp32 workspace, 也不需要单独的 reduce kernel ——
// 整个 splitK 归约只剩「一次 GEMM 启动」。
//
// 代价是 D 现在既是输出也是累加目标, 所以 n_split > 1 时它必须先清零。
// 这一步是 M*N*2 字节的写, 比原来「写 n_split 份 fp32 再读回来」省得多。
//
// 本文件是唯一使用 kernel 的地方, 所以它直接 include 模板头并在调用处实例化
// (见下面的 launch_standard<...>) —— 不需要额外的启动函数 wrapper, 也不需要
// 一个只做显式实例化的 .cu。
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#include "gemm_standard.cuh"

// kernel 模板在 atex::gemm::sm120 里; 本文件是它的唯一调用方, 直接引入名字。
using atex::gemm::sm120::launch_standard;
using atex::gemm::sm120::StandardTuned;

namespace {

using bf16_t = __nv_bfloat16;

// ---------------------------------------------------------------------------
// 按 (TileM, SWAP_AB) 实例化并启动 —— 把两件事收在一处:
//   1. 从 StandardTuned<TileM, SWAP_AB> 取出该形态的调优参数
//   2. 把对外接口的 void* 转成 kernel 要的具体类型
// 不加这层就要在三个分发分支里把上面两件事各抄一遍 (6 个 cast + 5 个模板实参)。
//
// 注意它**不是**导出的启动函数: 没有 extern "C", 返回 void, 启动失败由调用方
// 的 cudaGetLastError 统一报告 —— 所以不会像早先那版 wrapper 一样, 制造出
// "检查一个恒为 0 的返回值" 那种死逻辑。
// ---------------------------------------------------------------------------
template<int TileM, bool SWAP_AB>
inline void launch_tuned(const void*  A,
                         const void*  Wp,
                         const void*  Sc,
                         void*        D,
                         int          M,
                         int          N,
                         int          K,
                         int          Ks,
                         bool         atomic_out,
                         cudaStream_t st) {
    using T = StandardTuned<TileM, SWAP_AB>;
    launch_standard<TileM, T::kTileN, T::kTileK, T::kWarps, T::kStage, SWAP_AB>(static_cast<const bf16_t*>(A),
                                                                                static_cast<const int32_t*>(Wp),
                                                                                static_cast<const uint8_t*>(Sc),
                                                                                static_cast<bf16_t*>(D),
                                                                                M,
                                                                                N,
                                                                                K,
                                                                                Ks,
                                                                                atomic_out,
                                                                                st);
}

// 把 D 清零。用 4 字节 (bf16x2) 粒度写, 比逐元素快一倍。
// 只在 n_split > 1 时调用。
__global__ void zero_bf16_kernel(bf16_t* __restrict__ p, int n2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2)
        reinterpret_cast<uint32_t*>(p)[i] = 0u;
}

// ============================================================================
// splitK 的自动选择
//
// 各 split 把部分和以 bf16 直接累加进 D (见下), 所以 splitK 只影响**并行度**:
// 不切时只有 grid.x*grid.z 个 block, 可能喂不满 110 个 SM; 切了就有 n_split 倍。
// 但切分本身有代价 (流水重启 + 原子累加竞争 + D 要先清零), 所以切不够会饿、
// 切过头会亏。
//
// ---------------------------------------------------------------------------
// 规则: 让不切分时的 block 数补到 ~192 个所需的最小 splitK, 但不把 K 切碎
//
//     for sk in {1,2,4,8,16}:            # 只考虑 Ks >= 256 的候选
//         if base_blocks * n_split(sk) >= 192: return sk
//     return 满足 Ks >= 256 的最大 sk; 若没有则 1
//
// 其中 base_blocks = grid.x * grid.z 是**完全不切分时**的 block 数。
//
// 两个常数各有实测依据:
//   * 192  ≈ 110 个 SM 的 1.75 倍。略高于一块 GPU 的容量, 让最后一波也能填满,
//           同时不至于把 K 切得太碎。扫 64..2048 时它同时是最优的总时间与
//           最小的最差单点。
//   * 256  是每段 K 的下限 (Ks, 已含 128 对齐)。**没有它时小形状会崩**:
//           N=1024/K=2048/M=8 会被切到 Ks=128, 实测比最优慢 1.80x。加了这条
//           下限后降到 1.52x, 而目标形状的性能分毫未动 (仍 1.0098x)。
//           物理上: 每段 K 都要走一遍预取-计算流水, 段太短会放大流水启动
//           开销与原子累加竞争。
//
// ---------------------------------------------------------------------------
// 实测 (11 个目标形状 x **M=1..32 全部** = 352 个点; cold 口径见 bench_speedup.py):
//
//     sk 恒为 1 (调参前的行为)   总 38444us   比自动慢 2.47x
//     本规则 (自动)              总 15555us   --
//     每个点各自扫描最优 (上界)   总 15319us   自动只差 1.015x
//
// 也就是**离理论上界 1.5%**, 同时相对旧行为拿到 2.47x。逐形状看, 收益主要
// 来自 M 较大的点 (权重读占绝对主导时切分最划算):
//
//     N=4096  K=4096   自动/oracle 1.016x   sk1/自动 3.03x
//     N=4096  K=12288  自动/oracle 1.001x   sk1/自动 4.05x
//     N=5120  K=27648  自动/oracle 0.999x   sk1/自动 3.92x
//     N=17408 K=5120   自动/oracle 1.065x   sk1/自动 1.29x   <- 最差的一档
//     N=27648 K=5120   自动/oracle 1.024x   sk1/自动 1.06x
//
// 最差的一档是 N=17408/K=5120 (自动/oracle 1.065x): base=68 时规则选 sk=4
// 得 65.5us, 而 sk=8 是 57.3us —— 已复测 3 次确认可复现, 不是噪声。
//
// 另外用 9 个小形状 (N,K 从 256 到 4096) 做了稳健性检查: 联合 140 个点的总
// 时间在 1.036x 以内。这些形状不在目标范围内, 只要求不被切坏 ——
// 正是它们迫使加上了下面那条 Ks 下限。
//
// ---------------------------------------------------------------------------
// 为什么用这个"笨"规则, 而不是某个更物理的模型
//
// 试过 5 种更"有原理"的形式, **全部更差**, 记在这里避免重复尝试:
//
//     本规则 (floor=192)        目标 3289us  1.0098x  最差 1.14x
//     floor 按 M 分档 (192/896/576)   3275us  1.0053x  最差 1.20x  <- 总时间
//                                                更低但最差单点更差, 不值
//     "单波填满则不切" + floor         3322us  1.0199x  最差 1.22x
//     波次模型 ceil(W*nsp/S)/nsp       3378us  1.0371x  最差 1.18x
//     二参数拟合 + C*nsp 开销项        3374us  1.0358x  最差 1.40x
//
// 波次模型之所以失败: 它预测"base>=SM 时不该切", 但实测 N=17408/M=32
// (base=68) 在 sk=8 (544 blocks) 才最优, 而 N=27648/M=32 (base=108) 在
// sk=2 就最优 —— 同样是 M=32、权重体积也在同一量级, 最优切分深度却差 4 倍,
// 说明主导因素不是简单的 block 数或波次取整。既然模型拟合不出, 就用实测
// 最优的简单规则。
//
// 注: M > 32 时 grid.z >= 2 会额外提供并行度, base_blocks 已把它算进去,
// 所以大 M 会更倾向于不切 —— 这与"M 越大越不需要 splitK 补并行"的直觉一致。
// (大 M 没有专门调优, 这里只求不劣化。)
// ============================================================================
constexpr int kSplitTargetBlocks = 192;  // 希望达到的 block 数
constexpr int kSplitMinKs        = 256;  // 每段 K 的下限 (含 128 对齐)

// 不考虑 splitK 时的 block 数 = grid.x * grid.z。
// TileN 与 grid.z 的取法必须与下面各 launch 里的 grid 构造一致:
//     M <= 8  -> swapAB,  TileN=64
//     M <= 16 -> 标准,    TileN=128
//     else    -> 标准,    TileN=256, grid.z = ceil(M/32)
inline int base_blocks(int M, int N) {
    if (M <= 8)
        return (N + 63) / 64;
    if (M <= 16)
        return (N + 127) / 128;
    return ((N + 255) / 256) * ((M + 31) / 32);
}

// 给定 splitK 时的实际切分数。必须与主入口里的对齐逻辑一致 ——
// Ks 先对齐到 128 (保证 cp.async 16B), 对齐会让实际切分数可能小于请求值。
inline int n_split_of(int K, int splitK) {
    int ks = (K + splitK - 1) / splitK;
    ks     = ((ks + 127) / 128) * 128;
    return (K + ks - 1) / ks;
}

// 每段 K 的长度 (含 128 对齐)。与 n_split_of 用的是同一个 ks。
inline int k_seg_of(int K, int splitK) {
    int ks = (K + splitK - 1) / splitK;
    return ((ks + 127) / 128) * 128;
}

inline int pick_splitk(int M, int N, int K) {
    const int base    = base_blocks(M, N);
    int       last_ok = 0;  // 满足 Ks 下限的最大候选
    for (int sk = 1; sk <= 16; sk *= 2) {
        if (k_seg_of(K, sk) < kSplitMinKs)
            break;  // 再切就更碎了, 后面都不合适
        last_ok = sk;
        if (base * n_split_of(K, sk) >= kSplitTargetBlocks)
            return sk;
    }
    // 没有候选能满足 block 目标 (或全被 Ks 下限挡掉): 取最接近的
    return last_ok > 0 ? last_ok : 1;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// 主入口
//   A    : [M, K] bf16
//   Wp   : 权重 [K/16][N/2][4] int32 —— **只有这一种布局**, 任何 M 都用它
//   Sc   : [K/32][N][4] uint8, e4m3 存的 pow2 scale (见 common.cuh)
//   D    : [M, N] bf16 输出。n_split > 1 时会被预清零再由各 split 累加,
//          所以调用方不必自己清零, 但原有内容会被覆盖。
//   splitK : K 方向切分数。**传 0 表示自动选择** (推荐, 见 pick_splitk);
//            传正数则强制该值 —— 仅供测试用来覆盖各条 splitK 代码路径。
//   返回: 0 成功, 1 M 超范围, 2 权重为空, 3 参数非法, 4 启动失败
//
// 关于权重布局: 曾经为 M <= 8 的 swapAB 路径另备了一份 "A fragment 顺序" 的
// 权重 (WpAb), 使调用方要按 M 选布局。现已取消 —— swapAB 的 A fragment 可以
// 直接从标准 word 里按索引取出来 (每个 lane 读 2 个 word), 见 quant.cuh。
// 于是对外只剩一种权重, 与 M 无关。
int w4a16_gemm_splitk_sm120(
    const void* A, const void* Wp, const void* Sc, void* D, int M, int N, int K, int splitK, void* stream) {
    if (M < 1 || N < 1 || K < 1 || D == nullptr || Wp == nullptr)
        return 3;
    // M 无上限。最大的 TileM 是 32, 超出的行由 grid.z 分块覆盖
    // (epilogue 的行掩码负责边界)。M <= 32 时 grid.z 恒为 1。
    if (M > 32 * 1024)
        return 1;  // 兜底, 防止 grid.z 溢出 int
    // N 需能被 16 整除 (ldmatrix / 权重打包); K 需能被 32 整除, 因为 scale 的
    // 4 字节分组对应一个 k32 —— K 是 16 的倍数但非 32 的倍数时, 末组的高
    // 2 字节会越界。
    if (N % 16 != 0 || K % 32 != 0)
        return 3;

    if (splitK <= 0)
        splitK = pick_splitk(M, N, K);  // 0 = 自动
    int Ks            = (K + splitK - 1) / splitK;
    Ks                = ((Ks + 127) / 128) * 128;  // 对齐到 128, 保证 cp.async 16B
    const int n_split = (K + Ks - 1) / Ks;

    auto st = static_cast<cudaStream_t>(stream);

    // n_split > 1 时 D 被多路 split 累加, 必须先清零。
    // n_split == 1 时是直接覆盖写, 不需要。
    if (n_split > 1) {
        const int n2  = M * N / 2;  // 以 bf16x2 (4B) 为单位
        const int BLK = 256;
        zero_bf16_kernel<<<(n2 + BLK - 1) / BLK, BLK, 0, st>>>(static_cast<bf16_t*>(D), n2);
    }

    // ---- 按 M 分发: 三条实例化都是同一份模板, 只差 (TileM, SWAP_AB) ----
    // 用下面的 launch_tuned 直接实例化。它只是把"从 StandardTuned 取调优参数"
    // 与 6 个 void* -> 具体类型的 cast 收在一处 (否则要复制三遍) —— 不是导出的
    // 启动函数, 只是本文件的局部设施。
    if (M <= 8) {
        // swapAB: 权重当 A 操作数。用的还是同一份 Wp (读法不同而已)。
        launch_tuned<8, true>(A, Wp, Sc, D, M, N, K, Ks, n_split > 1, st);
    } else if (M <= 16) {
        launch_tuned<16, false>(A, Wp, Sc, D, M, N, K, Ks, n_split > 1, st);
    } else {
        launch_tuned<32, false>(A, Wp, Sc, D, M, N, K, Ks, n_split > 1, st);
    }

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("w4a16_gemm: CUDA error: %s\n", cudaGetErrorString(e));
        return 4;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 自省: 返回 **自动选择** (splitK=0) 会用的切分数。
//
// 用途有二:
//   1. 基准/调优时能**直接读回**实际选择, 而不是靠重算或猜 —— 这样验证
//      启发式与实现是否一致只需一次调用。
//   2. 线上 profiling 时想知道某个形状走了多深的切分, 不必改 kernel。
// 返回 0 表示参数非法 (与主入口的校验一致)。
// ---------------------------------------------------------------------------
int w4a16_gemm_pick_splitk(int M, int N, int K) {
    if (M < 1 || N < 1 || K < 1)
        return 0;
    if (M > 32 * 1024)
        return 0;
    if (N % 16 != 0 || K % 32 != 0)
        return 0;
    return pick_splitk(M, N, K);
}

}  // extern "C"
