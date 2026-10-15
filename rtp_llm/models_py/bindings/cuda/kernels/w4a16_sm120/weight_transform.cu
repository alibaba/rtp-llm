// weight_transform.cu — 权重的离线转换 (int4 量化 + fragment 重排) 与逆变换
//
// ============================== 转换流程 ==============================
//   W [N, K] bf16
//     │
//     ├─(1) 量化: group = 8, 有符号 int4 (值域 [-8, 7])
//     │      scale = 2^ceil(log2(amax / 7))  -> 2 的幂, 能被 e4m3 无损存
//     │
//     └─(2) 重排成 tensor-core fragment 顺序
//             标准布局 (M >= 9): B fragment -> Wp  [K/16][N/2][4]   int32
//             swapAB 布局 (M <= 8): A fragment -> Wp2 [K/16][N/16][32] int32
//
//   Sc [K/32][N][4] uint8 = e4m3 存的 scale (尾数位为 0, 见 common.cuh)
//      一个 4 字节组对应一个 k32 (即两个 k16), 字节 0-1 是前半个 k16 的
//      两个 k8 scale, 字节 2-3 是后半个。这样加载是纯 4 字节拷贝,
//      且 GEMM 内层一次 2 字节读就能拿到一个 mma 需要的两个 k-group scale。
//
// ============================== 逆变换 ==============================
//   反量化: nibble 乘回 scale 即得到 W。
//   注意: 这只是**给测试用的参考路径**, GEMM 本身不需要它。
//
// ============================ 设计约束 ============================
//   * 本文件**不使用任何内部持久缓冲**。工作区 (N*K 字节的 nibble 暂存)
//     完全由调用方通过 qbuf 参数提供, 避免 .so 内部维护跨调用的静态状态。
//   * 本文件**不做任何设备同步** (不调用 cudaDeviceSynchronize)。
//     nvcc --shared 静态链接独立的 cudart, 与调用方 (torch) 的不是同一实例;
//     在 .so 内同步会作用在错误的 runtime 状态上。同步交给调用方。
#include "include/common.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace atex {
namespace wconvert {
namespace sm120 {

using namespace atex::sm120;

constexpr int GS  = 8;   // 量化 group size
constexpr int BNT = 32;  // 每 block 处理的行数
constexpr int K16 = 16;  // 每线程处理的 K 方向块大小 (= 一个 k16)

// ---------------------------------------------------------------------------
// nibble 位置查表 (与 quant.cuh 的 dequant 严格对应)
//   s = 该 nibble 在 uint32 里的 bit 组序号 (0..7)
//   用 constexpr 函数而非数组: 数组在 device 代码里是外部链接符号,
//   --shared 构建时可能未注册; 函数形式能被完全内联成常量。
//   等价于 c_k_off = {0,8,0,8,1,9,1,9}, c_n_odd = {0,0,1,1,0,0,1,1}
// ---------------------------------------------------------------------------
__device__ __forceinline__ constexpr int nib_k_off(int s) {
    return ((s & 1) ? 8 : 0) + ((s & 4) ? 1 : 0);
}
__device__ __forceinline__ constexpr int nib_n_odd(int s) {
    return (s & 2) ? 1 : 0;
}

// ============================================================================
// Kernel 1: 量化
//   grid: x = n 方向块 (BNT 行), y = k16 块
//   每线程独立处理一个 (n, k16) 行对: 读 16 个 k -> 两个 group 各自
//   定 scale 并量化 -> 写 qbuf 与 Sc。
//
//   没有 __syncthreads, 所以边界线程可以直接 return (不会死锁)。
//
//   scale 写入: Sc 是 [K/32][N][4] 的 uint8 视图, 这里按字节写 ——
//   一个线程负责 (n, k16), 恰好就是该 4 字节组里的一半, 所以直接写 2 字节。
// ============================================================================
__global__ void
quantize_kernel(const bf16_t* __restrict__ W, uint8_t* __restrict__ qbuf, uint8_t* __restrict__ Sc, int N, int K) {
    const int n = blockIdx.x * BNT + threadIdx.x;
    if (n >= N)
        return;

    const int k16 = blockIdx.y * K16;

    float         x[K16];
    const bf16_t* src = W + static_cast<size_t>(n) * K + k16;
#pragma unroll
    for (int t = 0; t < K16; ++t)
        x[t] = __bfloat162float(src[t]);

    // ---- 两个 group 各自定 scale 并量化 ----
    uint8_t q[K16];
    uint8_t sc2[2];
#pragma unroll
    for (int gi = 0; gi < K16 / GS; ++gi) {
        float amax = 1e-12f;
#pragma unroll
        for (int t = 0; t < GS; ++t)
            amax = fmaxf(amax, fabsf(x[gi * GS + t]));
        // scale = 2^ceil(log2(amax/7)): 商必落在 [-8,7] 内, 且 2 的幂能被 e4m3
        // 无损表示 (尾数位恒 0), 又能被 bf16 精确表示。
        // clamp 到 [2^-6, 2^8] -> e4m3 指数 E = e+7 ∈ [1,15], 恒为正规数。
        const float e     = ceilf(log2f(amax / 7.0f));
        const float ec    = fminf(fmaxf(e, -6.0f), 8.0f);
        const float scale = exp2f(ec);
        sc2[gi]           = static_cast<uint8_t>(scale_exp_to_e4m3(static_cast<int>(ec)));

        const float inv = 1.0f / scale;
#pragma unroll
        for (int t = 0; t < GS; ++t) {
            const float v  = fminf(fmaxf(roundf(x[gi * GS + t] * inv), -8.0f), 7.0f);
            q[gi * GS + t] = static_cast<uint8_t>(static_cast<int>(v) + 8);
        }
    }

    // ---- 写 Sc: [K/32][N][4], 当前 (n, k16) 落在后半个或前半个 4 字节组 ----
    uint16_t* sc_dst =
        reinterpret_cast<uint16_t*>(Sc) + (static_cast<size_t>(k16 / 32) * N + n) * 2 + ((k16 % 32) / 16);
    *sc_dst = static_cast<uint16_t>(sc2[0] | (sc2[1] << 8));

    // ---- 写 qbuf (自然序 [N][K], pack 阶段做 gather) ----
    uint8_t* dst = qbuf + static_cast<size_t>(n) * K + k16;
#pragma unroll
    for (int t = 0; t < K16; ++t)
        dst[t] = q[t];
}
// ============================================================================
// Kernel 2a: 打包成标准布局 (B fragment) -> Wp [K/16][N/2][4] int32
//   一个 word 由 (k16, np, r) 唯一确定, 覆盖 n ∈ {2np, 2np+1} 与
//   k ∈ {2r, 2r+1, 2r+8, 2r+9}
// ============================================================================
__global__ void pack_std_kernel(const uint8_t* __restrict__ qbuf, int32_t* __restrict__ Wp, int N, int K) {
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (K / 16) * (N / 2) * 4;
    if (idx >= total)
        return;

    const int r   = idx & 3;
    const int np  = (idx >> 2) % (N / 2);
    const int k16 = idx / ((N / 2) * 4);

    uint32_t word = 0;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int      n   = 2 * np + nib_n_odd(s);
        const int      k   = k16 * 16 + 2 * r + nib_k_off(s);
        const uint32_t nib = qbuf[static_cast<size_t>(n) * K + k] & 0xFu;
        word |= nib << (4 * s);
    }
    Wp[idx] = static_cast<int32_t>(word);
}

// ============================================================================
// Kernel 3: 逆变换 (Wp, Sc) -> W [N, K] bf16   (仅供往返测试)
//
//   打包时的对应关系 (同一 word, 由整体右移 8 位区分 n 的奇偶):
//     bits  0-3  -> k = 2r       bits  4-7  -> k = 2r + 8
//     bits 16-19 -> k = 2r + 1   bits 20-23 -> k = 2r + 9
//   与 quant.cuh 的 dequant_b_frag_e4m3 读法完全一致。
//   ============================================================================
__global__ void
inverse_kernel(const int32_t* __restrict__ Wp, const uint8_t* __restrict__ Sc, bf16_t* __restrict__ W, int N, int K) {
    const int n   = blockIdx.x * blockDim.x + threadIdx.x;
    const int k16 = blockIdx.y;
    if (n >= N)
        return;

    const int      np    = n / 2;
    const int      shift = (n & 1) ? 8 : 0;
    const int32_t* row   = Wp + static_cast<size_t>(k16) * (N / 2) * 4;

    constexpr int k_bit[4] = {0, 4, 16, 20};
    constexpr int k_off[4] = {0, 8, 1, 9};

    float x[K16];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint32_t word = static_cast<uint32_t>(row[np * 4 + r]) >> shift;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int nib       = static_cast<int>((word >> k_bit[t]) & 0xFu);
            x[2 * r + k_off[t]] = static_cast<float>(nib - 8);
        }
    }

    // 乘回 scale: 从 [K/32][N][4] 里取出本 k16 的那 2 字节 e4m3。
    // 注意这里的 k16 是**块索引**, 而一个 k32 含 2 个 k16 块 —— 所以
    // k32 索引 = k16/2, 在 k32 内是前半还是后半 = k16%2。
    // (transform 侧的变量 k16 是元素坐标, 那里用 /32 是对的, 别照抄。)
    const uint8_t* sc_base = Sc + (static_cast<size_t>(k16 / 2) * N + n) * 4 + (k16 % 2) * 2;
#pragma unroll
    for (int gi = 0; gi < K16 / GS; ++gi) {
        const float s = scale_e4m3_to_f32(sc_base[gi]);
#pragma unroll
        for (int t = 0; t < GS; ++t)
            x[gi * GS + t] *= s;
    }

    bf16_t* dst = W + static_cast<size_t>(n) * K + k16 * K16;
#pragma unroll
    for (int t = 0; t < K16; ++t)
        dst[t] = __float2bfloat16(x[t]);
}

}  // namespace sm120
}  // namespace wconvert
}  // namespace atex

// ============================================================================
// 对外 C 接口 (供 ctypes 调用)
// ============================================================================
extern "C" {

// 转换: W [N,K] bf16 -> (Wp, Sc)
//   qbuf  : 调用方提供的 nibble 暂存区, 至少 N*K 字节 (不能为 nullptr)
//   Sc    : [K/32][N][4] uint8 (e4m3 存的 pow2 scale), 至少 (K/32)*N*4 字节
//   Wp    : [K/16][N/2][4] int32, 至少 (K/16)*(N/2)*4 个元素
//   返回 0 成功, 1 shape 不合法, 2 缺少 qbuf
//
// 只有一种权重布局。M <= 8 的 swapAB 路径直接复用这份 Wp (按不同索引读),
// 所以不需要为它另打一份 —— 见 quant.cuh。
int w4a16_weight_transform(const void* W, int N, int K, void* Wp, void* Sc, void* qbuf, void* stream) {
    using namespace atex::wconvert::sm120;
    using atex::sm120::bf16_t;

    // K 必须是 32 的倍数: Sc 按 k32 分组存 (一个 4 字节组), K 只是 16 的倍数时
    // 最后那组的高 2 字节会落在缓冲之外。
    if (K % 32 != 0 || N % 16 != 0)
        return 1;
    if (qbuf == nullptr || Sc == nullptr)
        return 2;

    auto  st = static_cast<cudaStream_t>(stream);
    auto* w  = static_cast<const bf16_t*>(W);
    auto* sc = static_cast<uint8_t*>(Sc);
    auto* qb = static_cast<uint8_t*>(qbuf);

    const dim3 g1((N + BNT - 1) / BNT, K / K16);
    quantize_kernel<<<g1, BNT, 0, st>>>(w, qb, sc, N, K);

    if (Wp) {
        const int total = (K / 16) * (N / 2) * 4;
        pack_std_kernel<<<(total + 255) / 256, 256, 0, st>>>(qb, static_cast<int32_t*>(Wp), N, K);
    }
    return 0;
}

// 逆变换: (Wp_std, Sc) -> W [N,K] bf16
int w4a16_weight_inverse(const void* Wp_std, const void* Sc, int N, int K, void* W, void* stream) {
    using namespace atex::wconvert::sm120;
    using atex::sm120::bf16_t;

    if (K % 32 != 0 || N % 2 != 0)
        return 1;
    const dim3 g((N + 255) / 256, K / K16);
    inverse_kernel<<<g, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int32_t*>(Wp_std), static_cast<const uint8_t*>(Sc), static_cast<bf16_t*>(W), N, K);
    return 0;
}

// ---------------------------------------------------------------------------
// 仅打包 (不量化): 用调用方**已有的** qbuf 生成 fragment 布局。
//
// 用途: 测试时可以把"量化"替换成 host 侧模拟的结果, 从而只验证 kernel 里的
// ops (打包 -> 反量化 -> mma) 是否正确, 不与量化实现的浮点细节耦合。
//
// 例如 host 侧模拟: 直接令 scale = 某个固定 pow2, nibble = 随机 0..15,
// 则 GEMM 输出应当精确等于 (A @ dequant(nibble)*scale)。
// 这样绕开了 log2/ceil 在 pow2 边界上的实现差异 (那个差异会让测试
// 无法区分"kernel 算错" 与 "量化边界不同")。
//
// 返回 0 成功, 1 shape 不合法
int w4a16_weight_pack(const void* qbuf, int N, int K, void* Wp, void* stream) {
    using namespace atex::wconvert::sm120;

    if (K % 32 != 0 || N % 16 != 0)
        return 1;
    if (qbuf == nullptr)
        return 1;

    auto  st = static_cast<cudaStream_t>(stream);
    auto* qb = static_cast<const uint8_t*>(qbuf);

    if (Wp) {
        const int total = (K / 16) * (N / 2) * 4;
        pack_std_kernel<<<(total + 255) / 256, 256, 0, st>>>(qb, static_cast<int32_t*>(Wp), N, K);
    }
    return 0;
}

}  // extern "C"
