// include/copy/sm80.cuh — cp.async 异步拷贝 (SM80 起可用)
//
// 说明: 本 kernel 的数据加载走 cp.async, 而不是 SM90 的 TMA
// (cp.async.bulk.tensor)。原因:
//   - 权重/激活在 n (或 k) 方向上是 16B 对齐的, cp.async 的 16B 粒度
//     已经能吃满带宽, 不需要 tensormap 的间接层;
//   - cp.async 不占用额外 smem 描述符, 对多级流水更省空间;
//   - TMA 的 mbarrier 管理与本 kernel 的 splitK 流水结合更复杂,
//     而实测瓶颈不在这里 (纯带宽瓶颈, 见 bench_speedup.py)。
// 若后续要换 TMA, 只需替换本文件的 load_* 函数, 上层接口不变。
#pragma once

#include "../common.cuh"

namespace atex {
namespace copy {
namespace sm80 {

using namespace atex::sm120;

// ============================================================================
// cp.async — 全局 -> 共享 的异步拷贝
//   .cg : 只走 L2, 不污染 L1 —— PTX 只允许 16 字节, 用于权重的流式加载
//   .ca : 走 L1, 支持 4/8/16 字节 —— 小粒度 (如 scale) 只能用这个
//
// 按字节粒度做模板分派: cp_async<16> / cp_async<8> / cp_async<4>。
// 因为 PTX 的 cp-size 必须是立即数, 这里用 if constexpr 而不是把 Bytes 塞进
// asm 的常量占位符 (后者编译器不接受)。
// ============================================================================
template<int Bytes>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp.async 只支持 4 / 8 / 16 字节");
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    if constexpr (Bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
    } else if constexpr (Bytes == 8) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(gmem_src));
    }
}

// ============================================================================
// 组管理 — 多级流水的同步基础
//   每次 commit 把之前发出的所有 cp.async 归为一组;
//   wait_group<N> 表示 "等待直到未完成的组数 <= N"。
// ============================================================================
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// 等待至多 N 个组未完成 (N 必须编译期已知)
template<int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---------------------------------------------------------------------------
// 运行期等待深度: cp.async.wait_group 的操作数必须是立即数, 所以这里算出
// 深度后交给下面的 switch 分发。
//
// 深度公式的推导 (这是流水线最容易错的地方):
//   记 npre = min(niter, Stage) 为预取组数。
//   第 it 轮开始时已 commit 的组数 C = min(npre + it, niter)。
//   我们需要的是第 it 组已完成, 而 cp.async.wait_group N 只保证
//   "除最新 N 组外全部完成" —— 也就是说允许最新的 N 组(即 it+1 .. C-1)未完成。
//   所以必须 N <= C - it - 1 = min(npre, niter - it) - 1。
//
//   稳态 (npre + it <= niter): 深度恒为 npre - 1, 与 it 无关;
//   尾段 (it 接近 niter):     深度必须收缩到 niter-it-1, 最后一轮为 0。
//   曾经写成"随 it 递增的 pending 计数器", 尾段会退化成过松的等待,
//   于是读到尚未就绪的 smem —— 表现为大 K/大网格时的 NaN 或随机错值,
//   而且小规模测试 (预取总能及时完成) 不会复现。
// ---------------------------------------------------------------------------
__device__ __forceinline__ int cp_wait_depth(int npre, int niter, int it) {
    const int remain = niter - it;
    return (npre < remain ? npre : remain) - 1;
}

__device__ __forceinline__ void cp_wait_pending(int Pending) {
    switch (Pending) {
#define ATEX_CP_CASE(D)                                                                                                \
    case D:                                                                                                            \
        cp_wait<D>();                                                                                                  \
        break;
        ATEX_CP_CASE(0)
        ATEX_CP_CASE(1)
        ATEX_CP_CASE(2)
        ATEX_CP_CASE(3)
        ATEX_CP_CASE(4)
        ATEX_CP_CASE(5)
        ATEX_CP_CASE(6)
        ATEX_CP_CASE(7)
        default:
            cp_wait<7>();
            break;
#undef ATEX_CP_CASE
    }
}

}  // namespace sm80
}  // namespace copy
}  // namespace atex
