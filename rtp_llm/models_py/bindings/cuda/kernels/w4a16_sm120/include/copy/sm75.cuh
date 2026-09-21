// include/copy/sm75.cuh — ldmatrix (SM75 起可用)
//
// ldmatrix.sync.aligned.m8n8.x{N}.shared.b16: 一次把 N 个 8x8 的 bf16 矩阵
// 从 smem 搬进寄存器, 并自动按 tensor core 的 fragment 布局分给各 lane。
// 相比逐个元素 LDS, 它省掉了地址计算与 bank 冲突的手工规避。
//
// 本 kernel 用它加载 mma 的 A 操作数 (16x16 = 4 个 8x8, 即 ldmatrix<4>)。
//
// ============================== 地址约定 ==============================
// 每个 8x8 矩阵由 8 个 lane 各提供一个**行首地址**: lane 提供的是该矩阵第
// (lane & 7) 行的起始地址。N 个矩阵依次由 lane 0-7, 8-15, 16-23, 24-31 提供。
//
// 16x16 的 4 个 8x8 子块与矩阵序号的对应 (A fragment 要求的排布):
//   矩阵 0 -> (行 0-7 , 列 0-7 )     矩阵 1 -> (行 8-15, 列 0-7 )
//   矩阵 2 -> (行 0-7 , 列 8-15)     矩阵 3 -> (行 8-15, 列 8-15)
// 即 行块 = 序号 & 1, 列块 = 序号 >> 1。
//
// lane 超出 Num*8 的部分 (如 x1 的 lane 8-31) 提供的地址会被硬件忽略;
// 这里统一套同一公式算, 重复值无害, 从而不必分支。
#pragma once

#include "../common.cuh"

namespace atex {
namespace copy {
namespace sm75 {

using namespace atex::sm120;

// Num        : 矩阵个数 (1 / 2 / 4), 同时决定输出寄存器个数
// r          : 输出, Num 个 b32
// base       : 第一个 8x8 矩阵左上角在 smem 的地址
// row_stride : 行跨距 (字节)
// lane       : 线程在 warp 内的下标
template<int Num>
__device__ __forceinline__ void ldmatrix(uint32_t* r, const char* base, int row_stride, int lane) {
    static_assert(Num == 1 || Num == 2 || Num == 4, "ldmatrix 只支持 1 / 2 / 4 个矩阵");
    const int      mat  = (lane >> 3) & (Num - 1);
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(base))
                          + static_cast<uint32_t>((lane & 7) + (mat & 1) * 8) * row_stride
                          + static_cast<uint32_t>(mat >> 1) * 16;
    if constexpr (Num == 1) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(r[0]) : "r"(addr));
    } else if constexpr (Num == 2) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
    }
}

}  // namespace sm75
}  // namespace copy
}  // namespace atex
