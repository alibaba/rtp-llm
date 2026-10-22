#pragma once
#include <cuda_bf16.h>
#include <cute/atom/copy_traits_sm100.hpp>

// Decode-only epilogue. The caller owns all outputs throughout Graph replay.
struct QBArgs {
    unsigned char* payload;
    unsigned char* scales;
    __nv_bfloat16* scaled_weights;
    const __nv_bfloat16* weights;
    const float* frequencies;
    const void* positions;
    bool positions_i64;
    unsigned m, max_position;
};

__device__ __forceinline__ unsigned fp4_pair(float even, float odd) {
    unsigned packed;
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;"
        " cvt.u32.u8 %0, b; }" : "=r"(packed) : "f"(even),"f"(odd));
    return packed;
}

// Non-swap TMEM: one lane owns one token and a complete 32-element SF group.
// Both RoPE partners and the amax reduction are register-local. No CTA exchange.
template<unsigned BM, unsigned BN, unsigned Overlap>
__device__ __forceinline__ void store_indexer_q(
    const QBArgs& args, unsigned tmem_base, unsigned base_m, unsigned base_n,
    unsigned warp, unsigned lane, bool reverse,
    const cutlass::arch::ClusterTransactionBarrier* overlap,
    const cutlass::arch::ClusterTransactionBarrier* empty) {
    static_assert(BM == 128 && (BN == 128 || BN == 256) && Overlap <= 32);
    const unsigned token = base_m + warp * 32 + lane;
    int64_t position = 0;
    if (token < args.m) {
        position = args.positions_i64 ? static_cast<const int64_t*>(args.positions)[token]
                                      : static_cast<const int*>(args.positions)[token];
        if (position < 0) position += args.max_position;
        if (position < 0 || position >= args.max_position) asm volatile("trap;");
    }
    // Keep SF groups as a loop: duplicating the full RoPE/quant
    // epilogue also enlarges the instruction footprint of every Main-Q CTA.
    #pragma unroll 1
    for (unsigned iteration = 0; iteration < BN / 32; ++iteration) {
        const unsigned column_group = reverse ? BN / 32 - 1 - iteration : iteration;
        const unsigned group = column_group % 4;
        const unsigned head = (base_n - 32768) / 128 + column_group / 4;
        unsigned bits[32];
        [&]<size_t... I>(cute::index_sequence<I...>) {
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_base + column_group * 32, bits[I]...);
        }(cute::make_index_sequence<32>{});
        cutlass::arch::fence_view_async_tmem_load();
        if constexpr (Overlap > 0) {
            if (iteration == 0) {
                deep_gemm::ptx::tcgen05_before_thread_sync();
                overlap->arrive(0u);
            }
        }
        if (iteration == BN / 32 - 1) {
            deep_gemm::ptx::tcgen05_before_thread_sync();
            empty->arrive(0u);
        }
        if (token < args.m) {
            float values[32];
            #pragma unroll
            for (unsigned i = 0; i < 32; ++i)
                values[i] = __bfloat162float(__float2bfloat16_rn(__uint_as_float(bits[i])));
            if (group >= 2) {
                #pragma unroll
                for (unsigned i = 0; i < 32; i += 2) {
                    const unsigned angle = (group - 2) * 16 + i / 2;
                    const float cosine = args.frequencies[position * 64 + angle * 2];
                    const float sine = args.frequencies[position * 64 + angle * 2 + 1];
                    const float real = values[i], imag = values[i + 1];
                    const float prod = __uint_as_float(__float_as_uint(__fmul_rn(imag, sine)) ^ 0x80000000u);
                    values[i] = __bfloat162float(__float2bfloat16_rn(__fmaf_rn(real, cosine, prod)));
                    values[i+1] = __bfloat162float(__float2bfloat16_rn(__fmaf_rn(imag, cosine, __fmul_rn(real, sine))));
                }
            }
            float maximum = 0x1.8p-124f; // 6 * 2^-126
            #pragma unroll
            for (unsigned i = 0; i < 32; ++i) maximum = fmaxf(maximum, fabsf(values[i]));
            unsigned b = __float_as_uint(__fmul_rn(maximum, 1.f / 6.f));
            unsigned exponent = ((b >> 23) & 255) + ((b & 0x7fffff) != 0);
            const float inverse = __uint_as_float(exponent < 254 ? (254-exponent)<<23 :
                                                  exponent == 254 ? 0x00400000u : 0u);
            const unsigned row = token * 32 + head;
            args.scales[row * 4 + group] = exponent;
            unsigned packed[4] = {};
            #pragma unroll
            for (unsigned i = 0; i < 32; i += 2) {
                packed[i / 8] |= fp4_pair(__fmul_rn(values[i],inverse),
                                         __fmul_rn(values[i+1],inverse)) << ((i % 8) * 4);
            }
            *reinterpret_cast<uint4*>(args.payload + row * 64 + group * 16) =
                make_uint4(packed[0], packed[1], packed[2], packed[3]);
            if (group == 0)
                args.scaled_weights[row] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(args.weights[row]), 1.f / 64.f));
        }
    }
}
