// Startup-built CPU Engram gather/dequantization. No Torch or CUDA dependency.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {
alignas(64) uint32_t lookup_table[256 * 256];
std::once_flag lookup_once;

float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        // NVIDIA cvt.rn.bf16.f32 canonical NaN; sign/payload are not retained.
        return 0x7fff;
    }
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

void initialize_lookup() {
#if defined(__x86_64__)
    const unsigned saved_mxcsr = _mm_getcsr();
    // LUT generation must not inherit a caller's denormal flush settings.
    _mm_setcsr(saved_mxcsr & ~((1U << 15) | (1U << 6)));
#endif
    for (unsigned scale = 0; scale < 256; ++scale) {
        const float multiplier = from_bits(scale << 23);
        for (unsigned raw = 0; raw < 256; ++raw) {
            const unsigned exponent = (raw >> 3) & 15;
            const unsigned mantissa = raw & 7;
            float          value    = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9) :
                                                      std::ldexp(static_cast<float>(8 + mantissa), static_cast<int>(exponent) - 10);
            if (exponent == 15 && mantissa == 7) {
                value = std::numeric_limits<float>::quiet_NaN();
            }
            if (raw & 128) {
                value = -value;
            }
            lookup_table[scale * 256 + raw] = to_bf16(value * multiplier);
        }
    }
#if defined(__x86_64__)
    _mm_setcsr(saved_mxcsr);
#endif
}

void row_scalar(const uint8_t* weight, const uint8_t* scales, uint16_t* out, int64_t dim) {
    for (int64_t block = 0; block < dim / 32; ++block) {
        const uint32_t* table = lookup_table + static_cast<unsigned>(scales[block]) * 256;
        for (int column = 0; column < 32; ++column) {
            out[block * 32 + column] = static_cast<uint16_t>(table[weight[block * 32 + column]]);
        }
    }
}

#if defined(__x86_64__)
__attribute__((target("avx512f,avx512bw"))) void
row_avx512(const uint8_t* weight, const uint8_t* scales, uint16_t* out, int64_t dim) {
    for (int64_t block = 0; block < dim / 32; ++block) {
        const uint32_t* table = lookup_table + static_cast<unsigned>(scales[block]) * 256;
        for (int column = 0; column < 32; column += 16) {
            const __m128i raw     = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + block * 32 + column));
            const __m512i indices = _mm512_cvtepu8_epi32(raw);
            const __m512i bits    = _mm512_i32gather_epi32(indices, table, 4);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + block * 32 + column), _mm512_cvtepi32_epi16(bits));
        }
    }
}
#endif

using RowFunction = void (*)(const uint8_t*, const uint8_t*, uint16_t*, int64_t);
RowFunction row_function() {
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
        return row_avx512;
    }
#endif
    return row_scalar;
}
}  // namespace

extern "C" int engram_cpu_prepare() {
    std::call_once(lookup_once, initialize_lookup);
    return row_function() == row_scalar ? 1 : 2;
}

extern "C" int engram_cpu_lookup(const uint8_t* weight,
                                 const uint8_t* scales,
                                 const int64_t* indices,
                                 uint16_t*      output,
                                 int64_t        table_rows,
                                 int64_t        dim,
                                 int64_t        weight_stride,
                                 int64_t        scale_stride,
                                 int64_t        rows,
                                 int            num_threads) {
    if (table_rows < 0 || dim <= 0 || dim > 4096 || dim % 32 || weight_stride < dim || scale_stride < dim / 32
        || rows < 0 || rows > std::numeric_limits<int64_t>::max() / dim / 2 || num_threads < 1 || num_threads > 16
        || (rows && (!indices || !output)) || (table_rows && (!weight || !scales))) {
        return 1;
    }
    if (table_rows
        && ((table_rows - 1) > (std::numeric_limits<int64_t>::max() - dim) / weight_stride
            || (table_rows - 1) > (std::numeric_limits<int64_t>::max() - dim / 32) / scale_stride)) {
        return 1;
    }
    const RowFunction convert = row_function();
    // Explicit local team size: never change Torch/OpenMP global thread settings.
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t index  = indices[row];
        uint16_t*     target = output + row * dim;
        if (index < 0 || index >= table_rows) {
            std::memset(target, 0, static_cast<size_t>(dim) * sizeof(uint16_t));
        } else {
            convert(weight + index * weight_stride, scales + index * scale_stride, target, dim);
        }
    }
    return 0;
}
