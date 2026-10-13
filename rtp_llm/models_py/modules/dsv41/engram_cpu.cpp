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

extern "C" int engram_cpu_lookup_packed(const uint8_t* weight,
                                        const uint8_t* scales,
                                        const int64_t* indices,
                                        uint8_t*       output,
                                        int64_t        table_rows,
                                        int64_t        dim,
                                        int64_t        weight_stride,
                                        int64_t        scale_stride,
                                        int64_t        rows,
                                        int            num_threads) {
    if (table_rows < 0 || dim <= 0 || dim > 4096 || dim % 32 || weight_stride < dim || scale_stride < dim / 32
        || rows < 0 || rows > std::numeric_limits<int64_t>::max() / (dim + dim / 32) || num_threads < 1
        || num_threads > 16 || (rows && (!indices || !output)) || (table_rows && (!weight || !scales))) {
        return 1;
    }
    if (table_rows
        && ((table_rows - 1) > (std::numeric_limits<int64_t>::max() - dim) / weight_stride
            || (table_rows - 1) > (std::numeric_limits<int64_t>::max() - dim / 32) / scale_stride)) {
        return 1;
    }
    const int64_t packed_dim = dim + dim / 32;
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t index  = indices[row];
        uint8_t*      target = output + row * packed_dim;
        if (index < 0 || index >= table_rows) {
            std::memset(target, 0, static_cast<size_t>(packed_dim));
        } else {
            // Preserve all FP8/UE8M0 bytes, including zero and NaN encodings.
            std::memcpy(target, weight + index * weight_stride, static_cast<size_t>(dim));
            std::memcpy(target + dim, scales + index * scale_stride, static_cast<size_t>(dim / 32));
        }
    }
    return 0;
}

extern "C" int engram_cpu_hash(const void*    windows,
                               const int64_t* token_map,
                               const int64_t* multipliers,
                               const int64_t* primes,
                               const int64_t* offsets,
                               int64_t*       output,
                               int64_t        tokens,
                               int64_t        ngram,
                               int64_t        layers,
                               int64_t        heads,
                               int64_t        map_size,
                               int64_t        window_stride0,
                               int64_t        window_stride1,
                               int64_t        pad_id,
                               int64_t        image_token,
                               int            window_bytes,
                               int            num_threads) {
    const int64_t max_elements = std::numeric_limits<int64_t>::max() / sizeof(int64_t);
    if (tokens < 0 || ngram < 2 || layers < 1 || heads < 1 || map_size < 1 || window_stride0 < 0 || window_stride1 < 0
        || (window_bytes != 4 && window_bytes != 8) || num_threads < 1 || num_threads > 16
        || ngram > max_elements / layers || heads > max_elements / (ngram - 1) / layers || !token_map || !multipliers
        || !primes || !offsets || (tokens && (!windows || !output))) {
        return 1;
    }
    const int64_t columns = (ngram - 1) * heads;
    if (tokens > max_elements / layers / columns || window_stride1 > max_elements / (ngram - 1)
        || (tokens > 1 && window_stride0 > (max_elements - (ngram - 1) * window_stride1) / (tokens - 1))) {
        return 1;
    }
    for (int64_t column = 0; column < layers * columns; ++column) {
        if (primes[column] <= 0) {
            return 1;
        }
    }
    // Hashing and lookup execute serially on the same Python worker. Both use
    // explicit bounded teams, without changing Torch's intra-op thread pool.
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t layer = 0; layer < layers; ++layer) {
            uint64_t rolling = 0;
            bool     blocked = false;
            for (int64_t shift = 0; shift < ngram; ++shift) {
                const int64_t position = token * window_stride0 + shift * window_stride1;
                const int64_t value    = window_bytes == 4 ? static_cast<const int32_t*>(windows)[position] :
                                                             static_cast<const int64_t*>(windows)[position];
                blocked                = blocked || value < 0 || value >= map_size || value == image_token;
                const int64_t mapped   = blocked ? pad_id : token_map[value];
                // Unsigned multiplication/XOR defines the wraparound that NumPy
                // int64 uses. memcpy preserves its signed two's-complement bits.
                rolling ^= static_cast<uint64_t>(mapped) * static_cast<uint64_t>(multipliers[layer * ngram + shift]);
                if (shift) {
                    int64_t signed_rolling;
                    std::memcpy(&signed_rolling, &rolling, sizeof(rolling));
                    for (int64_t head = 0; head < heads; ++head) {
                        const int64_t column    = layer * columns + (shift - 1) * heads + head;
                        const int64_t prime     = primes[column];
                        int64_t       remainder = signed_rolling % prime;
                        if (remainder < 0) {
                            remainder += prime;
                        }
                        const uint64_t result =
                            static_cast<uint64_t>(remainder) + static_cast<uint64_t>(offsets[column]);
                        std::memcpy(output + token * layers * columns + column, &result, sizeof(result));
                    }
                }
            }
        }
    }
    return 0;
}
