#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

// Host-only planning for Qwen GDN prefill. This is a conservative estimate,
// not an allocator guarantee: other kernels, graph pools and local ViT still
// need the ordinary runtime reserve. No model arithmetic or cache state changes.
struct GdnPrefillMemory {
    static size_t add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a) {
            throw std::overflow_error("GDN prefill memory estimate overflow");
        }
        return a + b;
    }

    static size_t multiply(size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max() / a) {
            throw std::overflow_error("GDN prefill memory estimate overflow");
        }
        return a * b;
    }

    static size_t estimate(size_t tokens,
                           size_t sequences,
                           size_t key_heads,
                           size_t value_heads,
                           size_t key_dim,
                           size_t value_dim,
                           size_t hidden_size,
                           size_t activation_bytes) {
        if (!tokens) {
            return 0;
        }
        if (!sequences || !key_heads || !value_heads || !key_dim || !value_dim || !activation_bytes) {
            throw std::invalid_argument("GDN prefill memory planning requires positive dimensions");
        }
        const auto k = multiply(key_heads, key_dim);
        const auto v = multiply(value_heads, value_dim);
        // QKVZ projection (2K+2V), convolution (2K+V), scatter (2K+V),
        // normalized Q/K (2K), output and norm output (2V).
        // Also budget native FLA w (Hv*K), u and v_new (2V), and three
        // hidden/residual buffers. Keeping these lifetimes overlapping is
        // intentionally conservative for the fused FlashInfer path.
        const auto tensor_elements =
            add(add(multiply(8, k), multiply(8, v)), add(multiply(value_heads, key_dim), multiply(3, hidden_size)));
        constexpr size_t chunk_size = 64;
        // Original a/b, FP32 gates/cumsum, plus KKT and triangular-solve
        // temporaries. Do not size runtime activations from FP8 KV/weight dtype.
        const auto gate_bytes =
            multiply(value_heads, add(multiply(2, activation_bytes), (3 + 2 * chunk_size) * sizeof(float)));
        const auto token_bytes = multiply(tokens, add(multiply(tensor_elements, activation_bytes), gate_bytes));
        // Native FLA materializes one FP32 state per 64-token chunk; ragged
        // sequences can each add a partial chunk. Initial/final states add 2N.
        // This also covers FlashInfer's sparser checkpoints (interval >=128).
        const auto chunks      = add(tokens / chunk_size, tokens % chunk_size != 0);
        const auto states      = add(chunks, multiply(3, std::min(sequences, tokens)));
        const auto state_bytes = multiply(multiply(value_heads, key_dim), multiply(value_dim, sizeof(float)));
        return add(token_bytes, multiply(states, state_bytes));
    }

    // Match the existing exclusive MAX_BATCH_TOKENS_SIZE boundary, including
    // expanded sequences, without overflowing length * width or the sum.
    static bool tryAdd(size_t& used_tokens,
                       size_t& used_sequences,
                       size_t  length,
                       size_t  width,
                       size_t  token_limit,
                       size_t  sequence_limit) {
        if (!width || used_sequences > sequence_limit || width > sequence_limit - used_sequences
            || used_tokens >= token_limit || length > (token_limit - used_tokens - 1) / width) {
            return false;
        }
        used_tokens += length * width;
        used_sequences += width;
        return true;
    }

    // An explicit KV size is a requested size, not permission to consume the
    // prefill workspace. Fail at startup instead of launching an unsafe engine.
    static size_t kvCacheBytes(size_t available, size_t runtime_required, size_t explicit_kv = 0) {
        if (runtime_required >= available || (explicit_kv && explicit_kv > available - runtime_required)) {
            throw std::invalid_argument("Insufficient memory for GDN prefill runtime reserve and KV cache; "
                                        "lower KV_CACHE_MEM_MB or MAX_BATCH_TOKENS_SIZE "
                                        "(RESERVER_RUNTIME_MEM_MB is an additional runtime floor)");
        }
        return explicit_kv ? explicit_kv : available - runtime_required;
    }
};

}  // namespace rtp_llm
