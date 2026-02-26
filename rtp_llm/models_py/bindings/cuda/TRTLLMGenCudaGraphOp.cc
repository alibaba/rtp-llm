#include "rtp_llm/models_py/bindings/cuda/TRTLLMGenCudaGraphOp.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

namespace {

/**
 * Update kv_cache_offset in-place via invokeConvertOffsetToBlockArrayData.
 *
 * kv_cache_offset  shape: [B_cap, 1, 2, M_cap]  — pre-allocated at warmup size
 * kv_cache_block_id shape: [B_act, M_act]         — actual batch this step
 *
 * Fast path (M_act == M_cap): writes directly into kv_cache_offset without any
 * allocation, since the memory layout [B, 1, 2, M] has batch_stride = 2*M which
 * matches what invokeConvertOffsetToBlockArrayData expects.
 *
 * Slow path (M_act != M_cap): allocates a small temporary, runs the kernel, then
 * copies the relevant slice into kv_cache_offset.
 */
static void updateKVCacheOffsetInPlace(torch::Tensor&       kv_cache_offset,
                                       const torch::Tensor& kv_cache_block_id,
                                       StreamType           stream) {
    const int B_act = static_cast<int>(kv_cache_block_id.size(0));
    const int M_act = static_cast<int>(kv_cache_block_id.size(1));
    const int M_cap = static_cast<int>(kv_cache_offset.size(3));

    if (M_act == M_cap) {
        invokeConvertOffsetToBlockArrayData(kv_cache_offset.data_ptr<int32_t>(),
                                            kv_cache_block_id.data_ptr<int32_t>(),
                                            B_act,
                                            M_act,
                                            stream);
    } else {
        auto tmp = torch::empty({B_act, 1, 2, M_act},
                                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        invokeConvertOffsetToBlockArrayData(tmp.data_ptr<int32_t>(),
                                            kv_cache_block_id.data_ptr<int32_t>(),
                                            B_act,
                                            M_act,
                                            stream);
        kv_cache_offset.slice(0, 0, B_act).slice(3, 0, M_act).copy_(tmp, /*non_blocking=*/true);
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Unified implementation
// ---------------------------------------------------------------------------
void trtllm_gen_update_cuda_graph_params(torch::Tensor                seq_lens,
                                         torch::Tensor                kv_cache_offset,
                                         torch::Tensor                kv_cache_block_id,
                                         torch::Tensor                lengths_a,
                                         c10::optional<torch::Tensor> lengths_b,
                                         c10::optional<torch::Tensor> cu_kv_seqlens,
                                         int64_t                      page_size) {
    StreamType stream = GET_CURRENT_STREAM();

    if (lengths_b.has_value()) {
        // ── SpecDecode / Prefill: seq_lens = lengths_a + lengths_b ──────────
        torch::add_out(seq_lens, lengths_a, lengths_b.value());

        if (cu_kv_seqlens.has_value()) {
            // ── Prefill extra: cu_kv_seqlens[1:] = cumsum(ceil(seq_lens/page_size))
            auto page_per_seq =
                (seq_lens + static_cast<int>(page_size) - 1).div(static_cast<int>(page_size), "trunc");
            auto cumsum = torch::cumsum(page_per_seq, /*dim=*/0, /*dtype=*/torch::kInt32);
            cu_kv_seqlens.value()
                .narrow(/*dim=*/0, /*start=*/1, seq_lens.size(0))
                .copy_(cumsum, /*non_blocking=*/true);
        }
    } else {
        // ── Decode: seq_lens = lengths_a (device-to-device copy) ────────────
        seq_lens.copy_(lengths_a, /*non_blocking=*/true);
    }

    // Always update kv_cache_offset in-place
    updateKVCacheOffsetInPlace(kv_cache_offset, kv_cache_block_id, stream);
}

// ---------------------------------------------------------------------------
// Pybind registration
// ---------------------------------------------------------------------------
void registerTRTLLMGenCudaGraphOp(py::module& m) {
    m.def("trtllm_gen_update_cuda_graph_params",
          &trtllm_gen_update_cuda_graph_params,
          "Unified in-place CUDA-graph param update for all FlashInferTRTLLM*Impl variants.\n"
          "Mode is selected by which optional args are provided:\n"
          "  Decode:    lengths_b=None, cu_kv_seqlens=None  => seq_lens = copy(lengths_a)\n"
          "  SpecDecode: lengths_b provided, cu_kv_seqlens=None => seq_lens = lengths_a + lengths_b\n"
          "  Prefill:   lengths_b provided, cu_kv_seqlens provided => seq_lens = sum, update cumsum",
          py::arg("seq_lens"),
          py::arg("kv_cache_offset"),
          py::arg("kv_cache_block_id"),
          py::arg("lengths_a"),
          py::arg("lengths_b")      = py::none(),
          py::arg("cu_kv_seqlens")  = py::none(),
          py::arg("page_size")      = 0);
}

}  // namespace rtp_llm
