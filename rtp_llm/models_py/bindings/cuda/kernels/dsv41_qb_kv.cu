#include "dsv41_qb_kv.h"

#if defined(RTP_DSV41_QB_KV_ENABLED) && !defined(USE_ROCM)
#include <cmath>
#include <climits>
// Match the existing DeepSelect compatibility guard: the pinned CUTLASS
// predates SM103, whose exact architecture cubin supports these SM90 TMA ops.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)
#ifndef CUTE_ARCH_TMA_SM90_ENABLED
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif
// The same pinned header otherwise emits brkpt for tcgen05.wait::ld/st.
// Enable only this instruction family, only for the exact SM103a cubin.
#ifndef CUTLASS_ARCH_TCGEN_ENABLED
#define CUTLASS_ARCH_TCGEN_ENABLED 1
#endif
#endif
#include "dsv41_qb_kv/q_nonswap.cuh"

namespace {
void check_q(torch::Tensor x, torch::Tensor xs, torch::Tensor w,
             torch::Tensor ws, torch::Tensor out, int64_t variant, int64_t clusters) {
    TORCH_CHECK(x.is_cuda(), "QB requires CUDA inputs");
    for (auto t : {xs, w, ws, out}) {
        TORCH_CHECK(t.is_cuda() && t.device() == x.device(), "QB tensors must share a CUDA device");
    }
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 1280 && x.size(0) >= 1 && x.size(0) <= 512,
                "QB input must be [M,1280], M in 1..512");
    TORCH_CHECK(w.dim() == 2 && (w.size(0) == 32768 || w.size(0) == 36864) && w.size(1) == 1280,
                "QB weight must be [32768 or 36864,1280]");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == w.size(0),
                "QB output must be [M,N]");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float8_e4m3fn && w.scalar_type() == x.scalar_type()
                    && out.scalar_type() == at::kBFloat16, "QB requires E4M3 operands and BF16 output");
    for (auto t : {x, w, out}) {
        TORCH_CHECK(t.is_contiguous() && uintptr_t(t.data_ptr()) % 16 == 0,
                    "QB payloads must be contiguous and 16-byte aligned");
    }
    for (auto pair : {std::make_pair(xs, x.size(0)), std::make_pair(ws, w.size(0))}) {
        auto sf = pair.first;
        TORCH_CHECK(sf.scalar_type() == at::kInt && sf.dim() == 2 && sf.size(0) == pair.second
                        && sf.size(1) == 10 && sf.stride(0) == 1 && sf.stride(1) >= sf.size(0)
                        && sf.stride(1) % 4 == 0 && uintptr_t(sf.data_ptr()) % 16 == 0,
                    "QB SF must be TMA-aligned MN-major packed UE8M0 [rows,10]");
    }
    for (auto input : {x, xs, w, ws}) {
        TORCH_CHECK(!out.is_alias_of(input), "QB output must not alias inputs");
    }
    TORCH_CHECK(variant >= 0 && variant <= 4, "Invalid QB variant");
    const auto* properties = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(properties->major == 10 && properties->minor == 3 && properties->multiProcessorCount == 148,
                "QB backend is qualified only for runtime SM103 / 148 SM");
    TORCH_CHECK(clusters > 0 && clusters <= properties->multiProcessorCount / 2, "Invalid QB cluster count");
}

template<bool Side>
void launch_q(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
              torch::Tensor out, int variant, int clusters, rtp_dsv41_qb::SideArgs side) {
    switch (variant) {
        case 0: rtp_dsv41_qb::launch<128, true, Side>(x, xs, w, ws, out, clusters, side); break;
        case 1: rtp_dsv41_qb::launch<192, true, Side>(x, xs, w, ws, out, clusters, side); break;
        case 2: rtp_dsv41_qb_ns::launch<224, true, 1, Side>(x, xs, w, ws, out, clusters, side); break;
        case 3: rtp_dsv41_qb_ns::launch<256, true, 1, Side>(x, xs, w, ws, out, clusters, side); break;
        case 4: rtp_dsv41_qb::launch<256, true, Side>(x, xs, w, ws, out, clusters, side); break;
    }
}
}  // namespace
#endif

namespace torch_ext {
bool dsv41_qb_kv_available() {
#if defined(RTP_DSV41_QB_KV_ENABLED) && !defined(USE_ROCM)
    return true;
#else
    return false;
#endif
}

void dsv41_qb_out(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
                  torch::Tensor out, int64_t variant, int64_t clusters, bool wait_pdl) {
#if defined(RTP_DSV41_QB_KV_ENABLED) && !defined(USE_ROCM)
    TORCH_CHECK(x.is_cuda(), "QB requires CUDA");
    const c10::cuda::CUDAGuard guard(x.device());
    check_q(x, xs, w, ws, out, variant, clusters);
    rtp_dsv41_qb::SideArgs side;
    side.wait_pdl = wait_pdl;
    launch_q<false>(x, xs, w, ws, out, variant, clusters, side);
#else
    TORCH_CHECK(false, "DSV41 QB was not built for this platform");
#endif
}

static void qb_kv_impl(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
                     torch::Tensor out, torch::Tensor kv, torch::Tensor gamma,
                     torch::Tensor positions, torch::Tensor frequencies, torch::Tensor pool,
                     torch::Tensor slots, torch::Tensor kv_out, double eps, int64_t variant,
                     int64_t clusters, bool wait_pdl, bool trap_invalid,
                     torch::Tensor req, torch::Tensor table, torch::Tensor local,
                     torch::Tensor global, int64_t logical_entries, int64_t index_entries) {
#if defined(RTP_DSV41_QB_KV_ENABLED) && !defined(USE_ROCM)
    TORCH_CHECK(x.is_cuda(), "QB requires CUDA");
    const c10::cuda::CUDAGuard guard(x.device());
    check_q(x, xs, w, ws, out, variant, clusters);
    for (auto t : {kv, gamma, positions, frequencies, pool, slots, kv_out}) {
        TORCH_CHECK(t.is_cuda() && t.device() == x.device(), "KV tensors must share the QB device");
    }
    const auto m = x.size(0);
    TORCH_CHECK(kv.dim() == 2 && kv.size(0) == m && kv.size(1) == 512 && kv.stride(1) == 1
                    && kv.stride(0) >= 512 && kv.scalar_type() == at::kBFloat16,
                "KV requires strided BF16 [M,512]");
    TORCH_CHECK(gamma.scalar_type() == at::kBFloat16 && gamma.dim() == 1 && gamma.numel() == 512
                    && gamma.is_contiguous(), "KV norm requires BF16[512]");
    for (auto t : {positions, slots}) {
        TORCH_CHECK(t.dim() == 1 && t.numel() == m && t.is_contiguous()
                        && (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong),
                    "KV metadata requires contiguous int32/int64 [M]");
    }
    TORCH_CHECK(frequencies.dim() == 2 && frequencies.size(1) == 64 && frequencies.size(0) > 0
                    && frequencies.scalar_type() == at::kFloat && frequencies.is_contiguous(),
                "KV frequencies require interleaved FP32[P,64]");
    TORCH_CHECK(pool.dim() == 3 && pool.size(0) > 0 && pool.size(1) > 0 && pool.size(2) == 528
                    && pool.scalar_type() == at::kByte && pool.stride(2) == 1 && pool.stride(1) == 528
                    && pool.stride(0) >= pool.size(1) * 528, "KV cache requires native 528-byte rows");
    TORCH_CHECK(kv_out.dim() == 2 && kv_out.size(0) == m && kv_out.size(1) == 512
                    && kv_out.scalar_type() == at::kBFloat16 && kv_out.is_contiguous(), "Invalid KV output");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0, "Invalid norm epsilon");
    for (auto output : {out, kv_out, pool}) {
        for (auto input : {x, xs, w, ws, kv, gamma, positions, frequencies, slots}) {
            TORCH_CHECK(!output.is_alias_of(input), "QB/KV output aliases input");
        }
    }
    TORCH_CHECK(!out.is_alias_of(pool) && !out.is_alias_of(kv_out) && !kv_out.is_alias_of(pool),
                "QB/KV outputs must have separate storage");
    if (global.defined()) {
        TORCH_CHECK(req.defined() && table.defined() && local.defined(), "Missing ready-index inputs");
        for (auto t : {req, table, local, global}) {
            TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.scalar_type() == at::kInt,
                        "Index tensors must be int32 on the QB device");
        }
        TORCH_CHECK(req.dim() == 1 && req.numel() == m && req.is_contiguous(), "Invalid request map");
        TORCH_CHECK(table.dim() == 2 && table.size(0) > 0 && table.size(1) > 0
                        && table.stride(1) == 1 && table.stride(0) >= table.size(1), "Invalid block table");
        TORCH_CHECK(local.dim() == 2 && local.size(0) == m && local.size(1) > 0
                        && local.size(1) % 128 == 0 && local.stride(1) == 1
                        && local.stride(0) >= local.size(1), "Invalid local TopK");
        TORCH_CHECK(global.sizes() == local.sizes() && global.is_contiguous(), "Invalid slot output");
        TORCH_CHECK(logical_entries > 0 && logical_entries <= INT_MAX && index_entries > 0 && index_entries <= INT_MAX,
                    "Invalid index page geometry");
        for (auto o : {out, kv_out, pool, global}) {
            for (auto i : {req, table, local}) TORCH_CHECK(!o.is_alias_of(i), "Index input/output alias");
        }
        for (auto t : {x, xs, w, ws, kv, gamma, positions, frequencies, slots, out, kv_out, pool})
            TORCH_CHECK(!global.is_alias_of(t), "Slot output aliases QB/KV storage");
    }
    rtp_dsv41_qb::SideArgs side;
    if (global.defined()) {
        side.req = req.data_ptr<int>(); side.block_table = table.data_ptr<int>();
        side.local_indices = local.data_ptr<int>(); side.global_indices = global.data_ptr<int>();
        side.topk = local.size(1); side.num_requests = table.size(0); side.blocks_per_req = table.size(1);
        side.logical_entries = logical_entries; side.index_entries = index_entries;
        if ((logical_entries & (logical_entries - 1)) == 0) side.logical_shift = __builtin_ctz(unsigned(logical_entries));
        if ((index_entries & (index_entries - 1)) == 0) side.entries_shift = __builtin_ctz(unsigned(index_entries));
        side.table_stride = table.stride(0); side.local_stride = local.stride(0);
    }
    side.wait_pdl = wait_pdl;
    side.trap_invalid = trap_invalid;
    side.kv = reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr());
    side.gamma = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
    if (positions.scalar_type() == at::kLong) side.pos64 = positions.data_ptr<int64_t>();
    else side.pos32 = positions.data_ptr<int>();
    if (slots.scalar_type() == at::kLong) side.slots64 = slots.data_ptr<int64_t>();
    else side.slots32 = slots.data_ptr<int>();
    side.freqs = frequencies.data_ptr<float>();
    side.pool = pool.data_ptr<uint8_t>();
    side.kv_out = reinterpret_cast<__nv_bfloat16*>(kv_out.data_ptr());
    side.rows = m;
    side.pages = pool.size(0);
    side.entries = pool.size(1);
    side.page_stride = pool.stride(0);
    side.kv_stride = kv.stride(0);
    side.num_freqs = frequencies.size(0);
    side.eps = eps;
    launch_q<true>(x, xs, w, ws, out, variant, clusters, side);
#else
    TORCH_CHECK(false, "DSV41 QB/KV was not built for this platform");
#endif
}

void dsv41_qb_kv_out(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
                     torch::Tensor out, torch::Tensor kv, torch::Tensor gamma,
                     torch::Tensor positions, torch::Tensor frequencies, torch::Tensor pool,
                     torch::Tensor slots, torch::Tensor kv_out, double eps, int64_t variant,
                     int64_t clusters, bool wait_pdl, bool trap_invalid) {
    qb_kv_impl(x,xs,w,ws,out,kv,gamma,positions,frequencies,pool,slots,kv_out,eps,
               variant,clusters,wait_pdl,trap_invalid,{},{},{},{},0,0);
}

void dsv41_qb_kv_index_out(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
                     torch::Tensor out, torch::Tensor kv, torch::Tensor gamma,
                     torch::Tensor positions, torch::Tensor frequencies, torch::Tensor pool,
                     torch::Tensor slots, torch::Tensor kv_out, double eps, int64_t variant,
                     int64_t clusters, bool wait_pdl, bool trap_invalid,
                     torch::Tensor req, torch::Tensor table, torch::Tensor local,
                     torch::Tensor global, int64_t logical_entries, int64_t index_entries) {
    qb_kv_impl(x,xs,w,ws,out,kv,gamma,positions,frequencies,pool,slots,kv_out,eps,
               variant,clusters,wait_pdl,trap_invalid,req,table,local,global,
               logical_entries,index_entries);
}
}  // namespace torch_ext
