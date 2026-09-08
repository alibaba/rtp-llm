// v32_ctx.cu — Scheme B offload context v2: staging + async fetch fully in C++.
// Python calls per layer: serve_layer() (build+sanitize+miss->fetch) and
// drain happens inside serve_layer. Fetch thread never touches the GIL.
#include <torch/extension.h>
#include "NoBlockCopy.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
constexpr int BS = 64;

// Shared per-device counters; defined further down next to the lossy state.
static unsigned long long* cnts_for(int device);
}  // namespace

// Opt-in event timing (defined further down); lets the PCIe gather be reported
// apart from launch overhead. File scope so the launch sites can call it.
bool v32_kt_begin(cudaStream_t stream);
void v32_kt_end(cudaStream_t stream, int which);

namespace {

__global__ void build_indices_kernel(const int* __restrict__ sel,
                                     const int* __restrict__ bt,
                                     const long* __restrict__ s_idx,
                                     const long* __restrict__ s_slot,
                                     const long* __restrict__ s_logical,
                                     long* __restrict__ s_seen,
                                     int* __restrict__ gidx,
                                     int* __restrict__ miss,      // gpu [1+topk], [0]=count
                                     int* __restrict__ counters,  // gpu [2]
                                     int  topk,
                                     int  S,
                                     int  hist,
                                     long step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= topk)
        return;
    int p = sel[i];
    if (p < 0 || p >= hist)
        return;
    int blk = bt[p / BS];
    if (blk > 0) {
        gidx[atomicAdd(&counters[0], 1)] = p;  // logical: convert translates via bt
        return;
    }
    long pp = (long)p;
    int  lo = 0, hi = S - 1;
    while (lo <= hi) {
        int  mid = (lo + hi) >> 1;
        long v   = s_idx[mid];
        if (v == pp) {
            gidx[atomicAdd(&counters[0], 1)] = (int)s_logical[mid];
            s_seen[mid]                      = step;
            return;
        }
        if (v < pp)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    miss[1 + atomicAdd(&counters[1], 1)] = p;
}

// place current-token logical position after valids; publish miss count.
// cur_pos < 0 disables the append (T1 path: cur is already inside sel).
__global__ void finalize_kernel(
    int* __restrict__ gidx, const int* __restrict__ counters, int* __restrict__ miss, int cur_pos, int topk) {
    if (threadIdx.x == 0) {
        int nv = counters[0];
        if (cur_pos >= 0 && nv < topk)
            gidx[nv] = cur_pos;
        miss[0] = counters[1];
    }
}

// copy one token's indexer-K (segregated block layout: [64*128 fp8][64*4 scale])
// from the main scale pool block bt[pos/64] into the side store at pos.
__global__ void append_tok_kernel(unsigned char* __restrict__ idxp,
                                  const unsigned char* __restrict__ pool_u8,
                                  const int* __restrict__ bt,
                                  int  pos,
                                  long pool_stride) {
    int blk = bt[pos / BS];
    if (blk <= 0)
        return;
    const unsigned char* src = pool_u8 + (size_t)blk * pool_stride;
    unsigned char*       dst = idxp + (size_t)(pos / BS) * (132 * BS);
    int                  off = pos % BS;
    int                  t   = threadIdx.x;
    if (t < 128)
        dst[off * 128 + t] = src[off * 128 + t];
    else if (t < 132)
        dst[BS * 128 + off * 4 + (t - 128)] = src[BS * 128 + off * 4 + (t - 128)];
}

// scatter fetched rows into staging slots chosen by LRU (victims precomputed)
__global__ void staging_write_kernel(const long* __restrict__ victims,    // [m] indices into slot arrays
                                     const long* __restrict__ stg_slots,  // [S]
                                     const long* __restrict__ new_pos,    // [m]
                                     const float* __restrict__ rows,      // [m, 576] fp32 staged->cast
                                     long* __restrict__ slot_pos,
                                     long* __restrict__ slot_seen,
                                     __nv_bfloat16* __restrict__ pool,  // [slots, 576] flat
                                     int  m,
                                     long step) {
    int r = blockIdx.x;
    if (r >= m)
        return;
    long v    = victims[r];
    long slot = stg_slots[v];
    for (int c = threadIdx.x; c < 576; c += blockDim.x) {
        pool[slot * 576 + c] = __float2bfloat16(rows[r * 576 + c]);
    }
    if (threadIdx.x == 0) {
        slot_pos[v]  = new_pos[r];
        slot_seen[v] = step;
    }
}

// DSA indexer scoring on the paged side store, one warp per token.
// score(t) = sum_h w[h] * relu(q[h] . dequant(k[t]))   (q_scale folded into w)
// v1 (fastest measured on H20): one warp per token, per-head shfl reduce.
__global__ void side_score_kernel(const unsigned char* __restrict__ q,     // [64,128] fp8
                                  const unsigned char* __restrict__ idxp,  // paged [blocks,64,132]
                                  const float* __restrict__ w,             // [64]
                                  float* __restrict__ out,
                                  int hist) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= hist)
        return;
    const unsigned char* base   = idxp + (size_t)(warp >> 6) * (132 * 64);
    const unsigned char* krow   = base + (size_t)(warp & 63) * 128;
    float                kscale = *reinterpret_cast<const float*>(base + 64 * 128 + (size_t)(warp & 63) * 4);
    float                kv[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        kv[d] = float(*reinterpret_cast<const __nv_fp8_e4m3*>(krow + lane * 4 + d)) * kscale;
    }
    float acc = 0.f;
    for (int h = 0; h < 64; ++h) {
        const unsigned char* qrow = q + h * 128 + lane * 4;
        float                p    = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            p += float(*reinterpret_cast<const __nv_fp8_e4m3*>(qrow + d)) * kv[d];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            p += __shfl_xor_sync(0xffffffffu, p, off);
        }
        acc += w[h] * fmaxf(p, 0.f);
    }
    if (lane == 0)
        out[warp] = acc;
}

struct StoreState {
    torch::Tensor kv_host;                         // pinned bf16 [cap, 576]
    torch::Tensor stg_slots, slot_pos, slot_seen;  // gpu long [S]
    torch::Tensor s_idx, s_slot;                   // sorted views (gpu long [S])
    torch::Tensor s_logical;                       // bt-logical alias per slot, aligned with s_slot
    torch::Tensor miss_buf;                        // mapped pinned int32 [1+topk]
    torch::Tensor gidx, counters;                  // gpu int32 [topk+1], [2]
    torch::Tensor pin_scratch;                     // pinned staging for fetched rows
    torch::Tensor scores;                          // gpu fp32 scratch [cap]
    std::mutex    mu;
    std::deque<std::pair<std::vector<int64_t>, torch::Tensor>> inbox;  // (pos, rows fp32 cpu)
    std::atomic<int>                                           inflight{0};
};

struct FetchJob {
    std::shared_ptr<StoreState> st;
    std::vector<int64_t>        miss;
};

struct Context {
    std::unordered_map<int64_t, std::shared_ptr<StoreState>> stores;
    std::mutex                                               mu;
    std::deque<FetchJob>                                     q;
    std::condition_variable                                  cv;
    std::thread                                              worker;
    std::atomic<bool>                                        stop{false};
    cudaStream_t copy_stream = nullptr;  // non-blocking: never syncs the device
    std::mutex   copy_mu;
    Context() {
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
        worker = std::thread([this] { run(); });
    }
    ~Context() {
        stop = true;
        cv.notify_all();
        if (worker.joinable())
            worker.join();
    }
    void run() {
        while (!stop) {
            FetchJob job;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop.load() || !q.empty(); });
                if (stop)
                    return;
                job = std::move(q.front());
                q.pop_front();
            }
            auto          st = job.st;
            const int64_t m  = (int64_t)job.miss.size();
            if (!m)
                continue;
            auto idx  = torch::tensor(job.miss, torch::kInt64);
            auto rows = st->kv_host.index_select(0, idx);  // cpu gather (bf16)
            {
                std::lock_guard<std::mutex> lk(st->mu);
                st->inbox.emplace_back(std::move(job.miss), std::move(rows));
                st->inflight -= (int)m;
            }
        }
    }
};

std::shared_ptr<Context> g_ctx;
int64_t                  key_of(int64_t req_key, int64_t layer) {
    return req_key * 128 + layer;
}
}  // namespace

void ctx_init() {
    if (!g_ctx)
        g_ctx = std::make_shared<Context>();
}

// register/replace a per (request,layer) store
void ctx_register(int64_t       req_key,
                  int64_t       layer,
                  torch::Tensor kv_host,
                  torch::Tensor stg_slots,
                  torch::Tensor stg_logical,
                  int64_t       topk) {
    ctx_init();
    auto st       = std::make_shared<StoreState>();
    st->kv_host   = kv_host;
    st->stg_slots = stg_slots.to(torch::kInt64);
    auto    dev   = stg_slots.device();
    int64_t S     = stg_slots.size(0);
    st->slot_pos  = -2 - torch::arange(S, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    st->slot_seen = torch::zeros({S}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    st->s_idx     = st->slot_pos.clone();
    st->s_slot    = st->stg_slots.clone();
    st->s_logical = stg_logical.to(torch::kInt64).clone();
    st->miss_buf  = torch::zeros({1 + topk}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    st->gidx      = torch::empty({topk + 1}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    st->counters  = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    std::lock_guard<std::mutex> lk(g_ctx->mu);
    g_ctx->stores[key_of(req_key, layer)] = st;
}

bool ctx_has(int64_t req_key, int64_t layer) {
    return g_ctx && g_ctx->stores.count(key_of(req_key, layer)) > 0;
}

void ctx_update_host(int64_t req_key, int64_t layer, torch::Tensor kv_host) {
    auto it = g_ctx->stores.find(key_of(req_key, layer));
    if (it != g_ctx->stores.end())
        it->second->kv_host = kv_host;
}

void ctx_release(int64_t req_key) {
    if (!g_ctx)
        return;
    std::lock_guard<std::mutex> lk(g_ctx->mu);
    for (int l = 0; l < 128; ++l)
        g_ctx->stores.erase(key_of(req_key, l));
}

// per layer per request: drain inbox -> staging, build global indices, launch
// miss fetch. Returns gidx int32 [topk] (-1 padded), all GPU-side, no syncs.
// cur_pos >= 0 forces that logical position into the first free slot.
torch::Tensor serve_impl(int64_t       req_key,
                         int64_t       layer,
                         torch::Tensor sel,
                         torch::Tensor bt_row,
                         torch::Tensor main_pool_flat,
                         int64_t       hist,
                         int64_t       step,
                         bool          want_fetch,
                         int64_t       cur_pos) {
    auto it = g_ctx->stores.find(key_of(req_key, layer));
    TORCH_CHECK(it != g_ctx->stores.end(), "store missing");
    auto st     = it->second;
    auto stream = at::cuda::getCurrentCUDAStream();
    // 1) drain inbox (batched)
    std::deque<std::pair<std::vector<int64_t>, torch::Tensor>> items;
    {
        std::lock_guard<std::mutex> lk(st->mu);
        items.swap(st->inbox);
    }
    if (!items.empty()) {
        std::vector<int64_t>       pos_all;
        std::vector<torch::Tensor> rows_all;
        for (auto& [pos, rows] : items) {
            pos_all.insert(pos_all.end(), pos.begin(), pos.end());
            rows_all.push_back(rows);
        }
        int64_t m = (int64_t)pos_all.size();
        if (m > 0) {
            auto    dev     = st->stg_slots.device();
            int64_t take    = std::min(m, st->slot_seen.size(0));
            auto    victims = std::get<1>(st->slot_seen.topk(take, /*dim=*/0, /*largest=*/false));
            auto    pos_gpu = torch::tensor(pos_all, torch::kInt64).to(dev, true);
            // victim slots to CPU once (small) to address pool rows directly
            auto victims_cpu = victims.to(torch::kCPU);
            auto slots_cpu   = st->stg_slots.index_select(0, victims).to(torch::kCPU);
            auto rows_cat    = torch::cat(rows_all);
            if (!st->pin_scratch.defined() || st->pin_scratch.size(0) < rows_cat.size(0)) {
                st->pin_scratch = torch::empty({std::max<int64_t>(rows_cat.size(0), 4096), 576},
                                               torch::TensorOptions().dtype(torch::kBFloat16).pinned_memory(true));
            }
            auto rows_host = st->pin_scratch.narrow(0, 0, rows_cat.size(0));
            rows_host.copy_(rows_cat);
            const size_t                     row_bytes = 576 * 2;
            char*                            pool_base = reinterpret_cast<char*>(main_pool_flat.data_ptr());
            char*                            src_base  = reinterpret_cast<char*>(rows_host.data_ptr());
            rtp_llm::BatchedMemoryCopyParams bp;
            bp.device_index = (int)main_pool_flat.get_device();
            bp.tiles.reserve(take);
            auto* slot_p = slots_cpu.data_ptr<long>();
            for (int64_t r = 0; r < take; ++r) {
                bp.tiles.push_back(
                    {pool_base + (size_t)slot_p[r] * row_bytes, src_base + (size_t)r * row_bytes, row_bytes});
            }
            rtp_llm::execBatchedMemoryCopy(bp);  // one runtime call, own stream, sync
            st->slot_pos.index_copy_(0, victims, pos_gpu.narrow(0, 0, take));
            st->slot_seen.index_fill_(0, victims, step);
            auto order    = st->slot_pos.argsort();
            st->s_idx     = st->slot_pos.index_select(0, order);
            st->s_slot    = st->stg_slots.index_select(0, order);
            st->s_logical = st->s_logical.index_select(0, order);
            // remap seen through order so kernel updates aligned array
            st->slot_seen = st->slot_seen.index_select(0, order);
            st->slot_pos  = st->s_idx.clone();
            st->stg_slots = st->s_slot.clone();
        }
    }
    // 2) build indices
    int topk = (int)sel.size(0);
    st->gidx.fill_(-1);
    st->counters.zero_();
    int threads = 256, blocks = (topk + threads - 1) / threads;
    build_indices_kernel<<<blocks, threads, 0, stream>>>(sel.data_ptr<int>(),
                                                         bt_row.data_ptr<int>(),
                                                         st->s_idx.data_ptr<long>(),
                                                         st->s_slot.data_ptr<long>(),
                                                         st->s_logical.data_ptr<long>(),
                                                         st->slot_seen.data_ptr<long>(),
                                                         st->gidx.data_ptr<int>(),
                                                         st->miss_buf.data_ptr<int>(),
                                                         st->counters.data_ptr<int>(),
                                                         topk,
                                                         (int)st->s_idx.size(0),
                                                         (int)hist,
                                                         step);
    finalize_kernel<<<1, 32, 0, stream>>>(
        st->gidx.data_ptr<int>(), st->counters.data_ptr<int>(), st->miss_buf.data_ptr<int>(), (int)cur_pos, topk);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "v32_ctx kernel launch failed");
    // 3) async fetch: read miss_buf on a recorded event from the fetch thread
    if (want_fetch && st->inflight.load() < 8192) {
        // capture completion via event; hand to worker with a tiny host callback
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming | cudaEventBlockingSync);
        cudaEventRecord(ev, stream);
        auto stc = st;
        auto ctx = g_ctx;
        std::thread([stc, ctx, ev]() {
            cudaEventSynchronize(ev);
            cudaEventDestroy(ev);
            const int        topk_n = (int)stc->miss_buf.size(0);
            std::vector<int> host(topk_n);
            {
                std::lock_guard<std::mutex> lk(ctx->copy_mu);
                cudaMemcpyAsync(host.data(),
                                stc->miss_buf.data_ptr<int>(),
                                sizeof(int) * topk_n,
                                cudaMemcpyDeviceToHost,
                                ctx->copy_stream);
                cudaStreamSynchronize(ctx->copy_stream);
            }
            int n = host[0];
            if (n <= 0)
                return;
            FetchJob job;
            job.st = stc;
            job.miss.assign(host.begin() + 1, host.begin() + 1 + n);
            stc->inflight += n;
            {
                std::lock_guard<std::mutex> lk(ctx->mu);
                if (ctx->q.size() < 512)
                    ctx->q.push_back(std::move(job));
            }
            ctx->cv.notify_one();
        }).detach();
    }
    return st->gidx.narrow(0, 0, topk);
}

torch::Tensor ctx_serve(int64_t       req_key,
                        int64_t       layer,
                        torch::Tensor sel,
                        torch::Tensor bt_row,
                        torch::Tensor main_pool_flat,
                        int64_t       hist,
                        int64_t       step,
                        bool          want_fetch) {
    return serve_impl(req_key, layer, sel, bt_row, main_pool_flat, hist, step, want_fetch, hist);
}

// T1: copy the current token's indexer-K bytes from the main scale pool into
// the side store so the native fused scorer sees it (lengths = kvlen).
void ctx_append_tok(int64_t       req_key,
                    int64_t       layer,
                    torch::Tensor idxp,
                    torch::Tensor pool_u8,
                    torch::Tensor bt_all,
                    int64_t       row_i,
                    int64_t       pos) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto bt_row = bt_all.select(0, row_i).to(torch::kInt32).contiguous();
    append_tok_kernel<<<1, 132, 0, stream>>>(reinterpret_cast<unsigned char*>(idxp.data_ptr()),
                                             reinterpret_cast<const unsigned char*>(pool_u8.data_ptr()),
                                             bt_row.data_ptr<int>(),
                                             (int)pos,
                                             (long)pool_u8.size(1));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "append_tok launch failed");
}

// T1: sel comes from the native fused topk over the full history (cur token
// included), so no forced cur append; write logical indices back into the
// kernel_topk row for the native convert-to-global.
void ctx_serve_wb(int64_t       req_key,
                  int64_t       layer,
                  torch::Tensor sel,
                  torch::Tensor bt_all,
                  torch::Tensor kernel_topk_all,
                  int64_t       row_i,
                  torch::Tensor main_pool_flat,
                  int64_t       kvlen,
                  int64_t       step,
                  bool          want_fetch) {
    auto it = g_ctx->stores.find(key_of(req_key, layer));
    TORCH_CHECK(it != g_ctx->stores.end(), "store missing");
    auto    bt_row  = bt_all.select(0, row_i).to(torch::kInt32).contiguous();
    auto    sel_row = sel.dim() == 2 ? sel.select(0, 0) : sel;
    auto    out     = serve_impl(req_key, layer, sel_row, bt_row, main_pool_flat, kvlen, step, want_fetch, -1);
    auto    ktr     = kernel_topk_all.select(0, row_i).reshape({-1});
    int64_t k = out.size(0), kw = ktr.size(0);
    if (k > kw)
        k = kw;
    ktr.narrow(0, 0, k).copy_(out.narrow(0, 0, k));
    if (kw > k)
        ktr.narrow(0, k, kw - k).fill_(-1);
}

std::vector<int64_t> ctx_debug(int64_t req_key, int64_t layer) {
    auto it = g_ctx->stores.find(key_of(req_key, layer));
    TORCH_CHECK(it != g_ctx->stores.end());
    auto    st     = it->second;
    auto    c      = st->counters.to(torch::kCPU);
    auto    staged = (st->slot_pos >= 0).sum().item<int64_t>();
    int64_t inbox;
    {
        std::lock_guard<std::mutex> lk(st->mu);
        inbox = (int64_t)st->inbox.size();
    }
    return {c[0].item<int64_t>(), c[1].item<int64_t>(), staged, inbox, (int64_t)st->inflight.load()};
}

// scattered pool rows (slots_cpu int64 [n]) -> contiguous pinned host rows.
// Uses the ported staged D2H path (gather kernel + single D2H, own stream).
void ctx_mirror_d2h(torch::Tensor pool_flat, torch::Tensor slots_cpu, torch::Tensor host_dst) {
    const int64_t n = slots_cpu.size(0);
    if (n == 0)
        return;
    const size_t                    row_bytes = (size_t)pool_flat.size(1) * pool_flat.element_size();
    rtp_llm::StagedMemoryCopyParams sp;
    sp.direction                   = rtp_llm::StagedMemoryCopyDirection::D2H;
    sp.device_index                = (int)pool_flat.get_device();
    sp.host_base                   = host_dst.data_ptr();
    sp.host_bytes                  = 0;
    sp.direct_pinned_host_segments = false;  // D2H ignores it and would bail out
    char* pool_base                = reinterpret_cast<char*>(pool_flat.data_ptr());
    auto* sl                       = slots_cpu.data_ptr<long>();
    sp.tiles.reserve(n);
    sp.host_segments.reserve(n);
    for (int64_t r = 0; r < n; ++r) {
        size_t off = (size_t)r * row_bytes;
        sp.tiles.push_back({pool_base + (size_t)sl[r] * row_bytes, off, row_bytes});
        sp.host_segments.push_back({reinterpret_cast<char*>(host_dst.data_ptr()) + off, off, row_bytes});
        sp.host_bytes = off + row_bytes;
    }
    static rtp_llm::StagedMemoryCopyScratch scratch;
    TORCH_CHECK(rtp_llm::execStagedMemoryCopy(sp, &scratch), "mirror d2h failed");
}

// Block-granular mirror: a whole 64-token block is contiguous in the paged pool,
// so aligned ranges copy straight D2H with no gather kernel and no staging.
// blk_cpu: int32 [n] physical block ids; host_dst: pinned [cap, row] mirror.
// Copies stay in flight across up to MIRROR_INFLIGHT calls so the launch thread
// is not blocked by PCIe; pass flush=true on the chunk that completes the
// history to make the whole mirror durable before it is served from.
constexpr int MIRROR_INFLIGHT = 8;

// Streams and events are device-scoped: a rank whose pools live on device r must
// not be served from a stream created on device 0.
struct DevCopy {
    cudaStream_t stream                       = nullptr;
    cudaEvent_t  mirror_ev[MIRROR_INFLIGHT]   = {};
    bool         mirror_live[MIRROR_INFLIGHT] = {};
    int          slot                         = 0;
    std::mutex   mu;
};

namespace {
std::unordered_map<int, std::shared_ptr<DevCopy>> g_devcopy;
std::mutex                                        g_devcopy_mu;
}  // namespace

static DevCopy& dev_copy(int device) {
    std::lock_guard<std::mutex> lk(g_devcopy_mu);
    auto&                       slot = g_devcopy[device];
    if (!slot) {
        slot = std::make_shared<DevCopy>();
        c10::cuda::CUDAGuard guard(device);
        cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking);
    }
    return *slot;
}

void ctx_mirror_blocks_d2h(torch::Tensor pool_flat,
                           torch::Tensor blk_cpu,
                           torch::Tensor host_dst,
                           int64_t       dst_token,
                           int64_t       block_tokens,
                           bool          flush) {
    const int64_t n = blk_cpu.size(0);
    if (n == 0)
        return;
    TORCH_CHECK(blk_cpu.scalar_type() == torch::kInt32 && blk_cpu.is_cpu(), "blk_cpu must be cpu int32");
    TORCH_CHECK(host_dst.is_pinned(), "host_dst must be pinned");
    TORCH_CHECK(pool_flat.size(1) == host_dst.size(1), "row width mismatch");
    const size_t  row_bytes = (size_t)pool_flat.size(1) * pool_flat.element_size();
    const size_t  blk_bytes = (size_t)block_tokens * row_bytes;
    const int64_t pool_blks = pool_flat.size(0) / block_tokens;
    TORCH_CHECK(dst_token + n * block_tokens <= host_dst.size(0), "mirror dst overflow");
    char* pool_base = reinterpret_cast<char*>(pool_flat.data_ptr());
    char* host_base = reinterpret_cast<char*>(host_dst.data_ptr());
    auto* b         = blk_cpu.data_ptr<int>();
    TORCH_CHECK(g_ctx != nullptr, "ctx_init not called");
    char*                       dst_base = host_base + (size_t)dst_token * row_bytes;
    const int                   device   = (int)pool_flat.get_device();
    c10::cuda::CUDAGuard        guard(device);
    DevCopy&                    dc = dev_copy(device);
    std::lock_guard<std::mutex> lk(dc.mu);
    const int                   slot = dc.slot;
    if (!dc.mirror_ev[slot])
        cudaEventCreateWithFlags(&dc.mirror_ev[slot], cudaEventDisableTiming);
    if (dc.mirror_live[slot])
        cudaEventSynchronize(dc.mirror_ev[slot]);  // oldest in-flight chunk is durable
    int64_t i = 0;
    while (i < n) {
        const int64_t blk = b[i];
        TORCH_CHECK(blk > 0 && blk < pool_blks, "mirror block id out of range");
        int64_t run = 1;  // physically consecutive blocks stay contiguous on both ends
        while (i + run < n && b[i + run] == blk + run && blk + run < pool_blks)
            ++run;
        cudaMemcpyAsync(dst_base + (size_t)i * blk_bytes,
                        pool_base + (size_t)blk * blk_bytes,
                        (size_t)run * blk_bytes,
                        cudaMemcpyDeviceToHost,
                        dc.stream);
        i += run;
    }
    cudaEventRecord(dc.mirror_ev[slot], dc.stream);
    dc.mirror_live[slot] = true;
    dc.slot              = (slot + 1) % MIRROR_INFLIGHT;
    if (flush) {
        cudaStreamSynchronize(dc.stream);
        for (int k = 0; k < MIRROR_INFLIGHT; ++k)
            dc.mirror_live[k] = false;
    }
}

// one call per layer per request: score (own kernel) + topk + drain + build
// + async miss fetch. q_fp8: [64,128] fp8 tensor; w: [64] float32.
torch::Tensor ctx_serve_full(int64_t       req_key,
                             int64_t       layer,
                             torch::Tensor q_all,
                             torch::Tensor w_all,
                             torch::Tensor idxp,
                             torch::Tensor bt_all,
                             torch::Tensor kernel_topk_all,
                             int64_t       row_i,
                             torch::Tensor main_pool_flat,
                             int64_t       hist,
                             int64_t       topk_sel,
                             int64_t       step,
                             bool          want_fetch) {
    auto q_fp8  = q_all.select(0, row_i);
    auto w      = w_all.select(0, row_i).reshape({-1}).to(torch::kFloat32);
    auto bt_row = bt_all.select(0, row_i).to(torch::kInt32).contiguous();
    auto it     = g_ctx->stores.find(key_of(req_key, layer));
    TORCH_CHECK(it != g_ctx->stores.end(), "store missing");
    auto st     = it->second;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (!st->scores.defined() || st->scores.size(0) < hist) {
        st->scores = torch::empty({hist + 8192}, torch::TensorOptions().dtype(torch::kFloat32).device(bt_row.device()));
    }
    int threads         = 256;
    int warps_per_block = threads / 32;
    int blocks          = (int)((hist + warps_per_block - 1) / warps_per_block);
    side_score_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const unsigned char*>(q_fp8.data_ptr()),
                                                      reinterpret_cast<const unsigned char*>(idxp.data_ptr()),
                                                      w.data_ptr<float>(),
                                                      st->scores.data_ptr<float>(),
                                                      (int)hist);
    int64_t k   = std::min<int64_t>(topk_sel, hist);
    auto    sel = std::get<1>(st->scores.narrow(0, 0, hist).topk(k)).to(torch::kInt32);
    auto    out = ctx_serve(req_key, layer, sel, bt_row, main_pool_flat, hist, step, want_fetch);
    // write logical indices straight into the native kernel_topk row; the
    // engine's convert-to-global then feeds attention (no python fwd hook).
    auto    ktr = kernel_topk_all.select(0, row_i).reshape({-1});
    int64_t kw  = ktr.size(0);
    ktr.narrow(0, 0, k).copy_(out.narrow(0, 0, k));
    if (kw > k)
        ktr.narrow(0, k, kw - k).fill_(-1);
    return out;
}

// ---- Tier-2 lossy attention: block-granularity hot pool with per-layer
// aliases. Scoring stays exact (full history); attention drops selections that
// are neither GPU-resident (engine block table > 0) nor in the per-layer hot
// pool built from the request's staging blocks. Hot-pool hits are remapped to
// staging logical coords (table_pos*64+off) so the native convert-to-global
// translates them through the UNCHANGED engine table. Misses are exported via
// mapped pinned memory (zero sync) and prefetched whole-block on the shared
// copy stream with >=1-step lag. Counters give the true warm-pool hit ratio.
namespace {
constexpr int LOSSY_MISS_CAP = 512;

__global__ void lossy_mask_kernel(int* __restrict__ ktr,                  // [kw] logical sel, in/out
                                  const int* __restrict__ bt,             // engine kbt row
                                  int* __restrict__ map_pos,              // [mw] logical block -> staging table pos
                                  const int* __restrict__ alias_lb,       // [n_alias] blocks to update
                                  const int* __restrict__ alias_val,      // [n_alias] new table pos (0 = retire)
                                  int* __restrict__ miss,                 // pinned [2+CAP]: tag, count, blocks
                                  unsigned long long* __restrict__ cnts,  // device [4]: tail,pool,miss,serves
                                  int kw,
                                  int w,
                                  int mw,
                                  int hist,
                                  int tag,
                                  int n_alias,
                                  int stg_lo,  // staging table positions [stg_lo, stg_hi]:
                                  int stg_hi,  // still non-zero in bt, but python owns them
                                  int diag,
                                  int want_miss) {  // 0: no consumer, skip the host export
    __shared__ int s_cnt[3];
    __shared__ int s_n;
    // Publishing and retiring aliases here rather than in a separate launch keeps
    // the update ordered ahead of the remap for free: one block, one barrier.
    for (int i = threadIdx.x; i < n_alias; i += blockDim.x) {
        int b = alias_lb[i];
        if (b >= 0 && b < mw)
            map_pos[b] = alias_val[i];
    }
    if (threadIdx.x < 3)
        s_cnt[threadIdx.x] = 0;
    if (threadIdx.x == 0)
        s_n = 0;
    __syncthreads();
    int c0 = 0, c1 = 0, c2 = 0;  // tallied per thread; a shared atomic per
                                 // selection would serialise ~1800 of them
    for (int i = threadIdx.x; i < kw; i += blockDim.x) {
        int p = ktr[i];
        if (p < 0)
            continue;
        if (p >= hist) {
            if (!diag)
                ktr[i] = -1;
            continue;
        }
        int j = p / BS;
        // A staging position's bt entry is still non-zero, but its contents are
        // the hot pool, not this logical block: it is not resident.
        if (j < w && bt[j] > 0 && !(j >= stg_lo && j <= stg_hi)) {
            ++c0;
            continue;
        }
        int mp = (j < mw) ? map_pos[j] : 0;
        if (mp > 0) {
            if (!diag)
                ktr[i] = mp * BS + (p % BS);
            ++c1;
            continue;
        }
        if (!diag)
            ktr[i] = -1;
        ++c2;
        if (want_miss) {
            int slot = atomicAdd(&s_n, 1);
            if (slot < LOSSY_MISS_CAP)
                miss[2 + slot] = j;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        c0 += __shfl_down_sync(0xffffffffu, c0, off);
        c1 += __shfl_down_sync(0xffffffffu, c1, off);
        c2 += __shfl_down_sync(0xffffffffu, c2, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&s_cnt[0], c0);
        atomicAdd(&s_cnt[1], c1);
        atomicAdd(&s_cnt[2], c2);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (want_miss) {
            miss[1] = s_n < LOSSY_MISS_CAP ? s_n : LOSSY_MISS_CAP;
            __threadfence_system();  // entries+count visible before the tag flips
            miss[0] = tag;
        }
        atomicAdd(&cnts[0], (unsigned long long)s_cnt[0]);
        atomicAdd(&cnts[1], (unsigned long long)s_cnt[1]);
        atomicAdd(&cnts[2], (unsigned long long)s_cnt[2]);
        atomicAdd(&cnts[3], 1ull);
    }
}

// Whole-block gather from the pinned host mirror (device-addressable under UVA)
// into pool slots: one launch replaces the per-block cudaMemcpyAsync loop, which
// at 8 blocks x 61 layers was ~500 CUDA calls per decode step of pure launch tax.
// vec_per_blk counts 16B units in one block (BS * row_w * 2B / 16).
__global__ void lossy_fetch_blocks_kernel(int4* __restrict__ pool,
                                          const int4* __restrict__ host_mirror,
                                          const int* __restrict__ src_blk,
                                          const int* __restrict__ dst_blk,
                                          int m,
                                          int vec_per_blk) {
    const long total = (long)m * vec_per_blk;
    for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long)gridDim.x * blockDim.x) {
        const int  i                               = (int)(t / vec_per_blk);
        const long off                             = t - (long)i * vec_per_blk;
        pool[(long)dst_blk[i] * vec_per_blk + off] = host_mirror[(long)src_blk[i] * vec_per_blk + off];
    }
}

// ---- Scheme C (lossless): instead of dropping non-resident selections, hand
// each one a scratch token slot inside the request's staging blocks and remap it
// there. Slot k lives at staging table position jpos[k / BS], offset k % BS, so
// the native convert-to-global reaches it through the unchanged engine table -
// same trick as the lossy hot pool, but at token rather than block granularity.
// Capacity is S*BS slots (32*64 = 2048), which covers a full top-2048 row.
// Multi-block: with one block this ran on a single SM while the rest of the GPU
// idled. The slot counter therefore has to be a device atomic, and it needs to be
// zero on entry - so need_n holds two slots and each call zeroes the one the next
// call will use. Same stream, so the previous call's fetch has already read it.
__global__ void lossless_mask_kernel(int* __restrict__ ktr,                  // [kw] logical sel, in/out
                                     const int* __restrict__ bt,             // engine kbt row
                                     const int* __restrict__ jpos,           // [S] staging table positions
                                     int* __restrict__ need_tok,             // [S*BS] out: wanted global token
                                     int* __restrict__ need_n,               // [2] ping-pong wanted count
                                     unsigned long long* __restrict__ cnts,  // tail,fetch,overflow,serves
                                     int kw,
                                     int w,
                                     int S,
                                     int stg_lo,  // staging positions hold scratch, not
                                     int stg_hi,  // the logical block bt still names
                                     int hist,
                                     int parity,
                                     int mode) {  // 0/1 remap misses; 2 drop misses (attribution only)
    if (blockIdx.x == 0 && threadIdx.x == 0)
        need_n[1 - parity] = 0;  // hand the next call a zeroed counter
    const int cap = S * BS;
    int       c0 = 0, c1 = 0, c2 = 0;  // per-thread; reduced through the warp below
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kw; i += blockDim.x * gridDim.x) {
        int p = ktr[i];
        if (p < 0)
            continue;
        if (p >= hist) {
            ktr[i] = -1;
            continue;
        }
        int j = p / BS;
        // A staging position's bt entry is still non-zero, but it holds scratch:
        // treat it as offloaded so it gets fetched rather than read stale.
        if (j < w && bt[j] > 0 && !(j >= stg_lo && j <= stg_hi)) {
            ++c0;
            continue;
        }
        if (mode == 2) {
            ktr[i] = -1;
            ++c2;
            continue;
        }
        int k = atomicAdd(&need_n[parity], 1);  // slot identity, not reducible
        if (k < cap) {
            need_tok[k] = p;
            ktr[i]      = jpos[k / BS] * BS + (k % BS);
            ++c1;
        } else {
            ktr[i] = -1;  // scratch exhausted: degrade to lossy rather than read garbage
            ++c2;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        c0 += __shfl_down_sync(0xffffffffu, c0, off);
        c1 += __shfl_down_sync(0xffffffffu, c1, off);
        c2 += __shfl_down_sync(0xffffffffu, c2, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&cnts[0], (unsigned long long)c0);
        atomicAdd(&cnts[1], (unsigned long long)c1);
        atomicAdd(&cnts[2], (unsigned long long)c2);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicAdd(&cnts[3], 1ull);
}

// Token-granular companion to lossless_mask_kernel: pull exactly the wanted
// tokens out of the pinned host mirror into their scratch slots. Runs on the
// compute stream right after the mask so attention is guaranteed to see real
// data - that ordering is what makes scheme C synchronous, and its cost is
// precisely the price of lossless offload we want to measure.
__global__ void lossless_fetch_kernel(int4* __restrict__ pool,
                                      const int4* __restrict__ host_mirror,
                                      const int* __restrict__ need_tok,
                                      const int* __restrict__ need_n,
                                      const int* __restrict__ sb,  // [S] physical staging block ids
                                      int cap,
                                      int vec_per_row,
                                      int parity) {
    int n = need_n[parity];
    if (n > cap)
        n = cap;
    const long total = (long)n * vec_per_row;
    for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long)gridDim.x * blockDim.x) {
        const int  k                  = (int)(t / vec_per_row);
        const long off                = t - (long)k * vec_per_row;
        const long dst                = (long)sb[k / BS] * BS + (k % BS);
        pool[dst * vec_per_row + off] = host_mirror[(long)need_tok[k] * vec_per_row + off];
    }
}

struct LossyState {
    torch::Tensor    map_pos;           // gpu int32 [mw]
    torch::Tensor    miss_hdr;          // pinned int32 [2+CAP]
    torch::Tensor    pin_vals;          // pinned int32 [4*S]: alias lb list | alias val list
    torch::Tensor    pin_blk;           // pinned int32 [2*S]: src host blocks | dst pool blocks
    torch::Tensor    jpos_dev, sb_dev;  // gpu int32 [S] (scheme C remap needs them on device)
    torch::Tensor    need_tok, need_n;  // gpu int32 [S*BS] / [1] (scheme C wanted tokens)
    torch::Tensor    kv_cur, kv_prev;   // host mirror refs (prev keeps in-flight src alive)
    std::vector<int> jpos, sb, occupant;
    // aliases whose block copy is still in flight; published once ev_copy fires
    std::vector<int> inflight_lb, inflight_pos;
    int              ring         = 0;
    int              consumed_tag = 0;
    unsigned         serve_seq    = 0;        // alternates the ping-pong counter slot
    int              stg_lo       = 1;        // table positions python owns as staging
    int              stg_hi       = 0;        // (inclusive; hi<lo means none)
    cudaEvent_t      ev_copy      = nullptr;  // copy stream: prefetch landed
    cudaEvent_t      ev_clear     = nullptr;  // compute stream: eviction visible
    int              device       = 0;
    bool             copy_live    = false;  // ev_copy holds an un-reaped prefetch
    bool             alias_live   = false;  // ev_clear guards pin_vals reuse
    ~LossyState() {
        if (ev_copy)
            cudaEventDestroy(ev_copy);
        if (ev_clear)
            cudaEventDestroy(ev_clear);
    }
};
std::unordered_map<int64_t, std::shared_ptr<LossyState>> g_lossy;
std::mutex                                               g_lossy_mu;
// Counters live in device memory: atomicAdd into pinned host memory would make
// the compute stream wait on a PCIe round trip once per counter per layer.
// [0..3] tail/pool-or-fetch/miss/serves, [4] admit tokens requested,
// [5] admit tokens silently skipped (sentinel source or invalid destination)
std::unordered_map<int, torch::Tensor> g_cnts;  // device -> int64 [6]
const int g_lossy_diag = std::getenv("V32_LOSSY_DIAG") ? atoi(std::getenv("V32_LOSSY_DIAG")) : 0;

static unsigned long long* cnts_for(int device) {
    auto& t = g_cnts[device];
    if (!t.defined()) {
        c10::cuda::CUDAGuard guard(device);
        t = torch::zeros({6}, torch::TensorOptions().dtype(torch::kInt64).device(torch::Device(torch::kCUDA, device)));
    }
    return reinterpret_cast<unsigned long long*>(t.data_ptr<int64_t>());
}
}  // namespace

void ctx_lossy_register(int64_t                      req_key,
                        int64_t                      layer,
                        torch::Tensor                jpos_cpu,  // cpu int32 [S] table positions of staging blocks
                        torch::Tensor                sb_cpu,    // cpu int32 [S] physical staging block ids
                        int64_t                      mw,        // map width (>= offloaded block count)
                        int64_t                      device,
                        c10::optional<torch::Tensor> kv_host) {
    ctx_init();
    auto st = std::make_shared<LossyState>();
    if (kv_host.has_value())
        st->kv_cur = *kv_host;
    auto jp = jpos_cpu.to(torch::kInt32).contiguous();
    auto sc = sb_cpu.to(torch::kInt32).contiguous();
    int  S  = (int)jp.size(0);
    TORCH_CHECK(S > 0 && S <= 1024, "lossy: bad staging slot count");
    st->jpos.assign(jp.data_ptr<int>(), jp.data_ptr<int>() + S);
    st->sb.assign(sc.data_ptr<int>(), sc.data_ptr<int>() + S);
    st->occupant.assign(S, -1);
    // The engine leaves these table entries pointing at real blocks while handing
    // their contents to python, so residency tests must exclude them.
    st->stg_lo               = *std::min_element(st->jpos.begin(), st->jpos.end());
    st->stg_hi               = *std::max_element(st->jpos.begin(), st->jpos.end());
    auto                 dev = torch::Device(torch::kCUDA, (int)device);
    c10::cuda::CUDAGuard guard((int)device);
    st->device   = (int)device;
    st->map_pos  = torch::zeros({mw}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    st->miss_hdr = torch::zeros({2 + LOSSY_MISS_CAP}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    st->pin_vals = torch::zeros({4 * S}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    st->pin_blk  = torch::zeros({2 * S}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    st->jpos_dev = jp.to(dev);
    st->sb_dev   = sc.to(dev);
    st->need_tok = torch::zeros({S * BS}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    st->need_n   = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
    cudaEventCreateWithFlags(&st->ev_copy, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&st->ev_clear, cudaEventDisableTiming);
    std::lock_guard<std::mutex> lk(g_lossy_mu);
    cnts_for((int)device);  // allocate off the hot path
    g_lossy[key_of(req_key, layer)] = st;
}

bool ctx_lossy_has(int64_t req_key, int64_t layer) {
    std::lock_guard<std::mutex> lk(g_lossy_mu);
    return g_lossy.count(key_of(req_key, layer)) > 0;
}

// one call per (row, layer): consume last landed miss batch -> async whole-
// block prefetch on the copy stream, then mask/remap the kernel_topk row in
// place on the compute stream. Zero host syncs.
void ctx_lossy_serve(int64_t       req_key,
                     int64_t       layer,
                     torch::Tensor kv_host,  // pinned bf16 [cap,576] mirror (prefetch source)
                     torch::Tensor kbt_all,
                     torch::Tensor kernel_topk_all,
                     int64_t       row_i,
                     torch::Tensor main_pool_flat,  // [slots,576] bf16 (this layer)
                     int64_t       kvlen,
                     int64_t       step,
                     int64_t       prefetch_cap) {
    std::shared_ptr<LossyState> st;
    {
        std::lock_guard<std::mutex> lk(g_lossy_mu);
        auto                        it = g_lossy.find(key_of(req_key, layer));
        TORCH_CHECK(it != g_lossy.end(), "lossy state missing");
        st = it->second;
    }
    c10::cuda::CUDAGuard guard(st->device);  // pools are device-scoped, so are we
    auto                 stream = at::cuda::getCurrentCUDAStream();
    if (!st->kv_cur.defined() || st->kv_cur.data_ptr() != kv_host.data_ptr()) {
        st->kv_prev = st->kv_cur;  // keep old pinned buffer alive across in-flight copies
        st->kv_cur  = kv_host;
    }
    const int mw      = (int)st->map_pos.size(0);
    const int S       = (int)st->jpos.size();
    int*      hdr     = st->miss_hdr.data_ptr<int>();
    int*      pv      = st->pin_vals.data_ptr<int>();  // [0,2S) lb list, [2S,4S) val list
    int       n_alias = 0;

    // pin_vals is read asynchronously by the alias kernel; ev_clear is recorded
    // right after that kernel, so an incomplete ev_clear means the previous batch
    // is still being read. Deferring a step is harmless - the block simply stays
    // a miss for one more step - and it keeps the buffer race-free without a sync.
    const bool alias_buf_free = !st->alias_live || cudaEventQuery(st->ev_clear) == cudaSuccess;
    if (alias_buf_free)
        st->alias_live = false;

    // (1) publish aliases whose block copy has landed. Polling the copy event on
    // the host costs nothing on the compute stream; the publish itself is ordered
    // before this step's mask kernel, so a remap can only ever name real data.
    if (alias_buf_free && st->copy_live && cudaEventQuery(st->ev_copy) == cudaSuccess) {
        st->copy_live = false;
        for (size_t i = 0; i < st->inflight_lb.size(); ++i) {
            pv[n_alias]         = st->inflight_lb[i];
            pv[2 * S + n_alias] = st->inflight_pos[i];
            ++n_alias;
        }
        st->inflight_lb.clear();
        st->inflight_pos.clear();
    }

    // (2) consume the last landed miss batch and pick victims. Only one prefetch
    // batch per (request, layer) is ever in flight, which is also what guarantees
    // the alias buffer above has been fully consumed before we rewrite it.
    int n_want = 0;
    // 8 blocks/layer/step already moves 36 MB per decode step, so this ceiling is
    // generous; it exists so the plan can live on the stack.
    constexpr int MAX_PREFETCH = 64;
    int           want[MAX_PREFETCH], dst_sb[MAX_PREFETCH];
    const int     max_want = (int)std::min<int64_t>(std::min<int64_t>(prefetch_cap, S), MAX_PREFETCH);
    const int     tag      = hdr[0];
    if (prefetch_cap > 0 && alias_buf_free && !st->copy_live && tag > st->consumed_tag) {
        st->consumed_tag = tag;
        int n            = hdr[1];
        if (n > LOSSY_MISS_CAP)
            n = LOSSY_MISS_CAP;
        const int64_t host_rows = st->kv_cur.defined() ? st->kv_cur.size(0) : 0;
        const int64_t pool_rows = main_pool_flat.size(0);
        for (int i = 0; i < n && n_want < max_want; ++i) {
            const int lb = hdr[2 + i];
            if (lb < 0 || lb >= mw || (int64_t)(lb + 1) * BS > host_rows)
                continue;
            bool dup = false;
            for (int v = 0; v < n_want && !dup; ++v)
                dup = want[v] == lb;
            for (int occ : st->occupant)
                if (occ == lb)
                    dup = true;
            if (dup)
                continue;
            const int s = st->ring;
            if ((int64_t)(st->sb[s] + 1) * BS > pool_rows)
                continue;  // staging slot outside this layer's pool
            st->ring            = (st->ring + 1) % S;
            pv[n_alias]         = st->occupant[s];  // retire the outgoing alias
            pv[2 * S + n_alias] = 0;
            ++n_alias;
            st->occupant[s] = lb;
            st->inflight_lb.push_back(lb);
            st->inflight_pos.push_back(st->jpos[s]);
            dst_sb[n_want] = st->sb[s];
            want[n_want]   = lb;
            ++n_want;
        }
    }

    // (3) the mask kernel applies both the publishes and the retirements as its
    // first act, so no separate alias launch is needed; ev_clear is recorded after
    // it and is what the copy stream waits on.
    TORCH_CHECK(kbt_all.scalar_type() == torch::kInt32 && kernel_topk_all.scalar_type() == torch::kInt32,
                "int32 expected");
    TORCH_CHECK(kbt_all.is_contiguous() && kernel_topk_all.is_contiguous(), "contiguous base tensors expected");
    TORCH_CHECK(row_i >= 0 && row_i < kbt_all.size(0) && row_i < kernel_topk_all.size(0), "row out of range");
    const int w       = (int)kbt_all.size(1);
    const int kw      = (int)(kernel_topk_all.numel() / kernel_topk_all.size(0));
    int*      bt_ptr  = kbt_all.data_ptr<int>() + row_i * (int64_t)w;
    int*      ktr_ptr = kernel_topk_all.data_ptr<int>() + row_i * (int64_t)kw;
    lossy_mask_kernel<<<1, 1024, 0, stream>>>(ktr_ptr,
                                              bt_ptr,
                                              st->map_pos.data_ptr<int>(),
                                              pv,
                                              pv + 2 * S,
                                              hdr,
                                              cnts_for(st->device),
                                              kw,
                                              w,
                                              mw,
                                              (int)kvlen,
                                              (int)step,
                                              n_alias,
                                              st->stg_lo,
                                              st->stg_hi,
                                              g_lossy_diag,
                                              prefetch_cap > 0 ? 1 : 0);
    if (n_alias > 0) {
        cudaEventRecord(st->ev_clear, stream);
        st->alias_live = true;
    }

    // (4) issue the prefetch. The copy stream waits for the retirement to be
    // visible, so it can never clobber a slot the compute stream may still alias;
    // the compute stream itself waits for nothing.
    if (n_want > 0) {
        const size_t row_bytes = (size_t)main_pool_flat.size(1) * main_pool_flat.element_size();
        TORCH_CHECK(row_bytes % 16 == 0, "lossy prefetch needs 16B-aligned rows");
        const int vec_per_blk = (int)((size_t)BS * row_bytes / 16);
        int*      pb          = st->pin_blk.data_ptr<int>();
        for (int i = 0; i < n_want; ++i) {
            pb[i]     = want[i];
            pb[S + i] = dst_sb[i];
        }
        DevCopy&                    dc = dev_copy(st->device);
        std::lock_guard<std::mutex> lk(dc.mu);
        cudaStreamWaitEvent(dc.stream, st->ev_clear, 0);
        const long total   = (long)n_want * vec_per_blk;
        const int  threads = 256;
        int        blocks  = (int)((total + threads - 1) / threads);
        if (blocks > 1024)
            blocks = 1024;
        lossy_fetch_blocks_kernel<<<blocks, threads, 0, dc.stream>>>(
            reinterpret_cast<int4*>(main_pool_flat.data_ptr()),
            reinterpret_cast<const int4*>(st->kv_cur.data_ptr()),
            pb,
            pb + S,
            n_want,
            vec_per_blk);
        cudaEventRecord(st->ev_copy, dc.stream);
        st->copy_live = true;
    }
}

// Opt-in CUDA-event timing for the three device-side pieces we add, so the actual
// PCIe gather can be reported separately from launch overhead and from the mask.
// Timing forces a sync per call, so it is only for attribution runs.
namespace {
struct KTime {
    cudaEvent_t a = nullptr, b = nullptr;
    double      mask = 0, fetch = 0, append = 0;
    long        n_mask = 0, n_fetch = 0, n_append = 0;
    bool        on = false;
};
KTime      g_kt;
const long g_ktime_budget = std::getenv("V32_KTIME") ? atol(std::getenv("V32_KTIME")) : 0;

// returns true if this call should be timed; caller must then call kt_end
bool v32_kt_begin_impl(cudaStream_t stream) {
    if (g_ktime_budget <= 0 || g_kt.n_mask + g_kt.n_fetch + g_kt.n_append >= 3 * g_ktime_budget)
        return false;
    if (!g_kt.a) {
        cudaEventCreate(&g_kt.a);
        cudaEventCreate(&g_kt.b);
    }
    cudaEventRecord(g_kt.a, stream);
    return true;
}

void v32_kt_end_impl(cudaStream_t stream, double* acc, long* cnt) {
    cudaEventRecord(g_kt.b, stream);
    cudaEventSynchronize(g_kt.b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, g_kt.a, g_kt.b);
    *acc += ms;
    ++*cnt;
}
}  // namespace

bool v32_kt_begin(cudaStream_t stream) {
    return v32_kt_begin_impl(stream);
}

void v32_kt_end(cudaStream_t stream, int which) {
    if (which == 0)
        v32_kt_end_impl(stream, &g_kt.mask, &g_kt.n_mask);
    else if (which == 1)
        v32_kt_end_impl(stream, &g_kt.fetch, &g_kt.n_fetch);
    else
        v32_kt_end_impl(stream, &g_kt.append, &g_kt.n_append);
}

std::vector<double> ctx_ktimings() {
    return {g_kt.mask, (double)g_kt.n_mask, g_kt.fetch, (double)g_kt.n_fetch, g_kt.append, (double)g_kt.n_append};
}

// Measures the floor serve is built on: device guard plus one empty launch on the
// compute stream, and nothing else. The gap between this and a real serve is what
// remains to be optimised.
__global__ void probe_empty_kernel() {}

void ctx_probe_launch(int64_t device) {
    c10::cuda::CUDAGuard guard((int)device);
    probe_empty_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>();
}

void ctx_lossy_reset_counters() {
    std::lock_guard<std::mutex> lk(g_lossy_mu);
    for (auto& kv : g_cnts)
        if (kv.second.defined())
            kv.second.zero_();
}

std::vector<int64_t> ctx_lossy_counters() {
    std::lock_guard<std::mutex> lk(g_lossy_mu);
    std::vector<int64_t>        acc{0, 0, 0, 0, 0, 0};
    for (auto& kv : g_cnts) {
        if (!kv.second.defined())
            continue;
        auto  host = kv.second.to(torch::kCPU);
        auto* p    = host.data_ptr<int64_t>();
        for (int i = 0; i < 6; ++i)
            acc[i] += p[i];
    }
    return acc;
}

// ---- Scheme C entry: lossless offload. Attention receives exactly the
// baseline's top-k; every non-resident selection is remapped into a scratch
// token slot and gathered from the pinned host mirror on the compute stream
// before attention runs. No hot pool, no aliases, no miss export to the host -
// the only cost is the gather sitting on the critical path, which is the number
// this scheme exists to measure. Counters become tail / fetched / overflow /
// serves, where a non-zero overflow means the scratch ran out and the step
// silently degraded to lossy.
// Shared core of scheme C, operating on an already-resolved state: the steady
// state calls this once per layer and must not pay a second map lookup. The pool
// may arrive flat [slots,576] or in its native [blocks,64,576] shape; only the
// row width (last dim) and base pointer matter.
static void lossless_serve_impl(LossyState&          st,
                                const torch::Tensor& kbt_all,
                                const torch::Tensor& kernel_topk_all,
                                int64_t              row_i,
                                const torch::Tensor& pool,
                                int64_t              kvlen,
                                int64_t              mode) {
    c10::cuda::CUDAGuard guard(st.device);
    auto                 stream    = at::cuda::getCurrentCUDAStream();
    const int64_t        row_elems = pool.size(-1);
    const size_t         row_bytes = (size_t)row_elems * pool.element_size();
    TORCH_CHECK(row_bytes % 16 == 0, "lossless fetch needs 16B-aligned rows");
    TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
    TORCH_CHECK(row_elems == st.kv_cur.size(1), "mirror row width mismatch");
    const int S   = (int)st.jpos.size();
    const int cap = S * BS;

    TORCH_CHECK(kbt_all.scalar_type() == torch::kInt32 && kernel_topk_all.scalar_type() == torch::kInt32,
                "int32 expected");
    TORCH_CHECK(kbt_all.is_contiguous() && kernel_topk_all.is_contiguous(), "contiguous base tensors expected");
    TORCH_CHECK(row_i >= 0 && row_i < kbt_all.size(0) && row_i < kernel_topk_all.size(0), "row out of range");
    const int w       = (int)kbt_all.size(1);
    const int kw      = (int)(kernel_topk_all.numel() / kernel_topk_all.size(0));
    int*      bt_ptr  = kbt_all.data_ptr<int>() + row_i * (int64_t)w;
    int*      ktr_ptr = kernel_topk_all.data_ptr<int>() + row_i * (int64_t)kw;

    // kw is 2048 selections; 8x256 gives one per thread across 8 SMs instead of
    // crowding a single one.
    const int  parity   = (int)(st.serve_seq++ & 1);
    const int  m_thr    = 256;
    const int  m_blocks = (kw + m_thr - 1) / m_thr;
    const bool kt_m     = v32_kt_begin(stream);
    lossless_mask_kernel<<<m_blocks, m_thr, 0, stream>>>(ktr_ptr,
                                                         bt_ptr,
                                                         st.jpos_dev.data_ptr<int>(),
                                                         st.need_tok.data_ptr<int>(),
                                                         st.need_n.data_ptr<int>(),
                                                         cnts_for(st.device),
                                                         kw,
                                                         w,
                                                         S,
                                                         st.stg_lo,
                                                         st.stg_hi,
                                                         (int)kvlen,
                                                         parity,
                                                         (int)mode);
    if (kt_m)
        v32_kt_end(stream, 0);

    // The gather is PCIe-bound, not compute-bound: it wants several loads in flight
    // per thread and a small SM footprint so it does not crowd attention. One int4
    // per thread across 256 blocks did the opposite.
    const int vec_per_row = (int)(row_bytes / 16);
    const int threads     = 256;
    int       blocks      = (int)(((long)cap * vec_per_row + threads - 1) / threads);
    if (blocks > 48)
        blocks = 48;
    if (mode == 0) {
        const bool kt_f = v32_kt_begin(stream);
        lossless_fetch_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<int4*>(pool.data_ptr()),
                                                              reinterpret_cast<const int4*>(st.kv_cur.data_ptr()),
                                                              st.need_tok.data_ptr<int>(),
                                                              st.need_n.data_ptr<int>(),
                                                              st.sb_dev.data_ptr<int>(),
                                                              cap,
                                                              vec_per_row,
                                                              parity);
        if (kt_f)
            v32_kt_end(stream, 1);
    }
}

void ctx_lossless_serve(int64_t       req_key,
                        int64_t       layer,
                        torch::Tensor kv_host,  // pinned bf16 [cap,576] mirror
                        torch::Tensor kbt_all,
                        torch::Tensor kernel_topk_all,
                        int64_t       row_i,
                        torch::Tensor main_pool_flat,  // [slots,576] bf16 (this layer)
                        int64_t       kvlen,
                        int64_t       mode) {
    TORCH_CHECK(mode >= 0 && mode <= 2, "lossless attribution mode must be 0..2");
    std::shared_ptr<LossyState> st;
    {
        std::lock_guard<std::mutex> lk(g_lossy_mu);
        auto                        it = g_lossy.find(key_of(req_key, layer));
        TORCH_CHECK(it != g_lossy.end(), "lossy state missing");
        st = it->second;
    }
    if (!st->kv_cur.defined() || st->kv_cur.data_ptr() != kv_host.data_ptr()) {
        st->kv_prev = st->kv_cur;
        st->kv_cur  = kv_host;
    }
    lossless_serve_impl(*st, kbt_all, kernel_topk_all, row_i, main_pool_flat, kvlen, mode);
}

bool ctx_lossless_try_serve(int64_t       req_key,
                            int64_t       layer,
                            torch::Tensor kbt_all,
                            torch::Tensor kernel_topk_all,
                            int64_t       row_i,
                            torch::Tensor main_pool_flat,
                            int64_t       kvlen,
                            int64_t       mode) {
    std::shared_ptr<LossyState> st;
    {
        std::lock_guard<std::mutex> lk(g_lossy_mu);
        auto                        it = g_lossy.find(key_of(req_key, layer));
        if (it == g_lossy.end() || !it->second->kv_cur.defined())
            return false;
        st = it->second;
    }
    lossless_serve_impl(*st, kbt_all, kernel_topk_all, row_i, main_pool_flat, kvlen, mode);
    return true;
}

// ---- Step-armed fast path. Python arms the plan once per decode step (layer 0)
// after its request-level checks; every layer then makes one 4-argument call that
// reads the armed plan and a per-layer cached state pointer. This removes the
// per-layer python dict/plan work and the map lookup that made the first fast
// path recover only 0.1ms of the measured 1.4ms.
namespace {
struct FastStep {
    int64_t                                  step  = -1;  // python _step this plan is valid for
    int64_t                                  key   = 0;
    int32_t                                  row_i = 0;
    int32_t                                  kvlen = 0;
    int32_t                                  mode  = 0;
    std::vector<std::shared_ptr<LossyState>> layers;  // lazily resolved per layer
};
FastStep g_fast_step;
}  // namespace

void ctx_step_plan(int64_t step, int64_t key, int64_t row_i, int64_t kvlen, int64_t mode, int64_t max_layers) {
    TORCH_CHECK(mode >= 0 && mode <= 2, "lossless attribution mode must be 0..2");
    TORCH_CHECK(max_layers > 0 && max_layers <= 4096, "bad layer count");
    auto& fs = g_fast_step;
    if (fs.key != key || (int64_t)fs.layers.size() != max_layers) {
        // new request (or first arm): drop cached states so a recycled block-0 key
        // can never serve from the previous request's mirror
        fs.layers.assign((size_t)max_layers, nullptr);
        fs.key = key;
    }
    fs.step  = step;
    fs.row_i = (int32_t)row_i;
    fs.kvlen = (int32_t)kvlen;
    fs.mode  = (int32_t)mode;
}

bool ctx_step_serve(
    int64_t step, int64_t layer, torch::Tensor kbt_all, torch::Tensor kernel_topk_all, torch::Tensor pool) {
    auto& fs = g_fast_step;
    if (fs.step != step || layer < 0 || (size_t)layer >= fs.layers.size())
        return false;
    auto& slot = fs.layers[(size_t)layer];
    if (!slot) {
        std::lock_guard<std::mutex> lk(g_lossy_mu);
        auto                        it = g_lossy.find(key_of(fs.key, layer));
        if (it == g_lossy.end() || !it->second->kv_cur.defined())
            return false;  // not registered yet: python slow path owns this layer
        slot = it->second;
    }
    lossless_serve_impl(*slot, kbt_all, kernel_topk_all, fs.row_i, pool, fs.kvlen, fs.mode);
    return true;
}

void ctx_lossy_release(int64_t req_key) {
    std::lock_guard<std::mutex> lk(g_lossy_mu);
    if (g_fast_step.key == req_key) {
        // the armed plan and its cached states alias this request; disarm before
        // the map entries go away so a stale step can never serve freed mirrors
        g_fast_step.step = -1;
        g_fast_step.layers.assign(g_fast_step.layers.size(), nullptr);
    }
    for (int l = 0; l < 128; ++l)
        g_lossy.erase(key_of(req_key, l));
}

// ---- v32 admission mirror adoption (engine-owned buffers, staging-ring
// admission). The engine wheel exports C symbols; we bind them at runtime so
// this extension needs no link-time dependency on the wheel.
#include <dlfcn.h>
typedef int (*v32_adm_lookup_fn)(
    int64_t, int32_t, void**, int64_t*, int64_t*, void**, int64_t*, int64_t*, int64_t*, int32_t*);
typedef void (*v32_adm_release_fn)(int64_t);
typedef int64_t (*v32_adm_generation_fn)(int64_t);
typedef void (*v32_adm_release_generation_fn)(int64_t, int64_t);
static v32_adm_lookup_fn             g_adm_lookup             = nullptr;
static v32_adm_release_fn            g_adm_release            = nullptr;
static v32_adm_generation_fn         g_adm_generation         = nullptr;
static v32_adm_release_generation_fn g_adm_release_generation = nullptr;

bool ctx_admission_open(const std::string& engine_so_path) {
    void* h = dlopen(engine_so_path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (!h)
        h = dlopen(engine_so_path.c_str(), RTLD_LAZY);
    if (!h)
        return false;
    g_adm_lookup             = (v32_adm_lookup_fn)dlsym(h, "rtp_v32_admission_lookup");
    g_adm_release            = (v32_adm_release_fn)dlsym(h, "rtp_v32_admission_release");
    g_adm_generation         = (v32_adm_generation_fn)dlsym(h, "rtp_v32_admission_generation");
    g_adm_release_generation = (v32_adm_release_generation_fn)dlsym(h, "rtp_v32_admission_release_generation");
    return g_adm_lookup != nullptr && g_adm_release != nullptr;
}

// Returns (kv_host [cap,576] bf16, None, durable_tokens, generation) or None.
// The second tuple slot is retained for compatibility with the retired indexer
// shadow buffer. The tensor aliases engine-owned memory.
py::object ctx_adopt(int64_t req_key, int64_t layer) {
    if (!g_adm_lookup)
        return py::none();
    void*   host_kv    = nullptr;
    void*   idxp       = nullptr;
    int64_t cap_tokens = 0, kv_bpt = 0, nb_cap = 0, idx_bb = 0, durable = 0;
    int32_t dev = 0;
    if (!g_adm_lookup(req_key, (int32_t)layer, &host_kv, &cap_tokens, &kv_bpt, &idxp, &nb_cap, &idx_bb, &durable, &dev))
        return py::none();
    if (host_kv == nullptr || kv_bpt != 1152)
        return py::none();  // main-KV layout tripwire
    auto kv = torch::from_blob(host_kv, {cap_tokens, 576}, torch::TensorOptions().dtype(torch::kBFloat16));
    return py::make_tuple(kv, py::none(), durable, nb_cap);
}

int64_t ctx_admission_generation(int64_t req_key) {
    return g_adm_generation ? g_adm_generation(req_key) : -1;
}

void ctx_admission_release(int64_t req_key, int64_t generation) {
    if (generation >= 0 && g_adm_release_generation) {
        g_adm_release_generation(req_key, generation);
    } else if (g_adm_release) {
        g_adm_release(req_key);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ctx_debug", &ctx_debug);
    m.def("ctx_mirror_d2h", &ctx_mirror_d2h, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_mirror_blocks_d2h",
          &ctx_mirror_blocks_d2h,
          py::arg("pool_flat"),
          py::arg("blk_cpu"),
          py::arg("host_dst"),
          py::arg("dst_token"),
          py::arg("block_tokens"),
          py::arg("flush") = true,
          py::call_guard<py::gil_scoped_release>());
    m.def("ctx_serve_full", &ctx_serve_full, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_init", &ctx_init);
    m.def("ctx_register", &ctx_register);
    m.def("ctx_has", &ctx_has);
    m.def("ctx_update_host", &ctx_update_host);
    m.def("ctx_release", &ctx_release);
    m.def("ctx_serve", &ctx_serve, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_append_tok", &ctx_append_tok, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_serve_wb", &ctx_serve_wb, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_admission_open", &ctx_admission_open);
    m.def("ctx_adopt", &ctx_adopt);
    m.def("ctx_admission_generation", &ctx_admission_generation);
    m.def("ctx_admission_release", &ctx_admission_release, py::arg("req_key"), py::arg("generation") = -1);
    m.def("ctx_lossy_register",
          &ctx_lossy_register,
          py::arg("req_key"),
          py::arg("layer"),
          py::arg("jpos_cpu"),
          py::arg("sb_cpu"),
          py::arg("mw"),
          py::arg("device"),
          py::arg("kv_host") = c10::nullopt);
    m.def("ctx_lossy_has", &ctx_lossy_has);
    m.def("ctx_lossy_serve", &ctx_lossy_serve, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_lossless_serve", &ctx_lossless_serve, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_lossless_try_serve", &ctx_lossless_try_serve, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_step_plan", &ctx_step_plan);
    m.def("ctx_step_serve", &ctx_step_serve, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_lossy_counters", &ctx_lossy_counters);
    m.def("ctx_lossy_reset_counters", &ctx_lossy_reset_counters);
    m.def("ctx_probe_launch", &ctx_probe_launch, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_ktimings", &ctx_ktimings);
    m.def("ctx_lossy_release", &ctx_lossy_release);
}
