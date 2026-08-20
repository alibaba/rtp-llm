// v32_ctx.cu — Scheme B offload context v2: staging + async fetch fully in C++.
// Python calls per layer: serve_layer() (build+sanitize+miss->fetch) and
// drain happens inside serve_layer. Fetch thread never touches the GIL.
#include <torch/extension.h>
#include "NoBlockCopy.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
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

// single-wave pool: copy every row's current token indexer-K main->side pool.
// expected[r] = bookkept block0 identity; mismatch sets ok[r]=0 (tripwire, no write).
__global__ void batch_append_kernel(unsigned char* __restrict__ pool,
                                    const unsigned char* __restrict__ src_pool,
                                    const int* __restrict__ kbt,
                                    const int* __restrict__ ibt,
                                    const int* __restrict__ kvlen,
                                    const int* __restrict__ expected,
                                    int* __restrict__ ok,
                                    int  B,
                                    int  kbt_w,
                                    int  ibt_w,
                                    long src_stride) {
    int r = blockIdx.x;
    if (r >= B)
        return;
    if (kbt[r * kbt_w] != expected[r]) {
        if (threadIdx.x == 0)
            ok[r] = 0;
        return;
    }
    int pos = kvlen[r] - 1;
    if (pos < 0 || pos / BS >= ibt_w)
        return;
    int sb = kbt[r * kbt_w + pos / BS];
    int db = ibt[r * ibt_w + pos / BS];
    if (sb <= 0 || db < 0)
        return;
    const unsigned char* s   = src_pool + (size_t)sb * src_stride;
    unsigned char*       d   = pool + (size_t)db * (132 * BS);
    int                  off = pos % BS;
    int                  t   = threadIdx.x;
    if (t < 128)
        d[off * 128 + t] = s[off * 128 + t];
    else if (t < 132)
        d[BS * 128 + off * 4 + (t - 128)] = s[BS * 128 + off * 4 + (t - 128)];
}

// single-wave pool: bulk copy positions [lo, upto) of one row (admission/backfill).
__global__ void bulk_admit_kernel(unsigned char* __restrict__ pool,
                                  const unsigned char* __restrict__ src_pool,
                                  const int* __restrict__ kbt_row,
                                  const int* __restrict__ ibt_row,
                                  int  lo,
                                  int  upto,
                                  int  ibt_w,
                                  long src_stride) {
    int pos = lo + blockIdx.x;
    if (pos >= upto || pos / BS >= ibt_w)
        return;
    int sb = kbt_row[pos / BS];
    int db = ibt_row[pos / BS];
    if (sb <= 0 || db < 0)
        return;
    const unsigned char* s   = src_pool + (size_t)sb * src_stride;
    unsigned char*       d   = pool + (size_t)db * (132 * BS);
    int                  off = pos % BS;
    int                  t   = threadIdx.x;
    if (t < 128)
        d[off * 128 + t] = s[off * 128 + t];
    else if (t < 132)
        d[BS * 128 + off * 4 + (t - 128)] = s[BS * 128 + off * 4 + (t - 128)];
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

// single-wave pool maintenance. kbt/ibt/kvlen/expected int32; ok int32 [B]
// (device; caller drains it asynchronously).
void ctx_batch_append(torch::Tensor pool_l,
                      torch::Tensor src_pool_u8,
                      torch::Tensor kbt,
                      torch::Tensor ibt,
                      torch::Tensor kvlen,
                      torch::Tensor expected,
                      torch::Tensor ok) {
    TORCH_CHECK(kbt.scalar_type() == torch::kInt32 && ibt.scalar_type() == torch::kInt32
                    && kvlen.scalar_type() == torch::kInt32,
                "int32 expected");
    int  B      = (int)ibt.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    batch_append_kernel<<<B, 132, 0, stream>>>(reinterpret_cast<unsigned char*>(pool_l.data_ptr()),
                                               reinterpret_cast<const unsigned char*>(src_pool_u8.data_ptr()),
                                               kbt.data_ptr<int>(),
                                               ibt.data_ptr<int>(),
                                               kvlen.data_ptr<int>(),
                                               expected.data_ptr<int>(),
                                               ok.data_ptr<int>(),
                                               B,
                                               (int)kbt.size(1),
                                               (int)ibt.size(1),
                                               (long)src_pool_u8.size(1));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "batch_append launch failed");
}

void ctx_bulk_admit(torch::Tensor pool_l,
                    torch::Tensor src_pool_u8,
                    torch::Tensor kbt,
                    int64_t       row_i,
                    torch::Tensor ibt_row,
                    int64_t       lo,
                    int64_t       upto) {
    if (upto <= lo)
        return;
    auto kbt_row = kbt.select(0, row_i).to(torch::kInt32).contiguous();
    auto stream  = at::cuda::getCurrentCUDAStream();
    bulk_admit_kernel<<<(unsigned)(upto - lo), 132, 0, stream>>>(
        reinterpret_cast<unsigned char*>(pool_l.data_ptr()),
        reinterpret_cast<const unsigned char*>(src_pool_u8.data_ptr()),
        kbt_row.data_ptr<int>(),
        ibt_row.data_ptr<int>(),
        (int)lo,
        (int)upto,
        (int)ibt_row.size(0),
        (long)src_pool_u8.size(1));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "bulk_admit launch failed");
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
    sp.direct_pinned_host_segments = true;
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
    rtp_llm::execStagedMemoryCopy(sp, &scratch);
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

// ---- v32 admission mirror adoption (engine-owned buffers, staging-ring
// admission). The engine wheel exports C symbols; we bind them at runtime so
// this extension needs no link-time dependency on the wheel.
#include <dlfcn.h>
typedef int (*v32_adm_lookup_fn)(
    int64_t, int32_t, void**, int64_t*, int64_t*, void**, int64_t*, int64_t*, int64_t*, int32_t*);
typedef void (*v32_adm_release_fn)(int64_t);
static v32_adm_lookup_fn  g_adm_lookup  = nullptr;
static v32_adm_release_fn g_adm_release = nullptr;

bool ctx_admission_open(const std::string& engine_so_path) {
    void* h = dlopen(engine_so_path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (!h)
        h = dlopen(engine_so_path.c_str(), RTLD_LAZY);
    if (!h)
        return false;
    g_adm_lookup  = (v32_adm_lookup_fn)dlsym(h, "rtp_v32_admission_lookup");
    g_adm_release = (v32_adm_release_fn)dlsym(h, "rtp_v32_admission_release");
    return g_adm_lookup != nullptr && g_adm_release != nullptr;
}

// Returns (kv_host [cap,576] bf16, idxp [nb,64,132] u8 cuda, durable_tokens)
// or None. Tensors alias engine memory: valid until the engine releases the
// request (stream end / purge), same staleness contract as the block table.
py::object ctx_adopt(int64_t req_key, int64_t layer) {
    if (!g_adm_lookup)
        return py::none();
    void*   host_kv    = nullptr;
    void*   idxp       = nullptr;
    int64_t cap_tokens = 0, kv_bpt = 0, nb_cap = 0, idx_bb = 0, durable = 0;
    int32_t dev = 0;
    if (!g_adm_lookup(req_key, (int32_t)layer, &host_kv, &cap_tokens, &kv_bpt, &idxp, &nb_cap, &idx_bb, &durable, &dev))
        return py::none();
    if (kv_bpt != 1152 || idx_bb != BS * 132)
        return py::none();  // layout tripwire
    auto kv = torch::from_blob(host_kv, {cap_tokens, 576}, torch::TensorOptions().dtype(torch::kBFloat16));
    auto ip = torch::from_blob(
        idxp, {nb_cap, BS, 132}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, dev));
    return py::make_tuple(kv, ip, durable);
}

void ctx_admission_release(int64_t req_key) {
    if (g_adm_release)
        g_adm_release(req_key);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ctx_debug", &ctx_debug);
    m.def("ctx_mirror_d2h", &ctx_mirror_d2h, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_serve_full", &ctx_serve_full, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_init", &ctx_init);
    m.def("ctx_register", &ctx_register);
    m.def("ctx_has", &ctx_has);
    m.def("ctx_update_host", &ctx_update_host);
    m.def("ctx_release", &ctx_release);
    m.def("ctx_serve", &ctx_serve, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_append_tok", &ctx_append_tok, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_serve_wb", &ctx_serve_wb, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_batch_append", &ctx_batch_append, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_bulk_admit", &ctx_bulk_admit, py::call_guard<py::gil_scoped_release>());
    m.def("ctx_admission_open", &ctx_admission_open);
    m.def("ctx_adopt", &ctx_adopt);
    m.def("ctx_admission_release", &ctx_admission_release);
}
