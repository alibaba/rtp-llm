// rdma_sentinel.cu - eRDMA arrival-detection microbench for routed2 "per-vector sentinel" (see
// comm/largeep_a2a/README.md, "ruled out").
//
// Ping-pong between two hosts, one GPU each. Side A (host-driven, like the routed2 sender proxy) fills a VRAM payload
// for iteration `it` and posts one RDMA write of `bytes` into side B's VRAM landing buffer. Side B detects arrival on
// the GPU, verifies every 4-byte word, then its GPU rings a host-mapped doorbell and B's host thread (the receiver
// proxy) answers with an 8-byte WRITE_WITH_IMM; A measures the round trip. Only B's arrival detection differs:
//   mode imm  (= routed2 flag_mode 2): WRITE_WITH_IMM; B's host thread polls the CQ, stores the sequence into a
//             host-mapped flag, one GPU thread polls that flag over PCIe, then the block reads the payload.
//   mode sent (per-vector sentinel):   plain RDMA WRITE; the landing buffer holds 0xFFFFFFFF in every word; GPU
//             threads poll their own 16-byte vectors in VRAM until all 4 words are non-sentinel, check them, re-arm.
// Checks: in mode sent every observed word must be either the sentinel or its final value (a third value = a torn
// 4-byte word); in mode imm the payload must be complete when the flag is seen (stale word = CQE-before-data).
// Reports (B): torn words, stale words, first-vector -> last-vector arrival spread; (A): one-way = RTT/2.
// usage: rdma_sentinel A|B peer_ip port dev gid bytes iters imm|sent
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define SENT 0xFFFFFFFFu
#define NTHR 1024
static void check(bool ok, const std::string& m) {
    if (!ok)
        throw std::runtime_error(m);
}
static void cc(cudaError_t e, const char* w) {
    check(e == cudaSuccess, std::string(w) + ": " + cudaGetErrorString(e));
}

__host__ __device__ __forceinline__ uint32_t val(uint32_t it, uint32_t w) {
    return ((it * 2654435761u) ^ (w * 40503u + 0x1234567u)) & 0x7FFFFFFFu;  // top bit clear: never the sentinel
}
__device__ __forceinline__ uint64_t gtime() {
    uint64_t g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    return g;
}

__device__ __forceinline__ uint4 vld4(const uint32_t* p) {
    uint4 r;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}
__device__ __forceinline__ void vst4(uint32_t* p, uint4 v) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__global__ void k_fill(uint32_t* p, uint32_t nw, uint32_t it) {
    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < nw; w += gridDim.x * blockDim.x)
        p[w] = val(it, w);
}
__global__ void k_arm(uint32_t* p, uint32_t nw) {
    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < nw; w += gridDim.x * blockDim.x)
        p[w] = SENT;
}

// stats: [0] torn words, [1] stale words (imm), [2] timeouts; spread_ns[it-1] = last - first vector arrival
__global__ void __launch_bounds__(NTHR) k_recv(const uint32_t*     land,
                                               uint32_t            nvec,
                                               uint32_t            iters,
                                               int                 sent_mode,
                                               volatile uint64_t*  hflag,
                                               volatile uint64_t*  db,
                                               unsigned long long* stats,
                                               uint64_t*           spread_ns) {
    __shared__ unsigned long long s_first;
    __shared__ int                s_ok;
    const int                     tid = threadIdx.x;
    for (uint32_t it = 1; it <= iters; ++it) {
        if (tid == 0)
            s_first = ~0ull;
        __syncthreads();
        if (!sent_mode) {
            if (tid == 0) {
                const uint64_t t0 = gtime();
                while (*hflag < it) {
                    if (gtime() - t0 > 2000000000ull) {
                        atomicAdd(stats + 2, 1ull);
                        break;
                    }
                }
                s_first = gtime();
            }
            __syncthreads();
            __threadfence();
            for (uint32_t v = tid; v < nvec; v += NTHR) {
                const uint4    x    = vld4(land + 4 * v);
                const uint32_t w[4] = {x.x, x.y, x.z, x.w};
                for (int q = 0; q < 4; ++q)
                    if (w[q] != val(it, 4 * v + q))
                        atomicAdd(stats + 1, 1ull);
            }
        } else {
            for (uint32_t v = tid; v < nvec; v += NTHR) {
                const uint64_t t0 = gtime();
                for (;;) {
                    const uint4    x    = vld4(land + 4 * v);
                    const uint32_t w[4] = {x.x, x.y, x.z, x.w};
                    int            done = 0;
                    for (int q = 0; q < 4; ++q) {
                        const uint32_t e = val(it, 4 * v + q);
                        if (w[q] == e)
                            ++done;
                        else if (w[q] != SENT)
                            atomicAdd(stats + 0, 1ull);
                    }
                    if (done == 4) {
                        atomicMin(&s_first, (unsigned long long)gtime());
                        break;
                    }
                    if (gtime() - t0 > 2000000000ull) {
                        atomicAdd(stats + 2, 1ull);
                        break;
                    }
                }
            }
        }
        __syncthreads();
        if (tid == 0)
            spread_ns[it - 1] = gtime() - s_first;
        if (sent_mode)  // re-arm: this thread's vectors only (it is their single reader)
            for (uint32_t v = tid; v < nvec; v += NTHR)
                vst4((uint32_t*)land + 4 * v, make_uint4(SENT, SENT, SENT, SENT));
        __threadfence_system();
        __syncthreads();
        if (tid == 0)
            *db = it;
    }
}

struct Info {
    uint64_t addr;
    uint32_t rkey, qpn, psn;
    uint8_t  gid[16];
};
static void xchg(int fd, void* p, size_t n, bool w) {
    char* c = (char*)p;
    while (n) {
        ssize_t k = w ? send(fd, c, n, MSG_NOSIGNAL) : recv(fd, c, n, 0);
        check(k > 0, "ctrl");
        c += k;
        n -= k;
    }
}

int main(int argc, char** argv) {
    try {
        check(argc == 9, "usage: rdma_sentinel A|B peer_ip port dev gid bytes iters imm|sent");
        const bool        isA  = std::string(argv[1]) == "A";
        const std::string peer = argv[2], dev = argv[4];
        const int         port = atoi(argv[3]), gidx = atoi(argv[5]);
        const uint32_t    bytes = (uint32_t)strtoul(argv[6], 0, 10), iters = (uint32_t)atoi(argv[7]);
        const bool        sent = std::string(argv[8]) == "sent";
        check(bytes % 16 == 0, "bytes % 16");
        const uint32_t nw = bytes / 4, nvec = bytes / 16;
        cc(cudaSetDevice(0), "setdev");
        // A: VRAM payload source + host reply landing (8B); B: VRAM landing + host reply source (8B)
        void* vbuf;
        cc(cudaMalloc(&vbuf, bytes), "malloc");
        unsigned one = 1;
        cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)vbuf);
        uint64_t* hsmall;
        cc(cudaHostAlloc((void**)&hsmall, 4096, cudaHostAllocMapped), "hostalloc");
        memset(hsmall, 0, 4096);
        int          n;
        ibv_device** l   = ibv_get_device_list(&n);
        ibv_context* ctx = nullptr;
        for (int i = 0; i < n; ++i)
            if (dev == ibv_get_device_name(l[i]))
                ctx = ibv_open_device(l[i]);
        check(ctx, "open dev");
        ibv_pd* pd  = ibv_alloc_pd(ctx);
        ibv_mr* vmr = ibv_reg_mr(pd, vbuf, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        ibv_mr* hmr = ibv_reg_mr(pd, hsmall, 4096, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        check(vmr && hmr, std::string("reg_mr: ") + strerror(errno));
        ibv_cq*          cq = ibv_create_cq(ctx, 512, 0, 0, 0);
        ibv_qp_init_attr ia{};
        ia.send_cq             = cq;
        ia.recv_cq             = cq;
        ia.qp_type             = IBV_QPT_RC;
        ia.cap.max_send_wr     = 128;
        ia.cap.max_recv_wr     = 128;
        ia.cap.max_send_sge    = 1;
        ia.cap.max_recv_sge    = 1;
        ia.cap.max_inline_data = 32;
        ibv_qp* qp             = ibv_create_qp(pd, &ia);
        check(qp, "qp");
        ibv_qp_attr a{};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = 1;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        check(!ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "init");
        ibv_sge     rs{(uint64_t)(hsmall + 64), 8, hmr->lkey};
        ibv_recv_wr rw{};
        rw.sg_list = &rs;
        rw.num_sge = 1;
        for (int i = 0; i < 64; ++i) {
            ibv_recv_wr* bad;
            check(!ibv_post_recv(qp, &rw, &bad), "post_recv");
        }
        ibv_gid g;
        check(!ibv_query_gid(ctx, 1, gidx, &g), "gid");
        Info me{isA ? (uint64_t)hsmall : (uint64_t)vbuf,
                isA ? hmr->rkey : vmr->rkey,
                qp->qp_num,
                (uint32_t)(getpid() * 7919u & 0xffffff),
                {}};
        memcpy(me.gid, g.raw, 16);
        int         fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port   = htons(port);
        if (!isA) {
            int o = 1;
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &o, 4);
            sa.sin_addr.s_addr = INADDR_ANY;
            check(!bind(fd, (sockaddr*)&sa, sizeof sa), "bind");
            listen(fd, 1);
            int c = accept(fd, 0, 0);
            close(fd);
            fd = c;
        } else {
            inet_pton(AF_INET, peer.c_str(), &sa.sin_addr);
            check(!connect(fd, (sockaddr*)&sa, sizeof sa), "connect");
        }
        Info rem;
        xchg(fd, &me, sizeof me, true);
        xchg(fd, &rem, sizeof rem, false);
        a                    = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = IBV_MTU_1024;
        a.dest_qp_num        = rem.qpn;
        a.rq_psn             = rem.psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 12;
        a.ah_attr.is_global  = 1;
        a.ah_attr.port_num   = 1;
        memcpy(a.ah_attr.grh.dgid.raw, rem.gid, 16);
        a.ah_attr.grh.sgid_index = gidx;
        a.ah_attr.grh.hop_limit  = 64;
        check(!ibv_modify_qp(qp,
                             &a,
                             IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
                                 | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
              "rtr");
        a               = {};
        a.qp_state      = IBV_QPS_RTS;
        a.timeout       = 14;
        a.retry_cnt     = 7;
        a.rnr_retry     = 7;
        a.sq_psn        = me.psn;
        a.max_rd_atomic = 1;
        check(!ibv_modify_qp(qp,
                             &a,
                             IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN
                                 | IBV_QP_MAX_QP_RD_ATOMIC),
              "rts");
        auto reap_or_recv = [&](bool want_recv, uint32_t* imm_out) -> bool {  // returns true if a recv CQE was seen
            ibv_wc    wc[16];
            const int k = ibv_poll_cq(cq, 16, wc);
            check(k >= 0, "poll");
            bool got = false;
            for (int z = 0; z < k; ++z) {
                check(wc[z].status == IBV_WC_SUCCESS, std::string("wc ") + ibv_wc_status_str(wc[z].status));
                if (wc[z].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                    ibv_recv_wr* bad;
                    check(!ibv_post_recv(qp, &rw, &bad), "repost");
                    if (imm_out)
                        *imm_out = ntohl(wc[z].imm_data);
                    got = true;
                }
            }
            (void)want_recv;
            return got;
        };
        char b = 1;
        if (!isA) {
            cc(cudaMemset(vbuf, 0xFF, bytes), "arm0");
            cc(cudaDeviceSynchronize(), "sync");
        }
        xchg(fd, &b, 1, true);
        xchg(fd, &b, 1, false);
        if (isA) {
            std::vector<double> rtt;
            for (uint32_t it = 1; it <= iters; ++it) {
                k_fill<<<64, 256>>>((uint32_t*)vbuf, nw, it);
                cc(cudaDeviceSynchronize(), "fill");
                ibv_sge     s{(uint64_t)vbuf, bytes, vmr->lkey};
                ibv_send_wr w{}, *bad;
                w.sg_list             = &s;
                w.num_sge             = 1;
                w.opcode              = sent ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_WRITE_WITH_IMM;
                w.imm_data            = htonl(it);
                w.send_flags          = IBV_SEND_SIGNALED;
                w.wr.rdma.remote_addr = rem.addr;
                w.wr.rdma.rkey        = rem.rkey;
                const auto t0         = std::chrono::steady_clock::now();
                check(!ibv_post_send(qp, &w, &bad), "post");
                uint32_t imm = 0;
                while (!reap_or_recv(true, &imm)) {}
                const auto t1 = std::chrono::steady_clock::now();
                check(imm == it, "reply seq");
                if (it > 50)
                    rtt.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(rtt.begin(), rtt.end());
            auto p = [&](double q) { return rtt[size_t(q * (rtt.size() - 1))]; };
            printf(
                "{\"side\":\"A\",\"mode\":\"%s\",\"bytes\":%u,\"oneway_p50_us\":%.2f,\"oneway_p90_us\":%.2f,\"oneway_p99_us\":%.2f}\n",
                sent ? "sent" : "imm",
                bytes,
                p(.5) / 2,
                p(.9) / 2,
                p(.99) / 2);
        } else {
            uint64_t *         hflag_d, *db_d;
            volatile uint64_t* hflag = hsmall + 8;
            volatile uint64_t* db    = hsmall + 16;
            cc(cudaHostGetDevicePointer((void**)&hflag_d, (void*)(hsmall + 8), 0), "devptr");
            cc(cudaHostGetDevicePointer((void**)&db_d, (void*)(hsmall + 16), 0), "devptr");
            unsigned long long* stats;
            uint64_t*           spread;
            cc(cudaMalloc(&stats, 64), "stats");
            cc(cudaMemset(stats, 0, 64), "stats0");
            cc(cudaMalloc(&spread, 8ull * iters), "spread");
            cudaStream_t st;
            cc(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
            k_recv<<<1, NTHR, 0, st>>>((const uint32_t*)vbuf, nvec, iters, sent ? 1 : 0, hflag_d, db_d, stats, spread);
            cc(cudaGetLastError(), "launch");
            uint64_t next = 1;
            while (next <= iters) {
                uint32_t imm = 0;
                if (reap_or_recv(true, &imm))
                    *hflag = imm;  // imm mode: receiver-proxy store into the host-mapped flag
                if (*db >= next) {
                    ibv_sge     s{(uint64_t)(hsmall + 32), 8, hmr->lkey};
                    ibv_send_wr w{}, *bad;
                    w.sg_list             = &s;
                    w.num_sge             = 1;
                    w.opcode              = IBV_WR_RDMA_WRITE_WITH_IMM;
                    w.imm_data            = htonl((uint32_t)next);
                    w.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
                    w.wr.rdma.remote_addr = rem.addr + 256;
                    w.wr.rdma.rkey        = rem.rkey;
                    check(!ibv_post_send(qp, &w, &bad), "post reply");
                    ++next;
                }
            }
            cc(cudaStreamSynchronize(st), "kernel");
            unsigned long long hs[3];
            cc(cudaMemcpy(hs, stats, 24, cudaMemcpyDeviceToHost), "stats d2h");
            std::vector<uint64_t> sp(iters);
            cc(cudaMemcpy(sp.data(), spread, 8ull * iters, cudaMemcpyDeviceToHost), "spread d2h");
            std::vector<uint64_t> spv(sp.begin() + 50, sp.end());
            std::sort(spv.begin(), spv.end());
            printf(
                "{\"side\":\"B\",\"mode\":\"%s\",\"bytes\":%u,\"iters\":%u,\"torn_words\":%llu,\"stale_words\":%llu,\"timeouts\":%llu,"
                "\"spread_first_to_last_vector_p50_us\":%.2f,\"spread_p90_us\":%.2f}\n",
                sent ? "sent" : "imm",
                bytes,
                iters,
                hs[0],
                hs[1],
                hs[2],
                spv[spv.size() / 2] / 1e3,
                spv[spv.size() * 9 / 10] / 1e3);
        }
        return 0;
    } catch (std::exception& e) {
        fprintf(stderr, "ERROR %s\n", e.what());
        return 1;
    }
}
