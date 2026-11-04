// RDMA WRITE_WITH_IMM ping-pong between two hosts, arrival = receive CQE (the routed2 flag_mode 2 path, minus
// the GPU flag mirror). Source and target buffers are placed independently in VRAM (cudaMalloc, GPUDirect) or
// host pinned memory, to split the GPUDirect DMA cost from the wire and the host software.
// usage: vram_imm server|client peer_ip port dev gid bytes iters src(vram|host) dst(vram|host)
// prints one-way p50/p99 = RTT/2.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static void check(bool ok, const std::string& m) {
    if (!ok)
        throw std::runtime_error(m);
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
static void* alloc(bool vram, size_t n) {
    void* p = nullptr;
    if (vram) {
        check(cudaMalloc(&p, n) == cudaSuccess, "cudaMalloc");
        cudaMemset(p, 0, n);
        unsigned one = 1;
        cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)p);
    } else {
        check(cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess, "cudaHostAlloc");
        memset(p, 0, n);
    }
    return p;
}
int main(int argc, char** argv) {
    try {
        check(argc == 10,
              "usage: vram_imm server|client peer_ip port dev gid bytes iters src(vram|host) dst(vram|host)");
        const bool        server = std::string(argv[1]) == "server";
        const std::string peer = argv[2], dev = argv[4];
        const int         port = atoi(argv[3]), gidx = atoi(argv[5]), iters = atoi(argv[7]);
        const size_t      bytes    = strtoull(argv[6], 0, 10);
        const bool        src_vram = std::string(argv[8]) == "vram", dst_vram = std::string(argv[9]) == "vram";
        cudaSetDevice(0);
        void*        sbuf = alloc(src_vram, bytes);
        void*        dbuf = alloc(dst_vram, bytes);
        int          n;
        ibv_device** l   = ibv_get_device_list(&n);
        ibv_context* ctx = nullptr;
        for (int i = 0; i < n; ++i)
            if (dev == ibv_get_device_name(l[i]))
                ctx = ibv_open_device(l[i]);
        check(ctx, "open dev");
        ibv_pd* pd  = ibv_alloc_pd(ctx);
        ibv_mr* smr = ibv_reg_mr(pd, sbuf, bytes, IBV_ACCESS_LOCAL_WRITE);
        ibv_mr* dmr = ibv_reg_mr(pd, dbuf, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        check(smr && dmr, std::string("reg_mr: ") + strerror(errno));
        ibv_cq*          cq = ibv_create_cq(ctx, 256, 0, 0, 0);
        ibv_qp_init_attr ia{};
        ia.send_cq          = cq;
        ia.recv_cq          = cq;
        ia.qp_type          = IBV_QPT_RC;
        ia.cap.max_send_wr  = 64;
        ia.cap.max_recv_wr  = 64;
        ia.cap.max_send_sge = 1;
        ia.cap.max_recv_sge = 1;
        ibv_qp* qp          = ibv_create_qp(pd, &ia);
        check(qp, "qp");
        ibv_qp_attr a{};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = 1;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        check(!ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "init");
        ibv_sge     rs{(uint64_t)dbuf, 8, dmr->lkey};
        ibv_recv_wr rw{};
        rw.sg_list = &rs;
        rw.num_sge = 1;
        for (int i = 0; i < 32; ++i) {
            ibv_recv_wr* bad;
            check(!ibv_post_recv(qp, &rw, &bad), "post_recv");
        }
        ibv_gid g;
        check(!ibv_query_gid(ctx, 1, gidx, &g), "gid");
        Info me{(uint64_t)dbuf, dmr->rkey, qp->qp_num, (uint32_t)(getpid() & 0xffffff), {}};
        memcpy(me.gid, g.raw, 16);
        int         fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port   = htons(port);
        if (server) {
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
        char b = 1;
        xchg(fd, &b, 1, true);
        xchg(fd, &b, 1, false);
        int  sends_out = 0;
        auto post      = [&](uint32_t seq) {
            ibv_sge     s{(uint64_t)sbuf, (uint32_t)bytes, smr->lkey};
            ibv_send_wr w{}, *bad;
            w.sg_list             = &s;
            w.num_sge             = 1;
            w.opcode              = IBV_WR_RDMA_WRITE_WITH_IMM;
            w.imm_data            = htonl(seq);
            w.send_flags          = IBV_SEND_SIGNALED;
            w.wr.rdma.remote_addr = rem.addr;
            w.wr.rdma.rkey        = rem.rkey;
            check(!ibv_post_send(qp, &w, &bad), "post");
            ++sends_out;
        };
        auto wait_recv = [&](uint32_t seq) {
            for (;;) {
                ibv_wc    wc[8];
                const int k = ibv_poll_cq(cq, 8, wc);
                check(k >= 0, "poll");
                for (int z = 0; z < k; ++z) {
                    check(wc[z].status == IBV_WC_SUCCESS, std::string("wc ") + ibv_wc_status_str(wc[z].status));
                    if (wc[z].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                        ibv_recv_wr* bad;
                        check(!ibv_post_recv(qp, &rw, &bad), "repost");
                        check(ntohl(wc[z].imm_data) == seq, "imm seq");
                        return;
                    }
                    --sends_out;
                }
            }
        };
        std::vector<double> rtt;
        for (int i = 1; i <= iters + 50; ++i) {
            if (!server) {
                auto t0 = Clock::now();
                post(i);
                wait_recv(i);
                auto t1 = Clock::now();
                if (i > 50)
                    rtt.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            } else {
                wait_recv(i);
                post(i);
            }
        }
        if (!server) {
            std::sort(rtt.begin(), rtt.end());
            auto p = [&](double q) { return rtt[size_t(q * (rtt.size() - 1))]; };
            std::cout << "{\"bytes\":" << bytes << ",\"src\":\"" << argv[8] << "\",\"dst\":\"" << argv[9]
                      << "\",\"oneway_p50_us\":" << p(.5) / 2 << ",\"oneway_p99_us\":" << p(.99) / 2 << "}"
                      << std::endl;
        }
        return 0;
    } catch (std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}
