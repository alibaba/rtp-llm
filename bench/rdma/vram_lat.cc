// GPU VRAM <-> GPU VRAM RDMA_WRITE ping-pong latency over verbs (eRDMA), no staging.
// Each side registers a cudaMalloc buffer (SYNC_MEMOPS) and the peer's buffer is the write target.
// Round trip: A writes N bytes into B's VRAM (last 8 bytes = seq), B polls its own VRAM copy of the flag
// (cudaMemcpy D2H of 8 bytes) and writes back. Reported one-way = RTT/2, includes the D2H flag poll cost.
// A second mode "cq" measures the sender CQ completion of one signaled WRITE (no peer software in the loop).
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
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
static void check(bool ok, const std::string& m) { if (!ok) throw std::runtime_error(m); }
static void cc(cudaError_t s, const char* w) { check(s == cudaSuccess, std::string(w) + ": " + cudaGetErrorString(s)); }
struct Info { uint64_t addr; uint32_t rkey, qpn, psn; uint8_t gid[16]; };
static void xchg(int fd, void* p, size_t n, bool w) {
  char* c = (char*)p; while (n) { ssize_t k = w ? send(fd, c, n, MSG_NOSIGNAL) : recv(fd, c, n, 0); check(k > 0, "ctrl"); c += k; n -= k; }
}
int main(int argc, char** argv) {
  try {
    check(argc == 9, "usage: vram_lat server|client peer_ip port dev gpu gid bytes iters");
    bool server = std::string(argv[1]) == "server";
    std::string peer = argv[2], dev = argv[4]; int port = atoi(argv[3]), gpu = atoi(argv[5]), gidx = atoi(argv[6]);
    size_t bytes = strtoull(argv[7], 0, 10); int iters = atoi(argv[8]);
    cc(cudaSetDevice(gpu), "setdev");
    size_t buf = std::max<size_t>(bytes, 64) * 2;
    void* d; cc(cudaMalloc(&d, buf), "malloc"); cc(cudaMemset(d, 0, buf), "memset");
    unsigned one = 1; check(cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)d) == CUDA_SUCCESS, "syncmemops");
    int n; ibv_device** l = ibv_get_device_list(&n); ibv_context* ctx = nullptr;
    for (int i = 0; i < n; ++i) if (dev == ibv_get_device_name(l[i])) ctx = ibv_open_device(l[i]);
    check(ctx, "open dev");
    ibv_pd* pd = ibv_alloc_pd(ctx);
    ibv_mr* mr = ibv_reg_mr(pd, d, buf, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    check(mr, std::string("reg_mr VRAM: ") + strerror(errno));
    ibv_cq* cq = ibv_create_cq(ctx, 64, 0, 0, 0);
    ibv_qp_init_attr ia{}; ia.send_cq = cq; ia.recv_cq = cq; ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = 16; ia.cap.max_recv_wr = 1; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
    ibv_qp* qp = ibv_create_qp(pd, &ia); check(qp, "qp");
    ibv_qp_attr a{}; a.qp_state = IBV_QPS_INIT; a.port_num = 1; a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    check(!ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "init");
    ibv_gid g; check(!ibv_query_gid(ctx, 1, gidx, &g), "gid");
    Info me{(uint64_t)d, mr->rkey, qp->qp_num, (uint32_t)(getpid() & 0xffffff), {}}; memcpy(me.gid, g.raw, 16);
    int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_port = htons(port);
    if (server) { int o = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &o, 4); sa.sin_addr.s_addr = INADDR_ANY;
      check(!bind(fd, (sockaddr*)&sa, sizeof sa), "bind"); listen(fd, 1); int c = accept(fd, 0, 0); close(fd); fd = c; }
    else { inet_pton(AF_INET, peer.c_str(), &sa.sin_addr); check(!connect(fd, (sockaddr*)&sa, sizeof sa), "connect"); }
    Info rem; xchg(fd, &me, sizeof me, true); xchg(fd, &rem, sizeof rem, false);
    a = {}; a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_1024; a.dest_qp_num = rem.qpn; a.rq_psn = rem.psn;
    a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12; a.ah_attr.is_global = 1; a.ah_attr.port_num = 1;
    memcpy(a.ah_attr.grh.dgid.raw, rem.gid, 16); a.ah_attr.grh.sgid_index = gidx; a.ah_attr.grh.hop_limit = 64;
    check(!ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), "rtr");
    a = {}; a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = me.psn; a.max_rd_atomic = 1;
    check(!ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), "rts");
    char b = 1; xchg(fd, &b, 1, true); xchg(fd, &b, 1, false);
    // staging host copy of the outgoing payload's flag word, written into our own VRAM before each send
    size_t flag_off = bytes - 8;
    auto post = [&](uint64_t seq) {
      cc(cudaMemcpy((char*)d + flag_off, &seq, 8, cudaMemcpyHostToDevice), "set seq");
      ibv_sge s{(uint64_t)d, (uint32_t)bytes, mr->lkey};
      ibv_send_wr w{}, *bad; w.sg_list = &s; w.num_sge = 1; w.opcode = IBV_WR_RDMA_WRITE; w.send_flags = IBV_SEND_SIGNALED;
      w.wr.rdma.remote_addr = rem.addr + bytes; w.wr.rdma.rkey = rem.rkey;  // into peer's second half
      check(!ibv_post_send(qp, &w, &bad), "post");
      ibv_wc wc; int k; while ((k = ibv_poll_cq(cq, 1, &wc)) == 0) {} check(k == 1 && wc.status == IBV_WC_SUCCESS, "wc");
    };
    auto wait = [&](uint64_t seq) {
      uint64_t v = 0; auto t0 = Clock::now();
      while (true) { cc(cudaMemcpy(&v, (char*)d + bytes + flag_off, 8, cudaMemcpyDeviceToHost), "poll");
        if (v == seq) return; check(Clock::now() - t0 < std::chrono::seconds(10), "timeout waiting peer write"); }
    };
    std::vector<double> rtt, cqt;
    for (int i = 1; i <= iters + 50; ++i) {
      if (!server) {
        auto t0 = Clock::now(); post(i); auto t1 = Clock::now(); wait(i); auto t2 = Clock::now();
        if (i > 50) { rtt.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
                      cqt.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()); }
      } else { wait(i); post(i); }
    }
    if (!server) {
      std::sort(rtt.begin(), rtt.end()); std::sort(cqt.begin(), cqt.end());
      auto p = [](std::vector<double>& v, double q) { return v[size_t(q * (v.size() - 1))]; };
      std::cout << "{\"bytes\":" << bytes << ",\"iters\":" << iters << ",\"rtt_p50_us\":" << p(rtt, .5) << ",\"rtt_p99_us\":" << p(rtt, .99)
                << ",\"oneway_p50_us\":" << p(rtt, .5) / 2 << ",\"post_to_cqe_p50_us\":" << p(cqt, .5) << ",\"post_to_cqe_p99_us\":" << p(cqt, .99) << "}" << std::endl;
    }
    return 0;
  } catch (std::exception& e) { std::cerr << "ERROR " << e.what() << std::endl; return 1; }
}
