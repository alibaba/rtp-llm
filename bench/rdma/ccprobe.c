/* ccprobe.c - multi-QP host-memory RDMA WRITE throughput probe with time series
 *
 * Build: gcc -O2 -I$BUILD/include -L$BUILD/lib -o ccprobe ccprobe.c -libverbs
 * Run:   LD_LIBRARY_PATH=$BUILD/lib ./ccprobe ...
 *
 *   ccprobe server <port> <nqps>
 *   ccprobe client <ip> <port> <nqps> <depth> <seconds> [msg_bytes]
 *
 * Client prints one CSV line per 200ms window: "CSV <elapsed_ms>,<window_gbps>"
 * plus a final SUMMARY line. Server is a pure WRITE target, exits on DONE.
 * Device from env RDMA_DEV (default erdma_0), GID index RDMA_GID (default 1), port 1, MTU 1024 (eRDMA
 * convention); both ends must use the same GID type.
 */
#define _POSIX_C_SOURCE 200809L
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define WIRE 256
#define MAX_QPS 64
#define SRV_MR_BYTES (16U << 20)
#define SAMPLE_MS 200

struct qctx {
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    uint64_t remote_addr;
    uint32_t rkey;
    int outstanding;
    uint64_t posted;
};

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int transfer(int fd, void *p, size_t n, int sending) {
    unsigned char *b = p;
    while (n) {
        ssize_t k = sending ? send(fd, b, n, MSG_NOSIGNAL) : recv(fd, b, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) { fprintf(stderr, "TCP xfer failed k=%zd errno=%d\n", k, errno); return -1; }
        b += k; n -= (size_t)k;
    }
    return 0;
}

static int exchange(int fd, const char *out, char *in) {
    if (transfer(fd, (void *)out, WIRE, 1) || transfer(fd, in, WIRE, 0)) return -1;
    in[WIRE - 1] = 0;
    return 0;
}

static int tcp_open(int server, const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), listener;
    if (fd < 0) return -1;
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    if (server) {
        int yes = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) || listen(fd, 1)) { perror("bind/listen"); close(fd); return -1; }
        fprintf(stderr, "TCP listening port=%d\n", port);
        listener = fd; fd = accept(listener, NULL, NULL); close(listener);
        if (fd < 0) { perror("accept"); return -1; }
    } else {
        if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1 || connect(fd, (struct sockaddr *)&addr, sizeof(addr))) { perror("connect"); close(fd); return -1; }
    }
    return fd;
}

static const char *rdma_dev(void) {
    const char *d = getenv("RDMA_DEV");
    return d && *d ? d : "erdma_0";
}

static int rdma_gid(void) {
    const char *g = getenv("RDMA_GID");
    return g && *g ? atoi(g) : 1;
}

static struct ibv_context *open_erdma(void) {
    struct ibv_device **devices = ibv_get_device_list(NULL);
    if (!devices) { perror("ibv_get_device_list"); return NULL; }
    struct ibv_context *ctx = NULL;
    for (int i = 0; devices[i]; i++)
        if (!strcmp(ibv_get_device_name(devices[i]), rdma_dev())) { ctx = ibv_open_device(devices[i]); break; }
    ibv_free_device_list(devices);
    return ctx;
}

/* bring one QP to RTS; exchanges metadata over fd. local side must have mr.
 * peer_out filled with peer qpn/psn/rkey/addr/gid. */
static int qp_handshake(struct ibv_context *ctx, struct ibv_pd *pd, struct qctx *q,
                        int depth, int fd, uint32_t rkey, uint64_t addr) {
    q->cq = ibv_create_cq(ctx, depth + 16, NULL, NULL, 0);
    if (!q->cq) { perror("create_cq"); return -1; }
    struct ibv_qp_init_attr qi = {.send_cq = q->cq, .recv_cq = q->cq, .qp_type = IBV_QPT_RC,
        .cap = {.max_send_wr = (uint32_t)depth + 16, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1}};
    q->qp = ibv_create_qp(pd, &qi);
    if (!q->qp) { perror("create_qp"); return -1; }
    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = 1,
        .qp_access_flags = IBV_ACCESS_REMOTE_WRITE};
    if (ibv_modify_qp(q->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        fprintf(stderr, "QP INIT failed\n"); return -1;
    }
    union ibv_gid gid;
    if (ibv_query_gid(ctx, 1, rdma_gid(), &gid)) { perror("query_gid"); return -1; }
    char gid_s[INET6_ADDRSTRLEN], peer_gid_s[INET6_ADDRSTRLEN], out[WIRE] = {0}, in[WIRE] = {0};
    inet_ntop(AF_INET6, &gid, gid_s, sizeof(gid_s));
    uint32_t psn = ((uint32_t)getpid() * 2654435761U + (uint32_t)q->qp->qp_num) & 0xffffff;
    uint32_t peer_qpn, peer_psn, peer_rkey;
    uint64_t peer_addr;
    snprintf(out, sizeof(out), "CCP1 %" SCNx32 " %" SCNx32 " %" SCNx32 " %" PRIx64 " %s",
             q->qp->qp_num, psn, rkey, addr, gid_s);
    if (exchange(fd, out, in) ||
        sscanf(in, "CCP1 %" SCNx32 " %" SCNx32 " %" SCNx32 " %" SCNx64 " %45s",
               &peer_qpn, &peer_psn, &peer_rkey, &peer_addr, peer_gid_s) != 5) {
        fprintf(stderr, "metadata exchange failed: '%s'\n", in); return -1;
    }
    union ibv_gid peer_gid;
    if (inet_pton(AF_INET6, peer_gid_s, &peer_gid) != 1) { fprintf(stderr, "bad peer gid\n"); return -1; }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024; attr.dest_qp_num = peer_qpn; attr.rq_psn = peer_psn;
    attr.max_dest_rd_atomic = 1; attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1; attr.ah_attr.port_num = 1;
    attr.ah_attr.grh.dgid = peer_gid; attr.ah_attr.grh.sgid_index = rdma_gid(); attr.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(q->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        fprintf(stderr, "QP RTR failed\n"); return -1;
    }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14; attr.retry_cnt = 7; attr.rnr_retry = 7; attr.sq_psn = psn; attr.max_rd_atomic = 1;
    if (ibv_modify_qp(q->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        fprintf(stderr, "QP RTS failed\n"); return -1;
    }
    q->remote_addr = peer_addr;
    q->rkey = peer_rkey;
    return 0;
}

static int run_server(int port, int nqps) {
    struct ibv_context *ctx = open_erdma();
    if (!ctx) { fprintf(stderr, "open %s failed\n", rdma_dev()); return 1; }
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) return 1;
    void *mem = NULL;
    if (posix_memalign(&mem, 4096, SRV_MR_BYTES)) return 1;
    memset(mem, 0xa5, SRV_MR_BYTES);
    struct ibv_mr *mr = ibv_reg_mr(pd, mem, SRV_MR_BYTES,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) { perror("reg_mr"); return 1; }
    int fd = tcp_open(1, NULL, port);
    if (fd < 0) return 1;
    static struct qctx q[MAX_QPS];
    for (int i = 0; i < nqps; i++)
        if (qp_handshake(ctx, pd, &q[i], 64, fd, mr->rkey, (uint64_t)(uintptr_t)mem)) return 1;
    fprintf(stderr, "server ready nqps=%d\n", nqps);
    char done[WIRE] = {0};
    if (transfer(fd, done, WIRE, 0)) return 1;   /* blocks until client DONE */
    fprintf(stderr, "server done\n");
    close(fd);
    return 0;
}

static int run_client(const char *ip, int port, int nqps, int depth, int seconds, uint32_t msg, int sample_ms) {
    struct ibv_context *ctx = open_erdma();
    if (!ctx) { fprintf(stderr, "open %s failed\n", rdma_dev()); return 1; }
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) return 1;
    unsigned char *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, (size_t)msg * nqps)) return 1;
    memset(buf, 0x5a, (size_t)msg * nqps);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, (size_t)msg * nqps, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) { perror("reg_mr"); return 1; }
    int fd = tcp_open(0, ip, port);
    if (fd < 0) return 1;
    static struct qctx q[MAX_QPS];
    for (int i = 0; i < nqps; i++)
        if (qp_handshake(ctx, pd, &q[i], depth, fd, 0, 0)) return 1;
    fprintf(stderr, "client qps up nqps=%d depth=%d msg=%u seconds=%d\n", nqps, depth, msg, seconds);

    double t0 = now(), t_end = t0 + seconds, t_sample = t0;
    uint64_t total = 0, window = 0;
    /* initial fill */
    for (int i = 0; i < nqps; i++) {
        struct ibv_sge sge = {.addr = (uintptr_t)(buf + (size_t)i * msg), .length = msg, .lkey = mr->lkey};
        for (int d = 0; d < depth; d++) {
            struct ibv_send_wr wr = {0}, *bad = NULL;
            wr.wr_id = i; wr.sg_list = &sge; wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = q[i].remote_addr + (uint64_t)(i % 16) * msg;
            wr.wr.rdma.rkey = q[i].rkey;
            if (ibv_post_send(q[i].qp, &wr, &bad)) { fprintf(stderr, "post fill failed qp=%d\n", i); return 1; }
            q[i].outstanding++;
        }
    }
    int inflight = 1;
    while (inflight) {
        double t = now();
        if (t >= t_end) {
            /* drain: stop reposting */
            for (int i = 0; i < nqps; i++) {
                struct ibv_wc wc[16];
                int n;
                while ((n = ibv_poll_cq(q[i].cq, 16, wc)) > 0) {
                    for (int k = 0; k < n; k++) if (wc[k].status != IBV_WC_SUCCESS) {
                        fprintf(stderr, "WC_ERR qp=%d status=%d(%s) vendor=%u\n", i, wc[k].status,
                                ibv_wc_status_str(wc[k].status), wc[k].vendor_err);
                        return 1;
                    }
                    q[i].outstanding -= n; total += (uint64_t)n * msg; window += (uint64_t)n * msg;
                }
            }
            inflight = 0;
            for (int i = 0; i < nqps; i++) if (q[i].outstanding > 0) inflight = 1;
            if (now() - t_end > 10) { fprintf(stderr, "drain timeout\n"); break; }
        } else {
            for (int i = 0; i < nqps; i++) {
                struct ibv_wc wc[16];
                int n = ibv_poll_cq(q[i].cq, 16, wc);
                if (n < 0) { fprintf(stderr, "poll failed\n"); return 1; }
                if (n == 0) continue;
                for (int k = 0; k < n; k++) if (wc[k].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "WC_ERR qp=%d status=%d(%s) vendor=%u\n", i, wc[k].status,
                            ibv_wc_status_str(wc[k].status), wc[k].vendor_err);
                    return 1;
                }
                q[i].outstanding -= n; total += (uint64_t)n * msg; window += (uint64_t)n * msg;
                struct ibv_sge sge = {.addr = (uintptr_t)(buf + (size_t)i * msg), .length = msg, .lkey = mr->lkey};
                for (int k = 0; k < n; k++) {
                    struct ibv_send_wr wr = {0}, *bad = NULL;
                    wr.wr_id = i; wr.sg_list = &sge; wr.num_sge = 1;
                    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
                    wr.wr.rdma.remote_addr = q[i].remote_addr + (uint64_t)(i % 16) * msg;
                    wr.wr.rdma.rkey = q[i].rkey;
                    if (ibv_post_send(q[i].qp, &wr, &bad)) { fprintf(stderr, "repost failed\n"); return 1; }
                    q[i].outstanding++;
                }
            }
        }
        t = now();
        if (t - t_sample >= sample_ms / 1000.0) {
            double dt = t - t_sample;
            printf("CSV %llu,%.3f\n", (unsigned long long)((t - t0) * 1000),
                   (double)window * 8.0 / dt / 1e9);
            fflush(stdout);
            window = 0; t_sample = t;
        }
    }
    double el = now() - t0;
    printf("SUMMARY nqps=%d depth=%d msg=%u elapsed_s=%.3f total_gbps=%.3f\n",
           nqps, depth, msg, el, (double)total * 8.0 / el / 1e9);
    char out[WIRE] = {0};
    snprintf(out, sizeof(out), "DONE");
    transfer(fd, out, WIRE, 1);
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc == 4 && !strcmp(argv[1], "server"))
        return run_server(atoi(argv[2]), atoi(argv[3]));
    if (argc >= 7 && !strcmp(argv[1], "client"))
        return run_client(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]),
                          argc >= 8 ? (uint32_t)atoi(argv[7]) : (1U << 20),
                          argc >= 9 ? atoi(argv[8]) : 200);
    fprintf(stderr, "usage:\n  %s server <port> <nqps>\n  %s client <ip> <port> <nqps> <depth> <seconds> [msg_bytes] [sample_ms]\n", argv[0], argv[0]);
    return 2;
}
