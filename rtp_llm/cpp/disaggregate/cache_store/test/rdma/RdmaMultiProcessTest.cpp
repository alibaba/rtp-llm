#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "aios/network/accl-barex/include/accl/barex/xchannel.h"
#include "autil/NetUtil.h"

#include <sys/wait.h>
#include <unistd.h>

namespace rtp_llm {

struct ConnectionEndpoint {
    char     ip[64];
    uint32_t port;
    uint64_t peer_channel;
};

class RdmaMultiProcessTest: public ::testing::Test {};

TEST_F(RdmaMultiProcessTest, testCrossPD_ProtectionError) {
    // 多进程复现线上 bug: Prefill 用 Decode B 的 QP 写 Decode A 的 rkey/remote_addr
    //
    // 线上场景：
    //   Decode A 在 PD_A 中注册 buffer → rkey_A
    //   Decode B 在 PD_B 中注册 buffer → rkey_B
    //   旧代码 port-only 索引 → Prefill 拿到 B 的 QP 但用 A 的 rkey_A
    //   B 的 PD 中找不到 rkey_A → mlx5 local protection error (0x3a)
    //
    // 多进程模拟：
    //   父进程 = Prefill + Decode A（同一个 PD_A）
    //   子进程 = Decode B（独立 PD_B）
    //   父进程通过 B 的 QP 写 A 的 rkey → PD_B 找不到 rkey_A → protection error

    int to_child[2], from_child[2];
    ASSERT_EQ(0, pipe(to_child));
    ASSERT_EQ(0, pipe(from_child));

    auto server_port = autil::NetUtil::randomPort();

    pid_t pid = fork();
    ASSERT_NE(-1, pid);

    if (pid == 0) {
        // ===== 子进程 (Decode Rank B) =====
        close(to_child[1]);
        close(from_child[0]);

        uint8_t cmd = 0;
        if (read(to_child[0], &cmd, 1) != 1 || cmd != 1) {
            _exit(1);
        }

        auto memory_util = createMemoryUtilImpl(true);
        if (!memory_util || !memory_util->isRdmaMode()) {
            _exit(2);
        }

        auto client = std::make_shared<RdmaClient>(memory_util);
        if (!client->init(1, 1)) {
            _exit(3);
        }

        auto local_ip = autil::NetUtil::getBindIp();
        auto rdma_connection = client->getConnection(local_ip, server_port);
        if (!rdma_connection) {
            _exit(4);
        }

        auto xconns = rdma_connection->getAllConnections();
        uint32_t count = xconns.size();
        write(from_child[1], &count, sizeof(count));
        for (auto& xc : xconns) {
            auto local = xc->GetLocalAddrAndPort();
            ConnectionEndpoint ep;
            memset(&ep, 0, sizeof(ep));
            strncpy(ep.ip, local.first.c_str(), sizeof(ep.ip) - 1);
            ep.port = local.second;
            ep.peer_channel = xc->GetChannel()->GetPeerChannel();
            write(from_child[1], &ep, sizeof(ep));
        }

        if (read(to_child[0], &cmd, 1) != 1 || cmd != 2) {
            _exit(5);
        }

        rdma_connection.reset();
        client.reset();
        memory_util.reset();
        close(to_child[0]);
        close(from_child[1]);
        _exit(0);
    }

    // ===== 父进程 (Prefill + Decode A) =====
    close(to_child[0]);
    close(from_child[1]);

    auto memory_util = createMemoryUtilImpl(true);
    ASSERT_TRUE(memory_util != nullptr);
    if (!memory_util->isRdmaMode()) {
        uint8_t cmd = 2;
        write(to_child[1], &cmd, 1);
        int status;
        waitpid(pid, &status, 0);
        close(to_child[1]);
        close(from_child[0]);
        return;
    }

    auto server = std::make_shared<RdmaServer>(memory_util);
    if (!server->init(server_port, 1, 1)) {
        uint8_t skip_cmd = 2;
        write(to_child[1], &skip_cmd, 1);
        int skip_status;
        waitpid(pid, &skip_status, 0);
        close(to_child[1]);
        close(from_child[0]);
        GTEST_SKIP() << "RDMA server init failed (CQ/NIC resource issue), skipping";
    }

    uint8_t cmd = 1;
    write(to_child[1], &cmd, 1);

    uint32_t conn_count = 0;
    ASSERT_EQ((ssize_t)sizeof(conn_count), read(from_child[0], &conn_count, sizeof(conn_count)));
    ASSERT_GT(conn_count, 0u);

    std::vector<AddrPortChannel> addr_ports;
    for (uint32_t i = 0; i < conn_count; i++) {
        ConnectionEndpoint ep;
        ASSERT_EQ((ssize_t)sizeof(ep), read(from_child[0], &ep, sizeof(ep)));
        addr_ports.push_back({std::string(ep.ip), ep.port, ep.peer_channel});
    }

    auto rdma_conn_B = server->getConnection(addr_ports);
    ASSERT_TRUE(rdma_conn_B != nullptr);

    uint32_t block_size = 64;

    void* local_buf = malloc(block_size);
    ASSERT_TRUE(local_buf != nullptr);
    memset(local_buf, 'L', block_size);
    ASSERT_TRUE(memory_util->regUserMr(local_buf, block_size, false));

    void* fake_remote_A = malloc(block_size);
    ASSERT_TRUE(fake_remote_A != nullptr);
    memset(fake_remote_A, 'A', block_size);
    ASSERT_TRUE(memory_util->regUserMr(fake_remote_A, block_size, false));

    ::accl::barex::memp_t mem_A;
    ASSERT_TRUE(memory_util->findMemoryMr(&mem_A, fake_remote_A, block_size, false, true));
    ASSERT_FALSE(mem_A.mrs.empty());

    auto local_block = std::make_shared<BlockBuffer>(
        "local", std::shared_ptr<void>(local_buf, [](void*) {}), block_size, false, true);

    auto peer_info = std::make_shared<BlockBufferInfo>();
    peer_info->set_key("peer_A");
    peer_info->set_len(block_size);
    auto rdma_info = peer_info->mutable_rdma_info();
    rdma_info->set_addr(reinterpret_cast<uint64_t>(fake_remote_A));
    for (auto& [nicid, mr] : mem_A.mrs) {
        auto nic_rkey = rdma_info->add_nic_rkeys();
        nic_rkey->set_nicid(nicid);
        nic_rkey->set_rkey(mr->rkey);
    }

    std::mutex sync_mutex;
    bool       write_success = true;
    auto       callback      = [&sync_mutex, &write_success](bool success) {
        write_success = success;
        sync_mutex.unlock();
    };

    auto write_block = std::make_shared<RdmaConnectionWriteBlock>(local_block, peer_info, callback);

    sync_mutex.lock();
    rdma_conn_B->writeBlocks({write_block});

    sync_mutex.lock();
    sync_mutex.unlock();

    // rkey_A 属于父进程 PD_A，在子进程 PD_B 中无效 → RDMA write 失败
    // dmesg: mlx5_bond_X/Y: QP XXXXX error: local protection error (0x3a 0x0 0x95)
    EXPECT_FALSE(write_success);

    cmd = 2;
    write(to_child[1], &cmd, 1);

    int status;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(0, WEXITSTATUS(status));

    memory_util->deregUserMr(local_buf, false);
    memory_util->deregUserMr(fake_remote_A, false);
    free(local_buf);
    free(fake_remote_A);

    rdma_conn_B.reset();
    server.reset();
    memory_util.reset();

    close(to_child[1]);
    close(from_child[0]);
}

}  // namespace rtp_llm
