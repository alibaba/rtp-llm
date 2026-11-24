#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/test/RemoteServerProcessTestUtils.h"
#include <gtest/gtest.h>
#include <future>
#include <thread>
#include <pthread.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockFormat.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/remote_mock_control.pb.h"
extern char** environ;
namespace rtp_llm {
namespace {
class ServerProcess {
public:
    ~ServerProcess() {
        stop();
    }
    bool start(uint32_t port) {
        int pipes[2];
        if (pipe(pipes))
            return false;
        std::string executable = std::string(getenv("TEST_SRCDIR"))
                                 + "/rtp_llm/rtp_llm/cpp/disaggregate/cache_store/remote_mock/remote_mock_server";
        std::vector<std::string> args = {executable,
                                         "--port=" + std::to_string(port),
                                         "--total-bytes=1048576",
                                         "--source-bytes=4096",
                                         "--write-slots=1",
                                         "--write-slot-bytes=64",
                                         "--read-staging-bytes=65536",
                                         "--write-staging-bytes=65536",
                                         "--max-rpc-bytes=64",
                                         "--max-blocks-per-rpc=4"};
        if (const char* output = getenv("TEST_UNDECLARED_OUTPUTS_DIR"))
            args.push_back("--event-path=" + std::string(output) + "/remote-loopback-events-" + std::to_string(starts++)
                           + ".jsonl");
        std::vector<char*> argv;
        for (auto& a : args)
            argv.push_back(a.data());
        argv.push_back(nullptr);
        std::vector<std::string> environment;
        for (char** e = environ; *e; ++e)
            if (std::string(*e).rfind("CUDA_VISIBLE_DEVICES=", 0) != 0
                && std::string(*e).rfind("NVIDIA_VISIBLE_DEVICES=", 0) != 0)
                environment.emplace_back(*e);
        environment.emplace_back("CUDA_VISIBLE_DEVICES=");
        environment.emplace_back("NVIDIA_VISIBLE_DEVICES=void");
        std::vector<char*> env;
        for (auto& e : environment)
            env.push_back(e.data());
        env.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, pipes[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, pipes[0]);
        posix_spawn_file_actions_addclose(&actions, pipes[1]);
        int result = posix_spawn(&pid, executable.c_str(), &actions, nullptr, argv.data(), env.data());
        posix_spawn_file_actions_destroy(&actions);
        ::close(pipes[1]);
        if (result) {
            ::close(pipes[0]);
            pid = -1;
            return false;
        }
        std::string output;
        const bool  ready =
            remote_mock_test::waitReady(pipes[0], std::chrono::steady_clock::now() + std::chrono::seconds(10), output);
        ::close(pipes[0]);
        if (ready)
            return true;
        std::cerr << output;
        return false;
    }
    void stop() {
        if (pid > 0) {
            kill(pid, SIGTERM);
            int status = 0;
            EXPECT_EQ(remote_mock_test::waitChild(pid, &status), pid);
            EXPECT_TRUE(WIFEXITED(status));
            EXPECT_EQ(0, WEXITSTATUS(status));
            pid = -1;
        }
    }

private:
    pid_t    pid    = -1;
    unsigned starts = 0;
};
TEST(RemoteMockReadinessTest, QuietChildCanBecomeReadyAfterOneSecond) {
    int pipes[2];
    ASSERT_EQ(pipe(pipes), 0);
    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(1300));
        EXPECT_EQ(write(pipes[1], "REMOTE_MOCK_READY\n", 18), 18);
        close(pipes[1]);
    });
    std::string output;
    EXPECT_TRUE(
        remote_mock_test::waitReady(pipes[0], std::chrono::steady_clock::now() + std::chrono::seconds(3), output));
    producer.join();
    close(pipes[0]);
}
TEST(RemoteMockReadinessTest, InterruptedPollAndWaitpidRetryOwnedChild) {
    struct sigaction action{}, previous{};
    action.sa_handler = [](int) {};
    sigemptyset(&action.sa_mask);
    ASSERT_EQ(sigaction(SIGUSR1, &action, &previous), 0);
    const auto target = pthread_self();
    int        pipes[2];
    ASSERT_EQ(pipe(pipes), 0);
    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        pthread_kill(target, SIGUSR1);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        EXPECT_EQ(write(pipes[1], "REMOTE_MOCK_READY\n", 18), 18);
        close(pipes[1]);
    });
    std::string output;
    EXPECT_TRUE(
        remote_mock_test::waitReady(pipes[0], std::chrono::steady_clock::now() + std::chrono::seconds(3), output));
    producer.join();
    close(pipes[0]);
    char  executable[] = "/bin/sleep", delay[] = "0.2";
    char* argv[] = {executable, delay, nullptr};
    pid_t child;
    ASSERT_EQ(posix_spawn(&child, executable, nullptr, nullptr, argv, environ), 0);
    std::thread interrupt([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        pthread_kill(target, SIGUSR1);
    });
    int         status = 0;
    const auto  waited = remote_mock_test::waitChild(child, &status);
    EXPECT_EQ(waited, child);
    interrupt.join();
    if (waited != child)
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    EXPECT_TRUE(WIFEXITED(status));
    sigaction(SIGUSR1, &previous, nullptr);
}
TEST(RemoteMockReadinessTest, EofAndAbsoluteDeadlineReturnFailure) {
    int pipes[2];
    ASSERT_EQ(pipe(pipes), 0);
    std::string output;
    const auto  start = std::chrono::steady_clock::now();
    EXPECT_FALSE(remote_mock_test::waitReady(pipes[0], start + std::chrono::milliseconds(30), output));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(200));
    close(pipes[1]);
    EXPECT_FALSE(remote_mock_test::waitReady(pipes[0], start + std::chrono::seconds(3), output));
    close(pipes[0]);
}

TEST(RemoteMockLoopbackTest, ExistingClientAndRegisteredHostSnapshotAcrossCpuProcess) {
    ServerProcess server;
    const auto    port = autil::NetUtil::randomPort();
    ASSERT_TRUE(server.start(port));
    TcpClient control_client;
    ASSERT_TRUE(control_client.init(1));
    auto channel = control_client.getChannel("127.0.0.1", port);
    ASSERT_TRUE(channel);
    remote_mock::RemoteMockControlService_Stub stub(channel.get());
    auto                                       buffers = std::make_shared<RequestBlockBufferStore>(nullptr);
    TcpMessager                                messenger(nullptr, buffers, nullptr);
    MessagerInitParams                         params;
    params.io_thread_count     = 1;
    params.worker_thread_count = 1;
    params.worker_queue_size   = 16;
    params.server_port         = autil::NetUtil::randomPort();
    params.device_id           = -1;
    ASSERT_TRUE(messenger.init(params));
    std::string                  stale_key;
    remote_mock::InitRankRequest last_manifest;
    for (uint32_t delay : {0, 5, 20}) {
        arpc::ANetRPCController hc;
        hc.SetExpireTime(2000);
        remote_mock::HealthRequest   h;
        remote_mock::ControlResponse health;
        stub.health(&hc, &h, &health, nullptr);
        ASSERT_FALSE(hc.Failed());
        remote_mock::InitRankRequest q;
        q.mutable_session()->set_server_epoch(health.server_epoch());
        q.mutable_session()->set_session_id(19);
        q.mutable_session()->set_generation(delay + 1);
        q.set_rank(0);
        q.set_expected_ranks(1);
        q.set_layout_hash(111);
        q.set_read_delay_ms(delay);
        q.set_write_delay_ms(delay);
        auto g = q.add_groups();
        g->set_group(0);
        g->set_pool(0);
        g->set_payload_bytes(16);
        g->set_storage_bytes(16);
        g->set_stride_bytes(16);
        g->set_source_count(2);
        g->set_alignment(16);
        g->set_dtype("fp16");
        g->add_member_offsets(0);
        g->set_tokens_per_block(1);
        g->set_cp_world_size(1);
        g->set_cp_rank(0);
        last_manifest.CopyFrom(q);
        arpc::ANetRPCController ic;
        ic.SetExpireTime(2000);
        remote_mock::ControlResponse init;
        stub.initRank(&ic, &q, &init, nullptr);
        ASSERT_FALSE(ic.Failed());
        ASSERT_EQ(EC_SUCCESS, init.error_code()) << init.error_message();
        ASSERT_TRUE(init.ready());
        RemoteMockKey key;
        key.session     = {init.session().server_epoch(), init.session().session_id(), init.session().generation()};
        key.layout_hash = 111;
        key.length      = 16;
        key.checksum    = init.sources(0).record_crc32c();
        auto read_key   = encodeRemoteMockKey(key);
        if (stale_key.empty())
            stale_key = read_key;
        auto destination = std::make_shared<std::string>(16, '?');
        auto request     = std::make_shared<RequestBlockBuffer>("existing-client-" + std::to_string(delay));
        request->addBlock(std::make_shared<BlockBuffer>(
            read_key, std::shared_ptr<void>(destination, destination->data()), 16, false, true));
        std::promise<bool> read_done;
        auto               read_future = read_done.get_future();
        auto               start       = std::chrono::steady_clock::now();
        messenger.load(std::make_shared<LoadRequest>(
                           "127.0.0.1",
                           port,
                           0,
                           request,
                           [&](bool ok, CacheStoreErrorCode) { read_done.set_value(ok); },
                           2000,
                           1,
                           0),
                       std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 16));
        ASSERT_EQ(std::future_status::ready, read_future.wait_for(std::chrono::seconds(3)));
        ASSERT_TRUE(read_future.get());
        auto read_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        EXPECT_GE(read_us + 1000, delay * 1000);
        EXPECT_EQ(std::string(16, '\0'), *destination);
        auto host         = std::make_shared<std::string>("A\0B\xff"
                                                          "0123456789xy!",
                                                  16);
        auto local_key    = "host-" + std::to_string(delay);
        auto registration = buffers->regUserBuffersScoped(
            {std::make_shared<BlockBuffer>(local_key, std::shared_ptr<void>(host, host->data()), 16, false, true)});
        ASSERT_TRUE(registration);
        key.write     = true;
        key.operation = 1;
        key.checksum  = remoteMockCrc32c(reinterpret_cast<const uint8_t*>(host->data()), host->size());
        auto transfer = std::make_shared<TransferRequest>("existing-transfer", "127.0.0.1", port, 0, 2000);
        transfer->buffer_pairs[local_key] = encodeRemoteMockKey(key);
        std::promise<bool> write_done;
        auto               write_future = write_done.get_future();
        transfer->callback              = [&](bool ok, CacheStoreErrorCode, const auto&) { write_done.set_value(ok); };
        start                           = std::chrono::steady_clock::now();
        messenger.transfer(transfer);
        ASSERT_EQ(std::future_status::ready, write_future.wait_for(std::chrono::seconds(3)));
        ASSERT_TRUE(write_future.get());
        auto write_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        EXPECT_GE(write_us + 1000, delay * 1000);
        std::cout << "two_process delay_ms=" << delay << " read_us=" << read_us << " write_us=" << write_us
                  << " slack_us=1000 host_bytes=16 callback_port=" << params.server_port << "\n";
        registration.reset();
        EXPECT_EQ(nullptr, buffers->findUserBuffer(local_key));
        remote_mock::SessionRequest cq;
        cq.mutable_session()->CopyFrom(init.session());
        remote_mock::ControlResponse cr;
        arpc::ANetRPCController      cc;
        cc.SetExpireTime(2000);
        stub.close(&cc, &cq, &cr, nullptr);
        ASSERT_FALSE(cc.Failed());
        ASSERT_EQ(EC_SUCCESS, cr.error_code());
    }
    // A new process epoch must reject keys from the prior executable instance.
    channel.reset();
    server.stop();
    ASSERT_TRUE(server.start(port));
    TcpClient restarted;
    ASSERT_TRUE(restarted.init(1));
    auto rc = restarted.getChannel("127.0.0.1", port);
    ASSERT_TRUE(rc);
    remote_mock::RemoteMockControlService_Stub restarted_control(rc.get());
    remote_mock::HealthRequest                 health_request;
    remote_mock::ControlResponse               health;
    arpc::ANetRPCController                    hc;
    hc.SetExpireTime(2000);
    restarted_control.health(&hc, &health_request, &health, nullptr);
    ASSERT_FALSE(hc.Failed());
    EXPECT_NE(last_manifest.session().server_epoch(), health.server_epoch());
    last_manifest.mutable_session()->set_server_epoch(health.server_epoch());
    last_manifest.mutable_session()->set_generation(1);
    last_manifest.set_read_delay_ms(0);
    last_manifest.set_write_delay_ms(0);
    remote_mock::ControlResponse init;
    arpc::ANetRPCController      ic;
    ic.SetExpireTime(2000);
    restarted_control.initRank(&ic, &last_manifest, &init, nullptr);
    ASSERT_FALSE(ic.Failed());
    ASSERT_EQ(EC_SUCCESS, init.error_code());
    KvCacheStoreService_Stub data(rc.get());
    CacheLoadRequest         q;
    q.set_timeout_ms(1000);
    auto b = q.add_blocks();
    b->set_key(stale_key);
    b->set_len(16);
    CacheLoadResponse       r;
    arpc::ANetRPCController c;
    c.SetExpireTime(2000);
    data.load(&c, &q, &r, nullptr);
    ASSERT_FALSE(c.Failed());
    EXPECT_NE(EC_SUCCESS, r.error_code());
    auto fresh = decodeRemoteMockKey(stale_key);
    ASSERT_TRUE(fresh.ok());
    fresh->session.server_epoch = health.server_epoch();
    q.mutable_blocks(0)->set_key(encodeRemoteMockKey(*fresh));
    arpc::ANetRPCController nc;
    nc.SetExpireTime(2000);
    CacheLoadResponse nr;
    data.load(&nc, &q, &nr, nullptr);
    ASSERT_FALSE(nc.Failed());
    EXPECT_EQ(EC_SUCCESS, nr.error_code());
}
}  // namespace
}  // namespace rtp_llm
