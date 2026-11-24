#include <gtest/gtest.h>
#include <future>
#include <atomic>
#include <execinfo.h>
#include <dlfcn.h>
#include <cstring>
#include <mutex>
#include <fstream>
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/test/RemoteMockTestPeer.h"
#include "autil/NetUtil.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockFormat.h"

namespace rtp_llm {
namespace {
using Clock = std::chrono::steady_clock;
std::atomic<bool> fail_timer_allocation{false};
struct SourceCopyGate {
    std::promise<void>       copying, release;
    std::shared_future<void> released = release.get_future().share();
};
thread_local SourceCopyGate* source_gate     = nullptr;
thread_local bool            source_resolved = false;
extern "C" absl::StatusOr<std::shared_ptr<const RemoteMockSourceLease>>
realResolve(const RemoteMockState*, const std::string&, uint64_t) asm(
    "__real__ZNK7rtp_llm15RemoteMockState7resolveERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm");
extern "C" absl::StatusOr<std::shared_ptr<const RemoteMockSourceLease>>
wrapResolve(const RemoteMockState*, const std::string&, uint64_t) asm(
    "__wrap__ZNK7rtp_llm15RemoteMockState7resolveERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm");
extern "C" absl::StatusOr<std::shared_ptr<const RemoteMockSourceLease>>
wrapResolve(const RemoteMockState* state, const std::string& key, uint64_t length) {
    auto result = realResolve(state, key, length);
    if (source_gate && result.ok())
        source_resolved = true;
    return result;
}
extern "C" void* __real__Znwm(size_t);
extern "C" void* __wrap__Znwm(size_t bytes) {
    // Heap response, one previously reserved source: the next allocation after
    // resolve builds the outgoing protobuf block, outside the lifecycle lock.
    if (source_gate && std::exchange(source_resolved, false)) {
        source_gate->copying.set_value();
        source_gate->released.wait();
    }
    if (fail_timer_allocation.load(std::memory_order_relaxed)) {
        void* frames[20];
        int   n = backtrace(frames, 20);
        for (int i = 1; i < n; ++i) {
            Dl_info info{};
            if (dladdr(frames[i], &info) && info.dli_sname
                && (std::strstr(info.dli_sname, "RemoteMockTimer3runEv")
                    || std::strstr(info.dli_sname, "RemoteMockTimer6invoke"))
                && fail_timer_allocation.exchange(false))
                throw std::bad_alloc();
        }
    }
    return __real__Znwm(bytes);
}

class RemoteMockServiceTest: public ::testing::Test {
protected:
    virtual void configureOptions(RemoteMockServiceOptions&) {}
    void         SetUp() override {
        RemoteMockServiceOptions options;
        options.limits.total_bytes         = 1 << 20;
        options.limits.source_bytes        = 4096;
        options.limits.write_slots         = 1;
        options.limits.write_slot_bytes    = 64;
        options.limits.read_staging_bytes  = 65536;
        options.limits.write_staging_bytes = 65536;
        options.limits.max_rpc_bytes       = 64;
        options.limits.max_blocks_per_rpc  = 4;
        options.reads_per_connection       = 1;
        event_path = std::string(getenv("TEST_UNDECLARED_OUTPUTS_DIR") ? getenv("TEST_UNDECLARED_OUTPUTS_DIR") : "/tmp")
                     + "/remote-service-events.jsonl";
        options.event_path      = event_path;
        options.max_event_bytes = 4096;
        configureOptions(options);
        service = std::make_unique<RemoteMockService>(options);
        ASSERT_TRUE(service->init());
        server = std::make_unique<TcpServer>();
        ASSERT_TRUE(server->init(1, 1, false));
        ASSERT_TRUE(server->registerService(service.get()));
        ASSERT_TRUE(server->registerService(service->controlService()));
        port = autil::NetUtil::randomPort();
        ASSERT_TRUE(server->start(port, "127.0.0.1"));
        ASSERT_TRUE(client.init(1));
        channel = client.getChannel("127.0.0.1", port);
        ASSERT_TRUE(channel);
    }
    void TearDown() override {
        service->shutdown();
        server.reset();
        service.reset();
    }
    remote_mock::ControlResponse control(const remote_mock::InitRankRequest& request) {
        remote_mock::RemoteMockControlService_Stub stub(channel.get());
        arpc::ANetRPCController                    c;
        c.SetExpireTime(2000);
        remote_mock::ControlResponse r;
        stub.initRank(&c, &request, &r, nullptr);
        EXPECT_FALSE(c.Failed()) << c.ErrorText();
        return r;
    }
    void initialize(uint32_t read_delay = 0, uint32_t write_delay = 0, uint64_t generation = 1) {
        remote_mock::RemoteMockControlService_Stub stub(channel.get());
        arpc::ANetRPCController                    c;
        c.SetExpireTime(2000);
        remote_mock::HealthRequest   h;
        remote_mock::ControlResponse health;
        stub.health(&c, &h, &health, nullptr);
        ASSERT_FALSE(c.Failed());
        remote_mock::InitRankRequest request;
        request.mutable_session()->set_server_epoch(health.server_epoch());
        request.mutable_session()->set_session_id(9);
        request.mutable_session()->set_generation(generation);
        request.set_rank(0);
        request.set_expected_ranks(1);
        request.set_layout_hash(11);
        request.set_read_delay_ms(read_delay);
        request.set_write_delay_ms(write_delay);
        auto g = request.add_groups();
        g->set_group(0);
        g->set_pool(0);
        g->set_payload_bytes(16);
        g->set_storage_bytes(16);
        g->set_stride_bytes(16);
        g->set_crc_enabled(false);
        g->set_source_count(2);
        g->set_alignment(16);
        g->set_dtype("fp16");
        g->add_member_offsets(0);
        g->set_tokens_per_block(1);
        g->set_cp_world_size(1);
        g->set_cp_rank(0);
        auto r = control(request);
        ASSERT_EQ(EC_SUCCESS, r.error_code()) << r.error_message();
        ASSERT_TRUE(r.ready());
        ASSERT_EQ(1, r.sources_size());
        session = {r.session().server_epoch(), r.session().session_id(), r.session().generation()};
        RemoteMockKey key;
        key.session     = session;
        key.layout_hash = 11;
        key.length      = 16;
        key.checksum    = r.sources(0).record_crc32c();
        source          = encodeRemoteMockKey(key);
    }
    CacheLoadResponse read(const std::string& key, uint32_t timeout = 2000) {
        KvCacheStoreService_Stub stub(channel.get());
        arpc::ANetRPCController  c;
        c.SetExpireTime(timeout + 500);
        CacheLoadRequest q;
        q.set_timeout_ms(timeout);
        q.set_requestid("no-production-watch");
        auto b = q.add_blocks();
        b->set_key(key);
        b->set_len(16);
        CacheLoadResponse r;
        stub.load(&c, &q, &r, nullptr);
        EXPECT_FALSE(c.Failed()) << c.ErrorText();
        return r;
    }
    CacheTransferResponse write(uint64_t sequence, uint32_t callback_port, uint32_t timeout_ms = 1500) {
        KvCacheStoreService_Stub stub(channel.get());
        arpc::ANetRPCController  c;
        c.SetExpireTime(2000);
        CacheTransferRequest q;
        q.set_timeout_ms(timeout_ms);
        q.set_client_ip("127.0.0.1");
        q.set_client_port(callback_port);
        q.set_request_id("snapshot");
        auto t = q.mutable_transfer_info();
        t->set_partition_count(1);
        t->set_partition_id(0);
        t->add_local_keys("host");
        RemoteMockKey k;
        k.write       = true;
        k.session     = session;
        k.layout_hash = 11;
        k.length      = 16;
        k.operation   = sequence;
        k.checksum    = remoteMockCrc32c(reinterpret_cast<const uint8_t*>(peer.payload.data()), 16);
        t->add_remote_keys(encodeRemoteMockKey(k));
        auto b = q.add_blocks();
        b->set_key("host");
        b->set_len(16);
        b->set_addr(1234);
        CacheTransferResponse r;
        stub.transfer(&c, &q, &r, nullptr);
        EXPECT_FALSE(c.Failed()) << c.ErrorText();
        return r;
    }
    uint32_t startPeer() {
        peer_server = std::make_unique<TcpServer>();
        EXPECT_TRUE(peer_server->init(1, 1, false));
        EXPECT_TRUE(peer_server->registerService(&peer));
        auto p = autil::NetUtil::randomPort();
        EXPECT_TRUE(peer_server->start(p, "127.0.0.1"));
        return p;
    }
    void close() {
        remote_mock::RemoteMockControlService_Stub stub(channel.get());
        arpc::ANetRPCController                    c;
        c.SetExpireTime(2000);
        remote_mock::SessionRequest q;
        q.mutable_session()->set_server_epoch(session.server_epoch);
        q.mutable_session()->set_session_id(session.session_id);
        q.mutable_session()->set_generation(session.generation);
        remote_mock::ControlResponse r;
        stub.close(&c, &q, &r, nullptr);
        EXPECT_FALSE(c.Failed());
        EXPECT_EQ(EC_SUCCESS, r.error_code());
    }
    std::string                           event_path;
    RemoteMockTestPeer                    peer;
    std::unique_ptr<TcpServer>            peer_server;
    std::unique_ptr<RemoteMockService>    service;
    std::unique_ptr<TcpServer>            server;
    TcpClient                             client;
    std::shared_ptr<arpc::RPCChannelBase> channel;
    uint32_t                              port;
    RemoteMockSession                     session;
    std::string                           source;
};

TEST_F(RemoteMockServiceTest, ExistingClientReadsRealPayload) {
    initialize();
    for (int i = 0; i < 3; ++i) {
        auto r = read(source);
        ASSERT_EQ(EC_SUCCESS, r.error_code());
        ASSERT_EQ(1, r.blocks_size());
        EXPECT_EQ(source, r.blocks(0).key());
        EXPECT_EQ(16, r.blocks(0).len());
        EXPECT_EQ(std::string(16, '\0'), std::string(r.blocks(0).content().data(), r.blocks(0).content().size()));
    }
    EXPECT_EQ(48, service->stats().read_payload_bytes);
}
TEST_F(RemoteMockServiceTest, ReadDelayDoesNotBlockOtherRpc) {
    initialize(200);
    auto                                       pending = std::async(std::launch::async, [&] { return read(source); });
    auto                                       start   = Clock::now();
    remote_mock::RemoteMockControlService_Stub stub(channel.get());
    arpc::ANetRPCController                    c;
    c.SetExpireTime(100);
    remote_mock::HealthRequest   q;
    remote_mock::ControlResponse r;
    stub.health(&c, &q, &r, nullptr);
    EXPECT_FALSE(c.Failed());
    EXPECT_TRUE(r.ready());
    EXPECT_LT(Clock::now() - start, std::chrono::milliseconds(150));
    EXPECT_EQ(EC_SUCCESS, pending.get().error_code());
}
TEST_F(RemoteMockServiceTest, DelayLowerBoundsZeroFiveTwenty) {
    for (uint32_t delay : {0, 5, 20}) {
        initialize(delay, 0, delay + 1);
        auto start = Clock::now();
        auto r     = read(source);
        ASSERT_EQ(EC_SUCCESS, r.error_code());
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
        std::cout << "read delay_ms=" << delay << " elapsed_us=" << elapsed << " slack_us=1000\n";
        EXPECT_GE(elapsed + 1000, delay * 1000);
        EXPECT_LT(elapsed, 2000000);
        close();
    }
}
TEST_F(RemoteMockServiceTest, TimeoutAndCloseDrain) {
    initialize(500);
    auto r = read(source, 10);
    EXPECT_NE(EC_SUCCESS, r.error_code());
    close();
    initialize(0, 0, 2);
    EXPECT_EQ(EC_SUCCESS, read(source).error_code());
}
TEST_F(RemoteMockServiceTest, PayloadLimitSplitsOrRejects) {
    initialize();
    KvCacheStoreService_Stub stub(channel.get());
    arpc::ANetRPCController  c;
    c.SetExpireTime(2000);
    CacheLoadRequest q;
    q.set_timeout_ms(1000);
    for (int i = 0; i < 5; ++i) {
        auto b = q.add_blocks();
        b->set_key(source);
        b->set_len(16);
    }
    CacheLoadResponse r;
    stub.load(&c, &q, &r, nullptr);
    EXPECT_FALSE(c.Failed());
    EXPECT_NE(EC_SUCCESS, r.error_code());
    EXPECT_EQ(0, r.blocks_size());
}
TEST_F(RemoteMockServiceTest, RestartRejectsOldKeys) {
    initialize();
    auto old = source;
    close();
    initialize(0, 0, 2);
    EXPECT_NE(EC_SUCCESS, read(old).error_code());
    EXPECT_EQ(EC_SUCCESS, read(source).error_code());
}
TEST_F(RemoteMockServiceTest, TransferPullsRegisteredHostSnapshot) {
    initialize();
    auto port = startPeer();
    auto r    = write(1, port);
    EXPECT_EQ(EC_SUCCESS, r.error_code()) << r.error_msg();
    EXPECT_EQ(1, peer.calls);
    EXPECT_EQ(16, service->stats().write_payload_bytes);
    EXPECT_NE(EC_SUCCESS, write(1, port).error_code());
    EXPECT_EQ(1, peer.calls);
}
TEST_F(RemoteMockServiceTest, FullWritePoolDoesNotCallBlockRead) {
    initialize(0, 200);
    auto port  = startPeer();
    auto first = std::async(std::launch::async, [&] { return write(1, port); });
    // Wait for the observable pull without using a fixed sleep as synchronization.
    auto deadline = Clock::now() + std::chrono::seconds(1);
    while (peer.calls == 0 && Clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_EQ(1, peer.calls);
    EXPECT_NE(EC_SUCCESS, write(2, port).error_code());
    EXPECT_EQ(1, peer.calls);
    EXPECT_EQ(EC_SUCCESS, read(source).error_code());
    EXPECT_EQ(EC_SUCCESS, first.get().error_code());
}
TEST_F(RemoteMockServiceTest, WriteAckDelayAfterPayloadOnly) {
    auto port = startPeer();
    for (uint32_t delay : {0, 5, 20}) {
        initialize(0, delay, delay + 1);
        auto start = Clock::now();
        ASSERT_EQ(EC_SUCCESS, write(1, port).error_code());
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
        std::cout << "write delay_ms=" << delay << " elapsed_us=" << elapsed << " slack_us=1000\n";
        EXPECT_GE(elapsed + 1000, delay * 1000);
        EXPECT_LT(elapsed, 2000000);
        close();
    }
}
TEST_F(RemoteMockServiceTest, RejectMalformedReverseReplyBeforeAck) {
    initialize();
    auto           port                  = startPeer();
    const uint64_t cumulative_received[] = {16, 25, 41, 57, 57, 73};
    for (int corrupt = 1; corrupt <= 6; ++corrupt) {
        peer.corruption = corrupt;
        EXPECT_NE(EC_SUCCESS, write(corrupt, port).error_code());
        EXPECT_EQ(cumulative_received[corrupt - 1], service->stats().write_payload_bytes);
        EXPECT_EQ(0, service->stats().write_stored_ns);
    }
    peer.corruption = 0;
    EXPECT_EQ(EC_SUCCESS, write(7, port).error_code());
}
TEST_F(RemoteMockServiceTest, ActualEnvelopeIncludesUnknownFields) {
    initialize();
    KvCacheStoreService_Stub stub(channel.get());
    arpc::ANetRPCController  c;
    c.SetExpireTime(2000);
    CacheLoadRequest q;
    q.set_timeout_ms(1000);
    auto b = q.add_blocks();
    b->set_key(source);
    b->set_len(16);
    q.GetReflection()->MutableUnknownFields(&q)->AddLengthDelimited(99, std::string(7000, 'x'));
    CacheLoadResponse r;
    stub.load(&c, &q, &r, nullptr);
    ASSERT_FALSE(c.Failed());
    EXPECT_NE(EC_SUCCESS, r.error_code());
    EXPECT_EQ(0, r.blocks_size());
}
TEST_F(RemoteMockServiceTest, PerConnectionReadLimitAndUndelayedBlockRead) {
    initialize(500);
    auto first    = std::async(std::launch::async, [&] { return read(source); });
    auto deadline = Clock::now() + std::chrono::seconds(2);
    while (!service->stats().read_admitted_ns && Clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_NE(0, service->stats().read_admitted_ns);
    EXPECT_NE(EC_SUCCESS, read(source).error_code());
    EXPECT_EQ(EC_SUCCESS, first.get().error_code());
    KvCacheStoreService_Stub stub(channel.get());
    arpc::ANetRPCController  c;
    c.SetExpireTime(200);
    BlockReadRequest q;
    auto             b = q.add_blocks();
    b->set_key(source);
    b->set_len(16);
    BlockReadResponse r;
    auto              start = Clock::now();
    stub.blockRead(&c, &q, &r, nullptr);
    EXPECT_FALSE(c.Failed());
    ASSERT_EQ(EC_SUCCESS, r.error_code());
    EXPECT_LT(Clock::now() - start, std::chrono::milliseconds(200));
    EXPECT_EQ(16, r.blocks(0).content().size());
}
TEST_F(RemoteMockServiceTest, CloseDuringPullRetainsSlotUntilNetworkDrain) {
    initialize(0, 500);
    auto port    = startPeer();
    peer.hold    = true;
    auto pending = std::async(std::launch::async, [&] { return write(1, port); });
    ASSERT_TRUE(peer.waitPull());
    close();
    remote_mock::RemoteMockControlService_Stub stub(channel.get());
    arpc::ANetRPCController                    c;
    c.SetExpireTime(1000);
    remote_mock::HealthRequest   q;
    remote_mock::ControlResponse r;
    stub.health(&c, &q, &r, nullptr);
    ASSERT_FALSE(c.Failed());
    EXPECT_GT(r.allocated_bytes(), 0);
    peer.release();
    EXPECT_NE(EC_SUCCESS, pending.get().error_code());
    EXPECT_EQ(16, service->stats().write_payload_bytes);
    initialize(0, 0, 2);
    EXPECT_EQ(EC_SUCCESS, read(source).error_code());
}
TEST_F(RemoteMockServiceTest, CloseCancelsPendingDelay) {
    initialize(1000);
    auto pending  = std::async(std::launch::async, [&] { return read(source); });
    auto deadline = Clock::now() + std::chrono::seconds(2);
    while (!service->stats().read_admitted_ns && Clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_NE(0, service->stats().read_admitted_ns);
    close();
    ASSERT_EQ(std::future_status::ready, pending.wait_for(std::chrono::milliseconds(300)));
    EXPECT_NE(EC_SUCCESS, pending.get().error_code());
    initialize(0, 0, 2);
    EXPECT_EQ(EC_SUCCESS, read(source).error_code());
}
TEST_F(RemoteMockServiceTest, BoundedSidecarUsesActualDelayBoundaries) {
    initialize(5, 20);
    auto port = startPeer();
    ASSERT_EQ(EC_SUCCESS, read(source).error_code());
    ASSERT_EQ(EC_SUCCESS, write(1, port).error_code());
    auto stats = service->stats();
    EXPECT_GE(stats.read_payload_ns, stats.read_admitted_ns + 5000000);
    EXPECT_GE(stats.write_ack_ns, stats.write_stored_ns + 20000000);
    for (int i = 0; i < 10; ++i)
        ASSERT_EQ(EC_SUCCESS, read(source).error_code());
    std::ifstream file(event_path);
    std::string   contents((std::istreambuf_iterator<char>(file)), {});
    EXPECT_LE(contents.size(), 4096);
    EXPECT_NE(std::string::npos, contents.find("read_admitted"));
    EXPECT_NE(std::string::npos, contents.find("write_stored_crc"));
    EXPECT_NE(std::string::npos, contents.find("write_ack"));
    EXPECT_NE(std::string::npos, contents.find("read_copy_start"));
    EXPECT_NE(std::string::npos, contents.find("write_copy_start"));
    EXPECT_NE(std::string::npos, contents.find("boundary_ns"));
    EXPECT_EQ('\n', contents.back());
}
TEST_F(RemoteMockServiceTest, OversizedControlCannotCloseSession) {
    initialize();
    remote_mock::RemoteMockControlService_Stub stub(channel.get());
    remote_mock::SessionRequest                q;
    q.mutable_session()->set_server_epoch(session.server_epoch);
    q.mutable_session()->set_session_id(session.session_id);
    q.mutable_session()->set_generation(session.generation);
    q.GetReflection()->MutableUnknownFields(&q)->AddLengthDelimited(99, std::string(8192, 'x'));
    arpc::ANetRPCController c;
    c.SetExpireTime(1000);
    remote_mock::ControlResponse r;
    stub.close(&c, &q, &r, nullptr);
    ASSERT_FALSE(c.Failed());
    EXPECT_NE(EC_SUCCESS, r.error_code());
    EXPECT_EQ(EC_SUCCESS, read(source).error_code());
}
TEST_F(RemoteMockServiceTest, MalformedInitializingRankRollsBack) {
    remote_mock::RemoteMockControlService_Stub stub(channel.get());
    remote_mock::HealthRequest                 h;
    remote_mock::ControlResponse               health;
    arpc::ANetRPCController                    hc;
    hc.SetExpireTime(1000);
    stub.health(&hc, &h, &health, nullptr);
    ASSERT_FALSE(hc.Failed());
    remote_mock::InitRankRequest q;
    q.mutable_session()->set_server_epoch(health.server_epoch());
    q.mutable_session()->set_session_id(9);
    q.mutable_session()->set_generation(1);
    q.set_rank(0);
    q.set_expected_ranks(2);
    q.set_layout_hash(11);
    auto g = q.add_groups();
    g->set_group(0);
    g->set_pool(0);
    g->set_payload_bytes(16);
    g->set_storage_bytes(16);
    g->set_stride_bytes(16);
    g->set_dtype("fp16");
    g->set_tokens_per_block(1);
    g->set_cp_world_size(1);
    g->set_cp_rank(0);
    g->add_member_offsets(0);
    g->set_source_count(2);
    auto first = control(q);
    ASSERT_EQ(EC_SUCCESS, first.error_code());
    ASSERT_FALSE(first.ready());
    q.set_rank(1);
    q.mutable_groups(0)->clear_payload_bytes();
    auto rejected = control(q);
    EXPECT_NE(EC_SUCCESS, rejected.error_code());
    EXPECT_EQ(0, rejected.allocated_bytes());
    initialize(0, 0, 2);
    EXPECT_EQ(EC_SUCCESS, read(source).error_code());
}
TEST_F(RemoteMockServiceTest, ReverseTimeoutAndLateReplySettlesOnce) {
    initialize();
    auto port    = startPeer();
    peer.hold    = true;
    auto pending = std::async(std::launch::async, [&] { return write(1, port, 20); });
    ASSERT_TRUE(peer.waitPull());
    ASSERT_EQ(std::future_status::ready, pending.wait_for(std::chrono::seconds(1)));
    EXPECT_NE(EC_SUCCESS, pending.get().error_code());
    peer.release();
    peer.hold = false;
    EXPECT_EQ(EC_SUCCESS, write(2, port).error_code());
    EXPECT_EQ(1, service->stats().failed_operations);
    EXPECT_EQ(16, service->stats().write_payload_bytes);
}
TEST_F(RemoteMockServiceTest, ShutdownWaitsForAlreadyStartedReversePull) {
    initialize();
    auto port    = startPeer();
    peer.hold    = true;
    auto pending = std::async(std::launch::async, [&] { return write(1, port); });
    ASSERT_TRUE(peer.waitPull());
    auto shutdown = std::async(std::launch::async, [&] { service->shutdown(); });
    EXPECT_EQ(std::future_status::timeout, shutdown.wait_for(std::chrono::milliseconds(10)));
    peer.release();
    ASSERT_EQ(std::future_status::ready, pending.wait_for(std::chrono::seconds(1)));
    EXPECT_NE(EC_SUCCESS, pending.get().error_code());
    ASSERT_EQ(std::future_status::ready, shutdown.wait_for(std::chrono::seconds(1)));
    shutdown.get();
    EXPECT_EQ(1, service->stats().failed_operations);
}
TEST_F(RemoteMockServiceTest, HealthReportsEffectiveLimitsAndPayloadOnce) {
    initialize();
    auto port = startPeer();
    ASSERT_EQ(EC_SUCCESS, read(source).error_code());
    ASSERT_EQ(EC_SUCCESS, write(1, port).error_code());
    remote_mock::RemoteMockControlService_Stub stub(channel.get());
    remote_mock::HealthRequest                 q;
    remote_mock::ControlResponse               r;
    arpc::ANetRPCController                    c;
    c.SetExpireTime(1000);
    stub.health(&c, &q, &r, nullptr);
    ASSERT_FALSE(c.Failed());
    EXPECT_EQ(16, r.read_payload_bytes());
    EXPECT_EQ(16, r.write_payload_bytes());
    EXPECT_EQ(1, r.reverse_rpcs());
    EXPECT_EQ(0, r.failed_operations());
    EXPECT_EQ(1, r.reads_per_connection());
    EXPECT_EQ(2, r.writes_per_connection());
    EXPECT_EQ(128, r.max_callback_endpoints());
    EXPECT_EQ(60000, r.max_timeout_ms());
}
class RemoteMockMetadataTest: public RemoteMockServiceTest {
protected:
    void configureOptions(RemoteMockServiceOptions& options) override {
        options.limits.max_rpc_bytes       = 32768;
        options.limits.max_blocks_per_rpc  = 32;
        options.limits.read_staging_bytes  = 32768;
        options.limits.write_staging_bytes = 32768;  // Fits a singleton, but not 32 long-key records.
    }
    CacheTransferRequest longKeyTransfer(uint32_t callback_port) {
        CacheTransferRequest q;
        q.set_timeout_ms(1000);
        q.set_client_ip("127.0.0.1");
        q.set_client_port(callback_port);
        auto t = q.mutable_transfer_info();
        t->set_partition_count(1);
        t->set_partition_id(0);
        for (int i = 0; i < 32; ++i) {
            auto local = std::to_string(i);
            local.resize(384, 'x');
            t->add_local_keys(local);
            RemoteMockKey key;
            key.write        = true;
            key.session      = session;
            key.layout_hash  = 11;
            key.operation    = 1;
            key.record_index = i;
            key.length       = 1;
            key.checksum     = remoteMockCrc32c(reinterpret_cast<const uint8_t*>(peer.payload.data()), 1);
            t->add_remote_keys(encodeRemoteMockKey(key));
            auto b = q.add_blocks();
            b->set_key(local);
            b->set_len(1);
            b->set_addr(1234);
        }
        return q;
    }
    CacheTransferResponse transfer(const CacheTransferRequest& q) {
        KvCacheStoreService_Stub stub(channel.get());
        arpc::ANetRPCController  c;
        c.SetExpireTime(2000);
        CacheTransferResponse r;
        stub.transfer(&c, &q, &r, nullptr);
        EXPECT_FALSE(c.Failed()) << c.ErrorText();
        return r;
    }
};
TEST_F(RemoteMockMetadataTest, TinyPayloadLongKeysReserveMetadataBeforePull) {
    initialize();
    auto port = startPeer();
    auto q    = longKeyTransfer(port);
    ASSERT_LT(q.ByteSizeLong(), 32768 + 32 * 512 + 4096);
    EXPECT_NE(EC_SUCCESS, transfer(q).error_code());
    EXPECT_EQ(0, peer.calls);
    EXPECT_EQ(0, service->stats().reverse_rpcs);
    // A metadata NACK must consume neither the slot nor the replay sequence.
    // Serialize the same operation identity as a singleton which fits the quota.
    while (q.blocks_size() > 1) {
        q.mutable_blocks()->RemoveLast();
        q.mutable_transfer_info()->mutable_local_keys()->RemoveLast();
        q.mutable_transfer_info()->mutable_remote_keys()->RemoveLast();
    }
    EXPECT_EQ(EC_SUCCESS, transfer(q).error_code());
    EXPECT_EQ(1, peer.calls);
    EXPECT_EQ(1, service->stats().write_payload_bytes);
}
class RemoteMockFundedMetadataTest: public RemoteMockMetadataTest {
protected:
    void configureOptions(RemoteMockServiceOptions& options) override {
        RemoteMockMetadataTest::configureOptions(options);
        options.limits.write_staging_bytes = 524288;
        options.limits.total_bytes         = 2 << 20;
    }
};
TEST_F(RemoteMockFundedMetadataTest, FundedLongKeyTransferStillWorks) {
    initialize();
    auto port = startPeer();
    EXPECT_EQ(EC_SUCCESS, transfer(longKeyTransfer(port)).error_code());
    EXPECT_EQ(1, peer.calls);
    EXPECT_EQ(32, service->stats().write_payload_bytes);
}
TEST_F(RemoteMockFundedMetadataTest, TinyOperationRejectsUnknownMetadataWithinGlobalEnvelope) {
    initialize();
    auto port = startPeer();
    auto q    = longKeyTransfer(port);
    while (q.blocks_size() > 1) {
        q.mutable_blocks()->RemoveLast();
        q.mutable_transfer_info()->mutable_local_keys()->RemoveLast();
        q.mutable_transfer_info()->mutable_remote_keys()->RemoveLast();
    }
    q.GetReflection()->MutableUnknownFields(&q)->AddLengthDelimited(99, std::string(25000, 'x'));
    ASSERT_LT(q.ByteSizeLong(), 32768 + 32 * 512 + 4096);
    EXPECT_NE(EC_SUCCESS, transfer(q).error_code());
    EXPECT_EQ(0, peer.calls);
}
TEST_F(RemoteMockFundedMetadataTest, TinyOperationRejectsNestedUnknownMetadataBeforeCopy) {
    initialize();
    auto port = startPeer();
    auto q    = longKeyTransfer(port);
    while (q.blocks_size() > 1) {
        q.mutable_blocks()->RemoveLast();
        q.mutable_transfer_info()->mutable_local_keys()->RemoveLast();
        q.mutable_transfer_info()->mutable_remote_keys()->RemoveLast();
    }
    auto b = q.mutable_blocks(0);
    b->GetReflection()->MutableUnknownFields(b)->AddLengthDelimited(99, std::string(25000, 'x'));
    EXPECT_NE(EC_SUCCESS, transfer(q).error_code());
    EXPECT_EQ(0, peer.calls);
}
TEST_F(RemoteMockFundedMetadataTest, TinyReadRejectsUnknownMetadataWithinGlobalEnvelope) {
    initialize();
    KvCacheStoreService_Stub stub(channel.get());
    arpc::ANetRPCController  c;
    c.SetExpireTime(2000);
    CacheLoadRequest q;
    q.set_timeout_ms(1000);
    auto b = q.add_blocks();
    b->set_key(source);
    b->set_len(16);
    q.GetReflection()->MutableUnknownFields(&q)->AddLengthDelimited(99, std::string(25000, 'x'));
    CacheLoadResponse r;
    stub.load(&c, &q, &r, nullptr);
    ASSERT_FALSE(c.Failed());
    EXPECT_NE(EC_SUCCESS, r.error_code());
    EXPECT_EQ(0, r.blocks_size());
}
TEST_F(RemoteMockFundedMetadataTest, TinyReadChargesActualKnownRequestMetadata) {
    initialize();
    KvCacheStoreService_Stub stub(channel.get());
    arpc::ANetRPCController  c;
    c.SetExpireTime(2000);
    CacheLoadRequest q;
    q.set_timeout_ms(1000);
    q.set_client_ip(std::string(25000, 'x'));
    auto b = q.add_blocks();
    b->set_key(source);
    b->set_len(16);
    ASSERT_LT(q.ByteSizeLong(), 32768 + 32 * 512 + 4096);
    CacheLoadResponse r;
    stub.load(&c, &q, &r, nullptr);
    ASSERT_FALSE(c.Failed());
    EXPECT_NE(EC_SUCCESS, r.error_code());
    EXPECT_EQ(0, r.blocks_size());
}
TEST(RemoteMockServiceLimitsTest, BothDirectionsRequireServiceMetadataProgressFloor) {
    for (bool read : {false, true}) {
        RemoteMockServiceOptions options;
        (read ? options.limits.read_staging_bytes : options.limits.write_staging_bytes) = 20544;
        RemoteMockService service(options);
        EXPECT_FALSE(service.init());
    }
}

class RemoteCaptureOverflowTest: public RemoteMockServiceTest {
    void configureOptions(RemoteMockServiceOptions& options) override {
        options.max_event_bytes = 256;
    }
};
TEST_F(RemoteCaptureOverflowTest, OverflowWritesBoundedIncompleteFooter) {
    initialize();
    for (int i = 0; i < 3; ++i)
        ASSERT_EQ(read(source).error_code(), EC_SUCCESS);
    service->shutdown();
    std::ifstream input(event_path);
    std::string   text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    EXPECT_LE(text.size(), 256u);
    EXPECT_NE(text.find("\"capture_complete\":false"), std::string::npos);
    EXPECT_EQ(text.find("\"capture_complete\":true"), std::string::npos);
}
TEST_F(RemoteMockServiceTest, DelayedHandoffFailureRepliesOnceAndReturnsAdmission) {
    initialize(30);
    CacheLoadRequest request;
    request.set_timeout_ms(1000);
    request.set_requestid("delay");
    auto block = request.add_blocks();
    block->set_key(source);
    block->set_len(16);
    CacheLoadResponse response;
    struct Reply: google::protobuf::Closure {
        std::atomic<unsigned> calls{0};
        std::promise<void>    done;
        void                  Run() override {
            if (++calls == 1)
                done.set_value();
        }
    } reply;
    auto ready            = reply.done.get_future();
    fail_timer_allocation = true;
    service->load(nullptr, &request, &response, &reply);
    if (ready.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
        ADD_FAILURE() << "no delayed terminal";
        std::_Exit(1);
    }
    EXPECT_FALSE(fail_timer_allocation.exchange(false));
    EXPECT_NE(response.error_code(), EC_SUCCESS);
    EXPECT_EQ(response.blocks_size(), 0);
    EXPECT_EQ(service->stats().failed_operations, 1u);
    EXPECT_EQ(read(source).error_code(), EC_SUCCESS);
    service->shutdown();
    EXPECT_EQ(reply.calls.load(), 1u);
}
TEST_F(RemoteMockServiceTest, ShutdownWaitsForAdmittedSourceReplyBeforeFooter) {
    initialize();
    BlockReadRequest request;
    auto             block = request.add_blocks();
    block->set_key(source);
    block->set_len(16);
    BlockReadResponse response;
    struct Reply: google::protobuf::Closure {
        std::promise<void> entered, release;
        void               Run() override {
            entered.set_value();
            release.get_future().wait();
        }
    } reply;
    auto           entered = reply.entered.get_future();
    SourceCopyGate gate;
    auto           copying   = gate.copying.get_future();
    auto           read_done = std::async(std::launch::async, [&] {
        source_gate = &gate;
        service->blockRead(nullptr, &request, &response, &reply);
        source_gate = nullptr;
    });
    if (copying.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        gate.release.set_value();
        reply.release.set_value();
        FAIL() << "source copy gate missing";
    }
    EXPECT_EQ(service->stats().read_payload_bytes, 0u);
    auto stopped = std::async(std::launch::async, [&] { service->shutdown(); });
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(150)), std::future_status::timeout);
    {
        std::ifstream file(event_path);
        std::string   contents((std::istreambuf_iterator<char>(file)), {});
        EXPECT_EQ(contents.find("capture_complete"), std::string::npos);
    }
    gate.release.set_value();
    if (entered.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        reply.release.set_value();
        FAIL() << "source reply missing";
    }
    EXPECT_EQ(service->stats().read_payload_bytes, 16u);
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    reply.release.set_value();
    read_done.get();
    stopped.get();
    ASSERT_EQ(response.error_code(), EC_SUCCESS);
    ASSERT_EQ(response.blocks_size(), 1);
    EXPECT_EQ(response.blocks(0).content().size(), 16u);
    EXPECT_EQ(service->stats().read_payload_bytes, 16u);
    std::ifstream file(event_path);
    std::string   contents((std::istreambuf_iterator<char>(file)), {});
    EXPECT_NE(contents.find("\"capture_complete\":true"), std::string::npos);
}
TEST_F(RemoteMockServiceTest, HealthyCaptureCertifiedOnlyAfterProducerDrain) {
    initialize();
    ASSERT_EQ(read(source).error_code(), EC_SUCCESS);
    {
        std::ifstream input(event_path);
        std::string   text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        EXPECT_EQ(text.find("capture_complete"), std::string::npos);
    }
    service->shutdown();
    std::ifstream input(event_path);
    std::string   text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("\"capture_complete\":true"), std::string::npos);
    EXPECT_NE(text.find("\"event_count\":5"), std::string::npos);
    EXPECT_NE(text.find("\"prefix_bytes\":"), std::string::npos);
}
class RemoteCaptureWriteFailureTest: public RemoteMockServiceTest {
    void configureOptions(RemoteMockServiceOptions& options) override {
        options.event_path = "/dev/full";
    }
};
TEST_F(RemoteCaptureWriteFailureTest, StreamFailureNeverProducesSuccessfulCertificate) {
    initialize();
    ASSERT_EQ(read(source).error_code(), EC_SUCCESS);
    service->shutdown();
    // /dev/full reports real ENOSPC on write. A failed sink cannot produce a
    // complete footer; existing diagnostics must expose the sticky failure too.
    EXPECT_TRUE(service->stats().event_capture_failed);
}
}  // namespace
}  // namespace rtp_llm
