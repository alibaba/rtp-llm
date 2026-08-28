#include "rtp_llm/cpp/distribute/RpcCpuTpBroadcaster.h"

#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

CpuTpBroadcastRequestPB makeRequest(uint64_t seq, int dst_tp_rank, const std::string& payload) {
    CpuTpBroadcastRequestPB request;
    request.set_group_key("tp_cpu_broadcast:dp=0:tp=2:world=2");
    request.set_seq(seq);
    request.set_root(0);
    request.set_src_tp_rank(0);
    request.set_dst_tp_rank(dst_tp_rank);
    request.set_nbytes(payload.size());
    request.set_payload(payload);
    return request;
}

TEST(RpcCpuTpBroadcasterTest, NonRootWaitsUntilServerThreadPublishesPayload) {
    auto& bcast = RpcCpuTpBroadcaster::instance();
    bcast.reset();
    bcast.initialize(/*tp_rank=*/1,
                     /*tp_size=*/2,
                     /*dp_rank=*/0,
                     /*world_size=*/2,
                     /*worker_grpc_addrs=*/{"unused-root", "unused-rank1"},
                     /*timeout_ms=*/1000);

    std::vector<char> recv(5, 0);
    std::thread       waiter([&] { bcast.broadcast(recv.data(), recv.size(), /*root=*/0); });

    CpuTpBroadcastResponsePB response;
    ASSERT_TRUE(bcast.handleBroadcastRequest(makeRequest(/*seq=*/0, /*dst_tp_rank=*/1, "hello"), &response));
    EXPECT_TRUE(response.success());

    waiter.join();
    EXPECT_EQ(std::string(recv.begin(), recv.end()), "hello");
    bcast.reset();
}

TEST(RpcCpuTpBroadcasterTest, RejectsWrongDestinationRank) {
    auto& bcast = RpcCpuTpBroadcaster::instance();
    bcast.reset();
    bcast.initialize(/*tp_rank=*/1,
                     /*tp_size=*/2,
                     /*dp_rank=*/0,
                     /*world_size=*/2,
                     /*worker_grpc_addrs=*/{"unused-root", "unused-rank1"},
                     /*timeout_ms=*/1000);

    CpuTpBroadcastResponsePB response;
    EXPECT_FALSE(bcast.handleBroadcastRequest(makeRequest(/*seq=*/0, /*dst_tp_rank=*/0, "bad"), &response));
    EXPECT_FALSE(response.success());
    bcast.reset();
}

}  // namespace
}  // namespace rtp_llm
