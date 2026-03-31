#include <gtest/gtest.h>
#include <torch/torch.h>
#include <future>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include "rtp_llm/cpp/core/DistributedComm.h"
#include "rtp_llm/cpp/testing/TestBase.h"

using namespace std;
using namespace rtp_llm;

static c10::intrusive_ptr<c10d::ProcessGroup> createNcclPG(
    c10::intrusive_ptr<c10d::Store> store, int rank, int world_size) {
    auto nccl_backend = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, world_size);
    auto pg = c10::make_intrusive<c10d::ProcessGroup>(store, rank, world_size);
    pg->setBackend(c10::DeviceType::CUDA, c10d::ProcessGroup::BackendType::NCCL, nccl_backend);
    return pg;
}

class DistributedTest: public DeviceTestBase {
public:
    void initTestDevices() override {}

    void runDirectPGTest(const size_t rank, const size_t world_size, const size_t port) {
        at::cuda::CUDAGuard guard(static_cast<int>(rank));

        c10d::TCPStoreOptions opts;
        opts.port       = static_cast<uint16_t>(port);
        opts.isServer   = (rank == 0);
        opts.numWorkers = world_size;
        auto store = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", opts);
        auto pg    = createNcclPG(store, static_cast<int>(rank), static_cast<int>(world_size));

        auto buf1_t = torch::empty({10}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        auto buf2_t = torch::empty({100}, torch::dtype(torch::kHalf).device(torch::kCUDA));

        const auto t1 = torch::arange(10, torch::kInt64);
        const auto t2 = torch::arange(0, -1, -0.01, torch::kFloat16);
        if (rank == 0) {
            buf1_t.copy_(t1);
            buf2_t.copy_(t2);
        }

        {
            std::vector<at::Tensor> tensors = {buf1_t};
            c10d::BroadcastOptions  bcast_opts;
            bcast_opts.rootRank = 0;
            pg->broadcast(tensors, bcast_opts)->wait();
        }
        {
            std::vector<at::Tensor> tensors = {buf2_t};
            c10d::BroadcastOptions  bcast_opts;
            bcast_opts.rootRank = 0;
            pg->broadcast(tensors, bcast_opts)->wait();
        }

        assertTensorClose(t1, buf1_t.cpu(), 1e-8, 1e-8);
        assertTensorClose(t2, buf2_t.cpu(), 1e-8, 1e-8);

        auto buf3_t = torch::arange(0, -1, -0.01, torch::kFloat32).to(torch::kCUDA) * ((int32_t)rank + 1);
        {
            std::vector<at::Tensor> tensors = {buf3_t};
            c10d::AllreduceOptions  ar_opts;
            ar_opts.reduceOp = c10d::ReduceOp::SUM;
            pg->allreduce(tensors, ar_opts)->wait();
        }
        auto expected3 = torch::arange(0, -1, -0.01, torch::kFloat32)
                         * (((int32_t)world_size * ((int32_t)world_size - 1) / 2) + (int32_t)world_size);
        assertTensorClose(expected3, buf3_t.cpu(), 1e-6, 1e-6);

        auto buf4_t = torch::zeros({4 * (int64_t)world_size, 128}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        buf4_t.slice(0, rank * 4, (rank + 1) * 4) = (int32_t)rank + 1;
        auto send_t = buf4_t.narrow(0, rank * 4, 4).contiguous();
        pg->_allgather_base(buf4_t, send_t)->wait();

        auto out4 = buf4_t.cpu();
        for (size_t i = 0; i < world_size; i++) {
            auto expected4 = torch::zeros({4, 128}, torch::kUInt8);
            expected4.fill_((int32_t)i + 1);
            assertTensorClose(expected4, out4.slice(0, i * 4, (i + 1) * 4), 1e-8, 1e-8);
        }
    }

    void testDirectPGForWorldSize(const size_t world_size) {
        std::vector<future<void>> futures;
        const auto                port = getFreePort();
        RTP_LLM_LOG_INFO("Direct PG test: world_size=%zu, port=%zu\n", world_size, port);
        for (size_t i = 0; i < world_size; i++) {
            futures.push_back(async(launch::async, [this, i, world_size, port]() {
                runDirectPGTest(i, world_size, port);
            }));
        }
        for (auto& f : futures) {
            f.get();
        }
    }
};

TEST_F(DistributedTest, testRegistrationAPI) {
    if (getenv("SKIP_DISTRIBUTED_TEST")) {
        RTP_LLM_LOG_INFO("DistributedTest skipped\n");
        return;
    }

    c10d::TCPStoreOptions opts;
    opts.port       = static_cast<uint16_t>(getFreePort());
    opts.isServer   = true;
    opts.numWorkers = 1;
    auto store = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", opts);
    auto pg    = createNcclPG(store, 0, 1);

    EXPECT_FALSE(hasProcessGroup(ParallelMode::TP));
    registerProcessGroup(ParallelMode::TP, pg);
    EXPECT_TRUE(hasProcessGroup(ParallelMode::TP));
    EXPECT_EQ(getProcessGroup(ParallelMode::TP)->getRank(), 0);
    EXPECT_EQ(getProcessGroup(ParallelMode::TP)->getSize(), 1);
    clearProcessGroups();
    EXPECT_FALSE(hasProcessGroup(ParallelMode::TP));
}

TEST_F(DistributedTest, testDeviceCommunication) {
    if (getenv("SKIP_DISTRIBUTED_TEST")) {
        RTP_LLM_LOG_INFO("DistributedTest skipped\n");
        return;
    }

    const int gpu_count = static_cast<int>(torch::cuda::device_count());
    RTP_LLM_LOG_INFO("Available GPUs: %d\n", gpu_count);

    if (gpu_count >= 4) {
        testDirectPGForWorldSize(4);
    }
    if (gpu_count >= 2) {
        testDirectPGForWorldSize(2);
    }
}
