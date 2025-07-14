#include <gtest/gtest.h>
#include <torch/torch.h>
#include <future>

#define private public
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;
using namespace rtp_llm;

class DistributedTest: public DeviceTestBase {
public:
    void initTestDevices() override {}

    void runTestThread(const size_t rank, const size_t world_size, const size_t port) {
        auto device_name = getenv("TEST_USING_DEVICE");
        ASSERT_TRUE(device_name);
        auto             device_type = getDeviceType(device_name);
        auto             device_reg  = DeviceFactory::getRegistrationMap().at(device_type);
        DeviceInitParams params;
        params.device_id      = rank;
        params.tp_rank        = rank;
        params.tp_size        = world_size;
        params.master_ip      = "127.0.0.1";
        params.tp_master_port = port;
        auto device           = device_reg.create(params);

#define copy_tensor_to_buffer(t, buf)                                                                                  \
    {                                                                                                                  \
        auto buf_host = torchTensor2Buffer(t);                                                                         \
        device->copy({*buf, *buf_host});                                                                               \
    }

        // test broadcast
        const auto t1                         = torch::arange(10, torch::kInt64);
        const auto t2                         = torch::arange(0, -1, -0.01, torch::kFloat16);
        const auto t3                         = torch::arange(0, -1, -0.01, torch::kFloat32) * ((int32_t)rank + 1);
        const auto t4                         = torch::zeros({4 * (int64_t)world_size, 128}, torch::kUInt8);
        t4.slice(0, rank * 4, (rank + 1) * 4) = (int32_t)rank + 1;
        auto buf1                             = device->allocateBuffer({DataType::TYPE_INT64, {10}});
        auto buf2                             = device->allocateBuffer({DataType::TYPE_FP16, {100}});
        auto buf3                             = device->allocateBuffer({DataType::TYPE_FP32, {100}});
        auto buf4                             = device->allocateBuffer({DataType::TYPE_UINT8, {4 * world_size, 128}});
        if (rank == 0) {
            copy_tensor_to_buffer(t1, buf1);
            copy_tensor_to_buffer(t2, buf2);
        }
        copy_tensor_to_buffer(t3, buf3);
        copy_tensor_to_buffer(t4, buf4);
        device->broadcast({{buf1, buf2}, 0});
        device->allReduce({buf3, ReduceOp::Sum});
        device->allGather({{buf4}});
        device->syncAndCheck();
        auto out1 = bufferToTensor(*buf1, device);
        auto out2 = bufferToTensor(*buf2, device);
        auto out3 = bufferToTensor(*buf3, device);
        auto out4 = bufferToTensor(*buf4, device);
        device->syncAndCheck();
        assertTensorClose(t1, out1, 1e-8, 1e-8);
        assertTensorClose(t2, out2, 1e-8, 1e-8);
        auto expected3 = torch::arange(0, -1, -0.01, torch::kFloat32)
                         * (((int32_t)world_size * ((int32_t)world_size - 1) / 2) + (int32_t)world_size);
        assertTensorClose(expected3, out3, 1e-6, 1e-6);
        for (size_t i = 0; i < world_size; i++) {
            auto expected4 = torch::zeros({4, 128}, torch::kUInt8);
            expected4.fill_((int32_t)i + 1);
            auto slice = out4.slice(0, i * 4, (i + 1) * 4);
            assertTensorClose(expected4, slice, 1e-8, 1e-8);
        }
    }

    void testForWorldSize(const size_t world_size) {
        std::vector<future<void>> futures;
        const auto                port = getFreePort();
        RTP_LLM_LOG_INFO("found free port %d\n", port);
        for (size_t i = 0; i < world_size; i++) {
            auto future =
                async(launch::async,
                      bind(&DistributedTest::runTestThread, this, placeholders::_1, placeholders::_2, placeholders::_3),
                      i,
                      world_size,
                      port);
            futures.push_back(move(future));
        }

        for (auto& future : futures) {
            future.get();
        }
    }
};

TEST_F(DistributedTest, testDeviceCommunication) {
    if (getenv("SKIP_DISTRIBUTED_TEST")) {
        RTP_LLM_LOG_INFO("DistributedTest skipped\n");
        return;
    }

    testForWorldSize(4);
    testForWorldSize(2);
    testForWorldSize(1);
}
