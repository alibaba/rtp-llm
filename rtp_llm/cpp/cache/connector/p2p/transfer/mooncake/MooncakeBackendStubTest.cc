#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include "autil/NetUtil.h"
#include <string>
#include <thread>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheSender.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {
namespace {

uint32_t nextTestPort() {
    return autil::NetUtil::randomPort();
}

bool waitTaskDone(const IKVCacheRecvTaskPtr& task, int64_t wait_timeout_ms) {
    const auto deadline_ms = currentTimeMs() + wait_timeout_ms;
    while (task && !task->done() && currentTimeMs() < deadline_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return task && task->done();
}

class FakeMooncakeAdapter : public IMooncakeTransferEngineAdapter {
public:
    bool init(const MooncakeBackendConfig& config) override {
        last_location = config.location;
        init_called   = true;
        return true;
    }

    bool registerLocalMemory(const BlockInfo& block_info, uint64_t aligned_size) override {
        last_reg_addr         = block_info.addr;
        last_reg_aligned_size = aligned_size;
        ++register_count;
        return true;
    }

    bool openSegment(const std::string& segment_name) override {
        last_segment_name = segment_name;
        return true;
    }

    uint64_t allocateBatchID(size_t request_count) override {
        last_batch_request_count = request_count;
        return 7;
    }

    void freeBatchID(uint64_t batch_id) override {
        last_freed_batch_id = batch_id;
        ++free_batch_count;
    }

    bool submitTransfer(uint64_t batch_id, const std::vector<MooncakeWriteRequest>& requests) override {
        last_batch_id = batch_id;
        last_submit_requests = requests.size();
        last_submit_segment_name = requests.empty() ? std::string() : requests.front().segment_name;
        return true;
    }

    TransferErrorCode getTransferStatus(uint64_t batch_id, bool* finished, std::string* error_message) override {
        last_status_batch_id = batch_id;
        if (finished) {
            *finished = true;
        }
        if (error_message) {
            *error_message = "";
        }
        return TransferErrorCode::OK;
    }

public:
    bool        init_called              = false;
    int         register_count           = 0;
    void*       last_reg_addr            = nullptr;
    uint64_t    last_reg_aligned_size    = 0;
    size_t      last_batch_request_count = 0;
    uint64_t    last_batch_id            = 0;
    uint64_t    last_freed_batch_id      = 0;
    uint64_t    last_status_batch_id     = 0;
    size_t      last_submit_requests     = 0;
    int         free_batch_count         = 0;
    std::string last_submit_segment_name;
    std::string last_segment_name;
    std::string last_location;
};

class FakeMooncakeControlPlaneClient : public IMooncakeControlPlaneClient {
public:
    bool init(int io_thread_count) override {
        last_io_thread_count = io_thread_count;
        init_called = true;
        return true;
    }

    bool prepare(const std::string& ip,
                 uint32_t port,
                 const std::string& unique_key,
                 int64_t deadline_ms,
                 MooncakeRemoteDescriptor* descriptor,
                 TransferErrorCode* error_code,
                 std::string* error_message) override {
        last_prepare_ip = ip;
        last_prepare_port = port;
        last_prepare_key = unique_key;
        last_prepare_deadline = deadline_ms;
        if (!prepare_success) {
            if (error_code) {
                *error_code = prepare_error_code;
            }
            if (error_message) {
                *error_message = prepare_error_message;
            }
            return false;
        }
        if (descriptor) {
            *descriptor = prepare_descriptor;
        }
        if (error_code) {
            *error_code = TransferErrorCode::OK;
        }
        if (error_message) {
            error_message->clear();
        }
        return true;
    }

    bool finish(const std::string& ip,
                uint32_t port,
                const std::string& unique_key,
                bool success,
                TransferErrorCode error_code,
                const std::string& error_message,
                TransferErrorCode* response_error_code,
                std::string* response_error_message) override {
        finish_called = true;
        last_finish_ip = ip;
        last_finish_port = port;
        last_finish_key = unique_key;
        last_finish_success = success;
        last_finish_error_code = error_code;
        last_finish_error_message = error_message;
        if (response_error_code) {
            *response_error_code = finish_response_code;
        }
        if (response_error_message) {
            *response_error_message = finish_response_message;
        }
        return finish_success;
    }

public:
    bool                     init_called = false;
    int                      last_io_thread_count = 0;
    bool                     prepare_success = true;
    TransferErrorCode        prepare_error_code = TransferErrorCode::UNKNOWN;
    std::string              prepare_error_message;
    MooncakeRemoteDescriptor prepare_descriptor;
    std::string              last_prepare_ip;
    uint32_t                 last_prepare_port = 0;
    std::string              last_prepare_key;
    int64_t                  last_prepare_deadline = 0;
    bool                     finish_called = false;
    bool                     finish_success = true;
    TransferErrorCode        finish_response_code = TransferErrorCode::OK;
    std::string              finish_response_message;
    std::string              last_finish_ip;
    uint32_t                 last_finish_port = 0;
    std::string              last_finish_key;
    bool                     last_finish_success = false;
    TransferErrorCode        last_finish_error_code = TransferErrorCode::UNKNOWN;
    std::string              last_finish_error_message;
};

TEST(MooncakeKVCacheSenderTest, RegMemDelegatesToAdapter) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    config.mooncake.location         = "tp0";
    config.messager_io_thread_count  = 5;
    ASSERT_TRUE(sender.init(config));
    EXPECT_TRUE(adapter->init_called);
    EXPECT_TRUE(control_plane_client->init_called);
    EXPECT_EQ(control_plane_client->last_io_thread_count, 5);
    EXPECT_EQ(adapter->last_location, "tp0");

    BlockInfo block_info{false, 0, 0, reinterpret_cast<void*>(0x1234), 64};
    EXPECT_TRUE(sender.regMem(block_info, 128));
    EXPECT_EQ(adapter->register_count, 1);
    EXPECT_EQ(adapter->last_reg_addr, reinterpret_cast<void*>(0x1234));
    EXPECT_EQ(adapter->last_reg_aligned_size, 128);
}

TEST(MooncakeKVCacheSenderTest, SendReportsControlPlaneMissing) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    MooncakeKVCacheSender sender(adapter, nullptr);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    SendRequest request;
    request.unique_key = "ukey";

    TransferErrorCode actual_code = TransferErrorCode::OK;
    std::string       actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg  = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::UNKNOWN);
    EXPECT_NE(actual_msg.find("not fully initialized"), std::string::npos);
}

TEST(MooncakeKVCacheSenderTest, SendBuildsWriteRequestsAndFinishesOnSuccess) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    char block0[16]{};
    char block1[32]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block1, sizeof(block1)});

    MooncakeRemoteBlockDescriptor descriptor_block0;
    descriptor_block0.cache_key = 11;
    descriptor_block0.block_index = 0;
    descriptor_block0.target_addr = 101;
    descriptor_block0.len = sizeof(block0);
    MooncakeRemoteBlockDescriptor descriptor_block1;
    descriptor_block1.cache_key = 11;
    descriptor_block1.block_index = 1;
    descriptor_block1.target_addr = 202;
    descriptor_block1.len = sizeof(block1);
    control_plane_client->prepare_descriptor.segment_name = "segment_a";
    control_plane_client->prepare_descriptor.blocks = {descriptor_block0, descriptor_block1};

    SendRequest request;
    request.ip = "127.0.0.1";
    request.port = 22345;
    request.unique_key = "send_ok";
    request.deadline_ms = 999999999;
    request.block_info[11] = key_block_info;

    TransferErrorCode actual_code = TransferErrorCode::UNKNOWN;
    std::string       actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg  = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::OK);
    EXPECT_TRUE(actual_msg.empty());
    EXPECT_EQ(adapter->last_segment_name, "segment_a");
    EXPECT_EQ(adapter->last_submit_segment_name, "segment_a");
    EXPECT_EQ(adapter->last_batch_request_count, 2u);
    EXPECT_EQ(adapter->last_batch_id, 7u);
    EXPECT_EQ(adapter->last_freed_batch_id, 7u);
    EXPECT_EQ(adapter->last_status_batch_id, 7u);
    EXPECT_EQ(adapter->last_submit_requests, 2u);
    EXPECT_EQ(adapter->free_batch_count, 1);
    EXPECT_TRUE(control_plane_client->finish_called);
    EXPECT_TRUE(control_plane_client->last_finish_success);
}

TEST(MooncakeKVCacheSenderTest, SendReturnsBufferMismatchWhenDescriptorLengthDiffers) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    char block0[16]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    MooncakeRemoteBlockDescriptor mismatch_block;
    mismatch_block.cache_key = 12;
    mismatch_block.block_index = 0;
    mismatch_block.target_addr = 303;
    mismatch_block.len = 8;
    control_plane_client->prepare_descriptor.segment_name = "segment_b";
    control_plane_client->prepare_descriptor.blocks = {mismatch_block};

    SendRequest request;
    request.ip = "127.0.0.1";
    request.port = 22346;
    request.unique_key = "send_mismatch";
    request.deadline_ms = 999999999;
    request.block_info[12] = key_block_info;

    TransferErrorCode actual_code = TransferErrorCode::OK;
    std::string       actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg  = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::BUFFER_MISMATCH);
    EXPECT_NE(actual_msg.find("mismatch"), std::string::npos);
    EXPECT_TRUE(control_plane_client->finish_called);
    EXPECT_FALSE(control_plane_client->last_finish_success);
}

TEST(MooncakeKVCacheReceiverTest, RecvStoresAndStealsTask) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    MooncakeKVCacheReceiver receiver(adapter);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(receiver.init(config));

    BlockInfo block_info{false, 0, 0, reinterpret_cast<void*>(0x5678), 256};
    EXPECT_TRUE(receiver.regMem(block_info, 512));
    EXPECT_EQ(adapter->register_count, 1);

    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(block_info);

    RecvRequest request;
    request.unique_key  = "recv_key";
    request.deadline_ms = 123456;
    request.block_info[42] = key_block_info;

    auto task = receiver.recv(request);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(receiver.getTransferTaskStore()->getTaskCount(), 1);
    EXPECT_NE(receiver.getTask("recv_key"), nullptr);

    receiver.stealTask("recv_key");
    EXPECT_EQ(receiver.getTransferTaskStore()->getTaskCount(), 0);
    EXPECT_EQ(receiver.getTask("recv_key"), nullptr);
}

TEST(MooncakeKVCacheReceiverTest, PrepareDescriptorStartsTransferAndFinishClearsDescriptor) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    MooncakeKVCacheReceiver receiver(adapter);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(receiver.init(config));

    char block0[24]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    RecvRequest request;
    request.unique_key = "prepare_key";
    request.deadline_ms = currentTimeMs() + 5000;
    request.block_info[22] = key_block_info;
    auto task = receiver.recv(request);
    ASSERT_NE(task, nullptr);

    MooncakeRemoteDescriptor descriptor;
    TransferErrorCode error_code = TransferErrorCode::UNKNOWN;
    std::string error_message;
    EXPECT_TRUE(receiver.prepareDescriptor(request.unique_key,
                                          request.deadline_ms,
                                          &descriptor,
                                          &error_code,
                                          &error_message));
    EXPECT_EQ(error_code, TransferErrorCode::OK);
    ASSERT_EQ(descriptor.blocks.size(), 1u);
    EXPECT_EQ(descriptor.blocks[0].cache_key, 22);
    EXPECT_EQ(descriptor.blocks[0].block_index, 0u);
    EXPECT_EQ(descriptor.blocks[0].len, sizeof(block0));

    TransferErrorCode finish_response_code = TransferErrorCode::UNKNOWN;
    std::string finish_response_message;
    EXPECT_TRUE(receiver.finishTransfer(request.unique_key,
                                        true,
                                        TransferErrorCode::OK,
                                        "",
                                        &finish_response_code,
                                        &finish_response_message));
    EXPECT_EQ(finish_response_code, TransferErrorCode::OK);
    ASSERT_NE(receiver.getTask(request.unique_key), nullptr);
    EXPECT_EQ(receiver.getTask(request.unique_key)->errorCode(), TransferErrorCode::OK);

    MooncakeRemoteDescriptor descriptor_after_finish;
    error_code = TransferErrorCode::UNKNOWN;
    error_message.clear();
    EXPECT_FALSE(receiver.prepareDescriptor(request.unique_key,
                                           request.deadline_ms,
                                           &descriptor_after_finish,
                                           &error_code,
                                           &error_message));
    EXPECT_EQ(error_code, TransferErrorCode::OK);
}

TEST(MooncakeKVCacheIntegrationTest, SenderAndReceiverRoundTripThroughRealControlPlane) {
    const auto control_plane_port = nextTestPort();
    const auto local_server_name = std::string("127.0.0.1:") + std::to_string(control_plane_port);

    TransferBackendConfig receiver_config;
    receiver_config.cache_store_mooncake_mode = true;
    receiver_config.mooncake.control_plane_port = control_plane_port;
    receiver_config.cache_store_listen_port = control_plane_port;
    receiver_config.messager_io_thread_count = 1;
    receiver_config.messager_worker_thread_count = 4;
    receiver_config.cache_store_tcp_anet_rpc_thread_num = 1;
    receiver_config.cache_store_tcp_anet_rpc_queue_num = 8;
    receiver_config.mooncake.classic.local_server_name = local_server_name;
    receiver_config.mooncake.classic.ip_or_host_name = "127.0.0.1";
    receiver_config.mooncake.classic.rpc_port = static_cast<uint16_t>(control_plane_port);

    auto receiver_adapter = std::make_shared<FakeMooncakeAdapter>();
    MooncakeKVCacheReceiver receiver(receiver_adapter);
    ASSERT_TRUE(receiver.init(receiver_config));

    char target_block[32]{};
    auto recv_block_info = std::make_shared<KeyBlockInfo>();
    recv_block_info->blocks.push_back(BlockInfo{false, 0, 0, target_block, sizeof(target_block)});

    RecvRequest recv_request;
    recv_request.unique_key = "round_trip_key";
    recv_request.deadline_ms = currentTimeMs() + 5000;
    recv_request.block_info[77] = recv_block_info;
    auto recv_task = receiver.recv(recv_request);
    ASSERT_NE(recv_task, nullptr);

    TransferBackendConfig sender_config = receiver_config;
    auto sender_adapter = std::make_shared<FakeMooncakeAdapter>();
    auto control_plane_client = createMooncakeControlPlaneClient();
    MooncakeKVCacheSender sender(sender_adapter, control_plane_client);
    ASSERT_TRUE(sender.init(sender_config));

    char source_block[32]{};
    auto send_block_info = std::make_shared<KeyBlockInfo>();
    send_block_info->blocks.push_back(BlockInfo{false, 0, 0, source_block, sizeof(source_block)});

    SendRequest send_request;
    send_request.ip = "127.0.0.1";
    send_request.port = 0;
    send_request.unique_key = recv_request.unique_key;
    send_request.deadline_ms = recv_request.deadline_ms;
    send_request.block_info[77] = send_block_info;

    TransferErrorCode actual_code = TransferErrorCode::UNKNOWN;
    std::string       actual_msg;
    sender.send(send_request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg  = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::OK);
    EXPECT_TRUE(actual_msg.empty());
    EXPECT_EQ(sender_adapter->last_segment_name, local_server_name);
    EXPECT_EQ(sender_adapter->last_submit_segment_name, local_server_name);
    EXPECT_TRUE(waitTaskDone(recv_task, 200));
    EXPECT_TRUE(recv_task->success());

    MooncakeRemoteDescriptor descriptor_after_finish;
    TransferErrorCode after_finish_code = TransferErrorCode::UNKNOWN;
    std::string after_finish_msg;
    EXPECT_FALSE(receiver.prepareDescriptor(recv_request.unique_key,
                                           recv_request.deadline_ms,
                                           &descriptor_after_finish,
                                           &after_finish_code,
                                           &after_finish_msg));
    EXPECT_EQ(after_finish_code, TransferErrorCode::OK);
}

}  // namespace
}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
