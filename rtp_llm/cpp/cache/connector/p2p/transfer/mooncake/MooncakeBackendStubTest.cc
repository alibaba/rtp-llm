#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <memory>
#include "autil/NetUtil.h"
#include <string>
#include <thread>
#include <utility>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngineAdapterProvider.h"
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
        local_server_name = config.classic.local_server_name.empty()
                                ? std::string("127.0.0.1:") + std::to_string(config.classic.rpc_port)
                                : config.classic.local_server_name;
        init_called   = true;
        return init_success;
    }

    bool registerLocalMemory(const BlockInfo& block_info, uint64_t aligned_size) override {
        last_reg_addr         = block_info.addr;
        last_reg_aligned_size = aligned_size;
        ++register_count;
        return register_success;
    }

    bool openSegment(const std::string& segment_name) override {
        last_segment_name = segment_name;
        ++open_segment_count;
        return open_segment_success;
    }

    std::string getLocalServerName() override {
        return local_server_name;
    }

    uint64_t allocateBatchID(size_t request_count) override {
        last_batch_request_count = request_count;
        ++allocate_batch_count;
        return allocated_batch_id;
    }

    void freeBatchID(uint64_t batch_id) override {
        last_freed_batch_id = batch_id;
        ++free_batch_count;
    }

    bool submitTransfer(uint64_t batch_id, const std::vector<MooncakeWriteRequest>& requests) override {
        last_batch_id = batch_id;
        last_submit_requests = requests.size();
        last_submit_segment_name = requests.empty() ? std::string() : requests.front().segment_name;
        return submit_success;
    }

    TransferErrorCode getTransferStatus(uint64_t batch_id, bool* finished, std::string* error_message) override {
        last_status_batch_id = batch_id;
        const size_t status_index = status_call_count++;
        TransferErrorCode code = status_code;
        bool local_finished = status_finished;
        std::string local_error_message = status_error_message;
        if (status_index < status_sequence.size()) {
            code = status_sequence[status_index].first;
            local_finished = status_sequence[status_index].second;
        }
        if (status_index < status_error_messages.size()) {
            local_error_message = status_error_messages[status_index];
        }
        if (finished) {
            *finished = local_finished;
        }
        if (error_message) {
            *error_message = local_error_message;
        }
        return code;
    }

public:
    bool        init_called              = false;
    bool        init_success             = true;
    bool        register_success         = true;
    bool        open_segment_success     = true;
    bool        submit_success           = true;
    int         register_count           = 0;
    int         open_segment_count       = 0;
    int         allocate_batch_count     = 0;
    void*       last_reg_addr            = nullptr;
    uint64_t    last_reg_aligned_size    = 0;
    size_t      last_batch_request_count = 0;
    uint64_t    allocated_batch_id       = 7;
    uint64_t    last_batch_id            = 0;
    uint64_t    last_freed_batch_id      = 0;
    uint64_t    last_status_batch_id     = 0;
    size_t      last_submit_requests     = 0;
    int         free_batch_count         = 0;
    size_t      status_call_count        = 0;
    TransferErrorCode status_code        = TransferErrorCode::OK;
    bool        status_finished          = true;
    std::string status_error_message;
    std::vector<std::pair<TransferErrorCode, bool>> status_sequence;
    std::vector<std::string>                        status_error_messages;
    std::string last_submit_segment_name;
    std::string last_segment_name;
    std::string last_location;
    std::string local_server_name;
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

TEST(MooncakeKVCacheSenderTest, SendPropagatesPrepareTimeoutWithoutCallingFinish) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    control_plane_client->prepare_success = false;
    control_plane_client->prepare_error_code = TransferErrorCode::TIMEOUT;
    control_plane_client->prepare_error_message = "prepare timeout";
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    char block0[16]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    SendRequest request;
    request.ip = "127.0.0.1";
    request.port = 22347;
    request.unique_key = "send_prepare_timeout";
    request.deadline_ms = currentTimeMs() + 100;
    request.block_info[13] = key_block_info;

    TransferErrorCode actual_code = TransferErrorCode::OK;
    std::string actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::TIMEOUT);
    EXPECT_EQ(actual_msg, "prepare timeout");
    EXPECT_FALSE(control_plane_client->finish_called);
    EXPECT_EQ(adapter->open_segment_count, 0);
}

TEST(MooncakeKVCacheSenderTest, SendCallsFinishWhenOpenSegmentFails) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    adapter->open_segment_success = false;
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    char block0[16]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    MooncakeRemoteBlockDescriptor descriptor_block;
    descriptor_block.cache_key = 14;
    descriptor_block.block_index = 0;
    descriptor_block.target_addr = 404;
    descriptor_block.len = sizeof(block0);
    control_plane_client->prepare_descriptor.segment_name = "segment_open_fail";
    control_plane_client->prepare_descriptor.blocks = {descriptor_block};

    SendRequest request;
    request.ip = "127.0.0.1";
    request.port = 22348;
    request.unique_key = "send_open_fail";
    request.deadline_ms = currentTimeMs() + 1000;
    request.block_info[14] = key_block_info;

    TransferErrorCode actual_code = TransferErrorCode::OK;
    std::string actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::UNKNOWN);
    EXPECT_NE(actual_msg.find("openSegment failed"), std::string::npos);
    EXPECT_TRUE(control_plane_client->finish_called);
    EXPECT_FALSE(control_plane_client->last_finish_success);
    EXPECT_EQ(adapter->allocate_batch_count, 0);
}

TEST(MooncakeKVCacheSenderTest, SendFreesBatchAndReportsSubmitFailure) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    adapter->submit_success = false;
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    char block0[16]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    MooncakeRemoteBlockDescriptor descriptor_block;
    descriptor_block.cache_key = 15;
    descriptor_block.block_index = 0;
    descriptor_block.target_addr = 505;
    descriptor_block.len = sizeof(block0);
    control_plane_client->prepare_descriptor.segment_name = "segment_submit_fail";
    control_plane_client->prepare_descriptor.blocks = {descriptor_block};

    SendRequest request;
    request.ip = "127.0.0.1";
    request.port = 22349;
    request.unique_key = "send_submit_fail";
    request.deadline_ms = currentTimeMs() + 1000;
    request.block_info[15] = key_block_info;

    TransferErrorCode actual_code = TransferErrorCode::OK;
    std::string actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::UNKNOWN);
    EXPECT_NE(actual_msg.find("submitTransfer failed"), std::string::npos);
    EXPECT_TRUE(control_plane_client->finish_called);
    EXPECT_FALSE(control_plane_client->last_finish_success);
    EXPECT_EQ(adapter->free_batch_count, 1);
    EXPECT_EQ(adapter->last_freed_batch_id, adapter->allocated_batch_id);
}

TEST(MooncakeKVCacheSenderTest, SendReportsTimeoutWhenTransferNeverFinishesBeforeDeadline) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    adapter->status_sequence = {
        {TransferErrorCode::OK, false},
        {TransferErrorCode::OK, false},
        {TransferErrorCode::OK, false},
    };
    adapter->status_finished = false;
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    char block0[16]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    MooncakeRemoteBlockDescriptor descriptor_block;
    descriptor_block.cache_key = 16;
    descriptor_block.block_index = 0;
    descriptor_block.target_addr = 606;
    descriptor_block.len = sizeof(block0);
    control_plane_client->prepare_descriptor.segment_name = "segment_wait_timeout";
    control_plane_client->prepare_descriptor.blocks = {descriptor_block};

    SendRequest request;
    request.ip = "127.0.0.1";
    request.port = 22350;
    request.unique_key = "send_wait_timeout";
    request.deadline_ms = currentTimeMs() + 5;
    request.block_info[16] = key_block_info;

    TransferErrorCode actual_code = TransferErrorCode::OK;
    std::string actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::TIMEOUT);
    EXPECT_NE(actual_msg.find("timed out"), std::string::npos);
    EXPECT_TRUE(control_plane_client->finish_called);
    EXPECT_FALSE(control_plane_client->last_finish_success);
    EXPECT_EQ(control_plane_client->last_finish_error_code, TransferErrorCode::TIMEOUT);
    EXPECT_EQ(adapter->free_batch_count, 1);
}

TEST(MooncakeKVCacheSenderTest, SendOverridesSuccessWhenFinishReportsFailure) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    auto control_plane_client = std::make_shared<FakeMooncakeControlPlaneClient>();
    control_plane_client->finish_success = false;
    control_plane_client->finish_response_code = TransferErrorCode::CANCELLED;
    control_plane_client->finish_response_message = "finish rejected";
    MooncakeKVCacheSender sender(adapter, control_plane_client);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(sender.init(config));

    char block0[16]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    MooncakeRemoteBlockDescriptor descriptor_block;
    descriptor_block.cache_key = 17;
    descriptor_block.block_index = 0;
    descriptor_block.target_addr = 707;
    descriptor_block.len = sizeof(block0);
    control_plane_client->prepare_descriptor.segment_name = "segment_finish_fail";
    control_plane_client->prepare_descriptor.blocks = {descriptor_block};

    SendRequest request;
    request.ip = "127.0.0.1";
    request.port = 22351;
    request.unique_key = "send_finish_fail";
    request.deadline_ms = currentTimeMs() + 1000;
    request.block_info[17] = key_block_info;

    TransferErrorCode actual_code = TransferErrorCode::UNKNOWN;
    std::string actual_msg;
    sender.send(request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::CANCELLED);
    EXPECT_EQ(actual_msg, "finish rejected");
    EXPECT_TRUE(control_plane_client->finish_called);
    EXPECT_TRUE(control_plane_client->last_finish_success);
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

TEST(MooncakeKVCacheReceiverTest, PrepareDescriptorFailsWhenDeadlineExpired) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    MooncakeKVCacheReceiver receiver(adapter);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(receiver.init(config));

    char block0[24]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    RecvRequest request;
    request.unique_key = "expired_key";
    request.deadline_ms = currentTimeMs() + 1000;
    request.block_info[23] = key_block_info;
    ASSERT_NE(receiver.recv(request), nullptr);

    MooncakeRemoteDescriptor descriptor;
    TransferErrorCode error_code = TransferErrorCode::OK;
    std::string error_message;
    EXPECT_FALSE(receiver.prepareDescriptor(request.unique_key,
                                           currentTimeMs() - 1,
                                           &descriptor,
                                           &error_code,
                                           &error_message));
    EXPECT_EQ(error_code, TransferErrorCode::TIMEOUT);
    EXPECT_NE(error_message.find("timed out"), std::string::npos);
}

TEST(MooncakeKVCacheReceiverTest, PrepareDescriptorFailsAfterSteal) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    MooncakeKVCacheReceiver receiver(adapter);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(receiver.init(config));

    char block0[24]{};
    auto key_block_info = std::make_shared<KeyBlockInfo>();
    key_block_info->blocks.push_back(BlockInfo{false, 0, 0, block0, sizeof(block0)});

    RecvRequest request;
    request.unique_key = "stolen_key";
    request.deadline_ms = currentTimeMs() + 1000;
    request.block_info[24] = key_block_info;
    ASSERT_NE(receiver.recv(request), nullptr);
    receiver.stealTask(request.unique_key);

    MooncakeRemoteDescriptor descriptor;
    TransferErrorCode error_code = TransferErrorCode::OK;
    std::string error_message;
    EXPECT_FALSE(receiver.prepareDescriptor(request.unique_key,
                                           request.deadline_ms,
                                           &descriptor,
                                           &error_code,
                                           &error_message));
    EXPECT_EQ(error_code, TransferErrorCode::CANCELLED);
}

TEST(MooncakeKVCacheReceiverTest, FinishTransferReportsCancelledWhenTaskNotFound) {
    auto adapter = std::make_shared<FakeMooncakeAdapter>();
    MooncakeKVCacheReceiver receiver(adapter);

    TransferBackendConfig config;
    config.cache_store_mooncake_mode = true;
    ASSERT_TRUE(receiver.init(config));

    TransferErrorCode response_error_code = TransferErrorCode::OK;
    std::string response_error_message;
    EXPECT_FALSE(receiver.finishTransfer("missing_key",
                                         false,
                                         TransferErrorCode::UNKNOWN,
                                         "missing",
                                         &response_error_code,
                                         &response_error_message));
    EXPECT_EQ(response_error_code, TransferErrorCode::CANCELLED);
    EXPECT_NE(response_error_message.find("task not found"), std::string::npos);
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


TEST(MooncakeKVCacheClassicTeTest, RealClassicTransferEngineCopiesPayloadOverTcpTransport) {
    auto receiver_adapter = createMooncakeTransferEngineAdapter();
    auto sender_adapter = createMooncakeTransferEngineAdapter();
    if (!receiver_adapter || !sender_adapter) {
        GTEST_SKIP() << "Mooncake classic TE is not enabled in this build";
    }

    const auto control_plane_port = nextTestPort();
    const auto receiver_rpc_port = nextTestPort();
    const auto sender_rpc_port = nextTestPort();
    const auto receiver_server_name = std::string("127.0.0.1:") + std::to_string(receiver_rpc_port);
    const auto sender_server_name = std::string("127.0.0.1:") + std::to_string(sender_rpc_port);

    TransferBackendConfig receiver_config;
    receiver_config.cache_store_mooncake_mode = true;
    receiver_config.mooncake.location = "tp0";
    receiver_config.mooncake.control_plane_port = control_plane_port;
    receiver_config.cache_store_listen_port = control_plane_port;
    receiver_config.messager_io_thread_count = 1;
    receiver_config.messager_worker_thread_count = 4;
    receiver_config.cache_store_tcp_anet_rpc_thread_num = 1;
    receiver_config.cache_store_tcp_anet_rpc_queue_num = 8;
    receiver_config.mooncake.classic.local_server_name = receiver_server_name;
    receiver_config.mooncake.classic.ip_or_host_name = "127.0.0.1";
    receiver_config.mooncake.classic.rpc_port = static_cast<uint16_t>(receiver_rpc_port);

    MooncakeKVCacheReceiver receiver(receiver_adapter);
    ASSERT_TRUE(receiver.init(receiver_config));

    char target_block[64]{};
    BlockInfo recv_mem{false, 0, 0, target_block, sizeof(target_block)};
    ASSERT_TRUE(receiver.regMem(recv_mem, sizeof(target_block)));

    auto recv_block_info = std::make_shared<KeyBlockInfo>();
    recv_block_info->blocks.push_back(recv_mem);

    RecvRequest recv_request;
    recv_request.unique_key = "classic_tcp_round_trip";
    recv_request.deadline_ms = currentTimeMs() + 5000;
    recv_request.block_info[88] = recv_block_info;
    auto recv_task = receiver.recv(recv_request);
    ASSERT_NE(recv_task, nullptr);

    TransferBackendConfig sender_config = receiver_config;
    sender_config.mooncake.classic.local_server_name = sender_server_name;
    sender_config.mooncake.classic.rpc_port = static_cast<uint16_t>(sender_rpc_port);

    MooncakeKVCacheSender sender(sender_adapter, createMooncakeControlPlaneClient());
    ASSERT_TRUE(sender.init(sender_config));

    char source_block[64];
    for (size_t i = 0; i < sizeof(source_block); ++i) {
        source_block[i] = static_cast<char>('A' + (i % 23));
    }
    BlockInfo send_mem{false, 0, 0, source_block, sizeof(source_block)};
    ASSERT_TRUE(sender.regMem(send_mem, sizeof(source_block)));

    auto send_block_info = std::make_shared<KeyBlockInfo>();
    send_block_info->blocks.push_back(send_mem);

    SendRequest send_request;
    send_request.ip = "127.0.0.1";
    send_request.port = control_plane_port;
    send_request.unique_key = recv_request.unique_key;
    send_request.deadline_ms = recv_request.deadline_ms;
    send_request.block_info[88] = send_block_info;

    TransferErrorCode actual_code = TransferErrorCode::UNKNOWN;
    std::string actual_msg;
    sender.send(send_request, [&](TransferErrorCode code, const std::string& msg) {
        actual_code = code;
        actual_msg = msg;
    });

    EXPECT_EQ(actual_code, TransferErrorCode::OK) << actual_msg;
    ASSERT_TRUE(waitTaskDone(recv_task, 2000));
    EXPECT_TRUE(recv_task->success());
    EXPECT_EQ(std::memcmp(source_block, target_block, sizeof(source_block)), 0);
}

}  // namespace
}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
