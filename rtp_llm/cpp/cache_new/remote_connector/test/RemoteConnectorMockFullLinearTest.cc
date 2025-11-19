#include "MockKVCMClient.h"
#include "rtp_llm/cpp/cache_new/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/cache_new/HybridLayerKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "autil/EnvUtil.h"

using namespace kv_cache_manager;
using namespace ::testing;
using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {
namespace test {

class FakeRpcService final: public RpcService::Service {
public:
    ::grpc::Status BroadcastTp(::grpc::ServerContext*        context,
                               const ::BroadcastTpRequestPB* request,
                               ::BroadcastTpResponsePB*      response) override {
        if (hack_) {
            return ::grpc::Status(::grpc::StatusCode::INTERNAL, "");
        }
        return remote_connector_->broadcastTp(context, request, response);
    }

    void set_remote_connector(std::shared_ptr<RemoteConnector> remote_connector) {
        remote_connector_ = remote_connector;
    }

    void hack_grpc_status(bool hack) {
        hack_ = hack;
    }

private:
    std::shared_ptr<RemoteConnector> remote_connector_;
    bool                             hack_ = false;
};

class FakeRpcServer {
public:
    FakeRpcServer(std::unique_ptr<FakeRpcService> service): service_(std::move(service)) {}
    ~FakeRpcServer() {
        shutdown();
    }

    void start() {
        ASSERT_NE(nullptr, service_);
        std::string         bind_addr = "0.0.0.0:0";
        grpc::ServerBuilder builder;
        builder.AddListeningPort(bind_addr, grpc::InsecureServerCredentials(), &listen_port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        ASSERT_NE(nullptr, server_);
        ASSERT_NE(0, listen_port_);
    }

    int listenPort() const {
        return listen_port_;
    }

    void set_remote_connector(const std::shared_ptr<RemoteConnector>& remote_connector) {
        service_->set_remote_connector(remote_connector);
    }

    void hack_grpc_status(bool hack) {
        service_->hack_grpc_status(hack);
    }

private:
    void shutdown() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
            server_.reset();
        }
    }

private:
    std::unique_ptr<FakeRpcService> service_;
    std::unique_ptr<grpc::Server>   server_;
    int                             listen_port_{0};
};
namespace {

constexpr static int    kFakeLayerNum = 4;
constexpr static size_t kFakeIovSize  = 1024;

std::string genLocationSpecName(int tp_rank, const std::string& group_name) {
    static std::string location_spec_name("tp");
    return location_spec_name + std::to_string(tp_rank) + "_" + group_name;
}

};  // namespace

// TODO : remove this, use ture HybridLayerKVCacheAllocator
class FakeHybridLayerKVCacheAllocator: public HybridLayerKVCacheAllocator {
public:
    FakeHybridLayerKVCacheAllocator(const CacheConfig&          config,
                                    rtp_llm::DeviceBase*        device,
                                    const std::vector<int32_t>& full_group_ids,
                                    const std::vector<int32_t>& other_group_ids):
        HybridLayerKVCacheAllocator(config, device) {
        for (int32_t full_group_id : full_group_ids) {
            for (int i = 0; i < kFakeLayerNum; i++) {
                fake_layout_.layer_to_groups.push_back(full_group_id);
            }
        }
        for (int32_t other_group_id : other_group_ids) {
            for (int i = 0; i < kFakeLayerNum; i++) {
                fake_layout_.layer_to_groups.push_back(other_group_id);
            }
        }
    }
    CacheLayerLayout layerCacheBase() const override {
        return fake_layout_;
    }

    BlockBufferInfo convertIndexToBuffer(int layer_id, int block_id) const override {
        return {BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                     DataType::TYPE_INT8,
                                     {1, fake_buffer_.size()},
                                     static_cast<const void*>(fake_buffer_.data()),
                                     nullptr)),
                BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                     DataType::TYPE_INT8,
                                     {1, fake_buffer_.size()},
                                     static_cast<const void*>(fake_buffer_.data()),
                                     nullptr))};
    }

private:
    CacheLayerLayout                        fake_layout_;
    constexpr static size_t                 fake_buffer_size_ = kFakeIovSize / sizeof(int8_t);
    inline static const std::vector<int8_t> fake_buffer_      = std::vector<int8_t>(fake_buffer_size_, 0);
};

class RemoteConnectorMockFullLinearTest: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autil::EnvUtil::setEnv("RECO_SERVER_ADDRESS", fake_address_);
        ClientWrapper::client_factory_  = std::make_unique<MockClientFactory>();
        mock_client_factory_            = dynamic_cast<MockClientFactory*>(ClientWrapper::client_factory_.get());
        auto transfer_client            = std::make_unique<kv_cache_manager::MockTransferClient>();
        transfer_client_                = transfer_client.get();
        ClientWrapper::transfer_client_ = std::move(transfer_client);
    }

    void SetUp() override {
        rtp_llm::initLogger();
        initDevice();
        initServer();
        initConnector();
    }

    void TearDown() override {}

private:
    constexpr static int    tp_size_                    = 1;
    constexpr static size_t device_reserve_memory_size_ = 1024L * 1024 * 1024;  // 1GB;
    constexpr static size_t host_reserve_memory_size_   = 1024L * 1024 * 1024;  // 1GB;

    void initDevice() {
        gpt_init_params_.device_resource_config.device_reserve_memory_bytes = device_reserve_memory_size_;
        gpt_init_params_.device_resource_config.host_reserve_memory_bytes   = host_reserve_memory_size_;
        rtp_llm::DeviceFactory::initDevices(gpt_init_params_);
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();
    }
    void initServer() {
        for (int i = 0; i < tp_size_; i++) {
            auto service = std::make_unique<FakeRpcService>();
            auto server  = std::make_unique<FakeRpcServer>(std::move(service));
            server->start();
            server_addrs_.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
            servers_.push_back(std::move(server));
        }
    }
    void initConnector() {
        int block_num          = 10;
        int seq_size_per_block = 8;
        initHybridLayerCacheConfig(kFakeLayerNum, block_num, seq_size_per_block);
        for (int i = 0; i < tp_size_; i++) {
            auto meta_client = std::make_unique<kv_cache_manager::MockMetaClient>();
            meta_clients_.push_back(meta_client.get());
            EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _)).WillOnce(Return(ByMove(std::move(meta_client))));
            auto allocator = std::make_shared<FakeHybridLayerKVCacheAllocator>(
                cache_config_, device_, full_group_ids_, linear_group_ids_);
            ASSERT_TRUE(allocator->init());
            remote_connectors_.push_back(
                std::make_shared<RemoteConnector>(cache_config_,
                                                  gpt_init_params_,
                                                  device_,
                                                  nullptr,
                                                  0,
                                                  allocator,
                                                  server_addrs_,
                                                  RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                                                  full_group_ids_,
                                                  linear_group_ids_,
                                                  1));
            ASSERT_TRUE(remote_connectors_[i]->init());
            servers_[i]->set_remote_connector(remote_connectors_[i]);
        }
    }

    void initHybridLayerCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
        cache_config_.layer_type_num     = 1;
        cache_config_.layer_num          = layer_num;
        cache_config_.block_num          = block_num;
        cache_config_.seq_size_per_block = seq_size_per_block;

        auto mha_spec                = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num          = layer_num;
        mha_spec->block_nums         = block_num;
        mha_spec->local_head_num_kv  = 8;
        mha_spec->size_per_head      = 128;
        mha_spec->seq_size_per_block = seq_size_per_block;
        mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
        mha_spec->type               = KVCacheType::MultiHeadAttention;

        cache_config_.layer_type_params.push_back(mha_spec);

        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        cache_config_.layer_ids.push_back(layer_ids);
    }

    UriStrVec genUris(const CacheKeysType&       cache_keys,
                      const std::vector<size_t>& linear_pos_vec,
                      const std::string&         uri_prefix = "") {
        UriStrVec res;
        size_t    pos_idx = 0;
        for (size_t i = 0; i < cache_keys.size(); i++) {
            for (auto group_id : full_group_ids_) {
                std::string full_group_name = "F" + std::to_string(group_id);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + full_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[i]);
                    res.push_back(uri);
                }
            }
            if (!linear_pos_vec.empty()) {
                if (i == linear_pos_vec[pos_idx]) {
                    for (auto group_id : linear_group_ids_) {
                        std::string linear_group_name = "L" + std::to_string(group_id);
                        for (int r = 0; r < tp_size_; r++) {
                            std::string uri = uri_prefix + "uri_" + linear_group_name + "_" + std::to_string(r) + "_"
                                              + std::to_string(cache_keys[i]);
                            res.push_back(uri);
                        }
                    }
                    pos_idx++;
                }
            }
        }
        return res;
    }

    kv_cache_manager::Locations genFullLinearLocations(const CacheKeysType&       cache_keys,
                                                       const std::vector<size_t>& linear_pos_vec,
                                                       const std::string&         uri_prefix = "") const {
        kv_cache_manager::Locations locations;
        locations.resize(cache_keys.size(), {});
        for (size_t i = 0; i < cache_keys.size(); i++) {
            for (auto group_id : full_group_ids_) {
                std::string full_group_name = "F" + std::to_string(group_id);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + full_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[i]);
                    locations[i].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, full_group_name), uri}));
                }
            }
        }
        for (auto pos : linear_pos_vec) {
            for (auto group_id : linear_group_ids_) {
                std::string linear_group_name = "L" + std::to_string(group_id);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + linear_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[pos]);
                    locations[pos].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, linear_group_name), uri}));
                }
            }
        }
        return locations;
    }

    CacheConfig                                         cache_config_;
    GptInitParameter                                    gpt_init_params_;
    DeviceBase*                                         device_ = nullptr;
    std::vector<std::shared_ptr<RemoteConnector>>       remote_connectors_;
    std::vector<std::unique_ptr<FakeRpcServer>>         servers_;
    std::vector<std::string>                            server_addrs_;
    inline static MockClientFactory*                    mock_client_factory_ = nullptr;
    std::vector<kv_cache_manager::MockMetaClient*>      meta_clients_;
    inline static kv_cache_manager::MockTransferClient* transfer_client_  = nullptr;
    inline static const std::vector<int32_t>            full_group_ids_   = {0};
    inline static const std::vector<int32_t>            linear_group_ids_ = {1, 2};

    constexpr static const char* fake_address_ = "fake_address";
    using MatchLocationReturnType              = std::pair<ClientErrorCode, Locations>;
    using StartWriteReturnType                 = std::pair<ClientErrorCode, WriteLocation>;
    using SaveKvCachesReturnType               = std::pair<ClientErrorCode, UriStrVec>;
};

struct BlockBuffersExpect {
    size_t block_buffers_size = 0;
    size_t iov_vec_size       = 0;
    size_t iov_size           = 0;
};

MATCHER_P(BlockBuffersMatcher, block_buffers_expect, "") {
    if (block_buffers_expect.block_buffers_size != arg.size()) {
        *result_listener << "BlockBuffers size mismatch: expected " << block_buffers_expect.block_buffers_size
                         << ", actual " << arg.size();
        return false;
    }
    for (size_t i = 0; i < arg.size(); ++i) {
        const auto& block_buffer = arg[i];
        if (block_buffers_expect.iov_vec_size != block_buffer.iovs.size()) {
            *result_listener << "At block buffer [" << i << "]: iovs size mismatch: expected "
                             << block_buffers_expect.iov_vec_size << ", actual " << block_buffer.iovs.size();
            return false;
        }
        for (size_t j = 0; j < block_buffer.iovs.size(); ++j) {
            const auto& iov = block_buffer.iovs[j];
            if (iov.size != block_buffers_expect.iov_size) {
                *result_listener << "At block buffer [" << i << "], iov [" << j << "]: iov.size mismatch: expected "
                                 << block_buffers_expect.iov_size << ", actual " << iov.size;
                return false;
            }
        }
    }
    return true;
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_match_all) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullLinearLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    UriStrVec          expected_uris        = genUris({1, 2, 3}, {2});
    BlockBuffersExpect block_buffers_expect = {5, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(3, kv_cache_resouce->reuse_len);
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_block_mask) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    kv_cache_resouce->reuse_len  = 1;
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullLinearLocations({2, 3}, {0, 1});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_2"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    UriStrVec          expected_uris        = genUris({2, 3}, {1});
    BlockBuffersExpect block_buffers_expect = {4, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(3, kv_cache_resouce->reuse_len);
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_part_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3, 4}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13, 14}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23, 24}}));
    kv_cache_resouce->reuse_len  = 1;
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullLinearLocations({2, 3, 4}, {1});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_2"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3, 4}),     // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    UriStrVec          expected_uris        = genUris({2, 3}, {1});
    BlockBuffersExpect block_buffers_expect = {4, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(3, kv_cache_resouce->reuse_len);
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_all_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3, 4}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13, 14}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23, 24}}));
    kv_cache_resouce->reuse_len  = 1;
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullLinearLocations({2, 3, 4}, {});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_2"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3, 4}),     // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(1, kv_cache_resouce->reuse_len);
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_actual_locations_different) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullLinearLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({1, 2, 3}, {0, 1, 2});
    UriStrVec actual_uris   = genUris({1, 2, 3}, {0, 1, 2}, "actual_");

    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullLinearLocations({1, 2, 3}, {0, 1, 2}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest,
       test_write_success_broadcast_success_actual_locations_different_with_block_mask) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullLinearLocations({2, 3}, {0, 1});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 3}, {0, 1});
    UriStrVec actual_uris   = genUris({2, 3}, {0, 1}, "actual_");

    BlockBuffersExpect block_buffers_expect = {6, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullLinearLocations({2, 3}, {0, 1}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(2))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_with_part_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3, 4}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, -1, 13, 14}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, -1, 23, 24}}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullLinearLocations({2, 3, 4}, {1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_1"),                                           // trace_id
                   std::vector<int64_t>({1, 2, 3, 4}),                                  // keys
                   _,                                                                   // tokens
                   Eq(std::vector<std::string>({"F0L1L2", "F0", "F0L1L2", "F0L1L2"})),  // location_spec_group_names
                   _                                                                    // write_timeout_seconds
                   ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 3, 4}, {1, 2});
    UriStrVec actual_uris   = genUris({2, 3, 4}, {1, 2}, "actual_");

    BlockBuffersExpect block_buffers_expect = {7, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullLinearLocations({2, 3, 4}, {1, 2}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

// In fact, this situation should not occur
TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_with_all_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3, 4}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{-1, -1, -1, -1}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{-1, -1, -1, -1}}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullLinearLocations({2, 3, 4}, {});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),                               // trace_id
                           std::vector<int64_t>({1, 2, 3, 4}),                      // keys
                           _,                                                       // tokens
                           Eq(std::vector<std::string>({"F0", "F0", "F0", "F0"})),  // location_spec_group_names
                           _                                                        // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 3, 4}, {});
    UriStrVec actual_uris   = genUris({2, 3, 4}, {}, "actual_");

    BlockBuffersExpect block_buffers_expect = {3, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullLinearLocations({2, 3, 4}, {}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_actual_locations_same) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_locations = genFullLinearLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({1, 2, 3}, {0, 1, 2});

    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, expected_uris})));

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto   meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t tp_rank = 0;
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));

    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);
    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_success_load_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullLinearLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    UriStrVec          expected_uris        = genUris({1, 2, 3}, {2});
    BlockBuffersExpect block_buffers_expect = {5, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_SDK_TIMEOUT));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_success_broadcast_grpc_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullLinearLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);

    servers_[tp_rank]->hack_grpc_status(true);
    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullLinearLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_invalid_block_ids) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, -1, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto   meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t tp_rank = 0;
    EXPECT_CALL(*meta_clients_[tp_rank], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_success_finish_write_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_write_locations = genFullLinearLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec          expected_uris        = genUris({1, 2, 3}, {0, 1, 2});
    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, expected_uris})));

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_INVALID_GRPCSTATUS));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_success_save_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_write_locations = genFullLinearLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec          expected_uris        = genUris({1, 2, 3}, {0, 1, 2});
    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum * 2, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_SDK_TIMEOUT, UriStrVec({})})));

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(0))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_grpc_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_write_locations = genFullLinearLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(0))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    servers_[tp_rank]->hack_grpc_status(true);
    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_threadpool_full) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{1, 2, 3}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{11, 12, 13}}));
    kv_cache_resouce->group_block_ids.push_back(std::shared_ptr<BlockIds>(new BlockIds{{21, 22, 23}}));
    auto   meta    = std::make_shared<RemoteConnectorMeta>("", "trace");
    size_t tp_rank = 0;
    remote_connectors_[tp_rank]->thread_pool_->stop();
    remote_connectors_[tp_rank]->thread_pool_->waitFinish();
    remote_connectors_[tp_rank]->thread_pool_.reset();
    remote_connectors_[tp_rank]->thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(1, /* queueSize= */ 0, nullptr, "RECOThreadPool");

    EXPECT_CALL(*meta_clients_[tp_rank], MatchLocation(_, _, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta));

    EXPECT_CALL(*meta_clients_[tp_rank], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta));
}

}  // namespace test
}  // namespace rtp_llm