#pragma once

#include "MockKVCMClient.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "autil/EnvUtil.h"

using namespace kv_cache_manager;
using namespace ::testing;
using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {
namespace test {

namespace {
constexpr static int    kFakeLayerNum = 4;
constexpr static size_t kFakeIovSize  = 1024;

}  // namespace

class FakeRpcService final: public RpcService::Service {
public:
    ::grpc::Status ExecuteFunction(::grpc::ServerContext*     context,
                                   const ::FunctionRequestPB* request,
                                   ::FunctionResponsePB*      response) override {
        if (hack_) {
            return ::grpc::Status(::grpc::StatusCode::INTERNAL, "grpc_error");
        }
        if (!remote_connector_->copyCache(request->remote_request(), *(response->mutable_remote_response()))) {
            return ::grpc::Status(::grpc::StatusCode::INTERNAL, "connector error");
        }
        return grpc::Status::OK;
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

class RemoteConnectorMockTestBase: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autil::EnvUtil::setEnv("RECO_SERVER_ADDRESS", fake_address_);
        autil::EnvUtil::setEnv("KVCM_SDK_CHECK", "1");
        ClientWrapper::client_factory_  = std::make_unique<MockClientFactory>();
        mock_client_factory_            = dynamic_cast<MockClientFactory*>(ClientWrapper::client_factory_.get());
        auto transfer_client            = std::make_unique<kv_cache_manager::MockTransferClient>();
        transfer_client_                = transfer_client.get();
        ClientWrapper::transfer_client_ = std::move(transfer_client);
    }

    static void TearDownTestSuite() {
        ClientWrapper::transfer_client_.reset();
        ClientWrapper::client_factory_.reset();
    }

    void SetUp() override {
        rtp_llm::initLogger();
        initDevice();
        initServer();
    }

    void TearDown() override {}

protected:
    constexpr static int    tp_size_                    = 1;
    constexpr static size_t device_reserve_memory_size_ = 1024L * 1024 * 1024;  // 1GB;
    constexpr static size_t host_reserve_memory_size_   = 1024L * 1024 * 1024;  // 1GB;

    std::string genLocationSpecName(int tp_rank, const std::string& group_name) const {
        static std::string location_spec_name("tp");
        return location_spec_name + std::to_string(tp_rank) + "_" + group_name;
    }

    void initDevice() {
        ParallelismConfig    parallelism_config;
        ModelConfig          model_config;
        EPLBConfig           eplb_config;
        FMHAConfig           fmha_config;
        DeviceResourceConfig device_resource_config;
        device_resource_config.device_reserve_memory_bytes = device_reserve_memory_size_;
        device_resource_config.host_reserve_memory_bytes   = host_reserve_memory_size_;
        MoeConfig                   moe_config;
        SpeculativeExecutionConfig  sp_config;
        MiscellaneousConfig         misc_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        HWKernelConfig              hw_kernel_config;
        ConcurrencyConfig           concurrency_config;
        FfnDisAggregateConfig       ffn_disaggregate_config;
        RuntimeConfig               runtime_config;
        ModelSpecificConfig         model_specific_config;
        NcclCommConfig              nccl_comm_config;

        DeviceFactory::initDevices(parallelism_config,
                                   model_config,
                                   eplb_config,
                                   fmha_config,
                                   device_resource_config,
                                   moe_config,
                                   sp_config,
                                   misc_config,
                                   profiling_debug_logging_config,
                                   hw_kernel_config,
                                   concurrency_config,
                                   ffn_disaggregate_config,
                                   runtime_config,
                                   model_specific_config,
                                   nccl_comm_config);
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
        runtime_config_.worker_grpc_addrs = server_addrs_;
    }

    UriStrVec genUris(const CacheKeysType&       cache_keys,
                      const std::vector<size_t>& other_pos_vec = {},
                      const std::string&         uri_prefix    = "") {
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
            if (!other_pos_vec.empty()) {
                if (i == other_pos_vec[pos_idx]) {
                    for (auto group_id : other_group_ids_) {
                        std::string other_group_name = "L" + std::to_string(group_id);
                        for (int r = 0; r < tp_size_; r++) {
                            std::string uri = uri_prefix + "uri_" + other_group_name + "_" + std::to_string(r) + "_"
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

    kv_cache_manager::Locations genFullotherLocations(const CacheKeysType&       cache_keys,
                                                      const std::vector<size_t>& other_pos_vec = {},
                                                      const std::string&         uri_prefix    = "") const {
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
        for (auto pos : other_pos_vec) {
            for (auto group_id : other_group_ids_) {
                std::string other_group_name = "L" + std::to_string(group_id);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + other_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[pos]);
                    locations[pos].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, other_group_name), uri}));
                }
            }
        }
        return locations;
    }

    std::shared_ptr<BlockIds> makeGroupBlockIds(const BlockIndicesType& block_indices) {
        auto result           = std::make_shared<BlockIds>();
        result->block_indices = block_indices;
        return result;
    }

    CacheConfig                                         cache_config_;
    KVCacheConfig                                       kv_cache_config_;
    RuntimeConfig                                       runtime_config_;
    ParallelismConfig                                   parallelism_config_;
    SpeculativeExecutionConfig                          sp_config_;
    DeviceBase*                                         device_ = nullptr;
    std::vector<std::shared_ptr<RemoteConnector>>       remote_connectors_;
    std::vector<std::unique_ptr<FakeRpcServer>>         servers_;
    std::vector<std::string>                            server_addrs_;
    inline static MockClientFactory*                    mock_client_factory_ = nullptr;
    std::vector<kv_cache_manager::MockMetaClient*>      meta_clients_;
    inline static kv_cache_manager::MockTransferClient* transfer_client_ = nullptr;
    inline static const std::vector<int32_t>            full_group_ids_  = {0};
    std::vector<int32_t>                                other_group_ids_ = {};

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

std::string StrVecToString(const std::vector<std::string>& vec) {
    std::string result;
    for (const auto& value : vec) {
        result += value + ",";
    }
    if (!result.empty()) {
        result.pop_back();
    }
    return result;
}

MATCHER_P(TransferTraceInfoMatcher, /*std::vector<std::string>*/ expect_block_ids, "") {
    const auto& real_block_ids = arg->block_ids;
    if (expect_block_ids != real_block_ids) {
        *result_listener << "block_ids size mismatch: expected " << StrVecToString(expect_block_ids) << ", actual "
                         << StrVecToString(real_block_ids);
        return false;
    }
    return true;
}

}  // namespace test
}  // namespace rtp_llm