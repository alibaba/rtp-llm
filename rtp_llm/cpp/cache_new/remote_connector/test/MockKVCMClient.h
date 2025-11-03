#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache_new/remote_connector/ClientFactory.h"
#include "rtp_llm/cpp/cache_new/remote_connector/DirectSubscriber.h"

namespace kv_cache_manager {

class MockMetaClient: public MetaClient {
public:
    ~MockMetaClient() override = default;
    MOCK_METHOD((std::pair<ClientErrorCode, Locations>),
                MatchLocation,
                (const std::string&              trace_id,
                 QueryType                       query_type,
                 const std::vector<int64_t>&     keys,
                 const std::vector<int64_t>&     tokens,
                 const BlockMask&                block_mask,
                 int32_t                         sw_size,
                 const std::vector<std::string>& location_spec_names),
                (override));

    MOCK_METHOD((std::pair<ClientErrorCode, WriteLocation>),
                StartWrite,
                (const std::string&              trace_id,
                 const std::vector<int64_t>&     keys,
                 const std::vector<int64_t>&     tokens,
                 const std::vector<std::string>& location_spec_group_names,
                 int64_t                         write_timeout_seconds),
                (override));

    MOCK_METHOD(ClientErrorCode,
                FinishWrite,
                (const std::string& trace_id,
                 const std::string& write_session_id,
                 const BlockMask&   success_block,
                 const Locations&   locations),
                (override));

    MOCK_METHOD(const std::string&, GetStorageConfig, (), (override, const));

private:
    // TODO : not test now
    MOCK_METHOD((std::pair<ClientErrorCode, Metas>),
                MatchMeta,
                (const std::string&          trace_id,
                 const std::vector<int64_t>& keys,
                 const std::vector<int64_t>& tokens,
                 const BlockMask&            block_mask,
                 int32_t                     detail_level),
                (override));

    MOCK_METHOD(ClientErrorCode,
                RemoveCache,
                (const std::string&          trace_id,
                 const std::vector<int64_t>& keys,
                 const std::vector<int64_t>& tokens,
                 const BlockMask&            block_mask),
                (override));

    MOCK_METHOD(ClientErrorCode, Init, (const std::string& config, const InitParams& init_params), (override));

    MOCK_METHOD(void, Shutdown, (), (override));
};

class MockTransferClient: public TransferClient {
public:
    MockTransferClient()           = default;
    ~MockTransferClient() override = default;
    MOCK_METHOD(ClientErrorCode, LoadKvCaches, (const UriStrVec& uri_str_vec, const BlockBuffers& block_buffers));

    MOCK_METHOD((std::pair<ClientErrorCode, UriStrVec>),
                SaveKvCaches,
                (const UriStrVec& uri_str_vec, const BlockBuffers& block_buffers),
                (override));

private:
    // TODO : not test now
    MOCK_METHOD(ClientErrorCode, Init, (const std::string& client_config, const InitParams& init_params), (override));
};

}  // namespace kv_cache_manager

namespace rtp_llm {
namespace remote_connector {

class MockSubscriber: public Subscriber {
public:
    MOCK_METHOD(bool, init, (const std::vector<std::string>& domains), (override));
    MOCK_METHOD(bool, getAddresses, (std::vector<std::string> & addresses), (override, const));
};

class MockClientFactory: public ClientFactory {
public:
    MOCK_METHOD(std::unique_ptr<kv_cache_manager::MetaClient>,
                CreateMetaClient,
                (const std::string& config, const kv_cache_manager::InitParams& init_params),
                (override, const));

    MOCK_METHOD(std::unique_ptr<kv_cache_manager::TransferClient>,
                CreateTransferClient,
                (const std::string& config, const kv_cache_manager::InitParams& init_params),
                (override, const));
};

}  // namespace remote_connector
}  // namespace rtp_llm
