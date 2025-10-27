#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"
#include "kvcm_client/common.h"

namespace kv_cache_manager {

class MockMetaClient: public MetaClient {
public:
    ~MockMetaClient() override = default;
    MOCK_METHOD((std::pair<ClientErrorCode, LocationsMap>),
                MatchLocation,
                (const std::string&              trace_id,
                 QueryType                       query_type,
                 const std::vector<int64_t>&     keys,
                 const std::vector<int64_t>&     tokens,
                 const BlockMask&                block_mask,
                 int32_t                         sw_size,
                 const std::vector<std::string>& location_spec_name),
                (override));

    MOCK_METHOD((std::pair<ClientErrorCode, WriteLocation>),
                StartWrite,
                (const std::string&              trace_id,
                 const std::vector<int64_t>&     keys,
                 const std::vector<int64_t>&     tokens,
                 const std::vector<std::string>& location_spec_names,
                 int64_t                         write_timeout_seconds),
                (override));

    MOCK_METHOD(ClientErrorCode,
                FinishWrite,
                (const std::string&  trace_id,
                 const std::string&  write_session_id,
                 const BlockMask&    success_block,
                 const LocationsMap& locations_map),
                (override));

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

    MOCK_METHOD(const std::string&, GetStorageConfig, (), (override, const));

    MOCK_METHOD(ClientErrorCode, Init, (const std::string& config, const InitParams& init_params), (override));

    MOCK_METHOD(void, Shutdown, (), (override));
};

class MockTransferClient: public TransferClient {
public:
    MockTransferClient()           = default;
    ~MockTransferClient() override = default;
    MOCK_METHOD(ClientErrorCode,
                LoadKvCaches,
                (const Locations& locations, const BlockBuffers& block_buffers),
                (override));

    MOCK_METHOD((std::pair<ClientErrorCode, Locations>),
                SaveKvCaches,
                (const Locations& locations, const BlockBuffers& block_buffers),
                (override));

private:
    // TODO : not test now
    MOCK_METHOD(ClientErrorCode, Init, (const std::string& client_config, const InitParams& init_params), (override));
};

}  // namespace kv_cache_manager