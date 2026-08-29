#pragma once

#include <gmock/gmock.h>

#include <memory>

#include "rtp_llm/cpp/cache/connector/remote_connector/ClientFactory.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/Subscriber.h"

namespace kv_cache_manager {

class MockMetaClient: public MetaClient {
public:
    MOCK_METHOD((std::pair<ClientErrorCode, Locations>),
                MatchLocation,
                (const std::string&,
                 QueryType,
                 const std::vector<int64_t>&,
                 const std::vector<int64_t>&,
                 const BlockMask&,
                 int32_t,
                 const std::vector<std::string>&),
                (override));
    MOCK_METHOD((std::pair<ClientErrorCode, WriteLocation>),
                StartWrite,
                (const std::string&,
                 const std::vector<int64_t>&,
                 const std::vector<int64_t>&,
                 const std::vector<std::string>&,
                 int64_t),
                (override));
    MOCK_METHOD(ClientErrorCode,
                FinishWrite,
                (const std::string&, const std::string&, const BlockMask&, const Locations&),
                (override));
    MOCK_METHOD(
        (std::pair<ClientErrorCode, Metas>),
        MatchMeta,
        (const std::string&, const std::vector<int64_t>&, const std::vector<int64_t>&, const BlockMask&, int32_t),
        (override));
    MOCK_METHOD((std::pair<ClientErrorCode, int64_t>),
                MatchLocationLen,
                (const std::string&, QueryType, const std::vector<int64_t>&, const std::vector<int64_t>&, int32_t),
                (override));
    MOCK_METHOD(ClientErrorCode,
                RemoveCache,
                (const std::string&, const std::vector<int64_t>&, const std::vector<int64_t>&, const BlockMask&),
                (override));
    MOCK_METHOD(const std::string&, GetStorageConfig, (), (const, override));

private:
    MOCK_METHOD(ClientErrorCode, Init, (const std::string&, const InitParams&), (override));
    MOCK_METHOD(void, Shutdown, (), (override));
};

class MockTransferClient: public TransferClient {
public:
    explicit MockTransferClient(std::shared_ptr<int> destruction_count):
        destruction_count_(std::move(destruction_count)) {}
    ~MockTransferClient() override {
        ++*destruction_count_;
    }

    MOCK_METHOD(ClientErrorCode,
                LoadKvCaches,
                (const UriStrVec&, const BlockBuffers&, std::shared_ptr<TransferTraceInfo>),
                (override));
    MOCK_METHOD((std::pair<ClientErrorCode, UriStrVec>),
                SaveKvCaches,
                (const UriStrVec&, const BlockBuffers&, std::shared_ptr<TransferTraceInfo>),
                (override));

private:
    MOCK_METHOD(ClientErrorCode, Init, (const std::string&, const InitParams&), (override));
    std::shared_ptr<int> destruction_count_;
};

}  // namespace kv_cache_manager

namespace rtp_llm {
namespace remote_connector {

class MockSubscriber: public Subscriber {
public:
    MOCK_METHOD(bool, init, (const std::vector<std::string>&), (override));
    MOCK_METHOD(bool, getAddresses, (std::vector<std::string>&), (const, override));
};

class MockClientFactory: public ClientFactory {
public:
    MOCK_METHOD(std::unique_ptr<kv_cache_manager::MetaClient>,
                createMetaClient,
                (const std::string&, const kv_cache_manager::InitParams&),
                (const, override));
    MOCK_METHOD(std::unique_ptr<kv_cache_manager::TransferClient>,
                createTransferClient,
                (const std::string&, const kv_cache_manager::InitParams&),
                (const, override));
    MOCK_METHOD(std::unique_ptr<Subscriber>, createSubscriber, (bool), (const, override));
};

}  // namespace remote_connector
}  // namespace rtp_llm
