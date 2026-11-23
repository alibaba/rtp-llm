#pragma once

#include <gmock/gmock.h>

#include <memory>
#include <type_traits>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/Subscriber.h"

namespace kv_cache_manager {

// The SDKs currently shipped for open and internal builds have different
// MetaClient ABIs. Derive the overrides from the selected SDK, not the GPU.
template<class Client, class Signature = decltype(&Client::StartWrite)>
class MockWriteClient;

template<class Client, class Result, class... Args>
class MockWriteClient<Client, Result (Client::*)(Args...)>: public Client {
public:
    Result StartWrite(Args... args) override {
        if constexpr (sizeof...(Args) == 5) {
            return OnStartWrite(args..., 0);  // Legacy SDK has no minimum replica count.
        } else {
            return OnStartWrite(args...);
        }
    }
    MOCK_METHOD((std::pair<ClientErrorCode, WriteLocation>),
                OnStartWrite,
                (const std::string&, const std::vector<int64_t>&, const std::vector<int64_t>&,
                 const std::vector<std::string>&, int64_t, int32_t));
};

template<class Client, class BackendQuery, class HostQuery>
class MockQueryClient;

template<class Client, class B, class... BA, class H, class... HA>
class MockQueryClient<Client, B (Client::*)(BA...), H (Client::*)(HA...)>: public MockWriteClient<Client> {
public:
    ::testing::MockFunction<B(BA...)> backend_locations;
    ::testing::MockFunction<H(HA...)> host_state;

    B GetCacheLocationsByBackend(BA... args) override { return backend_locations.Call(args...); }
    H GetHostCacheState(HA... args) override { return host_state.Call(args...); }
};

template<class Client, class = void>
struct MetaMockBase: MockWriteClient<Client> {};

template<class Client>
struct MetaMockBase<Client, std::void_t<decltype(&Client::GetCacheLocationsByBackend), decltype(&Client::GetHostCacheState)>>:
    MockQueryClient<Client, decltype(&Client::GetCacheLocationsByBackend), decltype(&Client::GetHostCacheState)> {};

class MockMetaClient: public MetaMockBase<MetaClient> {
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
namespace kvcm {

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

}  // namespace kvcm
}  // namespace rtp_llm
