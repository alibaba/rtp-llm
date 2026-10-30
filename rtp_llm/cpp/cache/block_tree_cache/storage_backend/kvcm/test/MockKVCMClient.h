#pragma once

#include <gmock/gmock.h>

#include <memory>
#include <tuple>
#include <type_traits>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/Subscriber.h"

namespace kv_cache_manager {

template<typename>
struct MetaClientMethodTypes;

template<typename ResultType, typename Client, typename... Args>
struct MetaClientMethodTypes<ResultType (Client::*)(Args...)> {
    using Result = ResultType;
    template<size_t Index>
    using Arg = std::tuple_element_t<Index, std::tuple<Args...>>;
};

// Internal service discovery and SDK API revisions are independent. Detect
// optional methods from the selected SDK instead of using KVCM_INTERNAL.
template<typename Client, typename = void>
class MockMetaClientQueries: public Client {};

template<typename Client>
class MockMetaClientQueries<
    Client,
    std::void_t<decltype(&Client::GetCacheLocationsByBackend), decltype(&Client::GetHostCacheState)>>: public Client {
    using Backend = MetaClientMethodTypes<decltype(&Client::GetCacheLocationsByBackend)>;
    using Host    = MetaClientMethodTypes<decltype(&Client::GetHostCacheState)>;

public:
    MOCK_METHOD(typename Backend::Result,
                GetCacheLocationsByBackend,
                (typename Backend::template Arg<0>,
                 typename Backend::template Arg<1>,
                 typename Backend::template Arg<2>,
                 typename Backend::template Arg<3>,
                 typename Backend::template Arg<4>,
                 typename Backend::template Arg<5>,
                 typename Backend::template Arg<6>),
                (override));
    MOCK_METHOD(typename Host::Result,
                GetHostCacheState,
                (typename Host::template Arg<0>,
                 typename Host::template Arg<1>,
                 typename Host::template Arg<2>,
                 typename Host::template Arg<3>,
                 typename Host::template Arg<4>,
                 typename Host::template Arg<5>),
                (override));
};

template<typename Client, typename Signature>
class MockMetaClientWrite;

template<typename Client, typename Result, typename Owner, typename... Args>
class MockMetaClientWrite<Client, Result (Owner::*)(Args...)>: public Client {
public:
    MOCK_METHOD((std::pair<ClientErrorCode, WriteLocation>),
                StartWriteRequest,
                (const std::string&,
                 const std::vector<int64_t>&,
                 const std::vector<int64_t>&,
                 const std::vector<std::string>&,
                 int64_t),
                ());
    Result StartWrite(Args... args) override {
        static_assert(sizeof...(Args) == 5 || sizeof...(Args) == 6, "unsupported StartWrite signature");
        auto arguments = std::forward_as_tuple(args...);
        if constexpr (sizeof...(Args) == 6) {
            EXPECT_EQ(std::get<5>(arguments), 0);
        }
        return StartWriteRequest(std::get<0>(arguments),
                                 std::get<1>(arguments),
                                 std::get<2>(arguments),
                                 std::get<3>(arguments),
                                 std::get<4>(arguments));
    }
};

class MockMetaClient: public MockMetaClientWrite<MockMetaClientQueries<MetaClient>, decltype(&MetaClient::StartWrite)> {
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
