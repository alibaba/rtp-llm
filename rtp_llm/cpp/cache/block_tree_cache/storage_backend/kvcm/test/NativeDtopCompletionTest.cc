#include <cuda_runtime_api.h>
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <future>
#include <map>
#include <tuple>
#include <typeinfo>
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientFactory.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/PackedSsdTransferClient.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

namespace rtp_llm {
namespace {
TEST(ClientFactoryFeatureBoundaryTest, DisabledAndUnsetDelegateConfigurationErrorsToTheSdk) {
    using namespace kv_cache_manager;
    // Invalid JSON cannot initiate network I/O. Compare against the SDK itself,
    // whose legacy error/exception contract the disabled feature must preserve.
    // Our SSD parser used to throw before the SDK was called, even when off.
    const std::string         config = "{";
    InitParams                params{};
    ClientMemoryRegistrations registrations;
    kvcm::ClientFactory       factory;
    const auto                observe = [](auto call) -> std::pair<bool, std::string> {
        try {
            return {static_cast<bool>(call()), {}};
        } catch (const std::exception& e) {
            return {false, std::string(typeid(e).name()) + ":" + e.what()};
        } catch (...) {
            return {false, "non-standard exception"};
        }
    };
    for (bool unset : {false, true}) {
        autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        if (unset) {
            ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"), 0);
        }
        EXPECT_EQ(observe([&] { return factory.createTransferClient(config, params); }),
                  observe([&] { return TransferClient::Create(config, params); }))
            << unset;
        EXPECT_EQ(observe([&] { return factory.createTransferClientWithMemory(config, params, registrations); }),
                  observe([&] { return TransferClient::Create(config, params, registrations); }))
            << unset;
    }
}
TEST(ClientFactoryFeatureBoundaryTest, EnabledSsdStillRejectsMissingGpuRegistrationAndMixedBackends) {
    autil::EnvGuard                             enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    kvcm::ClientFactory                         factory;
    kv_cache_manager::InitParams                params{};
    kv_cache_manager::ClientMemoryRegistrations registrations;
    const std::string                           ssd = R"({"sdk_config":{"sdk_backend_configs":[{"type":"pace_ssd"}]}})";
    const std::string mixed = R"({"sdk_config":{"sdk_backend_configs":[{"type":"pace"},{"type":"pace_ssd"}]}})";
    EXPECT_THROW(factory.createTransferClient(ssd, params), std::invalid_argument);
    EXPECT_THROW(factory.createTransferClientWithMemory(ssd, params, registrations), std::invalid_argument);
    EXPECT_THROW(factory.createTransferClientWithMemory(mixed, params, registrations), std::invalid_argument);
}
TEST(ClientFactoryFeatureBoundaryTest, EmptyDescriptorsCanReuseOnlySuccessfullyRegisteredGpuSubspans) {
    using Context      = kvcm::PackedSsdTransferClient::GpuPackingContext;
    const auto pointer = [](uintptr_t value) { return reinterpret_cast<void*>(value); };
    Context    context(std::shared_ptr<void>(pointer(0x10000), [](void*) {}));
    context.registered_gpu_spans = {{pointer(0x10000), 0x4000, 0}, {pointer(0x20000), 0x8000, 0}};
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x10000), 0x4000}, 0));
    context.registered = true;
    EXPECT_TRUE(context.coversRegisteredGpuSpan({pointer(0x10000), 0x4000}, 0));
    EXPECT_TRUE(context.coversRegisteredGpuSpan({pointer(0x22000), 0x6000}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x22000), 0x6001}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x28000), 1}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x0ffff), 1}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x10000), 0x4000}, 1));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({nullptr, 1}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(0x10000), 0}, 0));
    EXPECT_FALSE(context.coversRegisteredGpuSpan({pointer(UINTPTR_MAX - 1), 8}, 0));
}
TEST(DecodePublicationConfigTest, SingleSwitchNeedsNoCpSpecificFlag) {
    {
        autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
        EXPECT_FALSE(remoteCacheDecodePublicationEnabled());
    }
    {
        autil::EnvGuard on("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
        EXPECT_TRUE(remoteCacheDecodePublicationEnabled());
    }
}
TEST(DecodePublicationConfigTest, PendingBudgetsHaveBoundedDefaultsAndStrictOverrides) {
    EXPECT_EQ(DecodePublicationLimits{}.bytes, 512ULL * 1024 * 1024);
    EXPECT_EQ(DecodePublicationLimits{}.jobs, 4u);
    autil::EnvGuard mib("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "256");
    autil::EnvGuard jobs("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "8");
    auto            limits = remoteCacheDecodePublicationLimits();
    ASSERT_TRUE(limits.has_value());
    EXPECT_EQ(limits->bytes, 256ULL * 1024 * 1024);
    EXPECT_EQ(limits->jobs, 8u);
    for (const char* invalid : {"", "0", "-1", "1.5", " 8", "8x", "18446744073709551616"}) {
        autil::EnvGuard bad("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", invalid);
        EXPECT_FALSE(remoteCacheDecodePublicationLimits().has_value()) << invalid;
    }
    autil::EnvGuard too_many("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "65");
    EXPECT_FALSE(remoteCacheDecodePublicationLimits().has_value());
}
TEST(DecodePublicationConfigTest, DefaultBudgetTracksPhysicalPoolWithoutRoundingSmallPoolsUp) {
    autil::EnvGuard mib("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1");
    autil::EnvGuard jobs("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "4");
    ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB"), 0);
    ASSERT_EQ(unsetenv("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS"), 0);
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(0).has_value());
    for (const auto& sample :
         std::vector<std::pair<size_t, size_t>>{{8, 1},
                                                {8192, 1024},
                                                {4ULL * 1024 * 1024 * 1024, 512ULL * 1024 * 1024},
                                                {64ULL * 1024 * 1024 * 1024, 4ULL * 1024 * 1024 * 1024}}) {
        const auto limits = remoteCacheDecodePublicationLimits(sample.first);
        ASSERT_TRUE(limits.has_value());
        EXPECT_EQ(limits->bytes, sample.second);
        EXPECT_EQ(limits->jobs, 4u);
    }
}

TEST(NativeSsdGpuPackingTest, GathersAndScattersDifferentGpuIovsWithoutHostPayloadStaging) {
    using namespace kv_cache_manager;
    using M       = kv_cache_manager::MemoryType;
    auto allocate = [](size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess)
            throw std::runtime_error("test GPU allocation failed");
        return std::shared_ptr<void>(p, [](void* v) { cudaFree(v); });
    };
    auto                       arena = allocate(kvcm::PackedSsdTransferClient::max_operation_bytes);
    auto                       input = allocate(4096), output = allocate(4096), remote = allocate(4096);
    std::vector<unsigned char> expected(1280);
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = (i * 31 + 17) % 256;
    ASSERT_EQ(cudaMemcpy(input.get(), expected.data(), expected.size(), cudaMemcpyHostToDevice), cudaSuccess);
    auto*           src       = static_cast<unsigned char*>(input.get());
    auto*           dst       = static_cast<unsigned char*>(output.get());
    auto*           storage   = static_cast<unsigned char*>(remote.get());
    auto            destroyed = std::make_shared<int>(0);
    auto            delegate  = std::make_unique<MockTransferClient>(destroyed);
    const UriStrVec uris{"object0", "object1"};
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
        .WillOnce(Invoke([&](const UriStrVec& u, const BlockBuffers& b, auto) {
            EXPECT_EQ(u, uris);
            EXPECT_EQ(b.size(), 2u);
            for (size_t i = 0; i < b.size(); ++i) {
                EXPECT_EQ(b[i].iovs.size(), 1u);
                EXPECT_EQ(b[i].iovs[0].type, M::GPU);
                const size_t offset = i == 0 ? 0 : 257;
                EXPECT_EQ(b[i].iovs[0].size, i == 0 ? 257u : 1023u);
                EXPECT_EQ(cudaMemcpy(storage + offset, b[i].iovs[0].base, b[i].iovs[0].size,
                                     cudaMemcpyDeviceToDevice), cudaSuccess);
            }
            const auto first = reinterpret_cast<uintptr_t>(b[0].iovs[0].base);
            const auto second = reinterpret_cast<uintptr_t>(b[1].iovs[0].base);
            EXPECT_GE(second, first + b[0].iovs[0].size);
            return std::make_pair(ER_OK, u);
        }));
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
        .Times(2)
        .WillRepeatedly(Invoke([&](const UriStrVec& u, const BlockBuffers& b, auto) {
            EXPECT_EQ(b.size(), 1u);
            EXPECT_EQ(b[0].iovs.size(), 1u);
            EXPECT_EQ(b[0].iovs[0].type, M::GPU);
            size_t offset = u[0] == uris[0] ? 0 : 257;
            EXPECT_EQ(cudaMemcpy(b[0].iovs[0].base, storage + offset, b[0].iovs[0].size, cudaMemcpyDeviceToDevice),
                      cudaSuccess);
            return ER_OK;
        }));
    kvcm::PackedSsdTransferClient client(
        std::move(delegate),
        [](void* d, M dt, const void* s, M st, size_t n) {
            EXPECT_EQ(dt, M::GPU);
            EXPECT_EQ(st, M::GPU);
            return dt == M::GPU && st == M::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
        },
        std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena));
    BlockBuffers source(2), destination(2);
    source[0].iovs      = {{M::GPU, src, 17}, {M::GPU, src + 17, 240}};
    source[1].iovs      = {{M::GPU, src + 257, 513}, {M::GPU, src + 770, 510}};
    destination[0].iovs = {{M::GPU, dst + 2048, 19}, {M::GPU, dst + 2067, 238}};
    destination[1].iovs = {{M::GPU, dst, 13}, {M::GPU, dst + 13, 1010}};
    ASSERT_EQ(client.SaveKvCaches(uris, source).first, ER_OK);
    ASSERT_EQ(client.LoadKvCaches(uris, destination), ER_OK);
    std::vector<unsigned char> first(257), second(1023);
    ASSERT_EQ(cudaMemcpy(first.data(), dst + 2048, first.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(second.data(), dst, second.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::equal(first.begin(), first.end(), expected.begin()));
    EXPECT_TRUE(std::equal(second.begin(), second.end(), expected.begin() + 257));
}
TEST(NativeSsdGpuPackingTest, ConcurrentLargeLogicalBatchesReserveOnlyOneGpuObjectEach) {
    using namespace kv_cache_manager;
    using M                       = kv_cache_manager::MemoryType;
    constexpr size_t object_bytes = 16ULL * 1024 * 1024;
    void*            base         = nullptr;
    void*            source       = nullptr;
    ASSERT_EQ(cudaMalloc(&base, kvcm::PackedSsdTransferClient::max_operation_bytes), cudaSuccess);
    auto arena = std::shared_ptr<void>(base, [](void* p) { cudaFree(p); });
    ASSERT_EQ(cudaMalloc(&source, object_bytes), cudaSuccess);
    auto input = std::shared_ptr<void>(source, [](void* p) { cudaFree(p); });
    ASSERT_EQ(cudaMemset(source, 0x5a, object_bytes), cudaSuccess);
    auto               context   = std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena);
    auto               destroyed = std::make_shared<int>(0);
    auto               first     = std::make_unique<MockTransferClient>(destroyed);
    auto               second    = std::make_unique<MockTransferClient>(destroyed);
    std::promise<void> entered, release;
    auto               started  = entered.get_future();
    auto               released = release.get_future().share();
    std::atomic<int>   first_calls{0};
    EXPECT_CALL(*first, SaveKvCaches(_, _, _)).Times(4).WillRepeatedly(Invoke([&](const auto& u, const auto& b, auto) {
        EXPECT_EQ(b.size(), 1u);
        EXPECT_EQ(b[0].iovs[0].size, object_bytes);
        if (first_calls.fetch_add(1) == 0) {
            entered.set_value();
            released.wait();
        }
        return std::make_pair(ER_OK, u);
    }));
    EXPECT_CALL(*second, SaveKvCaches(_, _, _)).Times(4).WillRepeatedly(Invoke([&](const auto& u, const auto& b, auto) {
        EXPECT_EQ(b.size(), 1u);
        EXPECT_EQ(b[0].iovs[0].type, M::GPU);
        EXPECT_EQ(context->active_reservations, 2u);
        return std::make_pair(ER_OK, u);
    }));
    auto copy = [](void* d, M dt, const void* s, M st, size_t n) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
    };
    kvcm::PackedSsdTransferClient a(std::move(first), copy, context);
    kvcm::PackedSsdTransferClient b(std::move(second), copy, context);
    BlockBuffers                  buffers(4);
    for (auto& block : buffers)
        block.iovs = {{M::GPU, source, object_bytes}};
    UriStrVec  uris{"a", "b", "c", "d"};
    auto       first_result = std::async(std::launch::async, [&] { return a.SaveKvCaches(uris, buffers).first; });
    const auto ready        = started.wait_for(std::chrono::seconds(10));
    // Release before any fatal assertion so the test cannot strand the worker.
    const auto second_result =
        ready == std::future_status::ready ? b.SaveKvCaches(uris, buffers).first : ER_INVALID_PARAMS;
    release.set_value();
    EXPECT_EQ(first_result.get(), ER_OK);
    ASSERT_EQ(ready, std::future_status::ready);
    EXPECT_EQ(second_result, ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST(NativeSsdGpuPackingTest, UnknownCompletionRetainsItsRangeButOtherTagsContinue) {
    using M      = kv_cache_manager::MemoryType;
    void* base   = nullptr;
    void* source = nullptr;
    ASSERT_EQ(cudaMalloc(&base, kvcm::PackedSsdTransferClient::max_operation_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&source, 1024), cudaSuccess);
    auto arena     = std::shared_ptr<void>(base, [](void* p) { cudaFree(p); });
    auto input     = std::shared_ptr<void>(source, [](void* p) { cudaFree(p); });
    auto context   = std::make_shared<kvcm::PackedSsdTransferClient::GpuPackingContext>(arena);
    auto destroyed = std::make_shared<int>(0);
    auto first     = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    auto second    = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*first, SaveKvCaches(_, _, _)).WillOnce(::testing::Throw(std::runtime_error("uncertain DMA")));
    EXPECT_CALL(*second, SaveKvCaches(_, _, _)).WillOnce(Invoke([](const auto& u, const auto&, auto) {
        return std::make_pair(kv_cache_manager::ER_OK, u);
    }));
    auto copy = [](void* d, M dt, const void* s, M st, size_t n) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice) == cudaSuccess;
    };
    kvcm::PackedSsdTransferClient  a(std::move(first), copy, context);
    kvcm::PackedSsdTransferClient  b(std::move(second), copy, context);
    kv_cache_manager::BlockBuffers buffers(1);
    buffers[0].iovs = {{M::GPU, source, 1024}};
    ASSERT_EQ(cudaMemset(source, 0x5a, 1024), cudaSuccess);
    EXPECT_THROW(a.SaveKvCaches({"first"}, buffers), std::runtime_error);
    EXPECT_EQ(context->active_reservations, 1u);
    ASSERT_EQ(cudaMemset(source, 0x22, 1024), cudaSuccess);
    EXPECT_EQ(b.SaveKvCaches({"second"}, buffers).first, kv_cache_manager::ER_OK);
    unsigned char retained = 0;
    ASSERT_EQ(cudaMemcpy(&retained, arena.get(), 1, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(retained, 0x5a);
}
class NativeSsdGpuReadWindowTest: public ::testing::Test {
protected:
    using M       = kv_cache_manager::MemoryType;
    using Code    = kv_cache_manager::ClientErrorCode;
    using Buffers = kv_cache_manager::BlockBuffers;
    using Uris    = kv_cache_manager::UriStrVec;
    using Client  = kvcm::PackedSsdTransferClient;
    std::shared_ptr<void>                      arena, output;
    std::shared_ptr<Client::GpuPackingContext> context;
    std::shared_ptr<int>                       destroyed = std::make_shared<int>(0);
    void                                       SetUp() override {
        void* a = nullptr;
        void* b = nullptr;
        ASSERT_EQ(cudaMalloc(&a, Client::max_operation_bytes), cudaSuccess);
        arena = std::shared_ptr<void>(a, [](void* p) { cudaFree(p); });
        ASSERT_EQ(cudaMalloc(&b, 2 * 1024 * 1024), cudaSuccess);
        output = std::shared_ptr<void>(b, [](void* p) { cudaFree(p); });
        ASSERT_EQ(cudaMemset(b, 0, 2 * 1024 * 1024), cudaSuccess);
        context = std::make_shared<Client::GpuPackingContext>(arena);
    }
    static bool copy(void* dst, M dt, const void* src, M st, size_t bytes) {
        return dt == M::GPU && st == M::GPU && cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess
               && cudaStreamSynchronize(nullptr) == cudaSuccess;
    }
    Buffers destinations(const std::vector<size_t>& sizes) {
        Buffers b(sizes.size());
        size_t  offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            b[i].iovs = {{M::GPU, static_cast<unsigned char*>(output.get()) + offset, sizes[i], false}};
            offset += sizes[i];
        }
        return b;
    }
    template<typename C>
    void checkSafeSdkRetry() {
        if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<C>::value) {
            for (bool write : {false, true}) {
                auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
                if (write) {
                    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
                        .WillOnce(Return(std::make_pair(C::ER_SDK_ERROR_BUFFER_SAFE, Uris{})))
                        .WillOnce(Return(std::make_pair(C::ER_OK, Uris{})));
                } else {
                    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
                        .WillOnce(Return(C::ER_SDK_TIMEOUT_BUFFER_SAFE)).WillOnce(Return(C::ER_OK));
                }
                Client client(std::move(delegate), copy, context);
                auto buffers = destinations({17});
                if (write) {
                    EXPECT_EQ(client.SaveKvCaches({"safe0", "safe1"}, destinations({17, 33})).first,
                              C::ER_SDK_ERROR_BUFFER_SAFE);
                    EXPECT_EQ(context->active_reservations, 0u);
                    EXPECT_EQ(client.SaveKvCaches({"retry0", "retry1"}, destinations({17, 33})).first, C::ER_OK);
                } else {
                    EXPECT_EQ(client.LoadKvCaches({"safe"}, buffers), C::ER_SDK_TIMEOUT_BUFFER_SAFE);
                    EXPECT_EQ(context->active_reservations, 0u);
                    EXPECT_EQ(client.LoadKvCaches({"retry"}, buffers), C::ER_OK);
                }
                EXPECT_EQ(context->active_reservations, 0u);
            }
        } else {
            GTEST_SKIP() << "SDK has no explicit buffer-safe failure contract";
        }
    }
    template<typename C>
    void checkMixedPeerSafety() {
        if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<C>::value) {
            for (bool safe_first : {true, false}) {
                auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
                EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(2)
                    .WillRepeatedly(Invoke([](const Uris& uris, const Buffers&, auto) -> C {
                        return uris.front() == "safe" ? C::ER_SDK_TIMEOUT_BUFFER_SAFE : C::ER_SDK_TIMEOUT;
                    }));
                int device = 0;
                ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
                Client client(std::move(delegate), copy, context,
                              [device] { return cudaSetDevice(device) == cudaSuccess; });
                const Uris uris = safe_first ? Uris{"safe", "unknown"} : Uris{"unknown", "safe"};
                auto buffers = destinations({17, 17});
                EXPECT_EQ(client.LoadKvCaches(uris, buffers), C::ER_SDK_TIMEOUT);
                EXPECT_EQ(context->active_reservations, 1u);
                EXPECT_EQ(client.LoadKvCaches(uris, buffers), C::ER_INVALID_LOCAL_BUFFERS);
            }
        } else {
            GTEST_SKIP() << "SDK has no explicit buffer-safe failure contract";
        }
    }
};
TEST_F(NativeSsdGpuReadWindowTest, ExplicitSafeSdkFailureReleasesArenaAndPermitsRetry) {
    checkSafeSdkRetry<Code>();
}
TEST_F(NativeSsdGpuReadWindowTest, AnyUnknownPeerDominatesSafeFailureInEitherOrder) {
    checkMixedPeerSafety<Code>();
}
TEST_F(NativeSsdGpuReadWindowTest, ConcurrentWindowInitializesDeviceAndJoinsBeforeScatter) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    int  device   = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    std::mutex              mutex;
    std::condition_variable cv;
    int                     arrived[2] = {0, 0};
    std::atomic<int>        active{0}, peak{0}, initialized{0}, completed{0};
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
        .Times(8)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& gpu, auto t) {
            EXPECT_EQ(u.size(), 1u);
            int id = std::stoi(u[0]);
            EXPECT_EQ(t->block_ids[0], std::to_string(100 + id));
            int actual = -1;
            EXPECT_EQ(cudaGetDevice(&actual), cudaSuccess);
            EXPECT_EQ(actual, device);
            int count = active.fetch_add(1) + 1, prior = peak.load();
            while (prior < count && !peak.compare_exchange_weak(prior, count)) {}
            {
                std::unique_lock<std::mutex> guard(mutex);
                ++arrived[id / 4];
                cv.notify_all();
                EXPECT_TRUE(cv.wait_for(guard, std::chrono::seconds(5), [&] { return arrived[id / 4] == 4; }));
            }
            EXPECT_EQ(cudaMemset(gpu[0].iovs[0].base, 11 + id, gpu[0].iovs[0].size), cudaSuccess);
            active.fetch_sub(1);
            completed.fetch_add(1);
            return kv_cache_manager::ER_OK;
        }));
    auto scatter = [&](void* dst, M dt, const void* src, M st, size_t bytes) {
        EXPECT_EQ(active.load(), 0);
        return copy(dst, dt, src, st, bytes);
    };
    Client client(std::move(delegate), scatter, context, [&, device] {
        initialized.fetch_add(1);
        return cudaSetDevice(device) == cudaSuccess;
    });
    Uris   uris;
    auto   t = std::make_shared<kv_cache_manager::TransferTraceInfo>();
    for (int i = 0; i < 8; ++i) {
        uris.push_back(std::to_string(i));
        t->block_ids.push_back(std::to_string(100 + i));
    }
    ASSERT_EQ(client.LoadKvCaches(uris, destinations(std::vector<size_t>(8, 36 * 1024)), t), kv_cache_manager::ER_OK);
    EXPECT_EQ(peak.load(), 4);
    EXPECT_EQ(initialized.load(), 8);
    EXPECT_EQ(completed.load(), 8);
    std::vector<unsigned char> bytes(8 * 36 * 1024);
    ASSERT_EQ(cudaMemcpy(bytes.data(), output.get(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < 8; ++i)
        EXPECT_TRUE(std::all_of(bytes.begin() + i * 36 * 1024,
                                bytes.begin() + (i + 1) * 36 * 1024,
                                [i](unsigned char v) { return v == 11 + i; }));
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, WorkerExceptionDrainsEveryPeerAndQuarantinesWholeWindow) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    int  device   = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    std::atomic<int> completed{0};
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
        .Times(4)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& gpu, auto) -> Code {
            if (u[0] == "0")
                throw std::runtime_error("uncertain read DMA");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            EXPECT_EQ(cudaMemset(gpu[0].iovs[0].base, 0x55, gpu[0].iovs[0].size), cudaSuccess);
            completed.fetch_add(1);
            return kv_cache_manager::ER_OK;
        }));
    Client client(std::move(delegate), copy, context, [device] { return cudaSetDevice(device) == cudaSuccess; });
    std::atomic<size_t> confirmed{0}, unknown{0}, observations{0};
    client.setSdkTransferObserver([&](bool read, size_t bytes, bool uncertain) {
        EXPECT_TRUE(read);
        confirmed.fetch_add(bytes);
        unknown.fetch_add(uncertain);
        observations.fetch_add(1);
    });
    auto b = destinations({4096, 4096, 4096, 4096});
    EXPECT_THROW(client.LoadKvCaches({"0", "1", "2", "3"}, b), std::runtime_error);
    EXPECT_EQ(confirmed.load(), 3u * 4096);
    EXPECT_EQ(unknown.load(), 1u);
    EXPECT_EQ(observations.load(), 4u);
    EXPECT_EQ(completed.load(), 3);
    EXPECT_EQ(context->active_reservations, 1u);
    EXPECT_EQ(client.LoadKvCaches({"0", "1", "2", "3"}, b), kv_cache_manager::ER_INVALID_LOCAL_BUFFERS);
    unsigned char byte = 1;
    ASSERT_EQ(cudaMemcpy(&byte, output.get(), 1, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(byte, 0);
}
TEST_F(NativeSsdGpuReadWindowTest, PayloadBudgetSplitsAWindowBeforeTheObjectLimit) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    int  device   = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    std::atomic<int>        active{0}, peak{0};
    std::mutex              mutex;
    std::condition_variable cv;
    int                     arrived = 0;
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
        .Times(3)
        .WillRepeatedly(Invoke([&](const Uris& uris, const Buffers& buffers, auto) {
            EXPECT_EQ(uris.size(), 1u);
            int id    = std::stoi(uris[0]);
            int count = active.fetch_add(1) + 1, previous = peak.load();
            while (previous < count && !peak.compare_exchange_weak(previous, count)) {}
            if (id < 2) {
                std::unique_lock<std::mutex> guard(mutex);
                ++arrived;
                cv.notify_all();
                EXPECT_TRUE(cv.wait_for(guard, std::chrono::seconds(5), [&] { return arrived == 2; }));
            }
            {
                std::lock_guard<std::mutex> guard(context->mutex);
                size_t                      free = 0;
                for (const auto& range : context->free_ranges)
                    free += range.second;
                EXPECT_EQ(Client::max_operation_bytes - free, 200u * 1024);
            }
            EXPECT_EQ(cudaMemset(buffers[0].iovs[0].base, 11 + id, 100 * 1024), cudaSuccess);
            active.fetch_sub(1);
            return kv_cache_manager::ER_OK;
        }));
    Client client(std::move(delegate), copy, context, [device] { return cudaSetDevice(device) == cudaSuccess; });
    ASSERT_EQ(client.LoadKvCaches({"0", "1", "2"}, destinations({100 * 1024, 100 * 1024, 100 * 1024})),
              kv_cache_manager::ER_OK);
    EXPECT_EQ(peak.load(), 2);
    EXPECT_EQ(context->active_reservations, 0u);
    std::vector<unsigned char> bytes(300 * 1024);
    ASSERT_EQ(cudaMemcpy(bytes.data(), output.get(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int id = 0; id < 3; ++id)
        EXPECT_TRUE(std::all_of(bytes.begin() + id * 100 * 1024,
                                bytes.begin() + (id + 1) * 100 * 1024,
                                [id](unsigned char v) { return v == 11 + id; }));
}
TEST_F(NativeSsdGpuReadWindowTest, OversizedStateObjectIsReadAlone) {
    auto             delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    std::atomic<int> initialized{0};
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _))
        .Times(2)
        .WillRepeatedly(Invoke([](const Uris& uris, const Buffers&, auto) {
            EXPECT_EQ(uris.size(), 1u);
            return kv_cache_manager::ER_OK;
        }));
    Client client(std::move(delegate), copy, context, [&] {
        ++initialized;
        return true;
    });
    EXPECT_EQ(client.LoadKvCaches({"state", "small"}, destinations({354816, 4096})), kv_cache_manager::ER_OK);
    EXPECT_EQ(initialized.load(), 0);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, DeviceInitializationFailureProtectsTheEntireWindow) {
    auto delegate = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).Times(0);
    std::atomic<int> initialized{0};
    Client           client(std::move(delegate), copy, context, [&] {
        ++initialized;
        return false;
    });
    std::atomic<int> observed{0};
    client.setSdkTransferObserver([&](bool, size_t, bool) { ++observed; });
    auto buffers = destinations({4096, 4096, 4096, 4096});
    EXPECT_EQ(client.LoadKvCaches({"0", "1", "2", "3"}, buffers), kv_cache_manager::ER_CUDAMEMCPY_ERROR);
    EXPECT_EQ(initialized.load(), 4);
    EXPECT_EQ(observed.load(), 0);
    EXPECT_EQ(context->active_reservations, 1u);
    EXPECT_EQ(client.LoadKvCaches({"0", "1", "2", "3"}, buffers), kv_cache_manager::ER_INVALID_LOCAL_BUFFERS);
    unsigned char byte = 1;
    ASSERT_EQ(cudaMemcpy(&byte, output.get(), 1, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(byte, 0);
}
TEST_F(NativeSsdGpuReadWindowTest, SaveObserverKeepsConfirmedPrefixBeforeSdkFailure) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    Uris uris;
    std::vector<size_t> sizes;
    size_t first_window_bytes = 0;
    for (size_t i = 0; i < 129; ++i) {
        uris.push_back(std::to_string(i));
        sizes.push_back(17 + i % 7);
        if (i < 64)
            first_window_bytes += sizes.back();
    }
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
        .Times(2)
        .WillRepeatedly(Invoke([](const Uris& batch, const Buffers& buffers, auto) {
            EXPECT_EQ(batch.size(), 64u);
            EXPECT_EQ(buffers.size(), batch.size());
            EXPECT_TRUE(batch.front() == "0" || batch.front() == "64");
            return std::make_pair(batch.front() == "0" ? ER_OK : ER_SDK_TIMEOUT, batch);
        }));
    Client client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.SaveKvCaches(uris, destinations(sizes)).first, ER_SDK_TIMEOUT);
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0], std::make_tuple(false, first_window_bytes, false));
    EXPECT_EQ(seen[1], std::make_tuple(false, size_t(0), true));
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, PreSdkGeometryAndGatherRejectionsDoNotReportUnknownIo) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    Client client(std::move(delegate), [](void*, M, const void*, M, size_t) { return false; }, context);
    int    observed = 0;
    client.setSdkTransferObserver([&](bool, size_t, bool) { ++observed; });
    EXPECT_EQ(client.SaveKvCaches({"invalid"}, destinations({0})).first, ER_INVALID_LOCAL_BUFFERS);
    EXPECT_EQ(context->active_reservations, 0u);
    EXPECT_EQ(client.SaveKvCaches({"gather-fails"}, destinations({17})).first, ER_CUDAMEMCPY_ERROR);
    EXPECT_EQ(observed, 0);
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, SdkReadSuccessRemainsConfirmedAfterScatterFailure) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Return(ER_OK));
    Client client(std::move(delegate), [](void*, M, const void*, M, size_t) { return false; }, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.LoadKvCaches({"scatter-fails"}, destinations({17})), ER_CUDAMEMCPY_ERROR);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen[0], std::make_tuple(true, size_t(17), false));
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, SdkWriteSuccessRemainsConfirmedAfterActualUriShapeFailure) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_OK, Uris{"wrong", "count"})));
    Client                                      client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.SaveKvCaches({"one"}, destinations({17})).first, ER_INVALID_PARAMS);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen[0], std::make_tuple(false, size_t(17), false));
}
TEST_F(NativeSsdGpuReadWindowTest, ObserverExceptionsPreserveSdkSuccessAndFailureResults) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, LoadKvCaches(_, _, _)).WillOnce(Return(ER_OK)).WillOnce(Return(ER_SDK_TIMEOUT));
    Client                                      client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver([&](bool read, size_t bytes, bool unknown) {
        seen.emplace_back(read, bytes, unknown);
        throw std::logic_error("observer-failure");
    });
    EXPECT_EQ(client.LoadKvCaches({"success"}, destinations({17})), ER_OK);
    EXPECT_EQ(client.LoadKvCaches({"failed"}, destinations({17})), ER_SDK_TIMEOUT);
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0], std::make_tuple(true, size_t(17), false));
    EXPECT_EQ(seen[1], std::make_tuple(true, size_t(0), true));
    EXPECT_EQ(client.LoadKvCaches({"quarantined"}, destinations({17})), ER_INVALID_LOCAL_BUFFERS);
    EXPECT_EQ(seen.size(), 2u);
}
TEST_F(NativeSsdGpuReadWindowTest, ObserverExceptionCannotReplaceTheOriginalSdkException) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).WillOnce(::testing::Throw(std::runtime_error("sdk-sentinel")));
    Client client(std::move(delegate), copy, context);
    int    observed = 0;
    client.setSdkTransferObserver([&](bool read, size_t bytes, bool unknown) {
        ++observed;
        EXPECT_FALSE(read);
        EXPECT_EQ(bytes, 0u);
        EXPECT_TRUE(unknown);
        throw std::logic_error("observer-failure");
    });
    try {
        (void)client.SaveKvCaches({"throw"}, destinations({17}));
        FAIL() << "SDK exception must propagate";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "sdk-sentinel");
    }
    EXPECT_EQ(observed, 1);
    EXPECT_EQ(context->active_reservations, 1u);
}
TEST_F(NativeSsdGpuReadWindowTest, WriteBatchPreservesPositionsTracesAndGpuPayloadAcrossWindows) {
    using namespace kv_cache_manager;
    Uris uris;
    std::vector<size_t> sizes;
    std::vector<unsigned char> expected;
    auto trace = std::make_shared<TransferTraceInfo>();
    for (size_t i = 0; i < 130; ++i) {
        uris.push_back("uri" + std::to_string(i));
        sizes.push_back(17 + i % 7);
        trace->block_ids.push_back(std::to_string(1000 + i));
        for (size_t j = 0; j < sizes.back(); ++j)
            expected.push_back(static_cast<unsigned char>((i * 31 + j * 17) % 256));
    }
    ASSERT_EQ(cudaMemcpy(output.get(), expected.data(), expected.size(), cudaMemcpyHostToDevice), cudaSuccess);
    void* remote = nullptr;
    ASSERT_EQ(cudaMalloc(&remote, expected.size()), cudaSuccess);
    auto storage = std::shared_ptr<void>(remote, [](void* p) { cudaFree(p); });
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    size_t next = 0, remote_offset = 0, calls = 0;
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(3)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, const auto& slice) {
            EXPECT_EQ(u.size(), std::min(size_t(64), uris.size() - next));
            EXPECT_EQ(b.size(), u.size());
            EXPECT_NE(slice, nullptr);
            EXPECT_NE(slice, trace);
            if (!slice || b.size() != u.size() || slice->block_ids.size() != u.size())
                return std::make_pair(ER_INVALID_PARAMS, Uris{});
            uintptr_t previous_end = 0;
            Uris actual;
            for (size_t i = 0; i < u.size(); ++i, ++next) {
                EXPECT_EQ(u[i], uris[next]);
                EXPECT_EQ(slice->block_ids[i], std::to_string(1000 + next));
                EXPECT_EQ(b[i].iovs.size(), 1u);
                if (b[i].iovs.size() != 1 || next >= sizes.size())
                    return std::make_pair(ER_INVALID_PARAMS, Uris{});
                const auto& packed = b[i].iovs[0];
                EXPECT_EQ(packed.type, M::GPU);
                EXPECT_EQ(packed.size, sizes[next]);
                if (packed.size != sizes[next])
                    return std::make_pair(ER_INVALID_PARAMS, Uris{});
                const auto address = reinterpret_cast<uintptr_t>(packed.base);
                if (i) {
                    EXPECT_GE(address, previous_end);
                }
                previous_end = address + packed.size;
                EXPECT_EQ(cudaMemcpy(static_cast<unsigned char*>(storage.get()) + remote_offset,
                                     packed.base, packed.size, cudaMemcpyDeviceToDevice), cudaSuccess);
                remote_offset += packed.size;
                actual.push_back("actual" + std::to_string(next));
            }
            ++calls;
            return std::make_pair(ER_OK, actual);
        }));
    Client client(std::move(delegate), copy, context);
    const auto result = client.SaveKvCaches(uris, destinations(sizes), trace);
    ASSERT_EQ(result.first, ER_OK);
    ASSERT_EQ(result.second.size(), uris.size());
    for (size_t i = 0; i < uris.size(); ++i)
        EXPECT_EQ(result.second[i], "actual" + std::to_string(i));
    EXPECT_EQ(calls, 3u);
    EXPECT_EQ(next, uris.size());
    EXPECT_EQ(context->active_reservations, 0u);
    std::vector<unsigned char> observed(expected.size());
    ASSERT_EQ(cudaMemcpy(observed.data(), storage.get(), observed.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(observed, expected);
}
TEST_F(NativeSsdGpuReadWindowTest, WritePayloadBudgetSplitsBeforeObjectLimit) {
    using namespace kv_cache_manager;
    constexpr size_t bytes = 700 * 1024;
    Buffers source(3);
    for (auto& block : source)
        block.iovs = {{M::GPU, output.get(), bytes, false}};
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    size_t calls = 0;
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(2)
        .WillRepeatedly(Invoke([&](const Uris& u, const Buffers& b, auto) {
            EXPECT_EQ(u.size(), calls++ == 0 ? 2u : 1u);
            EXPECT_EQ(u.size(), b.size());
            size_t total = 0;
            for (const auto& block : b)
                total += block.iovs.front().size;
            EXPECT_LE(total, Client::max_write_window_bytes);
            return std::make_pair(ER_OK, u);
        }));
    Client client(std::move(delegate), copy, context);
    EXPECT_EQ(client.SaveKvCaches({"a", "b", "c"}, source).first, ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, WriteLeaseFailureSubmitsNothingAndDoesNotLeakProcessCharge) {
    using namespace kv_cache_manager;
    context->free_ranges = {{0, 16}};
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    Client failed(std::move(delegate), copy, context);
    Buffers large(1);
    large[0].iovs = {{M::GPU, arena.get(), Client::max_operation_bytes, false}};
    for (size_t i = 0; i < 9; ++i)
        EXPECT_EQ(failed.SaveKvCaches({"blocked"}, large).first, ER_INVALID_LOCAL_BUFFERS);
    EXPECT_EQ(context->active_reservations, 0u);
    context->free_ranges = {{0, Client::max_operation_bytes}};
    auto retry = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*retry, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_OK, Uris{})));
    Client accepted(std::move(retry), copy, context);
    EXPECT_EQ(accepted.SaveKvCaches({"retry"}, destinations({17})).first, ER_OK);
    EXPECT_EQ(context->active_reservations, 0u);
}
TEST_F(NativeSsdGpuReadWindowTest, BatchWriteBadUriShapeKeepsConfirmedBytesAndQuarantinesWindow) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _))
        .WillOnce(Return(std::make_pair(ER_OK, Uris{"only-one"})));
    Client client(std::move(delegate), copy, context);
    std::vector<std::tuple<bool, size_t, bool>> seen;
    client.setSdkTransferObserver(
        [&](bool read, size_t bytes, bool unknown) { seen.emplace_back(read, bytes, unknown); });
    EXPECT_EQ(client.SaveKvCaches({"one", "two"}, destinations({17, 33})).first, ER_INVALID_PARAMS);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen.front(), std::make_tuple(false, size_t(50), false));
    EXPECT_EQ(context->active_reservations, 1u);
    EXPECT_EQ(client.SaveKvCaches({"retry"}, destinations({17})).first, ER_INVALID_LOCAL_BUFFERS);
}
TEST_F(NativeSsdGpuReadWindowTest, GatherExceptionBeforeBatchSubmissionRetainsWholeWindow) {
    using namespace kv_cache_manager;
    auto delegate = std::make_unique<MockTransferClient>(destroyed);
    EXPECT_CALL(*delegate, SaveKvCaches(_, _, _)).Times(0);
    size_t copies = 0;
    Client client(std::move(delegate), [&](void* d, M dt, const void* s, M st, size_t n) {
        if (++copies == 2)
            throw std::runtime_error("uncertain-copy");
        return copy(d, dt, s, st, n);
    }, context);
    size_t observed = 0;
    client.setSdkTransferObserver([&](bool, size_t, bool) { ++observed; });
    EXPECT_THROW(client.SaveKvCaches({"one", "two"}, destinations({17, 33})), std::runtime_error);
    EXPECT_EQ(observed, 0u);
    EXPECT_EQ(context->active_reservations, 1u);
}
// GPU pages + recording SDK. CPU copies below are assertions only; the
// implementation must pass native GPU IOVs through the Prefill transfer path.
class EosGpuCompletionTest: public ::testing::TestWithParam<std::tuple<int, bool, int>> {
protected:
    autil::EnvGuard enable{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1"},
        pending{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "1"},
        // This tiny fixture intentionally holds most of its synthetic pool.
        // Production's 1/8 default is covered separately; allow these positive
        // lifetime/projection cases to reach transfer rather than admission.
        pending_bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "1"};
    // Failure cases deliberately freeze a large fraction of these tiny pools.
    autil::EnvGuard quarantine_abort{"RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0"};
    struct Scenario {
        int                                 cp, shards, count = 3;
        BackendEnvironment                  env;
        BlockIndicesType                    full, state;
        BatchKVCacheResource                resource;
        std::shared_ptr<MockClientWrapper>  client = std::make_shared<MockClientWrapper>();
        std::unique_ptr<MockQuiescentKVCMBackend> backend;
        std::map<std::string, std::string>  specs;
        std::atomic<int>                    starts{0}, saves{0}, finishes{0}, aborts{0};
        bool                                block_start = false, fail_save = false, lose_commit = false;
        std::promise<void>                  entered, released;
        std::shared_future<void>            release = released.get_future().share();
        std::future<void>                   started = entered.get_future();
        explicit Scenario(
            int target_cp, bool sharded, bool bounded = true, bool native_ssd_gdr = false, int decode_dp = 1):
            cp(target_cp), shards(sharded ? cp : 1) {
            // The target CP comes only from ParallelismConfig, not a CP-specific environment flag.
            auto f                     = defaultCacheGroupPolicy(CacheGroupType::FULL);
            auto s                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
            s.enable_prefix_reuse      = true;
            s.cp_mapping               = sharded ? CpBlockMappingMode::COMPACT_LAST_RANK : CpBlockMappingMode::NONE;
            s.cp_slice                 = sharded ? CpBlockSliceMode::EQUAL_BYTES : CpBlockSliceMode::NONE;
            s.sliding_window_size      = bounded ? 256 * shards : 0;
            env.cache_config.dtype     = DataType::TYPE_FP16;
            env.cache_config.layer_num = 2;
            env.cache_config.seq_size_per_block = 128;
            env.cache_config.fromGroupedSpecs({test::makeMhaSpec("full", 128, DataType::TYPE_FP16, 1, 2),
                                               test::makeMhaSpec("state", 128 * shards, DataType::TYPE_FP16, 1, 2)},
                                              {{0}, {1}},
                                              {CacheGroupType::FULL, CacheGroupType::SWA},
                                              {"full", "state"},
                                              {f, s});
            const int pool_blocks = std::max(32, (count + 1) * shards + 8);
            env.cache_config.finalizeBlockNums(pool_blocks, RuntimeConfig{});
            initializeEnvironmentPools(env, "eos_native_gpu", pool_blocks);
            full  = *env.pools_by_tag.at("full")->malloc((count + 1) * shards);
            state = *env.pools_by_tag.at("state")->malloc(count + 1);
            env.pools_by_tag.at("full")->incRef(full);
            env.pools_by_tag.at("state")->incRef(state);
            for (size_t i = 0; i < full.size(); ++i)
                fill(0, "full", full[i], i + 1);
            for (size_t i = 0; i < state.size(); ++i)
                fill(1, "state", state[i], 100 + i);
            resource.resetBatchSize(1);
            resource.initGroups(env.cache_config.topologyPtr());
            resource.setBatchBlocks(0, "full", full);
            resource.setBatchBlocks(0, "state", state);
            CacheKeysType keys;
            for (size_t i = 0; i < full.size(); ++i)
                keys.push_back(101 + i);
            resource.setBatchCacheKeys(0, keys);
            EXPECT_CALL(*client, initForPools(_, _, _, _))
                .WillOnce(Invoke([&](const auto&, auto, const auto& registrations, const auto& tags) {
                    EXPECT_EQ(registrations.size(), tags.size());
                    for (size_t i = 0; i < tags.size(); ++i)
                        specs[tags[i]] = registrations[i].location_spec_name;
                    return true;
                }));
            EXPECT_CALL(*client, shutdown()).Times(1);
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _))
                .WillRepeatedly(
                    Invoke([&](const auto&, const auto&, const auto& keys, const auto&, const auto&, auto, auto) {
                        ++starts;
                        if (block_start) {
                            entered.set_value();
                            EXPECT_EQ(release.wait_for(std::chrono::seconds(5)), std::future_status::ready);
                        }
                        kv_cache_manager::WriteLocation location;
                        location.write_session_id = "EOS";
                        location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
                        for (size_t key = 0; key < keys.size(); ++key) {
                            EXPECT_EQ(keys[key], 100 + (key + 1) * shards);
                            kv_cache_manager::Location one;
                            for (int peer = 0; peer < cp; ++peer) {
                                const auto suffix = std::to_string(key) + ":" + std::to_string(peer);
                                one.push_back({specs.at("full#cp" + std::to_string(peer)), "pace://F:" + suffix});
                                if (keys.size() - key
                                    <= env.cache_config.topology().group("state").reuseBlockCount(keys.size()))
                                    one.push_back({specs.at("state#cp" + std::to_string(peer)), "pace://S:" + suffix});
                            }
                            location.locations.push_back(one);
                        }
                        return std::make_pair(true, location);
                    }));
            EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
                .WillRepeatedly(Invoke([&](const auto& tag, const auto& uris, const auto& buffers, const auto&) {
                    ++saves;
                    EXPECT_EQ(uris.size(), buffers.size());
                    kv_cache_manager::UriStrVec actual;
                    for (size_t i = 0; i < uris.size(); ++i) {
                        const auto payload = uris[i].substr(7);
                        const int key = std::stoi(payload.substr(2)), peer = std::stoi(payload.substr(4));
                        // The first registered PACE client owns every GPU span.
                        EXPECT_EQ(tag, "full#cp0");
                        const int expected = payload[0] == 'F' ? key * shards + (shards > 1 ? peer : 0) + 1 : 100 + key;
                        for (const auto& iov : buffers[i].iovs) {
                            EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                            std::vector<unsigned char> bytes(iov.size);
                            EXPECT_EQ(cudaMemcpy(bytes.data(), iov.base, iov.size, cudaMemcpyDeviceToHost),
                                      cudaSuccess);
                            EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto b) { return b == expected; }));
                        }
                        actual.push_back("actual_" + uris[i]);
                    }
                    return std::make_pair(!fail_save, actual);
                }));
            EXPECT_CALL(*client, finishWrite(_, _, _, _, _))
                .WillRepeatedly(
                    Invoke([&](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                        if (locations.empty()) {
                            ++aborts;
                            return true;
                        }
                        ++finishes;
                        EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), locations.size());
                        for (const auto& row : locations)
                            for (const auto& item : row)
                                EXPECT_EQ(item.uri.rfind("actual_", 0), 0u);
                        return !lose_commit;
                    }));
            KVCacheConfig k;
            k.kvcm_server_address   = "unused";
            k.kvcm_instance_group   = "decode_remote_eos_gpu_test";
            k.kvcm_instance_id_salt = "eos_native_gpu";
            k.kvcm_model_sdk_config = "[{\"type\":\"pace\"}]";
            if (native_ssd_gdr) {
                k.kvcm_model_sdk_config  = "[{\"type\":\"pace_ssd\"}]";
                k.kvcm_read_backend_type = 9;
            }
            ParallelismConfig parallel;
            parallel.tp_size                            = 1;
            parallel.dp_size                            = decode_dp;
            parallel.ep_size                            = decode_dp;
            parallel.role_type                          = RoleType::DECODE;
            parallel.prefill_cp_config.prefill_cp_size  = cp;
            parallel.prefill_cp_config.kv_cache_sharded = sharded;
            RuntimeConfig rt;
            rt.model_name = "eos_gpu";
            backend       = std::make_unique<MockQuiescentKVCMBackend>(
                env.cache_config, k, rt, parallel, SpeculativeExecutionConfig{}, nullptr, true, nullptr, client);
            EXPECT_TRUE(initSingleRank(*backend, env));
        }
        ~Scenario() {
            unblock();
            if (backend) {
                backend->proveMockQuiescence();
                backend->shutdown();
            }
            dropRequestRefs();
        }
        void unblock() {
            try {
                released.set_value();
            } catch (...) {}
        }
        void fill(int layer, const std::string& tag, int block, int value) {
            for (const auto& part : environmentBuffers(env, layer, tag, block))
                EXPECT_EQ(cudaMemset(part.addr, value, part.size_bytes), cudaSuccess);
        }
        size_t end() const {
            return count * shards * 128;
        }
        bool refs = true;
        void dropRequestRefs() {
            if (refs) {
                env.pools_by_tag.at("full")->decRef(full);
                env.pools_by_tag.at("state")->decRef(state);
                refs = false;
            }
        }
    };
};
TEST_P(EosGpuCompletionTest, NoProgressSnapshotAndNativePagesSurviveRequestRelease) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.block_start = true;
    for (size_t i = 0; i < s.state.size(); ++i)
        s.fill(1, "state", s.state[i], 0xee);
    for (size_t n = 128; n <= s.end(); n += 128)
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, n, false));
    EXPECT_EQ(s.starts, 0);
    EXPECT_EQ(s.saves, 0);
    for (size_t i = 0; i < s.state.size(); ++i)
        s.fill(1, "state", s.state[i], 100 + i);
    // Change the final state after all progress callbacks. EOS must read this
    // GPU state, not any early host snapshot. Values remain same per-key here.
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    ASSERT_TRUE(s.resource.decodeCompletionSubmitted());
    s.dropRequestRefs();
    s.resource.clearBlocks();
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    EXPECT_TRUE(s.env.pools_by_tag.at("state")->isAllocated(s.state[1]));
    EXPECT_TRUE(s.env.pools_by_tag.at("state")->isAllocated(s.state[2]));
    EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(s.state[0]));
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
    for (auto block : s.state)
        EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
}
TEST_P(EosGpuCompletionTest, PartialTailUsesEarlierImmutableCheckpointNotFinalState) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    // Active, incomplete final slot intentionally differs from complete state.
    s.fill(1, "state", s.state.back(), 0xee);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 128 * s.shards - 1, true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
}
TEST_P(EosGpuCompletionTest, NativeSsdGdrKeepsGpuPagesThroughEosTransfer) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, true, std::get<2>(GetParam()));
    s.block_start = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    s.dropRequestRefs();
    s.resource.clearBlocks();
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
}
TEST_P(EosGpuCompletionTest, MissingStateBacksOffWithoutRelabeling) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[2]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
}
TEST_P(EosGpuCompletionTest, KnownPrefillBoundarySkipsCheckpointWithoutNewDecodeKv) {
    for (const bool checkpoint_backoff : {false, true}) {
        SCOPED_TRACE(checkpoint_backoff);
        Scenario     s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
        const size_t unit    = 128 * s.shards;
        const size_t handoff = checkpoint_backoff ? s.end() - unit : s.end();
        ASSERT_TRUE(s.resource.decodeCompletionLifecycle().recordPrefillComputedEnd(handoff));
        if (checkpoint_backoff) {
            auto state = s.state;
            state[2]   = NULL_BLOCK_IDX;
            s.resource.setBatchBlocks(0, "state", state);
        }
        // D has computed a tail beyond P. In the second case the newest full
        // boundary lacks its state page, so only P's older checkpoint is valid.
        ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end() + 17, true));
        ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
        EXPECT_EQ(s.starts, 0);
        EXPECT_EQ(s.saves, 0);
        EXPECT_EQ(s.finishes, 0);
        EXPECT_EQ(s.aborts, 0);
        EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
        for (const auto& [tag, pool] : s.env.pools_by_tag) {
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u) << tag;
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 0u) << tag;
        }
        s.dropRequestRefs();
        for (const auto block : s.full) {
            EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
        }
        for (const auto block : s.state) {
            EXPECT_FALSE(s.env.pools_by_tag.at("state")->isAllocated(block));
        }
    }
}

TEST_P(EosGpuCompletionTest, UnknownSdkCompletionRetainsPagesAndWaitsForExplicitMockQuiescence) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.fail_save = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 0);
    // Unknown DMA completion must retain the remote WRITING session as well as
    // local GPU pages. An immediate abort could recycle its destination.
    EXPECT_EQ(s.aborts, 0);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    EXPECT_TRUE(s.backend->protected_quarantine_);
    EXPECT_EQ(s.backend->quarantined_task_count_, 1u);
    BatchKVCacheResource other        = s.resource;
    other.decodeCompletionSubmitted() = false;
    ASSERT_TRUE(s.backend->captureCompletedDecode(other, s.end(), true));
    EXPECT_FALSE(other.decodeCompletionSubmitted());
    EXPECT_EQ(s.starts, 1);
    auto stopped = std::async(std::launch::async, [&] { s.backend->shutdown(); });
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_TRUE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
    s.backend->proveMockQuiescence();
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    stopped.get();
    EXPECT_FALSE(s.backend->protected_quarantine_);
    for (int i = 0; i < s.count * s.shards; ++i)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(s.full[i]));
}
TEST_P(EosGpuCompletionTest, DuplicateEosAndFullAdmissionDoNotDoublePinOrPublish) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.block_start = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_EQ(s.started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    const auto held = s.env.pools_by_tag.at("full")->refCount(s.full[0]);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    BatchKVCacheResource other        = s.resource;
    other.decodeCompletionSubmitted() = false;
    ASSERT_TRUE(s.backend->captureCompletedDecode(other, s.end(), true));
    EXPECT_FALSE(other.decodeCompletionSubmitted());
    EXPECT_EQ(s.env.pools_by_tag.at("full")->refCount(s.full[0]), held);
    s.unblock();
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 1);
    EXPECT_EQ(s.finishes, 1);
}
TEST_P(EosGpuCompletionTest, LostCommitAckDoesNotAbortOrQuarantineCompletedGpuTransfer) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), true, false, std::get<2>(GetParam()));
    s.lose_commit = true;
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    s.dropRequestRefs();
    EXPECT_EQ(s.finishes, 1);
    EXPECT_EQ(s.aborts, 0);
    for (auto block : s.full)
        EXPECT_FALSE(s.env.pools_by_tag.at("full")->isAllocated(block));
}
TEST_P(EosGpuCompletionTest, UnboundedSwaNeverPretendsTwoTailPagesAreEnough) {
    Scenario s(std::get<0>(GetParam()), std::get<1>(GetParam()), false, false, std::get<2>(GetParam()));
    auto     state = s.state;
    state[0]       = NULL_BLOCK_IDX;
    s.resource.setBatchBlocks(0, "state", state);
    ASSERT_TRUE(s.backend->captureCompletedDecode(s.resource, s.end(), true));
    ASSERT_TRUE(waitForBackendOperationsForTest(*s.backend));
    EXPECT_EQ(s.starts, 0);
    EXPECT_FALSE(s.resource.decodeCompletionSubmitted());
}
INSTANTIATE_TEST_SUITE_P(GenericCPAndIndependentDP,
                         EosGpuCompletionTest,
                         ::testing::Combine(::testing::Values(1, 2, 4, 8, 16),
                                            ::testing::Bool(),
                                            ::testing::Values(1, 3, 8)));
}  // namespace
}  // namespace rtp_llm
