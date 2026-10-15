#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <sys/resource.h>

#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {
namespace {

void pauseChecker(P2PConnectorAsyncWriteContextChecker& checker) {
    checker.stopping_.store(true);
    checker.notification_->notify();
    if (checker.thread_.joinable()) checker.thread_.join();
    checker.stopping_.store(false);
}

std::shared_ptr<P2PConnectorAsyncWriteContext> makeWriteContext(KVCacheResourcePtr                  resource,
                                                                std::string                         key,
                                                                int64_t                             deadline,
                                                                std::shared_ptr<P2PBroadcastClient> client,
                                                                int64_t                             cancel_timeout,
                                                                int64_t                             lease_timeout,
                                                                std::function<void()>               on_released = {}) {
    auto group = std::make_shared<p2p_internal::WriteTaskGroup>(deadline, std::move(resource));
    return std::make_shared<P2PConnectorAsyncWriteContext>(std::move(key),
                                                           deadline,
                                                           std::move(client),
                                                           cancel_timeout,
                                                           lease_timeout,
                                                           [group, on_released = std::move(on_released)] {
                                                               group->cancel();
                                                               group->releaseSourceResource();
                                                               if (on_released)
                                                                   on_released();
                                                           });
}

KVCacheResourcePtr contextResource(const std::shared_ptr<P2PSchedulerPrefillWrite::RequestState>& state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->resource;
}

bool receiveDone(const std::shared_ptr<P2PSchedulerPrefillWrite::RequestState>& state) {
    return !state->resourceHoldPending() || state->errorInfo().hasError();
}

bool receiveSuccess(const std::shared_ptr<P2PSchedulerPrefillWrite::RequestState>& state) {
    return !state->resourceHoldPending() && state->errorInfo().ok();
}

class WriteSchedulerService: public RpcService::Service {
public:
    std::function<ErrorInfo(const P2PConnectorBroadcastTpRequestPB&)> registered;
    using Handler = std::function<void(
        const P2PConnectorStartWriteRequestPB&, P2PConnectorStartWriteResponsePB&, std::function<bool()>)>;
    Handler           handshake;
    std::function<void(const P2PConnectorWriteReadyRequestPB&, P2PConnectorWriteReadyResponsePB&)>  ready;
    std::function<void(const P2PConnectorCancelWriteRequestPB&, P2PConnectorStartWriteResponsePB&)> cancel_write;
    std::atomic<int>     ready_calls{0}, peer_cancels{0};
    std::atomic<bool>    fail_start{false}, lose_handshake{false}, omit_status{false};
    std::atomic<int>     start_delay_ms{0};
    std::atomic<int>     starts{0}, cancels{0}, handshakes{0}, lease_queries{0};
    std::atomic<int64_t> control_deadline_ms{0};

    grpc::Status StartWrite(grpc::ServerContext*                   context,
                            const P2PConnectorStartWriteRequestPB* request,
                            P2PConnectorStartWriteResponsePB*      response) override {
        ++handshakes;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_handshake_ = *request;
        }
        handshake(*request, *response, [context]() { return context->IsCancelled(); });
        return lose_handshake ? grpc::Status(grpc::StatusCode::INTERNAL, "lost handshake") : grpc::Status::OK;
    }

    grpc::Status WriteReady(grpc::ServerContext*,
                            const P2PConnectorWriteReadyRequestPB* request,
                            P2PConnectorWriteReadyResponsePB*      response) override {
        ++ready_calls;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_ready_ = *request;
        }
        if (ready)
            ready(*request, *response);
        return grpc::Status::OK;
    }
    grpc::Status CancelWrite(grpc::ServerContext*,
                             const P2PConnectorCancelWriteRequestPB* request,
                             P2PConnectorStartWriteResponsePB*       response) override {
        ++peer_cancels;
        cancel_write(*request, *response);
        return grpc::Status::OK;
    }
    P2PConnectorWriteReadyRequestPB lastReady() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_ready_;
    }

    grpc::Status ExecuteFunction(grpc::ServerContext*     context,
                                 const FunctionRequestPB* request,
                                 FunctionResponsePB*      response) override {
        const auto& input = request->p2p_request();
        if (input.type() == CANCEL_HANDLE_WRITE) {
            control_deadline_ms.store(
                std::chrono::duration_cast<std::chrono::milliseconds>(context->deadline().time_since_epoch()).count());
        }
        if (input.type() == WRITE && start_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(start_delay_ms.load()));
        }
        std::unique_lock<std::mutex> lock(mutex_);
        auto&                        state = states_[input.unique_key()];
        if (draining_) {
            state.stopped   = true;
            state.cancelled = true;
        }
        if (input.type() == QUERY_LEASE_STATUS) {
            ++lease_queries;
        }
        const bool start = input.type() == WRITE || input.type() == HANDLE_WRITE;
        if (start) {
            ++starts;
            state.started = true;
            last_start_ = input;
            if (fail_start) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "lost worker START response");
            }
            lock.unlock();
            if (input.type() == HANDLE_WRITE && registered) {
                const auto error = registered(input);
                if (error.hasError()) {
                    return grpc::Status(grpc::StatusCode::INTERNAL, error.ToString());
                }
            }
            lock.lock();
            while (!state.stopped && (input.type() == WRITE || !state.cancelled) && !context->IsCancelled()) {
                cv_.wait_for(lock, std::chrono::milliseconds(2));
            }
            if (context->IsCancelled()) {
                return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "worker deadline expired");
            }
            if (state.cancelled || !state.success) {
                response->mutable_p2p_response()->set_error_code(ErrorCodePB::CANCELLED);
                response->mutable_p2p_response()->set_error_message("worker transfer failed or cancelled");
            }
        } else if (input.type() == CANCEL_WRITE || input.type() == CANCEL_HANDLE_WRITE) {
            ++cancels;
            state.cancelled = true;
            cv_.notify_all();
            if (input.type() == CANCEL_WRITE) {
                if (!state.started) {
                    state.stopped = true;
                }
                while (!state.stopped && !context->IsCancelled()) {
                    cv_.wait_for(lock, std::chrono::milliseconds(2));
                }
                if (context->IsCancelled()) {
                    return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "cancel stop confirmation expired");
                }
                if (!omit_status) {
                    response->mutable_p2p_response();
                }
                return grpc::Status::OK;
            }
        }
        if (omit_status && !start) {
            return grpc::Status::OK;
        }
        auto* lease = response->mutable_p2p_response()->mutable_lease_status();
        lease->set_sealed(true);
        lease->set_started_ops(1);
        lease->set_finished_ops(state.stopped ? 1 : 0);
        lease->set_stopped(state.stopped);
        return grpc::Status::OK;
    }

    void completeAll(bool success = true, bool draining = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        draining_ = draining_ || draining;
        for (auto& [key, state] : states_) {
            state.stopped = true;
            state.success = success;
        }
        cv_.notify_all();
    }
    P2PConnectorBroadcastTpRequestPB lastStart() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_start_;
    }
    P2PConnectorStartWriteRequestPB lastHandshake() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_handshake_;
    }

private:
    struct State {
        bool started{false};
        bool stopped{false};
        bool success{false};
        bool cancelled{false};
    };
    std::condition_variable          cv_;
    mutable std::mutex               mutex_;
    std::map<std::string, State>     states_;
    bool                             draining_{false};
    P2PConnectorBroadcastTpRequestPB last_start_;
    P2PConnectorStartWriteRequestPB  last_handshake_;
    P2PConnectorWriteReadyRequestPB  last_ready_;
};

class P2PConnectorWriteSchedulerTest: public ::testing::Test {
protected:
    using Allocator = test::BlockTreeCacheTestAllocator<SingleTypeKVCacheAllocator>;

    void SetUp() override {
        config_    = test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1, 2);
        allocator_ = std::make_shared<Allocator>(config_);
        ASSERT_TRUE(allocator_->init());
        for (int i = 0; i < 2; ++i) {
            launch(prefill_workers_[i], prefill_addrs_);
            launch(decode_workers_[i], decode_addrs_);
        }
        prefill_client_ = std::make_shared<P2PBroadcastClient>(prefill_addrs_, 100);
        decode_client_  = std::make_shared<P2PBroadcastClient>(decode_addrs_, 100);
        ASSERT_TRUE(prefill_client_->init());
        ASSERT_TRUE(decode_client_->init());
        prefill_config_ = schedulerConfig(RoleType::PREFILL, prefill_addrs_);
        decode_config_  = schedulerConfig(RoleType::DECODE, decode_addrs_);
        for (const auto& address : prefill_addrs_) {
            prefill_config_.worker_addrs.push_back(address + ":12345");
        }
        token_ids_               = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
        kv_ready_token_count_    = 12;
        input_length_            = 4;
        routing_.unique_key      = "write-scheduler";
        routing_.request_id      = 17;
        routing_.prefill_tp_size = 2;
        routing_.prefill_cp_size = 1;
        routing_.prefill_addr    = {"127.0.0.1", static_cast<uint32_t>(ports_.front())};
        keys_                    = calculateCacheKeys(token_ids_.data(), 12, 4);
        ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, {keys_.front()}).success);
        prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
        ASSERT_TRUE(prefill_->init());
        for (int i = 0; i < 2; ++i) {
            prefill_workers_[i].registered = [this, i](const auto& input) {
                auto ready = input;
                ready.set_worker_rank(i);
                return prefill_->workerRegistered(ready);
            };
        }
        prefill_workers_[0].handshake = [this](const auto& request, auto& response, auto cancelled) {
            prefill_->handleWrite(request, response, cancelled);
        };
        createDecode();
        prefill_workers_[0].cancel_write = [this](const auto& request, auto& response) {
            prefill_->cancelWrite(request, response);
        };
        decode_workers_[0].ready = [this](const auto& request, auto& response) {
            decode_->handleWriteReady(request, response);
        };
    }

    void TearDown() override {
        if (load_barrier_) {
            load_barrier_->release();
        }
        for (int i = 0; i < 2; ++i) {
            prefill_workers_[i].omit_status = false;
            decode_workers_[i].omit_status  = false;
        }
        completeAll(true);
        decode_.reset();
        for (const auto& worker : decode_workers_) {
            EXPECT_EQ(worker.lease_queries, 0);
        }
        prefill_.reset();
        for (auto& server : servers_) {
            server->Shutdown();
            server->Wait();
        }
    }

    void launch(WriteSchedulerService& service, std::vector<std::string>& addresses) {
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        servers_.push_back(std::move(server));
        ports_.push_back(port);
        addresses.push_back("127.0.0.1:" + std::to_string(port));
    }

    P2PConnectorSchedulerConfig schedulerConfig(RoleType role, const std::vector<std::string>& addresses) {
        P2PConnectorSchedulerConfig config;
        config.topology                                     = config_.topologyPtr();
        config.parallelism_config.tp_size                   = 2;
        config.parallelism_config.role_type                 = role;
        config.role_type                                    = role;
        config.worker_grpc_addrs                            = addresses;
        config.p2p_resource_store_timeout_check_interval_ms = 5;
        config.p2p_cancel_broadcast_timeout_ms              = 100;
        config.p2p_writeback_timeout_ms                     = 4000;
        return config;
    }

    void createDecode() {
        P2PConnectorWorkerConfig worker_config;
        worker_config.topology = decode_config_.topology;
        decode_source_worker_  = std::make_unique<P2PWorkerDecodeWrite>(worker_config, nullptr, nullptr);
        decode_ = std::make_unique<P2PSchedulerDecodeWrite>(decode_config_, decode_client_, *decode_source_worker_);
        ASSERT_TRUE(decode_->init());
    }

    static bool await(const std::function<bool()>& predicate, int timeout_ms = 5000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    }

    KVCacheResourcePtr source() {
        auto resource = std::make_shared<KVCacheResource>();
        resource->initGroups(config_.topologyPtr());
        resource->setCacheKeys(calculateCacheKeys(token_ids_.data(), token_ids_.size(), 4));
        resource->mutableBlockIds(0).assign({7, 8, 9, 10});
        return resource;
    }

    std::shared_ptr<P2PConnectorAsyncWriteContext> asyncWrite(KVCacheResourcePtr resource) {
        return decode_->asyncWrite(std::move(resource), token_ids_, input_length_, kv_ready_token_count_, routing_);
    }

    P2PConnectorStartWriteRequestPB request() {
        P2PConnectorStartWriteRequestPB request;
        request.set_unique_key(routing_.unique_key);
        request.set_request_id(17);
        request.set_timeout_ms(4000);
        request.set_deadline_ms(currentTimeMs() + 4000);
        request.set_decode_tp_size(2);
        request.set_decode_addr(decode_addrs_.front());
        request.set_input_length(4);
        request.set_layout_digest(ShardLayoutFactory::writebackLayoutDigest(*config_.topologyPtr(), 2));
        for (const auto token : token_ids_) {
            request.add_token_ids(token);
        }
        for (const auto key : keys_) {
            request.add_cache_keys(key);
        }
        return request;
    }

    void completeAll(bool draining = false) {
        for (int i = 0; i < 2; ++i) {
            prefill_workers_[i].completeAll(!draining, draining);
            decode_workers_[i].completeAll(!draining, draining);
        }
    }

    P2PConnectorStartWriteResponsePB handleWriteWhileAdmissionBlocked(bool expire) {
        auto input = request();
        if (expire) {
            input.set_timeout_ms(1000);
        }
        P2PConnectorStartWriteResponsePB response;
        std::atomic<bool>               cancelled{false};
        std::promise<void>              before_admission;
        auto                            reached_admission = before_admission.get_future();
        std::unique_lock<std::mutex>    tree_lock(allocator_->blockTreeCacheOwner()->mutex_);
        auto handling = std::async(std::launch::async, [&]() {
            int checks = 0;
            prefill_->handleWrite(input, response, [&]() {
                // Keep the pre-admission check successful while the tree lock delays admission.
                if (++checks == 2) {
                    before_admission.set_value();
                    return false;
                }
                return cancelled.load();
            });
        });
        const auto reached = reached_admission.wait_for(std::chrono::seconds(2));
        EXPECT_EQ(reached, std::future_status::ready);
        if (reached == std::future_status::ready) {
            if (expire) {
                const auto deadline_ms = currentTimeMs() + input.timeout_ms();
                EXPECT_TRUE(await([&]() { return currentTimeMs() >= deadline_ms; }));
            } else {
                cancelled = true;
            }
        }
        tree_lock.unlock();
        handling.get();
        return response;
    }

    void prepareLocalLoad(block_tree_cache_test::TransferCopyAction action, bool fully_cached = false) {
        prefill_.reset();
        allocator_ = std::make_shared<Allocator>(config_);
        KVCacheConfig tiered;
        tiered.enable_memory_cache  = true;
        tiered.memory_cache_size_mb = 1;
        allocator_->setBlockTreeCacheConfigForTest(tiered);
        ASSERT_TRUE(allocator_->init());
        ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, {keys_[0]}).success);
        const auto                                 cache = allocator_->blockTreeCacheOwner();
        const auto                                 group = cache->groupSets().front();
        const size_t                               count = fully_cached ? keys_.size() : keys_.size() - 1;
        std::vector<std::vector<GroupSetResource>> slots(count, std::vector<GroupSetResource>(1));
        for (size_t i = 1; i < count; ++i) {
            slots[i][0].host_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_NE(slots[i][0].host_block, NULL_BLOCK_IDX);
        }
        cache->tree()->insertNode(CacheKeysType(keys_.begin(), keys_.begin() + count), slots, /*collect_path=*/false, false);
        for (size_t i = 1; i < count; ++i) {
            group->releaseSingleBlock(Tier::HOST, slots[i][0].host_block, BlockTreeRefType::CACHE);
        }
        load_barrier_ = std::make_shared<block_tree_cache_test::CallbackBarrier>();
        load_engine_  = std::make_shared<block_tree_cache_test::ControlledPerRankBlockTransferEngine>(
            cache->groupSets(), action, load_barrier_);
        cache->transfer_dispatcher_->per_rank_engine_ = load_engine_;
        prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
        ASSERT_TRUE(prefill_->init());
    }

    std::shared_ptr<P2PSchedulerPrefillWrite::RequestState> receiveContext() {
        return prefill_->findRequest(routing_.unique_key);
    }

    static bool transferStopped(const std::shared_ptr<P2PSchedulerPrefillWrite::RequestState>& state) {
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->receive_stopped;
    }

    PlanResult mirroredWritePlan() const {
        const auto src = ShardLayoutFactory::fromTopology(
            *config_.topologyPtr(), decode_config_.parallelism_config, RoleType::DECODE);
        const auto dst = ShardLayoutFactory::peerOf(src, 2, false, RoleType::PREFILL);
        return KVCacheTransferPlanner::plan(src, dst, ShardLayoutFactory::tagsOf(*config_.topologyPtr()));
    }

    CacheConfig                                config_;
    P2PConnectorSchedulerConfig                prefill_config_, decode_config_;
    std::vector<int>                           token_ids_;
    int                                        input_length_{0};
    size_t                                     kv_ready_token_count_{0};
    Meta::P2PRoutingContext                    routing_;
    CacheKeysType                              keys_;
    std::shared_ptr<Allocator>                 allocator_;
    WriteSchedulerService                      prefill_workers_[2], decode_workers_[2];
    std::vector<std::unique_ptr<grpc::Server>> servers_;
    std::vector<int>                           ports_;
    std::vector<std::string>                   prefill_addrs_, decode_addrs_;
    std::shared_ptr<P2PBroadcastClient>        prefill_client_, decode_client_;
    std::unique_ptr<P2PSchedulerPrefillWrite>  prefill_;
    std::unique_ptr<P2PWorkerDecodeWrite>                                        decode_source_worker_;
    std::unique_ptr<P2PSchedulerDecodeWrite>   decode_;
    std::shared_ptr<block_tree_cache_test::CallbackBarrier>                      load_barrier_;
    std::shared_ptr<block_tree_cache_test::ControlledPerRankBlockTransferEngine> load_engine_;
};

TEST_F(P2PConnectorWriteSchedulerTest, MirroredRoutesAndAllRanksRequiredBeforePublicationAndSourceRelease) {
    const auto expected = mirroredWritePlan();
    ASSERT_TRUE(expected.ok());
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_NE(context, nullptr);
    const auto source_group = decode_source_worker_->groups_.at(routing_.unique_key);
    EXPECT_EQ(source_group->sourceResource(), weak.lock());
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    for (int rank = 0; rank < 2; ++rank) {
        const auto send    = decode_workers_[rank].lastStart();
        const auto receive = prefill_workers_[rank].lastStart();
        EXPECT_EQ(send.type(), WRITE);
        EXPECT_EQ(receive.type(), HANDLE_WRITE);
        EXPECT_EQ(send.plan_digest(), expected.plan.digest());
        EXPECT_EQ(send.deadline_ms(), decode_workers_[0].lastStart().deadline_ms());
        EXPECT_EQ(receive.deadline_ms(), prefill_workers_[0].lastStart().deadline_ms());
        EXPECT_EQ(send.plan_digest(), receive.plan_digest());
        ASSERT_EQ(send.routes_size(), 1);
        ASSERT_EQ(receive.routes_size(), 1);
        EXPECT_EQ(send.routes(0).route_id(), receive.routes(0).route_id());
        ASSERT_EQ(send.routes(0).layer_blocks_size(), 2);
        for (int layer = 0; layer < 2; ++layer) {
            const auto& s = send.routes(0).layer_blocks(layer);
            const auto& r = receive.routes(0).layer_blocks(layer);
            ASSERT_EQ(s.cache_keys_size(), 2);
            EXPECT_EQ(std::vector<int64_t>(s.cache_keys().begin(), s.cache_keys().end()),
                      std::vector<int64_t>(r.cache_keys().begin(), r.cache_keys().end()));
            for (int i = 0; i < s.cache_keys_size(); ++i) {
                EXPECT_EQ(s.block_ids(i), s.cache_keys(i) == keys_[1] ? 8 : 9);
            }
        }
    }
    prefill_workers_[0].completeAll();
    decode_workers_[0].completeAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(source_group->sourceResource(), weak.lock());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    completeAll();
    ASSERT_TRUE(await(
        [&]() { return context->done() && !context->resourceHoldPending() && prefill_->inflightContextCount() == 0; }));
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    ASSERT_TRUE(await([&]() { return weak.expired(); }));
    EXPECT_EQ(source_group->sourceResource(), nullptr);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
    for (int rank = 0; rank < 2; ++rank) {
        EXPECT_EQ(decode_workers_[rank].lease_queries, 0);
        EXPECT_EQ(prefill_workers_[rank].lease_queries, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, StartWriteSendsRemainingBudgetInsteadOfAnAbsoluteDeadline) {
    DecodeWriteHelper helper;
    auto input = request();
    const auto deadline_ms = currentTimeMs() + 2000;
    decode_workers_[0].ready      = {};
    auto result = helper.write(routing_.prefill_addr.first, routing_.prefill_addr.second, input, deadline_ms);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(await([&]() { return decode_workers_[0].ready_calls == 1; }));
    EXPECT_FALSE(result->done());
    completeAll();
    ASSERT_TRUE(result->waitDone(2000, false));
    ASSERT_TRUE(result->success());
    ASSERT_EQ(result->responses().size(), 1u);
    ASSERT_EQ(result->responses().front().error_code(), ErrorCodePB::NONE_ERROR);
    const auto handshake = prefill_workers_[0].lastHandshake();
    EXPECT_GT(handshake.timeout_ms(), 0);
    EXPECT_LE(handshake.timeout_ms(), 2000);
    EXPECT_LT(handshake.timeout_ms(), input.timeout_ms());
    completeAll();
    ASSERT_TRUE(await([&]() { return prefill_->inflightContextCount() == 0; }));
    input.set_unique_key("expired-before-handshake");
    EXPECT_EQ(helper.write(routing_.prefill_addr.first, routing_.prefill_addr.second, input, currentTimeMs() - 1),
              nullptr);
    EXPECT_EQ(prefill_workers_[0].handshakes, 1);
}

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverBuildsLocalDeadlineFromRemainingBudgetAndLocalLimits) {
    prefill_.reset();
    prefill_config_.p2p_writeback_timeout_ms = 1500;
    prefill_config_.load_cache_timeout_ms = 1000;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    const auto input = request();
    P2PConnectorStartWriteResponsePB response;
    const auto before_ms = currentTimeMs();
    decode_workers_[0].ready                   = {};
    auto handling = std::async(std::launch::async, [&]() { prefill_->handleWrite(input, response); });
    EXPECT_TRUE(await([&]() { return decode_workers_[0].ready_calls == 1; }));
    completeAll();
    handling.get();
    const auto after_ms = currentTimeMs();
    ASSERT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR) << response.error_message();
    for (const auto& worker : prefill_workers_) {
        const auto receive = worker.lastStart();
        EXPECT_GE(receive.deadline_ms(), before_ms + prefill_config_.load_cache_timeout_ms);
        EXPECT_LE(receive.deadline_ms(), after_ms + prefill_config_.load_cache_timeout_ms);
        EXPECT_EQ(receive.request_deadline_ms(), receive.deadline_ms());
    }
    completeAll();
    ASSERT_TRUE(await([&]() { return prefill_->inflightContextCount() == 0; }));
}

TEST_F(P2PConnectorWriteSchedulerTest, KvReadyBoundaryCapsCompleteTokenKeys) {
    kv_ready_token_count_ = 8;
    auto context          = asyncWrite(source());
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(await([&]() { return decode_workers_[1].starts == 1; }));
    const auto handshake = prefill_workers_[0].lastHandshake();
    EXPECT_EQ(handshake.token_ids_size(), 13);
    ASSERT_EQ(handshake.cache_keys_size(), 2);
    EXPECT_EQ(handshake.cache_keys(1), keys_[1]);
    EXPECT_EQ(decode_workers_[0].lastStart().routes(0).layer_blocks(0).cache_keys_size(), 1);
    completeAll();
    ASSERT_TRUE(await([&]() { return context->done() && prefill_->inflightContextCount() == 0; }));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 2u);
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyCachedHandshakeSkipsWorkersAndReleasesSource) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, keys_).success);
    const auto                     free    = allocator_->freeBlocksNum();
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
    EXPECT_TRUE(context->success());
    EXPECT_EQ(prefill_workers_[0].starts, 0);
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyCachedHandshakeRejectsDeadlineReachedDuringAdmission) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, keys_).success);
    const auto free     = allocator_->freeBlocksNum();
    const auto response = handleWriteWhileAdmissionBlocked(/*expire=*/true);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::GENERATE_TIMEOUT));
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyCachedHandshakeRejectsCancellationDuringAdmission) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, keys_).success);
    const auto free     = allocator_->freeBlocksNum();
    const auto response = handleWriteWhileAdmissionBlocked(/*expire=*/false);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::CANCELLED));
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyLocalHandshakeCancellationAbortsUnsubmittedLoad) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed, true);
    const auto free     = allocator_->freeBlocksNum();
    const auto response = handleWriteWhileAdmissionBlocked(/*expire=*/false);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::CANCELLED));
    EXPECT_EQ(load_engine_->submittedBatchCount(), 0u);
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, CancelBeforeReceiveSubmissionWaitsForLocalLoadAndReleasesTargets) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    const auto coordinator        = allocator_->blockTreeCacheOwner()->loader_.load_context_coordinator_;
    const auto commit             = coordinator->commit_callback_;
    coordinator->commit_callback_ = [&](const auto& load) {
        P2PConnectorCancelWriteRequestPB cancel;
        cancel.set_unique_key(routing_.unique_key);
        cancel.set_deadline_ms(currentTimeMs() + 4000);
        P2PConnectorStartWriteResponsePB response;
        prefill_->cancelWrite(cancel, response);
        EXPECT_FALSE(response.stopped());
        return commit(load);
    };
    const auto                       input = request();
    P2PConnectorStartWriteResponsePB response;
    auto handling = std::async(std::launch::async, [&]() { prefill_->handleWrite(input, response); });
    auto cleanup  = std::shared_ptr<void>(nullptr, [&](void*) {
        load_barrier_->release();
        if (auto receive = receiveContext()) {
            receive->fail(ErrorInfo(ErrorCode::CANCELLED, "test cleanup"));
        }
        coordinator->commit_callback_ = commit;
    });
    ASSERT_TRUE(load_barrier_->waitUntilEnteredFor(1, std::chrono::seconds(2)));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = contextResource(receive)->blocks(0);
    const auto pool   = allocator_->getDeviceBlockPool();
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_TRUE(pool->isAllocated(blocks.back()));
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
    }
    load_barrier_->release();
    const auto finished = handling.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(finished, std::future_status::ready);
    if (finished != std::future_status::ready) {
        receive->fail(ErrorInfo(ErrorCode::CANCELLED, "test cleanup"));
    }
    handling.get();
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::CANCELLED));
    EXPECT_TRUE(response.stopped());
    EXPECT_FALSE(receive->resourceHoldPending());
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_FALSE(pool->isAllocated(blocks.back()));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 2u);
}

TEST_F(P2PConnectorWriteSchedulerTest, LocalLoadRunsAlongsideReceiveAndPublicationWaitsForBoth) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = contextResource(receive)->blocks(0);
    const auto pool   = allocator_->getDeviceBlockPool();
    EXPECT_EQ(pool->refCount(blocks[0]), 2u);
    for (int rank = 0; rank < 2; ++rank) {
        const auto request = decode_workers_[rank].lastStart();
        ASSERT_EQ(request.routes_size(), 1);
        for (const auto& layer : request.routes(0).layer_blocks()) {
            ASSERT_EQ(layer.cache_keys_size(), 1);
            EXPECT_EQ(layer.cache_keys(0), keys_[2]);
        }
    }
    completeAll();
    ASSERT_TRUE(await([&]() { return transferStopped(receive); }));
    EXPECT_FALSE(receiveDone(receive));
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_TRUE(receiveSuccess(receive)) << receive->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
    EXPECT_EQ(pool->refCount(blocks[0]), 1u);
    EXPECT_EQ(pool->refCount(blocks[1]), 1u);
    EXPECT_EQ(pool->refCount(blocks[2]), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, LocalLoadFinishesFirstButReceiveStillHoldsResources) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = contextResource(receive)->blocks(0);
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return allocator_->devicePrefixBlocksForTest(keys_) == 2; }));
    EXPECT_FALSE(receiveDone(receive));
    EXPECT_EQ(allocator_->getDeviceBlockPool()->refCount(blocks[1]), 2u);
    completeAll();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_TRUE(receiveSuccess(receive));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, LocalLoadFailureCancelsReceiveAndDoesNotPublishTail) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Fail);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = contextResource(receive)->blocks(0);
    const auto pool   = allocator_->getDeviceBlockPool();
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return prefill_workers_[0].cancels > 0; }));
    EXPECT_TRUE(receiveDone(receive));
    EXPECT_FALSE(receiveSuccess(receive));
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_TRUE(pool->isAllocated(blocks[2]));
    completeAll();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    EXPECT_FALSE(pool->isAllocated(blocks[1]));
    EXPECT_FALSE(pool->isAllocated(blocks[2]));
    EXPECT_EQ(pool->refCount(blocks[0]), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, CancellationKeepsResourceUntilLocalLoadAlsoStops) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    const auto blocks = contextResource(receive)->blocks(0);
    const auto pool   = allocator_->getDeviceBlockPool();
    receive->fail(ErrorInfo(ErrorCode::CANCELLED, "test cancel"));
    completeAll();
    ASSERT_TRUE(await([&]() { return transferStopped(receive); }));
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_TRUE(pool->isAllocated(blocks[1]));
    EXPECT_TRUE(pool->isAllocated(blocks[2]));
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending(); }));
    EXPECT_FALSE(receiveSuccess(receive));
    EXPECT_FALSE(pool->isAllocated(blocks[2]));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 2u);
}

TEST_F(P2PConnectorWriteSchedulerTest, FullyLocalSuffixSkipsWorkersButWaitsForLoad) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed, true);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return load_engine_->submittedBatchCount() == 1; }));
    EXPECT_FALSE(context->done());
    EXPECT_EQ(prefill_workers_[0].starts, 0);
    EXPECT_EQ(decode_workers_[0].starts, 0);
    auto receive = receiveContext();
    ASSERT_NE(receive, nullptr);
    EXPECT_FALSE(receiveDone(receive));
    EXPECT_TRUE(receive->resourceHoldPending());
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return !receive->resourceHoldPending() && context->done(); }));
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_TRUE(receiveSuccess(receive)) << receive->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, ConcurrentWritesJoinLocalLoadAndPublishIdempotently) {
    prepareLocalLoad(block_tree_cache_test::TransferCopyAction::Succeed);
    auto first = asyncWrite(source());
    ASSERT_TRUE(await([&]() {
        return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1
               && load_engine_->submittedBatchCount() == 1;
    }));
    auto first_receive = receiveContext();
    ASSERT_NE(first_receive, nullptr);
    const auto first_blocks = contextResource(first_receive)->blocks(0);

    routing_.unique_key = "second-write-scheduler";
    auto second         = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 2 && decode_workers_[1].starts == 2; }));
    auto second_receive = receiveContext();
    ASSERT_NE(second_receive, nullptr);
    const auto second_blocks = contextResource(second_receive)->blocks(0);
    EXPECT_EQ(first_blocks[0], second_blocks[0]);
    EXPECT_EQ(first_blocks[1], second_blocks[1]);
    EXPECT_NE(first_blocks[2], second_blocks[2]);
    EXPECT_EQ(load_engine_->submittedBatchCount(), 1u);
    const auto pool = allocator_->getDeviceBlockPool();
    EXPECT_EQ(pool->refCount(first_blocks[0]), 3u);
    EXPECT_EQ(pool->refCount(first_blocks[1]), 3u);

    completeAll();
    load_barrier_->release();
    ASSERT_TRUE(
        await([&]() { return !first_receive->resourceHoldPending() && !second_receive->resourceHoldPending(); }));
    EXPECT_TRUE(receiveSuccess(first_receive)) << first_receive->errorInfo().ToString();
    EXPECT_TRUE(receiveSuccess(second_receive)) << second_receive->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
    EXPECT_EQ(pool->refCount(first_blocks[0]), 1u);
    EXPECT_EQ(pool->refCount(first_blocks[1]), 1u);
    EXPECT_NE(pool->isAllocated(first_blocks[2]), pool->isAllocated(second_blocks[2]));
}

TEST_F(P2PConnectorWriteSchedulerTest, InvalidPrefillAdmissionDoesNotAllocateOrRegister) {
    const auto                                                               free      = allocator_->freeBlocksNum();
    const std::vector<std::function<void(P2PConnectorStartWriteRequestPB&)>> mutations = {
        [](auto& r) { r.set_layout_digest(r.layout_digest() + 1); },
        [](auto& r) { r.set_decode_tp_size(1); },
        [](auto& r) { r.set_cache_keys(2, 999); },
        [](auto& r) { r.add_cache_keys(999); },
        [](auto& r) { r.set_input_length(100); },
        [](auto& r) { r.set_timeout_ms(0); },
        [](auto& r) { r.set_timeout_ms(-1); },
        [](auto& r) { r.set_timeout_ms(int64_t(std::numeric_limits<int>::max()) + 1); },
        [](auto& r) { r.set_input_length(12); }};
    for (const auto& mutate : mutations) {
        auto input = request();
        mutate(input);
        P2PConnectorStartWriteResponsePB response;
        prefill_->handleWrite(input, response);
        EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR) << input.DebugString();
    }
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, LostDecodeStartResponseRetainsSourceUntilEveryRankStops) {
    decode_workers_[1].fail_start          = true;
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(
        await([&]() { return context->done() && decode_workers_[0].cancels > 0 && decode_workers_[1].cancels > 0; }));
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(weak.expired());
    EXPECT_TRUE(context->resourceHoldPending());
    completeAll();
    ASSERT_TRUE(await([&]() { return !context->resourceHoldPending() && weak.expired(); }));
    EXPECT_FALSE(context->success());
}

TEST_F(P2PConnectorWriteSchedulerTest, PartialReceiveStartFailureOutlivesHandshakeAndBusinessDeadline) {
    prefill_workers_[1].fail_start = true;
    decode_.reset();
    decode_config_.p2p_writeback_timeout_ms = 200;
    createDecode();
    const auto                     free    = allocator_->freeBlocksNum();
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free - 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free - 2);
    completeAll();
    ASSERT_TRUE(await([&]() { return prefill_->inflightContextCount() == 0 && weak.expired(); }));
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, LostFinalResponseStillConfirmsPeerStop) {
    prefill_workers_[0].lose_handshake     = true;
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    EXPECT_FALSE(weak.expired());
    completeAll();
    ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
    EXPECT_FALSE(context->success());
    EXPECT_GT(prefill_workers_[0].peer_cancels, 0);
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, UnsupportedGroupsAndCpAreRejectedBeforeHandshake) {
    auto groups                         = config_.topologyPtr()->groups();
    groups[0].policy.group_type         = CacheGroupType::SWA;
    groups[0].policy.active_tail_blocks = 2;
    const auto unsupported              = CacheTopology::create(groups, config_.topologyPtr()->layers());
    EXPECT_TRUE(ShardLayoutFactory::validateWritebackLayout(*unsupported, decode_config_.parallelism_config, 2)
                    .hasError());
    auto cp                               = decode_config_.parallelism_config;
    cp.prefill_cp_config.kv_cache_sharded = true;
    EXPECT_TRUE(ShardLayoutFactory::validateWritebackLayout(*config_.topologyPtr(), cp, 2).hasError());
    decode_.reset();
    decode_config_.topology = unsupported;
    createDecode();
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(prefill_workers_[0].handshakes, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, LastSampleAndMtpAcceptanceDoNotOverstateReadyKv) {
    for (const auto& [token_count, ready_count] :
         std::vector<std::pair<int, int>>{{7, 7}, {8, 7}, {9, 8}, {16, 15}, {20, 17}}) {
        token_ids_.clear();
        for (int i = 0; i < token_count; ++i) {
            token_ids_.push_back(i + 1);
        }
        kv_ready_token_count_         = ready_count;
        routing_.unique_key           = "boundary-" + std::to_string(token_count);
        prefill_workers_[0].handshake = [&](const auto& request, auto& response, auto) {
            EXPECT_EQ(request.cache_keys_size(), ready_count / 4);
            response.set_stopped(true);
            response.set_accepted_start_block(request.cache_keys_size());
            const auto plan = mirroredWritePlan();
            response.set_plan_digest(plan.plan.digest());
        };
        auto context = asyncWrite(source());
        ASSERT_TRUE(await([&]() { return context->done(); }));
        EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    }
    EXPECT_EQ(decode_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, InvalidWriteInputAndCancellationSkipHandshake) {
    auto context = decode_->asyncWrite(source(), token_ids_, input_length_, token_ids_.size() + 1, routing_);
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    auto invalid_cp_routing            = routing_;
    invalid_cp_routing.unique_key     += "-invalid-cp";
    invalid_cp_routing.prefill_cp_size = 0;
    context = decode_->asyncWrite(
        source(), token_ids_, input_length_, kv_ready_token_count_, std::move(invalid_cp_routing));
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    auto owner            = source();
    owner->cacheKeys()[1] = 999;
    auto bad_keys_routing = routing_;
    bad_keys_routing.unique_key += "-bad-keys";
    context =
        decode_->asyncWrite(
            std::move(owner), token_ids_, input_length_, kv_ready_token_count_, std::move(bad_keys_routing));
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(prefill_workers_[0].handshakes, 0);
    auto                             req = request();
    P2PConnectorStartWriteResponsePB response;
    prefill_->handleWrite(req, response, []() { return true; });
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, MultipleFullGroupsRollBackEarlierAllocationsOnFailure) {
    auto config                        = config_;
    auto first                         = config.specForGroup(0)->clone();
    auto second                        = first->clone();
    first->tag                         = "first";
    second->tag                        = "second";
    config.use_independent_block_pools = true;
    config.fromGroupedSpecs(
        {first, second}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::FULL}, {"first", "second"});
    config.setGroupBlockLayout({8, 2}, {first->block_size_bytes(), second->block_size_bytes()}, {0, 0});
    using MultiAllocator = test::BlockTreeCacheTestAllocator<HybridPoolKVCacheAllocator>;
    auto allocator = std::make_shared<MultiAllocator>(config, AllocationType::DEVICE, nullptr, 0, RoleType::PREFILL);
    ASSERT_TRUE(allocator->init());
    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    const auto first_free  = pools[0]->freeBlocksNum();
    const auto second_free = pools[1]->freeBlocksNum();
    KVCacheResourcePtr                resource;
    size_t                            start = 0;
    std::shared_ptr<LoadAsyncContext> load_context;
    EXPECT_TRUE(allocator->admitWriteBackDecodeCache(keys_, 0, resource, start, load_context).hasError());
    EXPECT_EQ(resource, nullptr);
    EXPECT_EQ(pools[0]->freeBlocksNum(), first_free);
    EXPECT_EQ(pools[1]->freeBlocksNum(), second_free);
    EXPECT_TRUE(allocator->blockTreeCacheOwner()->tree()->findNode(keys_).empty());
}

TEST_F(P2PConnectorWriteSchedulerTest, InvalidReturnedRangesAndEndpointsNeverReachSenders) {
    const auto plan = mirroredWritePlan();
    const std::vector<std::function<void(P2PConnectorWriteReadyRequestPB&)>> mutations = {
        [](auto& r) { r.set_accepted_start_block(-1); },
        [](auto& r) { r.set_accepted_block_count(-1); },
        [](auto& r) { r.set_accepted_block_count(3); },
        [](auto& r) {
            r.set_accepted_start_block(0);
            r.set_accepted_block_count(3);
        },
        [](auto& r) { r.set_accepted_block_count(0); },
        [](auto& r) { r.set_plan_digest(0); },
        [](auto& r) { r.clear_prefill_worker_transfer_addrs(); },
        [](auto& r) { r.set_prefill_worker_transfer_addrs(0, "bad-endpoint"); }};
    for (size_t i = 0; i < mutations.size(); ++i) {
        prefill_workers_[0].handshake = [&, i](const auto& input, auto& response, auto) {
            P2PConnectorWriteReadyRequestPB ready;
            ready.set_unique_key(input.unique_key());
            ready.set_deadline_ms(input.deadline_ms());
            ready.set_accepted_start_block(1);
            ready.set_accepted_block_count(2);
            ready.set_plan_digest(plan.plan.digest());
            ready.add_prefill_worker_transfer_addrs("127.0.0.1:12345");
            ready.add_prefill_worker_transfer_addrs("[::1]:12345");
            mutations[i](ready);
            DecodeWriteHelper helper;
            auto              result = helper.ready(input.decode_addr(), ready, input.deadline_ms());
            EXPECT_NE(result, nullptr);
            if (result) {
                EXPECT_TRUE(result->waitDone(1000, false));
            }
            response.set_stopped(true);
            response.set_error_code(ErrorCodePB::INVALID_PARAMS);
        };
        auto iteration_routing = routing_;
        iteration_routing.unique_key += std::to_string(i);
        auto                           owner   = source();
        std::weak_ptr<KVCacheResource> weak    = owner;
        auto                           context = decode_->asyncWrite(
            std::move(owner), token_ids_, input_length_, kv_ready_token_count_, std::move(iteration_routing));
        ASSERT_TRUE(await([&]() { return context->done() && weak.expired(); }));
        EXPECT_FALSE(context->success()) << i;
    }
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, SendCompletionRetainsSourceUntilReceiverStops) {
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    pauseChecker(decode_->checker_);
    decode_workers_[0].completeAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_FALSE(weak.expired());
    decode_workers_[1].completeAll();
    EXPECT_FALSE(weak.expired());
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    context->checkDone();
    EXPECT_FALSE(context->done());
    for (auto& worker : prefill_workers_)
        worker.completeAll();
    ASSERT_TRUE(await([&]() { context->checkDone(); return context->done(); }));
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, PeerTimeoutRetainsSourceUntilReceiverStops) {
    decode_.reset();
    decode_config_.p2p_writeback_timeout_ms = 150;
    createDecode();
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    for (auto& worker : decode_workers_)
        worker.completeAll();
    ASSERT_TRUE(await([&]() { return context->done() && prefill_workers_[0].peer_cancels > 0; }));
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(weak.expired());
    for (auto& worker : prefill_workers_)
        worker.completeAll();
    ASSERT_TRUE(await([&]() { return !context->resourceHoldPending(); }));
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, CancelledSendRetainsSourceAfterAllRanksStopUntilReceiverStops) {
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    context->cancel();
    ASSERT_TRUE(await([&]() { return context->done() && decode_workers_[0].cancels > 0
                                    && decode_workers_[1].cancels > 0; }));
    EXPECT_FALSE(weak.expired());
    decode_workers_[0].completeAll(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_FALSE(weak.expired());
    decode_workers_[1].completeAll(false);
    EXPECT_FALSE(weak.expired());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    for (auto& worker : prefill_workers_)
        worker.completeAll(false);
    ASSERT_TRUE(await([&]() { return !context->resourceHoldPending(); }));
    EXPECT_TRUE(weak.expired());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverStopRetainsSourceUntilEverySenderStops) {
    auto owner = source();
    std::weak_ptr<KVCacheResource> weak = owner;
    auto context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    context->cancel();
    for (auto& worker : prefill_workers_) {
        worker.completeAll(false);
    }
    ASSERT_TRUE(await([&]() { return context->done() && prefill_->inflightContextCount() == 0; }));
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(weak.expired());
    decode_workers_[0].completeAll(false);
    ASSERT_TRUE(await([&]() { return decode_workers_[1].cancels > 0; }));
    EXPECT_FALSE(weak.expired());
    decode_workers_[1].completeAll(false);
    ASSERT_TRUE(await([&]() { return weak.expired(); }));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, FailedSendCallbackRetainsSourceWhileRemoteReadIsActive) {
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    // A failed RDMA notification RPC can complete the sender while the receiver still reads.
    for (auto& worker : decode_workers_)
        worker.completeAll(false);
    ASSERT_TRUE(await([&]() { return context->done() && prefill_workers_[0].peer_cancels > 0; }));
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    prefill_workers_[0].completeAll(false);
    ASSERT_TRUE(await([&]() { return prefill_workers_[0].lease_queries > 0; }));
    EXPECT_FALSE(weak.expired());
    prefill_workers_[1].completeAll(false);
    ASSERT_TRUE(await([&]() { return !context->resourceHoldPending() && weak.expired(); }));
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
    EXPECT_EQ(decode_workers_[0].lease_queries, 0);
    EXPECT_EQ(decode_workers_[1].lease_queries, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, WorkerStartRpcTimeoutStillConfirmsBothSidesStopped) {
    decode_.reset();
    decode_config_.p2p_writeback_timeout_ms = 150;
    createDecode();
    decode_workers_[0].start_delay_ms      = 250;
    auto                           owner   = source();
    std::weak_ptr<KVCacheResource> weak    = owner;
    auto                           context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&]() { return decode_workers_[1].starts == 1; }));
    ASSERT_TRUE(await([&]() { return context->done() && decode_workers_[0].cancels > 0; }));
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(weak.expired());
    completeAll(true);
    ASSERT_TRUE(await([&]() { return !context->resourceHoldPending() && weak.expired(); }));
}

TEST_F(P2PConnectorWriteSchedulerTest, DuplicateReadyDoesNotStartWorkersAgain) {
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    const auto                       ready = decode_workers_[0].lastReady();
    P2PConnectorWriteReadyResponsePB response;
    decode_->handleWriteReady(ready, response);
    EXPECT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR);
    auto changed = ready;
    changed.set_accepted_block_count(1);
    decode_->handleWriteReady(changed, response);
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(decode_workers_[0].starts, 1);
    EXPECT_EQ(decode_workers_[1].starts, 1);
    context->cancel();
    decode_->handleWriteReady(ready, response);
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    completeAll();
    ASSERT_TRUE(await([&]() { return !context->resourceHoldPending(); }));
}

TEST_F(P2PConnectorWriteSchedulerTest, CancelBeforeStartWriteFencesLateStartWithoutAllocation) {
    const auto                       free = allocator_->freeBlocksNum();
    P2PConnectorCancelWriteRequestPB cancel;
    cancel.set_unique_key(routing_.unique_key);
    cancel.set_deadline_ms(currentTimeMs() + 4000);
    P2PConnectorStartWriteResponsePB response;
    prefill_->cancelWrite(cancel, response);
    EXPECT_TRUE(response.stopped());
    prefill_->handleWrite(request(), response);
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_TRUE(response.stopped());
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
    EXPECT_EQ(decode_workers_[0].ready_calls, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, RegistrationBarrierChecksEveryRankAndRejectsLateReady) {
    prefill_workers_[1].registered = [](const auto&) { return ErrorInfo::OkStatus(); };
    decode_workers_[0].ready = [](const auto&, auto& response) { response.set_error_code(ErrorCodePB::NONE_ERROR); };
    auto                             input = request();
    P2PConnectorStartWriteResponsePB response;
    auto handling = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    EXPECT_TRUE(await([&] { return prefill_workers_[0].starts == 1 && prefill_workers_[1].starts == 1; }));
    const auto state = receiveContext();
    EXPECT_NE(state, nullptr);
    auto registration = prefill_workers_[0].lastStart();
    EXPECT_TRUE(prefill_->workerRegistered(registration).ok());
    EXPECT_EQ(decode_workers_[0].ready_calls, 0);
    registration.set_worker_rank(2);
    EXPECT_TRUE(prefill_->workerRegistered(registration).hasError());
    registration.set_worker_rank(1);
    registration.set_plan_digest(registration.plan_digest() + 1);
    EXPECT_TRUE(prefill_->workerRegistered(registration).hasError());
    EXPECT_EQ(decode_workers_[0].ready_calls, 0);
    registration.set_plan_digest(registration.plan_digest() - 1);
    EXPECT_TRUE(prefill_->workerRegistered(registration).ok());
    EXPECT_TRUE(await([&] { return decode_workers_[0].ready_calls == 1; }));
    completeAll();
    handling.get();
    EXPECT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_TRUE(prefill_->workerRegistered(registration).hasError());
}

TEST(P2PConnectorAsyncWriteContextTest, CompletionCallbacksRunOnceOutsideStateLock) {
    auto context        = makeWriteContext(nullptr, "completion-callback", currentTimeMs() + 4000, nullptr, 100, 1000);
    int callback_count = 0;
    context->onDone([&](ErrorInfo error) {
        ++callback_count;
        EXPECT_TRUE(error.ok());
        EXPECT_TRUE(context->done());
        EXPECT_TRUE(context->success());
        context->onDone([&](ErrorInfo late_error) {
            ++callback_count;
            EXPECT_TRUE(late_error.ok());
        });
    });
    EXPECT_EQ(callback_count, 0);
    context->finishWithoutTransfer();
    EXPECT_EQ(callback_count, 2);
    context->checkDone();
    context->cancel();
    context->onDone({});
    EXPECT_EQ(callback_count, 2);
}

TEST(P2PConnectorAsyncWriteContextTest, CancellationNotifiesBeforeResourceHoldIsReleased) {
    auto context = makeWriteContext(nullptr, "cancel-callback", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    int callback_count = 0;
    context->onDone([&](ErrorInfo error) {
        ++callback_count;
        EXPECT_EQ(error.code(), ErrorCode::CANCELLED);
        EXPECT_TRUE(context->done());
        EXPECT_TRUE(context->resourceHoldPending());
    });
    context->cancel();
    EXPECT_EQ(callback_count, 1);
    context->finishWithoutTransfer(context->errorInfo());
    EXPECT_FALSE(context->resourceHoldPending());
    context->onDone([&](ErrorInfo error) {
        ++callback_count;
        EXPECT_EQ(error.code(), ErrorCode::CANCELLED);
    });
    EXPECT_EQ(callback_count, 2);
}

TEST(P2PConnectorAsyncWriteContextTest, CancelPreservesEarlierWorkerErrorUntilReceiversStop) {
    auto resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner = resource;
    auto                           context =
        makeWriteContext(std::move(resource), "cancel-after-worker-error", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    ASSERT_TRUE(context->beginWorkerCall());
    auto failed = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    failed->response.mutable_p2p_response()->set_error_code(transErrorCodeToRPC(ErrorCode::CACHE_STORE_STORE_FAILED));
    failed->response.mutable_p2p_response()->set_error_message("receiver failed before cancellation");
    auto pending = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    pending->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);
    auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{failed, pending});
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("cancel-after-worker-error", workers, false));
    int callback_count = 0;
    context->onDone([&](ErrorInfo error) {
        ++callback_count;
        EXPECT_EQ(error.code(), ErrorCode::CACHE_STORE_STORE_FAILED);
        EXPECT_NE(error.ToString().find("receiver failed before cancellation"), std::string::npos);
    });
    workers->finishRank(0, true);
    ASSERT_FALSE(workers->done());
    ASSERT_TRUE(context->errorInfo().ok());

    context->cancel();
    EXPECT_EQ(callback_count, 1);
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    workers->finishRank(1, true);
    context->checkDone();
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());

    auto stopped = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
    stopped->response.set_stopped(true);
    auto peer = std::make_shared<DecodeWriteHelper::Result>(
        std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{stopped});
    context->setPeerCallResult(peer);
    peer->finishRank(0, true);
    context->checkDone();
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(callback_count, 1);
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
}

TEST(P2PConnectorAsyncWriteContextTest, CancelPreservesCompletedStartWriteResult) {
    for (const auto expected : {ErrorCode::NONE_ERROR, ErrorCode::CACHE_STORE_STORE_FAILED}) {
        auto context =
            makeWriteContext(nullptr, "cancel-after-peer-result", currentTimeMs() + 4000, nullptr, 100, 1000);
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        auto peer = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
        peer->request.set_unique_key("cancel-after-peer-result");
        peer->response.set_error_code(transErrorCodeToRPC(expected));
        peer->response.set_stopped(true);
        if (expected != ErrorCode::NONE_ERROR) {
            peer->response.set_error_message("prefill failed before cancellation");
        }
        auto result = std::make_shared<DecodeWriteHelper::Result>(
            std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{peer});
        context->setPeerCallResult(result);
        result->finishRank(0, true);
        int callback_count = 0;
        context->onDone([&](ErrorInfo error) {
            ++callback_count;
            EXPECT_EQ(error.code(), expected);
        });

        context->cancel();
        EXPECT_EQ(context->errorInfo().code(), expected);
        if (expected != ErrorCode::NONE_ERROR) {
            EXPECT_NE(context->errorInfo().ToString().find("prefill failed before cancellation"), std::string::npos);
        }
        EXPECT_TRUE(context->done());
        EXPECT_EQ(context->success(), expected == ErrorCode::NONE_ERROR);
        EXPECT_FALSE(context->resourceHoldPending());
        context->cancel();
        EXPECT_EQ(callback_count, 1);
    }
}

TEST(P2PConnectorAsyncWriteContextTest, CompletedBeforeDeadlineObservationRemainsSuccessful) {
    for (const bool empty_range : {false, true}) {
        auto                           resource = std::make_shared<KVCacheResource>();
        std::weak_ptr<KVCacheResource> owner    = resource;
        auto                           context  = makeWriteContext(
            std::move(resource), "completed-before-observation", currentTimeMs() + 4000, nullptr, 100, 1000);
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        if (!empty_range) {
            ASSERT_TRUE(context->beginWorkerCall());
            auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
            worker->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);
            auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
                std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
            context->setCallResults(
                std::make_shared<P2PBroadcastClient::Result>("completed-before-observation", workers));
            workers->finishRank(0, true);
        }
        auto peer = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
        peer->response.set_stopped(true);
        peer->response.set_accepted_block_count(empty_range ? 0 : 2);
        auto result = std::make_shared<DecodeWriteHelper::Result>(
            std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{peer});
        context->setPeerCallResult(result);
        result->finishRank(0, true);
        {
            std::lock_guard<std::mutex> lock(context->mutex_);
            context->ready_request_ = empty_range ? "" : "registered";
            context->deadline_ms_   = currentTimeMs() - 1;
        }
        context->checkDone();
        EXPECT_TRUE(context->done());
        EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
        EXPECT_FALSE(context->resourceHoldPending());
        EXPECT_TRUE(owner.expired());
    }
}

TEST(P2PConnectorAsyncWriteContextTest, CompletedSendStillTimesOutWhilePeerIsPending) {
    auto                           resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner    = resource;
    auto                           context =
        makeWriteContext(std::move(resource), "peer-pending-at-deadline", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    ASSERT_TRUE(context->beginWorkerCall());
    auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    worker->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);
    auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("peer-pending-at-deadline", workers));
    auto peer = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
    peer->response.set_stopped(true);
    peer->response.set_accepted_block_count(2);
    auto result = std::make_shared<DecodeWriteHelper::Result>(
        std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{peer});
    context->setPeerCallResult(result);
    workers->finishRank(0, true);
    {
        std::lock_guard<std::mutex> lock(context->mutex_);
        context->ready_request_ = "registered";
        context->deadline_ms_   = currentTimeMs() - 1;
    }
    context->expireTransferDeadlineIfNeeded();
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    result->finishRank(0, true);
    context->checkDone();
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_TRUE(owner.expired());
}

TEST(P2PConnectorAsyncWriteContextTest, StartWriteBusinessErrorParticipatesInFirstErrorOrdering) {
    for (const bool prefill_first : {true, false}) {
        auto                           resource = std::make_shared<KVCacheResource>();
        std::weak_ptr<KVCacheResource> owner    = resource;
        auto                           context =
            makeWriteContext(std::move(resource), "first-business-error", currentTimeMs() + 4000, nullptr, 100, 1000);
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        ASSERT_TRUE(context->beginWorkerCall());
        auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
        worker->response.mutable_p2p_response()->set_error_code(ErrorCodePB::CANCELLED);
        worker->response.mutable_p2p_response()->set_error_message("decode worker failed");
        auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
            std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
        context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("first-business-error", workers));
        auto peer = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
        peer->request.set_unique_key("first-business-error");
        peer->response.set_error_code(transErrorCodeToRPC(ErrorCode::CACHE_STORE_STORE_FAILED));
        peer->response.set_error_message("prefill receive failed");
        peer->response.set_stopped(true);
        auto result = std::make_shared<DecodeWriteHelper::Result>(
            std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{peer});
        context->setPeerCallResult(result);
        if (prefill_first) {
            result->finishRank(0, true);
            workers->finishRank(0, true);
        } else {
            workers->finishRank(0, true);
            result->finishRank(0, true);
        }
        EXPECT_TRUE(result->rpcSuccess());
        EXPECT_FALSE(result->success());
        EXPECT_EQ(result->firstError().error.code(), ErrorCode::CACHE_STORE_STORE_FAILED);
        context->checkDone();
        const auto expected = prefill_first ? ErrorCode::CACHE_STORE_STORE_FAILED : ErrorCode::CANCELLED;
        EXPECT_EQ(context->errorInfo().code(), expected);
        EXPECT_NE(
            context->errorInfo().ToString().find(prefill_first ? "prefill receive failed" : "decode worker failed"),
            std::string::npos);
        EXPECT_TRUE(context->peer_stopped_);
        EXPECT_EQ(context->control_result_, nullptr);
        EXPECT_FALSE(context->resourceHoldPending());
        EXPECT_TRUE(owner.expired());
    }
}

TEST(P2PConnectorAsyncWriteContextTest, ExpiredQueuedWriteReleasesWithoutFailStop) {
    auto resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner = resource;
    auto context = makeWriteContext(std::move(resource), "expired-queued", currentTimeMs() - 1000, nullptr, 100, 1);
    context->expireTransferDeadlineIfNeeded();
    context->checkCancelDone();
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_TRUE(owner.expired());
}

TEST_F(P2PConnectorWriteSchedulerTest, RejectedReceiveSubmissionNeedsNoWorkerCancellation) {
    prefill_->control_pool_->stop();
    P2PConnectorStartWriteResponsePB response;
    prefill_->handleWrite(request(), response);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::OUTPUT_QUEUE_FULL));
    EXPECT_TRUE(response.stopped());
    EXPECT_EQ(prefill_->inflightContextCount(), 0);
    for (const auto& worker : prefill_workers_) {
        EXPECT_EQ(worker.starts, 0);
        EXPECT_EQ(worker.cancels, 0);
        EXPECT_EQ(worker.lease_queries, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, MissingWorkerLeaseStatusTriggersFailStopWithoutReleasingResources) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            const rlimit core_limit{};
            setrlimit(RLIMIT_CORE, &core_limit);
            prefill_.reset();
            prefill_config_.p2p_lease_query_timeout_ms = 100;
            prefill_ =
                std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
            if (!prefill_->init())
                std::_Exit(43);
            for (auto& worker : prefill_workers_) {
                worker.fail_start  = true;
                worker.omit_status = true;
            }
            auto input = request();
            input.set_timeout_ms(100);
            input.set_deadline_ms(currentTimeMs() + 100);
            P2PConnectorStartWriteResponsePB response;
            prefill_->handleWrite(input, response);
            std::_Exit(42);
        },
        ::testing::KilledBySignal(SIGABRT),
        "");
}

TEST_F(P2PConnectorWriteSchedulerTest, MissingSenderAcknowledgmentsFailStopWithoutReleasingSource) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    for (auto& worker : decode_workers_) {
        worker.omit_status = true;
    }
    EXPECT_EXIT({
        const rlimit core_limit{};
        setrlimit(RLIMIT_CORE, &core_limit);
        auto resource = KVCacheResourcePtr(new KVCacheResource, [](auto*) { std::_Exit(42); });
        auto context  = makeWriteContext(
            std::move(resource), "source-stop-unconfirmed", currentTimeMs() + 100, decode_client_, 20, 100);
        context->beginKickoff();
        context->beginWorkerCall();
        context->setCallResults(nullptr);
        P2PConnectorAsyncWriteContextChecker checker;
        if (!checker.init() || !checker.addContext(context))
            std::_Exit(43);
        checker.stop();
        std::_Exit(45);
    }, ::testing::KilledBySignal(SIGABRT), "");
}

TEST_F(P2PConnectorWriteSchedulerTest, CompletedCancelAcknowledgmentIsConsumedBeforeGraceFailStop) {
    completeAll(true);
    const auto deadline = currentTimeMs() + 3000;
    auto       result   = prefill_client_->controlWriteAsync("stopped-before-grace", CANCEL_HANDLE_WRITE, deadline, 1000);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->waitDone(2000, false));
    ASSERT_TRUE(P2PBroadcastClient::leaseStatus(result).allStopped());
    auto context = makeWriteContext(source(), "stopped-before-grace", deadline, prefill_client_, 100, 100);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginWorkerCall());
    context->setCallResults(nullptr);
    {
        std::lock_guard<std::mutex> lock(context->mutex_);
        context->deadline_ms_    = currentTimeMs() - 500;
        context->control_result_ = result;
    }
    context->checkCancelDone();
    context->failStopIfLeaseUnconfirmed();
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED);
}

TEST_F(P2PConnectorWriteSchedulerTest, SlowControlConnectionDoesNotBlockAdmissionOrDeadline) {
    decode_workers_[0].ready = [](const auto&, auto& response) { response.set_error_code(ErrorCodePB::NONE_ERROR); };
    auto first               = request();
    first.set_timeout_ms(200);
    first.set_deadline_ms(currentTimeMs() + 200);
    P2PConnectorStartWriteResponsePB first_response, second_response;
    auto first_call = std::async(std::launch::async, [&] { prefill_->handleWrite(first, first_response); });
    EXPECT_TRUE(await([&] { return prefill_workers_[0].starts == 1 && prefill_workers_[1].starts == 1; }));
    const auto state = receiveContext();
    EXPECT_NE(state, nullptr);
    std::unique_lock<std::mutex> connection_lock(prefill_client_->tp_broadcast_manager_->rpc_pool_->mutex_);
    auto                         second = request();
    second.set_unique_key("new-admission");
    auto second_call = std::async(std::launch::async, [&] { prefill_->handleWrite(second, second_response); });
    EXPECT_TRUE(await([&] { return prefill_->findRequest(second.unique_key()) != nullptr; }, 100));
    EXPECT_TRUE(await([&] { return state->errorInfo().hasError(); }, 1000));
    EXPECT_GE(currentTimeMs(), first.deadline_ms());
    EXPECT_TRUE(state->resourceHoldPending());
    connection_lock.unlock();
    completeAll(true);
    first_call.get();
    second_call.get();
    EXPECT_FALSE(state->resourceHoldPending());
}

TEST_F(P2PConnectorWriteSchedulerTest, QueuedReceiveControlRecomputesRemainingBudget) {
    prefill_.reset();
    prefill_config_.p2p_cancel_broadcast_timeout_ms = 10000;
    prefill_config_.p2p_lease_query_timeout_ms      = 1000;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    decode_workers_[0].ready = [](const auto&, auto& response) { response.set_error_code(ErrorCodePB::NONE_ERROR); };
    auto input               = request();
    input.set_deadline_ms(currentTimeMs() + 2000);
    const auto                       hold_until = prefill_->holdUntilMs(input.deadline_ms());
    P2PConnectorStartWriteResponsePB response;
    auto               call = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    std::promise<void> resume;
    auto               gate    = resume.get_future().share();
    bool               resumed = false;
    auto               cleanup = std::shared_ptr<void>(nullptr, [&](void*) {
        if (!resumed)
            resume.set_value();
        completeAll(true);
    });
    ASSERT_TRUE(await([&] { return decode_workers_[0].ready_calls != 0; }, 1000));
    auto blocked = std::make_shared<std::atomic<size_t>>(0);
    auto pool    = prefill_->control_pool_;
    for (size_t i = 0; i < pool->getThreadNum(); ++i) {
        ASSERT_EQ(pool->pushTask(
                      [gate, blocked] {
                          ++*blocked;
                          gate.wait();
                      },
                      false,
                      false),
                  autil::ThreadPoolBase::ERROR_NONE);
    }
    ASSERT_TRUE(await([&] { return blocked->load() == pool->getThreadNum(); }, 1000));
    P2PConnectorCancelWriteRequestPB cancel;
    cancel.set_unique_key(input.unique_key());
    cancel.set_deadline_ms(input.deadline_ms());
    P2PConnectorStartWriteResponsePB cancel_response;
    prefill_->cancelWrite(cancel, cancel_response);
    ASSERT_TRUE(await([&] { return pool->getItemCount() != 0; }, 1000));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    resume.set_value();
    resumed = true;
    ASSERT_TRUE(await(
        [&] { return prefill_workers_[0].control_deadline_ms != 0 && prefill_workers_[1].control_deadline_ms != 0; },
        1000));
    for (const auto& worker : prefill_workers_) {
        EXPECT_LE(worker.control_deadline_ms.load(), hold_until + 20);
    }
    cleanup.reset();
    call.get();
    EXPECT_TRUE(response.stopped());
}

TEST_F(P2PConnectorWriteSchedulerTest, PeerOnlyCancellationUsesBackoffUntilReceiverStops) {
    auto context = makeWriteContext(source(), "peer-cancel-backoff", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginWorkerCall());
    std::atomic<int>     polls{0};
    std::atomic<bool>    stopped{false};
    std::mutex           times_mutex;
    std::vector<int64_t> times;
    ASSERT_TRUE(context->beginPeerCall([&] {
        {
            std::lock_guard<std::mutex> lock(times_mutex);
            times.push_back(currentTimeMs());
        }
        ++polls;
        auto call = std::make_shared<DecodeWriteHelper::CancelResult::WorkerRpcContext>();
        call->response.set_stopped(stopped.load());
        auto result = std::make_shared<DecodeWriteHelper::CancelResult>(
            std::vector<std::shared_ptr<DecodeWriteHelper::CancelResult::WorkerRpcContext>>{call});
        result->finishRank(0, true);
        return result;
    }));
    auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    worker->response.mutable_p2p_response();
    auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("peer-cancel-backoff", workers));
    workers->finishRank(0, true);
    context->cancel();
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init());
    ASSERT_TRUE(checker.addContext(context));
    EXPECT_TRUE(await([&] { return polls >= 4; }, 1000));
    {
        std::lock_guard<std::mutex> lock(times_mutex);
        if (times.size() >= 4) {
            EXPECT_GE(times[3] - times[0], 60);
        }
    }
    EXPECT_TRUE(context->resourceHoldPending());
    stopped = true;
    EXPECT_TRUE(await([&] { return !context->resourceHoldPending(); }, 1000));
    checker.stop();
}

TEST(P2PConnectorAsyncWriteContextTest, RpcProgressNotifiesInEitherRegistrationOrder) {
    for (const bool result_first : {false, true}) {
        auto context = makeWriteContext(nullptr, "write-notification", currentTimeMs() + 4000, nullptr, 100, 1000);
        auto first = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
        auto second = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
        first->response.mutable_p2p_response()->set_error_code(ErrorCodePB::CANCELLED);
        second->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);
        auto result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
            std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{first, second});
        auto workers = std::make_shared<P2PBroadcastClient::Result>("write-notification", result, false);
        auto notification = std::make_shared<P2PNotification>();
        if (result_first) context->setCallResults(workers);
        context->setNotification(notification);
        if (!result_first) context->setCallResults(workers);
        auto generation = notification->generation();
        result->finishRank(0, false);
        EXPECT_GT(notification->generation(), generation);
        EXPECT_FALSE(result->done());
        generation = notification->generation();
        result->finishRank(1, true);
        EXPECT_GT(notification->generation(), generation);
        EXPECT_TRUE(result->done());
        auto late = std::make_shared<P2PNotification>();
        context->setNotification(late);
        EXPECT_GT(late->generation(), 0u);
    }
}

TEST(P2PConnectorAsyncWriteContextTest, StaleCancelMustNotEraseStoppedStartWriteProof) {
    auto context = std::make_shared<P2PConnectorAsyncWriteContext>(
        "stopped-proof-race", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginWorkerCall());
    ASSERT_TRUE(context->beginPeerCall({}));
    context->cancel();
    auto final_call = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
    final_call->response.set_stopped(true);
    final_call->response.set_error_code(ErrorCodePB::CANCELLED);
    auto final_result = std::make_shared<DecodeWriteHelper::Result>(
        std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{final_call});
    final_result->finishRank(0, true);
    context->setPeerCallResult(final_result);
    context->checkDone();
    ASSERT_TRUE(context->peer_stopped_);
    ASSERT_TRUE(context->resourceHoldPending());
    auto stale_call = std::make_shared<DecodeWriteHelper::CancelResult::WorkerRpcContext>();
    stale_call->response.set_stopped(false);
    stale_call->response.set_error_code(ErrorCodePB::CANCELLED);
    auto stale_result = std::make_shared<DecodeWriteHelper::CancelResult>(
        std::vector<std::shared_ptr<DecodeWriteHelper::CancelResult::WorkerRpcContext>>{stale_call});
    stale_result->finishRank(0, true);
    context->peer_cancel_result_ = stale_result;
    context->checkCancelDone();
    EXPECT_TRUE(context->peer_stopped_) << "stale cancellation erased the newer final stop proof";
    auto local_call = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    local_call->response.mutable_p2p_response();
    auto local_result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{local_call});
    local_result->finishRank(0, true);
    context->control_result_ = local_result;
    context->checkCancelDone();
    EXPECT_FALSE(context->resourceHoldPending()) << "both sides stopped, but source hold is still retained";
}

TEST_F(P2PConnectorWriteSchedulerTest, StopKeepsCheckingWhileKickoffPoolIsBlocked) {
    auto released       = std::make_shared<std::promise<void>>();
    auto release_result = released->get_future();
    auto context =
        makeWriteContext(source(), "stop-with-blocked-kickoff", currentTimeMs() + 4000, nullptr, 100, 1000, [released] {
            released->set_value();
        });
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginWorkerCall());
    auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    worker->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);
    auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>(context->uniqueKey(), workers));
    ASSERT_TRUE(decode_->checker_.addContext(context));

    std::promise<void> resume;
    auto               gate           = resume.get_future().share();
    auto               entered        = std::make_shared<std::promise<void>>();
    auto               entered_result = entered->get_future();
    ASSERT_EQ(decode_->async_write_pool_->pushTask(
                  [gate, entered] {
                      entered->set_value();
                      gate.wait();
                  },
                  false),
              autil::ThreadPoolBase::ERROR_NONE);
    EXPECT_EQ(entered_result.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto stop    = std::async(std::launch::async, [&] { decode_->stop(); });
    auto unblock = std::shared_ptr<void>(nullptr, [&](void*) { resume.set_value(); });
    ASSERT_TRUE(await([&] {
        std::lock_guard<std::mutex> lock(decode_->checker_.mutex_);
        return !decode_->checker_.accepting_;
    }));
    EXPECT_FALSE(decode_->checker_.addContext(
        makeWriteContext(nullptr, "stop-rejects-new", currentTimeMs() + 4000, nullptr, 100, 1000)));
    EXPECT_EQ(stop.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    workers->finishRank(0, true);
    EXPECT_EQ(release_result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_FALSE(context->resourceHoldPending());
    unblock.reset();
    stop.get();
}

TEST(P2PConnectorAsyncWriteContextTest, CheckerWakesAtDeadlineWithoutRpcCompletion) {
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init());
    auto released       = std::make_shared<std::promise<void>>();
    auto release_result = released->get_future();
    auto context =
        makeWriteContext(nullptr, "deadline-notification", currentTimeMs() + 100, nullptr, 100, 1000, [released]() {
            released->set_value();
        });
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginWorkerCall());
    auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    worker->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);
    auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("deadline-notification", workers));
    auto completed = std::make_shared<std::promise<ErrorInfo>>();
    auto result = completed->get_future();
    context->onDone([completed](ErrorInfo error) { completed->set_value(error); });
    ASSERT_TRUE(checker.addContext(context));
    ASSERT_EQ(result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(result.get().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_TRUE(context->resourceHoldPending());
    workers->finishRank(0, true);
    ASSERT_EQ(release_result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_FALSE(context->resourceHoldPending());
}

TEST(P2PConnectorAsyncWriteContextTest, ReadyHandlerCompletionWakesCheckerAfterBothRpcsCompleted) {
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init());
    auto context = makeWriteContext(nullptr, "ready-notification", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    std::promise<void> entered;
    auto entering = entered.get_future();
    std::promise<void> release;
    auto released = release.get_future();
    context->setReadyHandler([&](const auto&) {
        context->beginWorkerCall();
        entered.set_value();
        released.wait();
        return ErrorInfo::OkStatus();
    });
    ASSERT_TRUE(checker.addContext(context));
    P2PConnectorWriteReadyRequestPB ready;
    ready.set_deadline_ms(context->deadlineMs());
    auto accepting = std::async(std::launch::async, [&] { return context->acceptWriteReady(ready); });
    auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) { release.set_value(); });
    ASSERT_EQ(entering.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    worker->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);
    auto workers = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("ready-notification", workers, false));
    auto peer = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
    peer->response.set_stopped(true);
    peer->response.set_accepted_block_count(1);
    auto result = std::make_shared<DecodeWriteHelper::Result>(
        std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{peer});
    context->setPeerCallResult(result);
    workers->finishRank(0, true);
    result->finishRank(0, true);
    auto completed = std::make_shared<std::promise<ErrorInfo>>();
    auto completion = completed->get_future();
    context->onDone([completed](ErrorInfo error) { completed->set_value(error); });
    context->checkDone();
    EXPECT_FALSE(context->done());
    cleanup.reset();
    EXPECT_TRUE(accepting.get().ok());
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(completion.get().ok());
    EXPECT_FALSE(context->resourceHoldPending());
}

}  // namespace
}  // namespace rtp_llm
