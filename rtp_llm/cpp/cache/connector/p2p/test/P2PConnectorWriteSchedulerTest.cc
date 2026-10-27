#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <sys/resource.h>

#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {
namespace {

P2PConnectorSchedulerConfig writeCheckerConfig() {
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1, 2).topologyPtr();
    return config;
}

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
                                                           }, nullptr, group->sourceResource());
}

std::shared_ptr<DecodeWriteHelper::Result>
peerResult(ErrorCode error = ErrorCode::NONE_ERROR, bool stopped = true, int count = 2) {
    auto call = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
    call->response.set_error_code(transErrorCodeToRPC(error));
    call->response.set_error_message(error == ErrorCode::NONE_ERROR ? "" : "prefill transfer failed");
    call->response.set_stopped(stopped);
    call->response.set_accepted_block_count(count);
    return std::make_shared<DecodeWriteHelper::Result>(
        std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{call});
}

void confirmSourceStopped(const std::shared_ptr<P2PConnectorAsyncWriteContext>& context) {
    auto call = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    auto* lease = call->response.mutable_p2p_response()->mutable_lease_status();
    lease->set_sealed(true);
    lease->set_stopped(true);
    auto result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{call});
    result->finishRank(0, true);
    {
        std::lock_guard<std::mutex> lock(context->mutex_);
        context->lease_result_ = result;
    }
    context->checkCancelDone();
}

bool receiveDone(const std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest>& state) {
    return !state->resourceHoldPending() || state->errorInfo().hasError();
}

bool receiveSuccess(const std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest>& state) {
    return !state->resourceHoldPending() && state->errorInfo().ok();
}

class RejectingWriteControlPool: public autil::ThreadPool {
public:
    RejectingWriteControlPool(): autil::ThreadPool(1, 8, nullptr, "RejectWriteControl") {}

    ERROR_TYPE pushWorkItem(autil::WorkItem* item, bool blocked = true) override {
        if (reject_submissions) {
            ++rejections;
            return ERROR_POOL_QUEUE_FULL;
        }
        return autil::ThreadPool::pushWorkItem(item, blocked);
    }

    std::atomic<bool> reject_submissions{false};
    std::atomic<int>  rejections{0};
};

class LeaseQueryFaultClient: public P2PBroadcastClient {
public:
    using P2PBroadcastClient::P2PBroadcastClient;
    std::atomic<int> failure_mode{0};
    std::atomic<int> failures{0};
    std::function<void()> before_query;

    std::shared_ptr<TpBroadcastResult>
    queryLeaseStatusAsync(const std::string& key, P2PConnectorBroadcastType operation, int64_t timeout_ms) override {
        if (before_query)
            before_query();
        const auto mode = failure_mode.load();
        if (mode != 0) {
            ++failures;
            if (mode == 2)
                throw std::runtime_error("injected lease query submission failure");
            return nullptr;
        }
        return P2PBroadcastClient::queryLeaseStatusAsync(key, operation, timeout_ms);
    }
};

class SchedulerWriteReceiver: public transfer::IKVCacheReceiver {
public:
    bool regMem(const BlockInfo&, uint64_t = 0) override {
        return true;
    }
    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override {
        auto task = tasks_.addTask(request.unique_key, request.block_info, request.deadline_ms);
        if (task && task->startTransfer()) {
            std::lock_guard<std::mutex> lock(mutex_);
            active_.push_back(task);
        }
        return task;
    }
    void stealTask(const std::string& key) override {
        tasks_.stealTask(key);
    }
    transfer::IKVCacheRecvTaskPtr getTask(const std::string& key) override {
        return tasks_.getTask(key);
    }
    void finish() {
        std::vector<std::shared_ptr<transfer::TransferTask>> active;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active.swap(active_);
        }
        for (const auto& task : active)
            task->notifyDone(false, transfer::TransferErrorCode::CANCELLED, "physical transfer stopped");
    }

private:
    transfer::TransferTaskStore                          tasks_;
    std::mutex                                           mutex_;
    std::vector<std::shared_ptr<transfer::TransferTask>> active_;
};

class WriteSchedulerService: public RpcService::Service {
public:
    std::function<ErrorInfo(const P2PConnectorBroadcastTpRequestPB&)> registered;
    using Handler = std::function<void(
        const P2PConnectorStartWriteBackRequestPB&, P2PConnectorStartWriteBackResponsePB&, std::function<bool()>)>;
    Handler           handshake;
    std::function<void(const P2PConnectorCancelWriteBackRequestPB&, P2PConnectorStartWriteBackResponsePB&)> cancel_write;
    std::atomic<int>                                                                                peer_cancels{0};
    std::atomic<bool>    fail_start{false}, lose_handshake{false}, omit_status{false};
    std::atomic<int>     start_delay_ms{0};
    std::atomic<int>     starts{0}, cancels{0}, handshakes{0}, lease_queries{0};
    std::atomic<int64_t> control_deadline_ms{0};

    std::function<void(const FunctionRequestPB&, FunctionResponsePB&)> execute_write;

    grpc::Status StartWriteBack(grpc::ServerContext*                   context,
                            const P2PConnectorStartWriteBackRequestPB* request,
                            P2PConnectorStartWriteBackResponsePB*      response) override {
        ++handshakes;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_handshake_ = *request;
        }
        handshake(*request, *response, [context]() { return context->IsCancelled(); });
        return lose_handshake ? grpc::Status(grpc::StatusCode::INTERNAL, "lost handshake") : grpc::Status::OK;
    }

    grpc::Status CancelWriteBack(grpc::ServerContext*,
                             const P2PConnectorCancelWriteBackRequestPB* request,
                             P2PConnectorStartWriteBackResponsePB*       response) override {
        ++peer_cancels;
        cancel_write(*request, *response);
        return grpc::Status::OK;
    }

    grpc::Status ExecuteFunction(grpc::ServerContext*     context,
                                 const FunctionRequestPB* request,
                                 FunctionResponsePB*      response) override {
        const auto& input = request->p2p_request();
        if (execute_write) {
            if (input.type() == HANDLE_WRITE)
                ++starts;
            else if (input.type() == CANCEL_HANDLE_WRITE)
                ++cancels;
            else if (input.type() == QUERY_LEASE_STATUS)
                ++lease_queries;
            execute_write(*request, *response);
            return grpc::Status::OK;
        }
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
    bool transferStopped(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = states_.find(key);
        return it != states_.end() && it->second.stopped;
    }
    P2PConnectorBroadcastTpRequestPB lastStart() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_start_;
    }
    P2PConnectorStartWriteBackRequestPB lastHandshake() const {
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
    P2PConnectorStartWriteBackRequestPB  last_handshake_;
};

class P2PConnectorWriteSchedulerTest: public ::testing::Test {
protected:
    using Allocator = test::BlockTreeCacheTestAllocator<CoordinatorCacheManager>;

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
        config.cache_key_tokens_per_block                   = config_.seq_size_per_block;
        config.parallelism_config.tp_size                   = 2;
        config.parallelism_config.role_type                 = role;
        config.role_type                                    = role;
        config.worker_grpc_addrs                            = addresses;
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
        for (const auto& tag : resource->groupTags()) {
            resource->mutableBlockIds(tag).assign({7, 8, 9, 10});
        }
        return resource;
    }

    std::shared_ptr<P2PConnectorAsyncWriteContext> asyncWrite(KVCacheResourcePtr resource) {
        return decode_->asyncWrite(std::move(resource), token_ids_, input_length_, kv_ready_token_count_, routing_);
    }

    P2PConnectorStartWriteBackRequestPB request() {
        P2PConnectorStartWriteBackRequestPB request;
        request.set_unique_key(routing_.unique_key);
        request.set_request_id(17);
        request.set_timeout_ms(4000);
        request.set_deadline_ms(currentTimeMs() + 4000);
        request.set_decode_tp_size(2);
        request.set_input_length(4);
        for (const auto token : token_ids_) {
            request.add_token_ids(token);
        }
        for (const auto key : keys_) {
            request.add_cache_keys(key);
        }
        const auto plan = mirroredWritePlan();
        request.set_plan_digest(plan.plan.digest());
        auto                           owner = source();
        P2PBroadcastClient::RankRoutes routes(decode_addrs_.size());
        EXPECT_TRUE(decode_->buildDecodeRankRoutes(plan.plan, *owner, 1, keys_.size() - 1, routes, nullptr).ok());
        for (size_t rank = 0; rank < routes.size(); ++rank) {
            auto* worker = request.add_decode_workers();
            worker->set_worker_addr(decode_addrs_[rank]);
            for (const auto& route : routes[rank])
                *worker->add_routes() = route;
        }
        return request;
    }

    void completeAll(bool draining = false) {
        for (int i = 0; i < 2; ++i) {
            prefill_workers_[i].completeAll(!draining, draining);
            decode_workers_[i].completeAll(!draining, draining);
        }
    }

    P2PConnectorStartWriteBackResponsePB handleWriteWhileAdmissionBlocked(bool expire) {
        auto input = request();
        if (expire) {
            input.set_timeout_ms(1000);
        }
        P2PConnectorStartWriteBackResponsePB response;
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
            const auto& device_blocks =
                cache->tree()->findNode({keys_[0]}).front()->group_set_resources[group->groupSetId()].device_blocks;
            ASSERT_TRUE(cache->executeTransfer(TransferTask(
                {TransferDescriptor::deviceToHost(group->groupSetId(), device_blocks, slots[i][0].host_block)},
                std::chrono::seconds(5))).ok());
        }
        cache->tree()->insertNode(CacheKeysType(keys_.begin(), keys_.begin() + count), slots, /*collect_path=*/false, false);
        for (size_t i = 1; i < count; ++i) {
            group->releaseSingleBlock(Tier::HOST, slots[i][0].host_block, BlockTreeRefType::CACHE);
        }
        const auto coordinator        = cache->loader_.load_context_coordinator_;
        const auto commit             = coordinator->commit_callback_;
        coordinator->commit_callback_ = [this, commit](const auto& load) {
            {
                std::lock_guard<std::mutex> lock(load_targets_mutex_);
                for (const auto& desc : load->loadDescs()) {
                    load_targets_[desc.path_index] = desc.target_blocks.front();
                }
            }
            return commit(load);
        };
        load_barrier_ = std::make_shared<block_tree_cache_test::CallbackBarrier>();
        load_engine_  = std::make_shared<block_tree_cache_test::ControlledPerRankBlockTransferEngine>(
            cache->groupSets(), action, load_barrier_);
        cache->transfer_dispatcher_->per_rank_engine_ = load_engine_;
        prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
        ASSERT_TRUE(prefill_->init());
    }

    std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest> receiveContext() {
        return prefill_->findRequest(routing_.unique_key);
    }

    BlockIndicesType receiveBlocks() const {
        BlockIndicesType blocks(keys_.size(), NULL_BLOCK_IDX);
        const auto       cache = allocator_->blockTreeCacheOwner();
        {
            std::lock_guard<std::mutex> lock(cache->mutex_);
            const auto                  path = cache->tree()->findNode(keys_);
            for (size_t i = 0; i < path.size(); ++i) {
                const auto& device_blocks = path[i]->group_set_resources.front().device_blocks;
                if (!device_blocks.empty())
                    blocks[i] = device_blocks.front();
            }
        }
        {
            std::lock_guard<std::mutex> lock(load_targets_mutex_);
            for (const auto& [index, block] : load_targets_)
                blocks[index] = block;
        }
        const auto request = prefill_workers_[0].lastStart();
        for (const auto& route : request.routes()) {
            for (const auto& layer : route.layer_blocks()) {
                for (int i = 0; i < layer.cache_keys_size(); ++i) {
                    const auto it = std::find(keys_.begin(), keys_.end(), layer.cache_keys(i));
                    if (it != keys_.end())
                        blocks[std::distance(keys_.begin(), it)] = layer.block_ids(i);
                }
            }
        }
        return blocks;
    }

    bool transferStopped() const {
        return std::all_of(std::begin(prefill_workers_), std::end(prefill_workers_), [this](const auto& worker) {
            return worker.transferStopped(routing_.unique_key);
        });
    }

    PlanResult mirroredWritePlan() const {
        const auto src = ShardLayoutFactory::fromTopology(
            *config_.topologyPtr(), decode_config_.parallelism_config, RoleType::DECODE, config_.seq_size_per_block);
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
    mutable std::mutex                                                           load_targets_mutex_;
    std::map<size_t, BlockIdxType>                                               load_targets_;
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
    EXPECT_TRUE(await([&] {
        std::lock_guard<std::mutex> lock(decode_->checker_.mutex_);
        return decode_->checker_.contexts_.empty();
    }));
    EXPECT_EQ(asyncWrite(source()), nullptr);
    for (int rank = 0; rank < 2; ++rank) {
        EXPECT_EQ(decode_workers_[rank].lease_queries, 0);
        EXPECT_EQ(prefill_workers_[rank].lease_queries, 0);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverSelectsSuffixWithoutReturningRangeToSource) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, {keys_[0], keys_[1]}).success);
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&] { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    for (int rank = 0; rank < 2; ++rank) {
        const auto send    = decode_workers_[rank].lastStart();
        const auto receive = prefill_workers_[rank].lastStart();
        ASSERT_EQ(send.routes_size(), 1);
        ASSERT_EQ(receive.routes_size(), 1);
        for (int layer = 0; layer < 2; ++layer) {
            ASSERT_EQ(send.routes(0).layer_blocks(layer).cache_keys_size(), 1);
            EXPECT_EQ(send.routes(0).layer_blocks(layer).cache_keys(0), keys_[2]);
            EXPECT_EQ(send.routes(0).layer_blocks(layer).block_ids(0), 9);
            ASSERT_EQ(receive.routes(0).layer_blocks(layer).cache_keys_size(), 1);
            EXPECT_EQ(receive.routes(0).layer_blocks(layer).cache_keys(0), keys_[2]);
        }
    }
    completeAll();
    ASSERT_TRUE(await([&] { return !context->resourceHoldPending(); }));
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST_F(P2PConnectorWriteSchedulerTest, ConcurrentBusySuffixFailsPublicationAndReleasesSource) {
    auto owner = source();
    std::weak_ptr<KVCacheResource> weak = owner;
    auto context = asyncWrite(std::move(owner));
    ASSERT_TRUE(await([&] { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, {keys_[0], keys_[1]}).success);
    const auto cache = allocator_->blockTreeCacheOwner();
    {
        std::lock_guard<std::mutex> lock(cache->mutex_);
        cache->tree()->findNode({keys_[0], keys_[1]}).back()->group_set_resources.front().transfer_state =
            GroupSetTransferState::DEMOTING;
    }
    completeAll();
    EXPECT_TRUE(await([&] { return !context->resourceHoldPending() && prefill_->inflightContextCount() == 0; }));
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
    EXPECT_NE(context->errorInfo().ToString().find("publication incomplete"), std::string::npos);
    EXPECT_TRUE(await([&] { return weak.expired(); }));
    {
        std::lock_guard<std::mutex> lock(cache->mutex_);
        EXPECT_EQ(cache->tree()->findNode(keys_).size(), 2u);
        cache->tree()->findNode({keys_[0], keys_[1]}).back()->group_set_resources.front().transfer_state =
            GroupSetTransferState::IDLE;
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, SourceSelectionKeepsOnlyCandidateBlocksAtOriginalPositions) {
    auto resource = source();
    ASSERT_TRUE(decode_->selectSourceBlocks(*resource, input_length_, kv_ready_token_count_, routing_).ok());
    EXPECT_EQ(resource->cacheKeys(), keys_);
    EXPECT_EQ(resource->blocks("default"), (BlockIndicesType{NULL_BLOCK_IDX, 8, 9}));
}

TEST_F(P2PConnectorWriteSchedulerTest, SourceSelectionUsesGroupTailPoliciesAndSkipsNonReusableGroups) {
    for (const bool reuse_linear : {true, false}) {
        auto config                          = test::makeFullSwaLinearCacheConfig();
        auto groups                          = config.topology().groups();
        groups[2].policy.enable_prefix_reuse = reuse_linear;
        config.setTopology(std::move(groups), config.topology().layers());
        decode_.reset();
        decode_config_.topology = config.topologyPtr();
        createDecode();
        KVCacheResource resource;
        resource.initGroups(config.topologyPtr());
        resource.setCacheKeys({11, 22, 33, 44, 55, 66});
        for (const auto& tag : resource.groupTags()) {
            resource.mutableBlockIds(tag).assign({7, 8, 9, 10, 11, 12});
        }
        const auto error = decode_->selectSourceBlocks(resource, 4, 24, routing_);
        ASSERT_TRUE(error.ok()) << error.ToString();
        EXPECT_EQ(resource.blocks("full"), (BlockIndicesType{NULL_BLOCK_IDX, 8, 9, 10, 11, 12}));
        EXPECT_EQ(resource.blocks("swa"),
                  (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 11, 12}));
        if (reuse_linear) {
            EXPECT_EQ(
                resource.blocks("linear"),
                (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 12}));
        } else {
            EXPECT_TRUE(resource.blocks("linear").empty());
        }
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, SenderRouteSelectionRejectsIncompleteKeyBlockPairs) {
    auto                           input = request();
    P2PBroadcastClient::RankRoutes receive_routes(2), send_routes;
    auto                           owner   = source();
    const auto                     planned = prefill_->planFor(2);
    ASSERT_TRUE(planned->ok());
    ASSERT_TRUE(prefill_->buildPrefillRankRoutes(planned->plan, *owner, 2, 1, receive_routes, nullptr).ok());
    ASSERT_TRUE(prefill_->selectSendRoutes(input, receive_routes, send_routes).ok());
    ASSERT_EQ(send_routes.size(), 2u);
    for (const auto& rank_routes : send_routes) {
        ASSERT_EQ(rank_routes.size(), 1u);
        for (const auto& layer : rank_routes.front().layer_blocks()) {
            ASSERT_EQ(layer.cache_keys_size(), 1);
            ASSERT_EQ(layer.block_ids_size(), 1);
            EXPECT_EQ(layer.cache_keys(0), keys_[2]);
            EXPECT_EQ(layer.block_ids(0), 9);
        }
    }
    input.mutable_decode_workers(0)->mutable_routes(0)->mutable_layer_blocks(0)->clear_block_ids();
    EXPECT_TRUE(prefill_->selectSendRoutes(input, receive_routes, send_routes).hasError());
    input = request();
    input.mutable_decode_workers(0)->clear_routes();
    EXPECT_TRUE(prefill_->selectSendRoutes(input, receive_routes, send_routes).hasError());
}

TEST_F(P2PConnectorWriteSchedulerTest, WritebackBudgetIsIndependentOfGenerationDeadline) {
    prefill_workers_[0].handshake = [](const auto&, auto& response, auto) { response.set_stopped(true); };
    for (const auto& [writeback_timeout, load_timeout] :
         std::vector<std::pair<int64_t, int64_t>>{{2000, 4000}, {4000, 2000}}) {
        decode_.reset();
        decode_config_.p2p_writeback_timeout_ms = writeback_timeout;
        decode_config_.load_cache_timeout_ms   = load_timeout;
        createDecode();
        for (const int64_t generation_remaining : {-1000, 100, 10000, 0}) {
            SCOPED_TRACE(std::to_string(writeback_timeout) + ":" + std::to_string(generation_remaining));
            routing_.unique_key = "independent-budget-" + std::to_string(writeback_timeout) + "-"
                                  + std::to_string(generation_remaining);
            routing_.deadline_ms = generation_remaining == 0 ? 0 : currentTimeMs() + generation_remaining;
            auto                           resource = source();
            std::weak_ptr<KVCacheResource> owner    = resource;
            const auto before_ms = currentTimeMs();
            auto context = asyncWrite(std::move(resource));
            const auto after_ms = currentTimeMs();
            ASSERT_NE(context, nullptr);
            const auto budget = std::min(writeback_timeout, load_timeout);
            EXPECT_GE(context->deadline_ms_, before_ms + budget);
            EXPECT_LE(context->deadline_ms_, after_ms + budget);
            ASSERT_TRUE(await([&] { return context->done() && owner.expired(); }));
            EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
            const auto handshake = prefill_workers_[0].lastHandshake();
            EXPECT_EQ(handshake.unique_key(), routing_.unique_key);
            EXPECT_GT(handshake.timeout_ms(), 0);
            EXPECT_LE(handshake.timeout_ms(), budget);
        }
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, StartWriteBackSendsRemainingBudgetInsteadOfAnAbsoluteDeadline) {
    DecodeWriteHelper helper;
    auto input = request();
    const auto deadline_ms = currentTimeMs() + 2000;
    auto result = helper.write(routing_.prefill_addr.first, routing_.prefill_addr.second, input, deadline_ms);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
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
    decode_workers_[1].start_delay_ms = 20;
    prefill_config_.p2p_writeback_timeout_ms = 1500;
    prefill_config_.load_cache_timeout_ms = 1000;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    const auto input = request();
    P2PConnectorStartWriteBackResponsePB response;
    const auto before_ms = currentTimeMs();
    auto handling = std::async(std::launch::async, [&]() { prefill_->handleWrite(input, response); });
    EXPECT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
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

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverRejectsWritebackThatWouldEatKvReserve) {
    prefill_.reset();
    prefill_config_.p2p_writeback_kv_reserve_ratio = 1.0;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    const auto                       prefix_blocks = allocator_->devicePrefixBlocksForTest(keys_);
    P2PConnectorStartWriteBackResponsePB response;
    prefill_->handleWrite(request(), response);
    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::MALLOC_FAILED)) << response.error_message();
    EXPECT_TRUE(response.stopped());
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), prefix_blocks);
}

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverUsesRemainingBudgetDespiteEarlierRemoteClock) {
    auto input = request();
    input.set_deadline_ms(currentTimeMs() - 60000);
    input.set_timeout_ms(2000);
    P2PConnectorStartWriteBackResponsePB response;
    const auto before_ms = currentTimeMs();
    auto call = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) { completeAll(true); });
    ASSERT_TRUE(await([&] { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    for (const auto& worker : decode_workers_) {
        const auto start = worker.lastStart();
        EXPECT_GT(start.deadline_ms(), before_ms);
        EXPECT_GT(start.transfer_timeout_ms(), 0);
        EXPECT_LE(start.transfer_timeout_ms(), 2000);
    }
    completeAll();
    call.get();
    EXPECT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR) << response.error_message();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 3u);
}

TEST(P2PConnectorAsyncWriteContextTest, TransferBudgetDoesNotFollowWireClockChanges) {
    auto context = makeWriteContext(nullptr, "steady-write", currentTimeMs() + 4000, nullptr, 100, 1000);
    const auto stop_at = context->nextWakeupMs();
    for (const int64_t offset_ms : {-60000, 60000}) {
        context->deadline_ms_ = currentTimeMs() + offset_ms;
        EXPECT_EQ(context->nextWakeupMs(), stop_at);
        context->expireTransferDeadlineIfNeeded();
        EXPECT_FALSE(context->done());
    }
    context->finishWithoutTransfer();
}

TEST(P2PWritebackRetainedKvTest, ThresholdDeduplicatesSharedBlocksAndUsesStrictGreaterThan) {
    const auto config = test::makeSimpleMhaCacheConfig(2, 10, 4, DataType::TYPE_FP16, 1, 2);
    auto first = std::make_shared<KVCacheResource>();
    first->initGroups(config.topologyPtr());
    first->mutableBlockIds("default").assign({1, 2});
    auto second = std::make_shared<KVCacheResource>();
    second->initGroups(config.topologyPtr());
    second->mutableBlockIds("default").assign({2, 3});
    for (const bool prefill : {false, true}) {
        WritebackRetainedKv retained;
        EXPECT_FALSE(retained.init(config.topologyPtr(), 0, prefill));
        EXPECT_FALSE(retained.init(config.topologyPtr(), 101, prefill));
        ASSERT_TRUE(retained.init(config.topologyPtr(), 20, prefill));
        retained.check({first, first});
        ::testing::FLAGS_gtest_death_test_style = "threadsafe";
        EXPECT_EXIT({
            const rlimit core_limit{};
            setrlimit(RLIMIT_CORE, &core_limit);
            retained.check({first, second});
            std::_Exit(42);
        }, ::testing::KilledBySignal(SIGABRT), "");
        retained.check({});
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverChecksEachIndependentPoolAndReleasesRejectedTargets) {
    prefill_.reset();
    decode_.reset();
    const auto full                     = test::makeMhaSpec("full", 4, DataType::TYPE_FP16, 1, 2);
    const auto swa                      = test::makeMhaSpec("swa", 4, DataType::TYPE_FP16, 1, 2);
    auto       policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.enable_prefix_reuse          = true;
    policy.sliding_window_size          = 8;
    config_.fromGroupedSpecs({full, swa},
                             {{0}, {1}},
                             {CacheGroupType::FULL, CacheGroupType::SWA},
                             {"full", "swa"},
                             {defaultCacheGroupPolicy(CacheGroupType::FULL), policy});
    config_.setGroupBlockLayout({"full", "swa"}, {8, 32}, {full->block_size_bytes(), swa->block_size_bytes()}, {0, 0});
    using MultiAllocator = test::BlockTreeCacheTestAllocator<CoordinatorCacheManager>;
    auto allocator = std::make_shared<MultiAllocator>(config_, AllocationType::DEVICE, nullptr, 0, RoleType::PREFILL);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator, {keys_.front()}).success);
    prefill_config_.topology                       = config_.topologyPtr();
    prefill_config_.p2p_writeback_kv_reserve_ratio = 0.75;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    decode_config_.topology = config_.topologyPtr();
    createDecode();
    const auto& pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    ASSERT_EQ(pools[0]->totalBlocksNum(), 7u);
    ASSERT_EQ(pools[1]->totalBlocksNum(), 31u);
    const auto full_free      = pools[0]->freeBlocksNum();
    const auto swa_free       = pools[1]->freeBlocksNum();
    const auto full_available = pools[0]->availableBlocksNum();
    const auto swa_available  = pools[1]->availableBlocksNum();
    ASSERT_LT(full_available - keys_.size(), 6u);
    ASSERT_GE(full_available + swa_available - 2 * keys_.size(), 29u);

    P2PConnectorStartWriteBackResponsePB response;
    prefill_->handleWrite(request(), response);

    EXPECT_EQ(response.error_code(), transErrorCodeToRPC(ErrorCode::MALLOC_FAILED)) << response.error_message();
    EXPECT_TRUE(response.stopped());
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(decode_workers_[0].starts, 0);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
    EXPECT_EQ(pools[0]->freeBlocksNum(), full_free);
    EXPECT_EQ(pools[1]->freeBlocksNum(), swa_free);
    EXPECT_EQ(pools[0]->availableBlocksNum(), full_available);
    EXPECT_EQ(pools[1]->availableBlocksNum(), swa_available);
    EXPECT_EQ(allocator->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, OutOfRangeKvReserveRatioFailsInit) {
    for (const double ratio : {-0.1, 1.5}) {
        auto prefill_config                           = prefill_config_;
        prefill_config.p2p_writeback_kv_reserve_ratio = ratio;
        EXPECT_FALSE(P2PSchedulerPrefillWrite(prefill_config, allocator_, nullptr, prefill_client_).init());
        auto decode_config                           = decode_config_;
        decode_config.p2p_writeback_kv_reserve_ratio = ratio;
        EXPECT_FALSE(P2PSchedulerDecodeWrite(decode_config, decode_client_, *decode_source_worker_).init());
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, KvReadyBoundaryCapsCompleteTokenKeys) {
    decode_workers_[0].start_delay_ms = 20;
    kv_ready_token_count_ = 8;
    auto context          = asyncWrite(source());
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    const auto handshake = prefill_workers_[0].lastHandshake();
    EXPECT_EQ(handshake.token_ids_size(), 13);
    ASSERT_EQ(handshake.cache_keys_size(), 2);
    EXPECT_EQ(handshake.cache_keys(1), keys_[1]);
    const auto sent = decode_workers_[0].lastStart();
    ASSERT_EQ(sent.routes_size(), 1);
    ASSERT_GT(sent.routes(0).layer_blocks_size(), 0);
    EXPECT_EQ(sent.routes(0).layer_blocks(0).cache_keys_size(), 1);
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

TEST_F(P2PConnectorWriteSchedulerTest, FullyCachedHandshakeNeedsNoKvReserve) {
    prefill_.reset();
    prefill_config_.p2p_writeback_kv_reserve_ratio = 1.0;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, keys_).success);
    P2PConnectorStartWriteBackResponsePB response;
    prefill_->handleWrite(request(), response);
    EXPECT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR) << response.error_message();
    EXPECT_EQ(response.accepted_block_count(), 0);
    EXPECT_EQ(decode_workers_[0].starts, 0);
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
        P2PConnectorCancelWriteBackRequestPB cancel;
        cancel.set_unique_key(routing_.unique_key);
        cancel.set_deadline_ms(currentTimeMs() + 4000);
        P2PConnectorStartWriteBackResponsePB response;
        prefill_->cancelWrite(cancel, response);
        EXPECT_FALSE(response.stopped());
        return commit(load);
    };
    const auto                       free_before = allocator_->freeBlocksNum();
    const auto                       input = request();
    P2PConnectorStartWriteBackResponsePB response;
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
    const auto blocks = receiveBlocks();
    const auto pool   = allocator_->groupBlockPools().front();
    EXPECT_TRUE(receive->resourceHoldPending());
    EXPECT_TRUE(pool->isAllocated(blocks[1]));
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
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
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 1);
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
    const auto blocks = receiveBlocks();
    const auto pool   = allocator_->groupBlockPools().front();
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
    ASSERT_TRUE(await([&]() { return transferStopped(); }));
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
    const auto blocks = receiveBlocks();
    load_barrier_->release();
    ASSERT_TRUE(await([&]() { return allocator_->devicePrefixBlocksForTest(keys_) == 2; }));
    EXPECT_FALSE(receiveDone(receive));
    EXPECT_EQ(allocator_->groupBlockPools().front()->refCount(blocks[1]), 2u);
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
    const auto blocks = receiveBlocks();
    const auto pool   = allocator_->groupBlockPools().front();
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
    ASSERT_TRUE(await([&] { return !context->resourceHoldPending(); }));
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
    const auto blocks = receiveBlocks();
    const auto pool   = allocator_->groupBlockPools().front();
    receive->fail(ErrorInfo(ErrorCode::CANCELLED, "test cancel"));
    completeAll();
    ASSERT_TRUE(await([&]() { return transferStopped(); }));
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
    const auto first_blocks = receiveBlocks();

    routing_.unique_key = "second-write-scheduler";
    auto second         = asyncWrite(source());
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 2 && decode_workers_[1].starts == 2; }));
    auto second_receive = receiveContext();
    ASSERT_NE(second_receive, nullptr);
    const auto second_blocks = receiveBlocks();
    EXPECT_EQ(first_blocks[0], second_blocks[0]);
    EXPECT_EQ(first_blocks[1], second_blocks[1]);
    EXPECT_NE(first_blocks[2], second_blocks[2]);
    EXPECT_EQ(load_engine_->submittedBatchCount(), 1u);
    const auto pool = allocator_->groupBlockPools().front();
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
    const std::vector<std::function<void(P2PConnectorStartWriteBackRequestPB&)>> mutations = {
        [](auto& r) { r.set_plan_digest(r.plan_digest() + 1); },
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
        P2PConnectorStartWriteBackResponsePB response;
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
    ASSERT_TRUE(await([&]() { return context->done(); }));
    EXPECT_FALSE(weak.expired());
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

TEST_F(P2PConnectorWriteSchedulerTest, WritebackPlanPreservesConfiguredActiveTail) {
    for (const auto& [window, tail] :
         std::vector<std::pair<int, uint32_t>>{{0, 2}, {4, 2}, {8, 2}, {16, 2}, {16, 4}, {16, 0}}) {
        SCOPED_TRACE("window=" + std::to_string(window) + " tail=" + std::to_string(tail));
        auto config                          = test::makeFullSwaLinearCacheConfig();
        auto groups                          = config.topology().groups();
        groups[1].policy.sliding_window_size = window;
        groups[1].policy.active_tail_blocks  = tail;
        const auto topology                  = CacheTopology::create(groups, config.topology().layers());
        auto       decode_config             = decode_config_;
        decode_config.topology               = topology;
        auto prefill_config                  = prefill_config_;
        prefill_config.topology              = topology;
        P2PSchedulerDecodeWrite  decode_scheduler(decode_config, decode_client_, *decode_source_worker_);
        P2PSchedulerPrefillWrite prefill_scheduler(prefill_config, allocator_, nullptr, prefill_client_);
        const auto               decode  = decode_scheduler.planFor(2);
        const auto               prefill = prefill_scheduler.planFor(2);
        ASSERT_TRUE(decode->ok()) << decode->error.ToString();
        ASSERT_TRUE(prefill->ok()) << prefill->error.ToString();
        EXPECT_EQ(decode->plan.digest(), prefill->plan.digest());
        for (const auto& route : decode->plan.routes) {
            const auto positions = KVCacheTransferPlanner::resolveKeys(route.src_keys, 8);
            if (route.cache_tag == "swa") {
                const size_t        count = tail == 0 ? 8u : std::min<size_t>(tail, 8);
                std::vector<size_t> expected;
                for (size_t i = 8 - count; i < 8; ++i) {
                    expected.push_back(i);
                }
                EXPECT_EQ(positions, expected);
            } else if (route.cache_tag == "linear") {
                EXPECT_EQ(positions, std::vector<size_t>({7}));
            }
        }
    }
    const auto config        = test::makeFullSwaLinearCacheConfig(false);
    auto       decode_config = decode_config_;
    decode_config.topology   = config.topologyPtr();
    P2PSchedulerDecodeWrite scheduler(decode_config, decode_client_, *decode_source_worker_);
    const auto              plan = scheduler.planFor(2);
    ASSERT_TRUE(plan->ok());
    for (const auto& route : plan->plan.routes) {
        EXPECT_NE(route.cache_tag, "swa");
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, WritebackMatchesForwardTpCpSupportFromRealSpecs) {
    AttentionConfigs attention;
    attention.kv_head_num   = 4;
    attention.size_per_head = 2;
    attention.kv_lora_rank  = 4;
    attention.rope_head_dim = 2;
    LinearAttentionConfig linear;
    linear.linear_num_key_heads   = 4;
    linear.linear_num_value_heads = 4;
    linear.linear_key_head_dim    = 2;
    linear.linear_value_head_dim  = 2;
    linear.linear_conv_kernel_dim = 2;
    linear.ssm_state_dtype        = DataType::TYPE_FP32;
    linear.conv_state_dtype       = DataType::TYPE_FP16;
    for (const auto& [spec_type, group_type] :
         std::vector<std::pair<KVCacheSpecType, CacheGroupType>>{
             {KVCacheSpecType::MultiHeadAttention, CacheGroupType::FULL},
             {KVCacheSpecType::MultiHeadAttention, CacheGroupType::SWA},
             {KVCacheSpecType::MultiHeadLatentAttention, CacheGroupType::FULL},
             {KVCacheSpecType::LinearAttention, CacheGroupType::LINEAR}}) {
        for (int prefill_tp : {1, 2, 4}) {
            for (int decode_tp : {1, 2, 4}) {
                for (auto method : {CPRotateMethod::DISABLED, CPRotateMethod::ALL_GATHER, CPRotateMethod::ALLTOALL}) {
                    for (bool sharded : {false, true}) {
                        if (method == CPRotateMethod::DISABLED && sharded) {
                            continue;
                        }
                        SCOPED_TRACE(::testing::Message() << "spec=" << static_cast<int>(spec_type)
                                                          << " group=" << static_cast<int>(group_type)
                                                          << " P=" << prefill_tp << " D=" << decode_tp
                                                          << " method=" << static_cast<int>(method)
                                                          << " sharded=" << sharded);
                        auto prefill_config = prefill_config_;
                        auto decode_config  = decode_config_;
                        auto& prefill_pc = prefill_config.parallelism_config;
                        auto& decode_pc  = decode_config.parallelism_config;
                        prefill_pc.tp_size = prefill_tp;
                        decode_pc.tp_size  = decode_tp;
                        prefill_pc.prefill_cp_config.method = method;
                        decode_pc.prefill_cp_config.method = method == CPRotateMethod::DISABLED ?
                                                                 CPRotateMethod::DISABLED : CPRotateMethod::PREFILL_CP;
                        prefill_pc.prefill_cp_config.kv_cache_sharded = sharded;
                        decode_pc.prefill_cp_config.kv_cache_sharded  = sharded;
                        const int prefill_cp = sharded ? prefill_tp : 1;
                        decode_pc.prefill_cp_config.prefill_cp_size = prefill_cp;
                        for (auto* config : {&prefill_config, &decode_config}) {
                            KVCacheSpecDesc desc;
                            desc.tag        = "default";
                            desc.cache_type = spec_type;
                            desc.dtype      = DataType::TYPE_FP16;
                            SpecBuildContext build;
                            build.seq_size_per_block      = 4;
                            build.attn_config             = &attention;
                            build.linear_attention_config = &linear;
                            build.parallelism_config      = &config->parallelism_config;
                            auto group   = config_.topology().groups().front();
                            group.spec   = SpecBuilder::build(desc, build);
                            group.policy = defaultCacheGroupPolicy(group_type);
                            group.policy.enable_prefix_reuse = true;
                            config->topology = CacheTopology::create({group}, config_.topology().layers());
                        }
                        P2PSchedulerPrefillRead forward_prefill(prefill_config, nullptr, prefill_client_);
                        P2PSchedulerDecodeRead forward_decode(decode_config, nullptr, decode_client_);
                        P2PSchedulerPrefillWrite reverse_prefill(prefill_config, allocator_, nullptr, prefill_client_);
                        P2PSchedulerDecodeWrite reverse_decode(decode_config, decode_client_, *decode_source_worker_);
                        const auto forward = forward_prefill.planFor(decode_tp, routing_.unique_key);
                        const auto forward_peer = forward_decode.planFor(prefill_tp, prefill_cp, routing_.unique_key);
                        const auto reverse = reverse_decode.planFor(prefill_tp);
                        const auto reverse_peer = reverse_prefill.planFor(decode_tp);
                        const bool supported = spec_type != KVCacheSpecType::LinearAttention
                                               || prefill_pc.get_attn_tp_size() == decode_pc.get_attn_tp_size();
                        ASSERT_EQ(forward->ok(), supported) << forward->error.ToString();
                        ASSERT_EQ(forward_peer->ok(), supported) << forward_peer->error.ToString();
                        ASSERT_EQ(reverse->ok(), supported) << reverse->error.ToString();
                        ASSERT_EQ(reverse_peer->ok(), supported) << reverse_peer->error.ToString();
                        if (supported) {
                            EXPECT_FALSE(reverse->plan.routes.empty());
                            EXPECT_EQ(forward->plan.digest(), forward_peer->plan.digest());
                            EXPECT_EQ(reverse->plan.digest(), reverse_peer->plan.digest());
                        }
                    }
                }
            }
        }
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, WriteRangeRejectsExplicitPeerCpLayoutDriftLikeForward) {
    auto config = decode_config_;
    config.parallelism_config.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    config.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
    config.parallelism_config.prefill_cp_config.prefill_cp_size = 4;
    routing_.prefill_cp_size = 2;
    P2PSchedulerDecodeRead forward(config, nullptr, decode_client_);
    P2PSchedulerDecodeWrite reverse(config, decode_client_, *decode_source_worker_);
    EXPECT_TRUE(forward.checkPeerCpLayout(2, 2).hasError());
    EXPECT_FALSE(reverse.writeRange(input_length_, kv_ready_token_count_, routing_).has_value());
    routing_.prefill_tp_size = 4;
    routing_.prefill_cp_size = 4;
    EXPECT_TRUE(forward.checkPeerCpLayout(4, 4).ok());
    EXPECT_TRUE(reverse.writeRange(input_length_, kv_ready_token_count_, routing_).has_value());
}

TEST_F(P2PConnectorWriteSchedulerTest, CpWritebackIncludesPromptBoundaryAndPublishesCanonicalKeys) {
    decode_.reset();
    prefill_.reset();
    token_ids_.clear();
    for (int token = 1; token <= 19; ++token) {
        token_ids_.push_back(token);
    }
    input_length_            = 12;
    kv_ready_token_count_    = 19;
    keys_                    = calculateCacheKeys(token_ids_.data(), 16, 4);
    routing_.prefill_cp_size = 2;
    allocator_               = std::make_shared<Allocator>(config_);
    ASSERT_TRUE(allocator_->init());
    allocator_->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, 2, 4));
    ASSERT_TRUE(test::seedCompleteBlockTreePath(allocator_, {keys_[1]}).success);
    prefill_config_.parallelism_config.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    prefill_config_.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    decode_config_.parallelism_config.tp_size                            = 1;
    decode_config_.parallelism_config.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_config_.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
    decode_config_.parallelism_config.prefill_cp_config.prefill_cp_size  = 2;
    decode_config_.worker_grpc_addrs.resize(1);
    decode_client_ = std::make_shared<P2PBroadcastClient>(decode_config_.worker_grpc_addrs, 100);
    ASSERT_TRUE(decode_client_->init());
    createDecode();
    ASSERT_TRUE(decode_->planFor(2)->ok());
    ASSERT_TRUE(prefill_->planFor(1)->ok());
    EXPECT_EQ(decode_->planFor(2)->plan.digest(), prefill_->planFor(1)->plan.digest());
    auto selected = source();
    ASSERT_TRUE(decode_->selectSourceBlocks(*selected, input_length_, kv_ready_token_count_, routing_).ok());
    EXPECT_EQ(selected->blocks("default"), (BlockIndicesType{NULL_BLOCK_IDX, NULL_BLOCK_IDX, 9, 10}));
    auto context = asyncWrite(source());
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(await([&]() { return decode_workers_[0].starts == 1 || context->done(); }));
    ASSERT_EQ(decode_workers_[0].starts, 1) << context->errorInfo().ToString();
    EXPECT_EQ(decode_workers_[1].starts, 0);
    const auto handshake = prefill_workers_[0].lastHandshake();
    EXPECT_EQ(handshake.cache_keys_size(), 4);
    const auto sent = decode_workers_[0].lastStart();
    ASSERT_EQ(sent.routes_size(), 2);
    for (int rank = 0; rank < 2; ++rank) {
        const auto received = prefill_workers_[rank].lastStart();
        ASSERT_EQ(received.routes_size(), 1);
        for (const auto& layer : received.routes(0).layer_blocks()) {
            ASSERT_EQ(layer.cache_keys_size(), 1);
            EXPECT_EQ(layer.cache_keys(0), keys_[2 + rank]);
        }
        for (const auto& layer : sent.routes(rank).layer_blocks()) {
            ASSERT_EQ(layer.cache_keys_size(), 1);
            EXPECT_EQ(layer.cache_keys(0), keys_[2 + rank]);
            EXPECT_EQ(layer.block_ids(0), 9 + rank);
        }
    }
    completeAll();
    ASSERT_TRUE(await([&]() { return context->done() && prefill_->inflightContextCount() == 0; }));
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest({keys_[1], keys_[3]}), 2u);
}

TEST_F(P2PConnectorWriteSchedulerTest, CompactStateProjectsGlobalKeysToGroupLocalBlocks) {
    auto group               = config_.topology().groups().front();
    group.policy             = defaultCacheGroupPolicy(CacheGroupType::SWA);
    group.spec = test::makeMhaSpec(group.tag, 8, DataType::TYPE_FP16, 1, 2);
    KVCacheResource resource;
    resource.initGroups(config_.topologyPtr());
    resource.setCacheKeys({11, 22, 33, 44, 55, 66});
    resource.mutableBlockIds(group.tag).assign({7, 8, 9});
    const std::vector<size_t> positions{3, 5};
    const auto decode  = LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, positions, 0, 1, 4);
    const auto prefill = LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, positions, 1, 2, 4);
    ASSERT_TRUE(decode.ok()) << decode.status().ToString();
    ASSERT_TRUE(prefill.ok()) << prefill.status().ToString();
    EXPECT_EQ(decode.value()->blockIdMap(), prefill.value()->blockIdMap());
    EXPECT_EQ(decode.value()->getBlockId(44), 8);
    EXPECT_EQ(decode.value()->getBlockId(66), 9);
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
    P2PConnectorStartWriteBackResponsePB response;
    prefill_->handleWrite(req, response, []() { return true; });
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, MultipleFullGroupsRollBackEarlierAllocationsOnFailure) {
    auto config                        = config_;
    auto first                         = config.group("default").spec->clone();
    auto second                        = first->clone();
    first->tag                         = "first";
    second->tag                        = "second";
    config.fromGroupedSpecs(
        {first, second}, {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::FULL}, {"first", "second"});
    config.setGroupBlockLayout({"first", "second"}, {8, 2}, {first->block_size_bytes(), second->block_size_bytes()}, {0, 0});
    using MultiAllocator = test::BlockTreeCacheTestAllocator<CoordinatorCacheManager>;
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

TEST_F(P2PConnectorWriteSchedulerTest, InvalidSourceWorkersAndPlanNeverReachWorkers) {
    const std::vector<std::function<void(P2PConnectorStartWriteBackRequestPB&)>> mutations = {
        [](auto& r) { r.set_plan_digest(0); },
        [](auto& r) { r.clear_decode_workers(); },
        [](auto& r) { r.set_decode_tp_size(1); },
        [](auto& r) { r.mutable_decode_workers(0)->set_worker_addr("invalid-endpoint"); }};
    const auto free = allocator_->freeBlocksNum();
    for (const auto& mutate : mutations) {
        auto input = request();
        mutate(input);
        P2PConnectorStartWriteBackResponsePB response;
        prefill_->handleWrite(input, response);
        EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
        EXPECT_TRUE(response.stopped());
        EXPECT_EQ(allocator_->freeBlocksNum(), free);
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
    EXPECT_GT(decode_workers_[0].lease_queries, 0);
    EXPECT_GT(decode_workers_[1].lease_queries, 0);
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

TEST_F(P2PConnectorWriteSchedulerTest, DuplicateStartWriteBackDoesNotStartWorkersAgain) {
    auto context = asyncWrite(source());
    ASSERT_TRUE(await([&] { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    P2PConnectorStartWriteBackResponsePB response;
    prefill_->handleWrite(prefill_workers_[0].lastHandshake(), response);
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_FALSE(response.stopped());
    EXPECT_EQ(decode_workers_[0].starts, 1);
    EXPECT_EQ(decode_workers_[1].starts, 1);
    completeAll();
    ASSERT_TRUE(await([&] { return !context->resourceHoldPending(); }));
    EXPECT_TRUE(context->success());
}

TEST_F(P2PConnectorWriteSchedulerTest, CancelBeforeStartWriteBackFencesLateStartWithoutAllocation) {
    const auto                       free = allocator_->freeBlocksNum();
    P2PConnectorCancelWriteBackRequestPB cancel;
    cancel.set_unique_key(routing_.unique_key);
    cancel.set_deadline_ms(currentTimeMs() + 4000);
    P2PConnectorStartWriteBackResponsePB response;
    prefill_->cancelWrite(cancel, response);
    EXPECT_TRUE(response.stopped());
    prefill_->handleWrite(request(), response);
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_TRUE(response.stopped());
    EXPECT_EQ(allocator_->freeBlocksNum(), free);
    EXPECT_EQ(prefill_workers_[0].starts, 0);
    EXPECT_EQ(decode_workers_[0].starts, 0);
}

TEST_F(P2PConnectorWriteSchedulerTest, RegistrationBarrierChecksEveryRankAndRejectsLateReady) {
    prefill_workers_[1].registered = [](const auto&) { return ErrorInfo::OkStatus(); };
    decode_workers_[1].start_delay_ms = 20;
    auto                             input = request();
    P2PConnectorStartWriteBackResponsePB response;
    auto handling = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    EXPECT_TRUE(await([&] { return prefill_workers_[0].starts == 1 && prefill_workers_[1].starts == 1; }));
    const auto state = receiveContext();
    EXPECT_NE(state, nullptr);
    auto registration = prefill_workers_[0].lastStart();
    EXPECT_TRUE(prefill_->workerRegistered(registration).ok());
    EXPECT_EQ(decode_workers_[0].starts, 0);
    registration.set_worker_rank(2);
    EXPECT_TRUE(prefill_->workerRegistered(registration).hasError());
    registration.set_worker_rank(1);
    registration.set_plan_digest(registration.plan_digest() + 1);
    EXPECT_TRUE(prefill_->workerRegistered(registration).hasError());
    EXPECT_EQ(decode_workers_[0].starts, 0);
    registration.set_plan_digest(registration.plan_digest() - 1);
    EXPECT_TRUE(prefill_->workerRegistered(registration).ok());
    EXPECT_TRUE(await([&] { return decode_workers_[0].starts == 1 && decode_workers_[1].starts == 1; }));
    completeAll();
    handling.get();
    EXPECT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_TRUE(prefill_->workerRegistered(registration).hasError());
}

TEST(P2PConnectorAsyncWriteContextTest, CancelAcknowledgmentNeedsValidStoppedLeasesFromEverySender) {
    int releases = 0;
    auto resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner = resource;
    auto context = makeWriteContext(std::move(resource), "cancel-is-not-stop", currentTimeMs() + 4000,
                                    nullptr, 100, 100, [&] { ++releases; });
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    context->cancel();
    context->peer_stopped_ = true;
    auto cancel_call = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    auto* cancel_lease = cancel_call->response.mutable_p2p_response()->mutable_lease_status();
    cancel_lease->set_sealed(true);
    cancel_lease->set_stopped(true);
    auto cancel_result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{cancel_call});
    cancel_result->finishRank(0, true);
    context->control_result_ = cancel_result;
    context->checkCancelDone();
    EXPECT_TRUE(context->worker_cancelled_);
    EXPECT_FALSE(context->transfer_stopped_);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(releases, 0);
    for (const auto failure : {"active", "missing", "invalid", "rpc_failed", "timeout", "stopped"}) {
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>> calls;
        for (int rank = 0; rank < 2; ++rank) {
            auto call = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
            auto* lease = call->response.mutable_p2p_response()->mutable_lease_status();
            lease->set_sealed(true);
            lease->set_started_ops(1);
            lease->set_finished_ops(1);
            lease->set_stopped(true);
            calls.push_back(std::move(call));
        }
        const std::string status = failure;
        auto* lease = calls.back()->response.mutable_p2p_response()->mutable_lease_status();
        if (status == "active" || status == "invalid") {
            lease->set_finished_ops(0);
            lease->set_stopped(status == "invalid");
        }
        if (status == "missing")
            calls.back()->response.mutable_p2p_response()->clear_lease_status();
        auto result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(calls);
        result->finishRank(0, true);
        if (status == "rpc_failed" || status == "timeout")
            calls.back()->status =
                grpc::Status(status == "timeout" ? grpc::StatusCode::DEADLINE_EXCEEDED : grpc::StatusCode::UNAVAILABLE,
                             "lease failed");
        result->finishRank(1, status != "rpc_failed" && status != "timeout");
        context->worker_cancelled_ = true;
        context->lease_result_     = result;
        context->checkCancelDone();
        EXPECT_EQ(context->worker_cancelled_, status == "active" || status == "stopped") << status;
        EXPECT_EQ(context->resourceHoldPending(), status != "stopped") << status;
        EXPECT_EQ(owner.expired(), status == "stopped") << status;
        EXPECT_EQ(releases, status == "stopped" ? 1 : 0) << status;
    }
    confirmSourceStopped(context);
    EXPECT_EQ(releases, 1);
}

TEST(P2PConnectorAsyncWriteContextTest, CheckerDoesNotCompleteQueuedOrStartingRequest) {
    auto                           resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner    = resource;
    auto context = makeWriteContext(std::move(resource), "queued-source", currentTimeMs() + 4000, nullptr, 100, 1000);
    context->checkDone();
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(owner.expired());
    ASSERT_TRUE(context->beginKickoff());
    context->checkDone();
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(owner.expired());
    context->finishWithoutTransfer();
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(owner.expired());
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
    EXPECT_TRUE(context->metrics_.no_transfer);
    EXPECT_EQ(callback_count, 2);
    context->checkDone();
    context->cancel();
    context->onDone({});
    EXPECT_EQ(callback_count, 2);
}

TEST(P2PConnectorAsyncWriteContextTest, ThrowingCompletionCallbacksDoNotStopRemainingCallbacks) {
    auto write = makeWriteContext(nullptr, "throwing-done", currentTimeMs() + 4000, nullptr, 100, 1000);
    auto read = std::make_shared<P2PConnectorAsyncReadContext>(
        nullptr, "throwing-read-done", nullptr, 1000);
    for (const std::shared_ptr<AsyncContext>& context : {std::static_pointer_cast<AsyncContext>(write),
                                                        std::static_pointer_cast<AsyncContext>(read)}) {
        int completed = 0;
        context->onDone([](ErrorInfo) { throw std::runtime_error("callback failure"); });
        context->onDone([](ErrorInfo) { throw 1; });
        context->onDone([&](ErrorInfo) { ++completed; });
        if (context == write)
            write->finishWithoutTransfer();
        else
            read->markStartFailed(ErrorInfo(ErrorCode::INVALID_PARAMS, "test"));
        EXPECT_EQ(completed, 1);
        EXPECT_NO_THROW(context->onDone([](ErrorInfo) { throw std::runtime_error("late callback failure"); }));
        context->onDone([&](ErrorInfo) { ++completed; });
        EXPECT_EQ(completed, 2);
    }
}

TEST(P2PConnectorAsyncWriteContextTest, ReleaseCallbackMayReenterAndKeepsHoldUntilCleanupCompletes) {
    std::promise<void> entered;
    std::promise<void> resume;
    auto gate = resume.get_future().share();
    int release_count = 0;
    std::shared_ptr<P2PConnectorAsyncWriteContext> context;
    context = makeWriteContext(nullptr, "reentrant-release", currentTimeMs() + 4000, nullptr, 100, 1000, [&] {
        ++release_count;
        EXPECT_FALSE(context->done());
        EXPECT_TRUE(context->resourceHoldPending());
        context->checkDone();
        context->cancel();
        context->onDone([&](ErrorInfo error) {
            EXPECT_TRUE(error.ok());
            EXPECT_FALSE(context->resourceHoldPending());
        });
        entered.set_value();
        gate.wait();
    });
    auto finishing = std::async(std::launch::async, [&] { context->finishWithoutTransfer(); });
    EXPECT_EQ(entered.get_future().wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(context->done());
    context->checkDone();
    context->cancel();
    resume.set_value();
    finishing.get();
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_EQ(release_count, 1);
}

TEST(P2PConnectorAsyncWriteContextTest, FailedReleaseRetainsOwnershipAndRetriesBeforeReportingReleased) {
    auto source = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> weak = source;
    int attempts = 0;
    int done_callbacks = 0;
    auto context = std::make_shared<P2PConnectorAsyncWriteContext>(
        "retry-release", currentTimeMs() + 4000, nullptr, 100, 1000,
        [source = std::move(source), &attempts]() mutable {
            if (++attempts == 1)
                throw std::runtime_error("release failure");
            source.reset();
        });
    context->onDone([&](ErrorInfo) { ++done_callbacks; });
    EXPECT_NO_THROW(context->finishWithoutTransfer());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(done_callbacks, 0);
    context->checkDone();
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(attempts, 2);
    EXPECT_EQ(done_callbacks, 1);
}

TEST(P2PConnectorAsyncWriteContextTest, PeerRejectionSurvivesCleanupButDoesNotReplaceEarlierTimeout) {
    for (const bool timeout_first : {false, true}) {
        auto context = makeWriteContext(nullptr, "peer-rejection", currentTimeMs() + 4000, nullptr, 100, 1000);
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        auto call = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
        call->response.set_error_code(transErrorCodeToRPC(ErrorCode::MALLOC_FAILED));
        call->response.set_stopped(true);
        auto result = std::make_shared<DecodeWriteHelper::Result>(
            std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{call});
        context->setPeerCallResult(result);
        if (timeout_first)
            context->cancel(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "earlier timeout"));
        result->finishRank(0, true);
        context->checkDone();
        confirmSourceStopped(context);
        EXPECT_FALSE(context->resourceHoldPending());
        EXPECT_EQ(context->errorInfo().code(), timeout_first ? ErrorCode::GENERATE_TIMEOUT : ErrorCode::MALLOC_FAILED);
    }
}

TEST(P2PConnectorAsyncWriteContextTest, SuccessResponseWithoutStopProofRetainsSource) {
    auto context = makeWriteContext(nullptr, "unstopped-peer", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    auto call = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
    auto result = std::make_shared<DecodeWriteHelper::Result>(
        std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{call});
    context->setPeerCallResult(result);
    result->finishRank(0, true);
    context->checkDone();
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED);
    auto stopped = peerResult();
    context->setPeerCallResult(stopped);
    stopped->finishRank(0, true);
    context->checkDone();
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_FALSE(context->success());
}

TEST(P2PConnectorAsyncWriteContextTest, RpcFailureKeepsOriginalError) {
    for (const auto status : {grpc::StatusCode::DEADLINE_EXCEEDED, grpc::StatusCode::CANCELLED}) {
        auto context = makeWriteContext(nullptr, "rpc-first-error", currentTimeMs() + 4000, nullptr, 100, 1000);
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        auto call = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
        call->status = grpc::Status(status, "original transport failure");
        auto result = std::make_shared<DecodeWriteHelper::Result>(
            std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{call});
        context->setPeerCallResult(result);
        result->finishRank(0, true);
        const auto expected = result->firstError().error;
        ASSERT_TRUE(expected.hasError());
        context->checkDone();
        EXPECT_EQ(context->errorInfo().code(), expected.code());
        EXPECT_EQ(context->errorInfo().ToString(), expected.ToString());
        EXPECT_TRUE(context->resourceHoldPending());
        auto stopped = peerResult();
        context->setPeerCallResult(stopped);
        stopped->finishRank(0, true);
        context->checkDone();
        EXPECT_FALSE(context->resourceHoldPending());
        EXPECT_EQ(context->errorInfo().ToString(), expected.ToString());
    }
}

TEST(P2PConnectorAsyncWriteContextTest, OnlyUnimplementedStartProvesPeerNeverStartedAndStillRequiresLocalStop) {
    for (const bool start : {true, false}) {
        for (const auto code : {grpc::StatusCode::UNIMPLEMENTED, grpc::StatusCode::UNAVAILABLE,
                                grpc::StatusCode::INTERNAL, grpc::StatusCode::UNKNOWN,
                                grpc::StatusCode::DEADLINE_EXCEEDED, grpc::StatusCode::CANCELLED,
                                grpc::StatusCode::INVALID_ARGUMENT, grpc::StatusCode::RESOURCE_EXHAUSTED}) {
            SCOPED_TRACE(std::to_string(start) + ":" + std::to_string(code));
            auto resource = std::make_shared<KVCacheResource>();
            std::weak_ptr<KVCacheResource> owner = resource;
            int releases = 0;
            auto context = makeWriteContext(std::move(resource), "legacy-peer", currentTimeMs() + 60000,
                                            nullptr, 100, 1000, [&] { ++releases; });
            ASSERT_TRUE(context->beginKickoff());
            ASSERT_TRUE(context->beginPeerCall({}));
            if (start) {
                auto call = std::make_shared<DecodeWriteHelper::Result::WorkerRpcContext>();
                call->status = grpc::Status(code, "grpc_code=12 is not a stop proof");
                auto result = std::make_shared<DecodeWriteHelper::Result>(
                    std::vector<std::shared_ptr<DecodeWriteHelper::Result::WorkerRpcContext>>{call});
                context->setPeerCallResult(result);
                context->checkDone();
                EXPECT_FALSE(context->peer_stopped_);
                result->finishRank(0, true);
                context->checkDone();
            } else {
                context->cancel();
                auto call = std::make_shared<DecodeWriteHelper::CancelResult::WorkerRpcContext>();
                call->status = grpc::Status(code, "grpc_code=12 is not a stop proof");
                auto result = std::make_shared<DecodeWriteHelper::CancelResult>(
                    std::vector<std::shared_ptr<DecodeWriteHelper::CancelResult::WorkerRpcContext>>{call});
                context->peer_cancel_result_ = result;
                context->checkCancelDone();
                EXPECT_FALSE(context->peer_stopped_);
                result->finishRank(0, true);
                context->checkCancelDone();
            }
            const bool peer_never_started = start && code == grpc::StatusCode::UNIMPLEMENTED;
            EXPECT_EQ(context->peer_stopped_, peer_never_started);
            EXPECT_FALSE(context->transfer_stopped_);
            EXPECT_TRUE(context->resourceHoldPending());
            EXPECT_FALSE(owner.expired());
            EXPECT_EQ(releases, 0);
            confirmSourceStopped(context);
            EXPECT_EQ(owner.expired(), peer_never_started);
            EXPECT_EQ(releases, peer_never_started ? 1 : 0);
            if (!peer_never_started) {
                EXPECT_TRUE(context->resourceHoldPending());
                auto stopped = peerResult();
                context->setPeerCallResult(stopped);
                stopped->finishRank(0, true);
                context->checkDone();
            }
            context->checkDone();
            context->checkCancelDone();
            confirmSourceStopped(context);
            EXPECT_FALSE(context->resourceHoldPending());
            EXPECT_TRUE(owner.expired());
            EXPECT_EQ(releases, 1);
        }
    }
}

TEST(P2PConnectorAsyncWriteContextTest, CancelUnimplementedRetainsSourceAfterInflightOrAcceptedStart) {
    for (const bool accepted : {false, true}) {
        SCOPED_TRACE(accepted);
        auto                           resource = std::make_shared<KVCacheResource>();
        std::weak_ptr<KVCacheResource> owner    = resource;
        int                            releases = 0;
        auto                           context  = makeWriteContext(
            std::move(resource), "cancel-incompatible", currentTimeMs() + 60000, nullptr, 100, 1000, [&] {
                ++releases;
            });
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        auto start = peerResult(ErrorCode::NONE_ERROR, false, 1);
        context->setPeerCallResult(start);
        if (accepted)
            start->finishRank(0, true);
        context->checkDone();
        context->cancel();
        auto call      = std::make_shared<DecodeWriteHelper::CancelResult::WorkerRpcContext>();
        call->status   = grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "cancel route is incompatible");
        auto cancelled = std::make_shared<DecodeWriteHelper::CancelResult>(
            std::vector<std::shared_ptr<DecodeWriteHelper::CancelResult::WorkerRpcContext>>{call});
        cancelled->finishRank(0, true);
        context->peer_cancel_result_ = cancelled;
        context->checkCancelDone();
        confirmSourceStopped(context);
        EXPECT_FALSE(context->peer_stopped_);
        EXPECT_TRUE(context->resourceHoldPending());
        EXPECT_FALSE(owner.expired());
        EXPECT_EQ(releases, 0);
        auto stopped = peerResult();
        context->setPeerCallResult(stopped);
        stopped->finishRank(0, true);
        context->checkDone();
        EXPECT_FALSE(context->resourceHoldPending());
        EXPECT_TRUE(owner.expired());
        context->checkCancelDone();
        context->cancel();
        EXPECT_EQ(releases, 1);
    }
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
    EXPECT_FALSE(context->metrics_.no_transfer);
    context->onDone([&](ErrorInfo error) {
        ++callback_count;
        EXPECT_EQ(error.code(), ErrorCode::CANCELLED);
    });
    EXPECT_EQ(callback_count, 2);
}

TEST(P2PConnectorAsyncWriteContextTest, CancelPreservesAggregatedTransferErrorUntilBothSidesStop) {
    auto resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner = resource;
    auto                           context =
        makeWriteContext(std::move(resource), "aggregated-error", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    auto peer = peerResult(ErrorCode::CACHE_STORE_STORE_FAILED, false);
    context->setPeerCallResult(peer);
    peer->finishRank(0, true);
    context->cancel();
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    confirmSourceStopped(context);
    EXPECT_TRUE(context->resourceHoldPending());
    auto stopped = peerResult(ErrorCode::CACHE_STORE_STORE_FAILED);
    context->setPeerCallResult(stopped);
    stopped->finishRank(0, true);
    context->checkDone();
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
}

TEST(P2PConnectorAsyncWriteContextTest, CancelPreservesCompletedStartWriteBackResult) {
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
        if (expected != ErrorCode::NONE_ERROR)
            confirmSourceStopped(context);
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
    for (const int count : {0, 2}) {
        auto context =
            makeWriteContext(nullptr, "completed-before-deadline", currentTimeMs() + 4000, nullptr, 100, 1000);
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        auto peer = peerResult(ErrorCode::NONE_ERROR, true, count);
        context->setPeerCallResult(peer);
        peer->finishRank(0, true);
        context->transfer_stop_at_ms_ = P2PNotification::nowMs() - 1;
        context->expireTransferDeadlineIfNeeded();
        EXPECT_TRUE(context->done());
        EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
        EXPECT_FALSE(context->resourceHoldPending());
    }
}

TEST(P2PConnectorAsyncWriteContextTest, PendingStartWriteBackTimesOutWithoutReleasingSource) {
    auto                           resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner    = resource;
    auto context = makeWriteContext(std::move(resource), "pending-peer", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    auto peer = peerResult();
    context->setPeerCallResult(peer);
    context->transfer_stop_at_ms_ = P2PNotification::nowMs() - 1;
    context->expireTransferDeadlineIfNeeded();
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    peer->finishRank(0, true);
    context->checkDone();
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_TRUE(owner.expired());
}

TEST(P2PConnectorAsyncWriteContextTest, StartWriteBackBusinessErrorSurvivesCancellationAndSourceCleanup) {
    auto                           resource = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> owner    = resource;
    auto                           context =
        makeWriteContext(std::move(resource), "first-business-error", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    auto peer = peerResult(ErrorCode::CACHE_STORE_STORE_FAILED);
    context->setPeerCallResult(peer);
    peer->finishRank(0, true);
    context->cancel();
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
    EXPECT_TRUE(context->peer_stopped_);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    confirmSourceStopped(context);
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_STORE_STORE_FAILED);
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
    P2PConnectorStartWriteBackResponsePB response;
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

TEST_F(P2PConnectorWriteSchedulerTest, MissingWorkerLeaseStatusRetainsTargetsUntilRecovery) {
    prefill_.reset();
    prefill_config_.p2p_lease_query_timeout_ms = 20;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    for (auto& worker : prefill_workers_) {
        worker.fail_start = true;
        worker.omit_status = true;
    }
    auto input = request();
    input.set_timeout_ms(100);
    const auto free_before = allocator_->freeBlocksNum();
    P2PConnectorStartWriteBackResponsePB response;
    auto call = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    EXPECT_TRUE(await([&] { return prefill_workers_[0].starts && prefill_workers_[1].starts; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(call.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
    for (auto& worker : prefill_workers_) {
        worker.omit_status = false;
    }
    completeAll();
    call.get();
    EXPECT_TRUE(response.stopped());
    EXPECT_EQ(response.accepted_block_count(), 0);
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverQuerySubmissionFailuresRepeatCancellationBeforeReleasingTargets) {
    auto client = std::make_shared<LeaseQueryFaultClient>(prefill_addrs_, 100);
    ASSERT_TRUE(client->init());
    prefill_.reset();
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, client);
    ASSERT_TRUE(prefill_->init());
    for (auto& worker : prefill_workers_)
        worker.fail_start = true;
    for (const auto mode : {1, 2}) {
        SCOPED_TRACE(mode);
        auto input = request();
        input.set_unique_key("receiver-query-submission-" + std::to_string(mode));
        const auto                           free_before    = allocator_->freeBlocksNum();
        const auto                           queries_before = prefill_workers_[0].lease_queries.load();
        P2PConnectorStartWriteBackResponsePB response;
        auto call    = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
        auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) {
            client->failure_mode = 0;
            completeAll();
        });
        ASSERT_TRUE(await([&] { return prefill_workers_[0].lease_queries > queries_before; }));
        const auto cancels_before  = prefill_workers_[0].cancels.load();
        const auto failures_before = client->failures.load();
        client->failure_mode       = mode;
        ASSERT_TRUE(await([&] { return client->failures > failures_before; }));
        EXPECT_TRUE(await([&] { return prefill_workers_[0].cancels > cancels_before; }));
        EXPECT_EQ(call.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
        client->failure_mode = 0;
        completeAll();
        ASSERT_EQ(call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        call.get();
        EXPECT_TRUE(response.stopped());
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, SenderQuerySubmissionFailuresClearCancelConfirmationAndRetry) {
    auto client = std::make_shared<LeaseQueryFaultClient>(decode_addrs_, 100);
    ASSERT_TRUE(client->init());
    auto pool = std::make_shared<autil::ThreadPool>(1, 8, nullptr, "QueryFailureControl");
    ASSERT_TRUE(pool->start());
    for (const auto scenario : {0, 1, 2, 3}) {
        SCOPED_TRACE(scenario);
        const auto                     mode              = scenario / 2 + 1;
        const bool                     already_cancelled = scenario % 2 != 0;
        auto                           resource          = source();
        std::weak_ptr<KVCacheResource> owner             = resource;
        std::atomic<int>               releases{0};
        auto                           context = makeWriteContext(std::move(resource),
                                        "sender-query-submission-" + std::to_string(scenario),
                                        currentTimeMs() + 4000,
                                        client,
                                        100,
                                        100,
                                        [&] { ++releases; });
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        context->cancel();
        context->peer_stopped_     = true;
        context->worker_cancelled_ = already_cancelled;
        const auto cancels_before  = decode_workers_[0].cancels.load();
        const auto failures_before = client->failures.load();
        client->failure_mode       = mode;
        context->runControl(pool);
        ASSERT_TRUE(await([&] {
            std::lock_guard<std::mutex> lock(context->mutex_);
            return !context->control_submitting_ && client->failures > failures_before;
        }));
        context->checkCancelDone();
        EXPECT_FALSE(context->worker_cancelled_);
        EXPECT_TRUE(context->resourceHoldPending());
        EXPECT_FALSE(owner.expired());
        EXPECT_EQ(releases, 0);
        if (!already_cancelled) {
            EXPECT_TRUE(await([&] { return decode_workers_[0].cancels > cancels_before; }));
        }
        const auto cancels_after_failure = decode_workers_[0].cancels.load();
        EXPECT_EQ(cancels_after_failure - cancels_before, already_cancelled ? 0 : 1);
        client->failure_mode = 0;
        EXPECT_TRUE(await([&] {
            context->checkCancelDone();
            context->runControl(pool);
            return !context->resourceHoldPending();
        }));
        EXPECT_GT(decode_workers_[0].cancels, cancels_after_failure);
        EXPECT_TRUE(owner.expired());
        EXPECT_EQ(releases, 1);
        context->checkCancelDone();
        context->cancel();
        EXPECT_EQ(releases, 1);
    }
    pool->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
}

TEST_F(P2PConnectorWriteSchedulerTest, ReceiverQueryQueueRejectionRepeatsConfirmedCancelAndRetainsTargetsUntilStop) {
    auto client = std::make_shared<LeaseQueryFaultClient>(prefill_addrs_, 100);
    ASSERT_TRUE(client->init());
    auto pool = std::make_shared<RejectingWriteControlPool>();
    ASSERT_TRUE(pool->start());
    auto reject_queries  = std::make_shared<std::atomic<bool>>(true);
    client->before_query = [pool, reject_queries] {
        if (*reject_queries)
            pool->reject_submissions = true;
    };
    prefill_.reset();
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, client);
    ASSERT_TRUE(prefill_->init());
    prefill_->control_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    prefill_->control_pool_ = pool;
    for (auto& worker : prefill_workers_)
        worker.fail_start = true;
    auto input = request();
    input.set_unique_key("receiver-query-queue-rejection");
    const auto                           free_before = allocator_->freeBlocksNum();
    P2PConnectorStartWriteBackResponsePB response;
    auto call    = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) {
        *reject_queries          = false;
        pool->reject_submissions = false;
        completeAll();
    });
    ASSERT_TRUE(await([&] { return pool->rejections > 0; }));
    ASSERT_GT(prefill_workers_[0].cancels, 0);
    ASSERT_GT(prefill_workers_[0].lease_queries, 0);
    auto state = prefill_->findRequest(input.unique_key());
    ASSERT_NE(state, nullptr);
    std::weak_ptr<KVCacheResource> owner;
    {
        std::lock_guard<std::mutex> lock(prefill_->requests_mutex_);
        owner = prefill_->retained_resources_.at(input.unique_key());
    }
    EXPECT_FALSE(owner.expired());
    EXPECT_TRUE(state->resourceHoldPending());
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
    const auto cancels_before = prefill_workers_[0].cancels.load();
    *reject_queries           = false;
    pool->reject_submissions  = false;
    ASSERT_TRUE(await([&] { return prefill_workers_[0].cancels > cancels_before; }));
    EXPECT_FALSE(owner.expired());
    EXPECT_TRUE(state->resourceHoldPending());
    prefill_workers_[0].completeAll();
    const auto queries_before = prefill_workers_[1].lease_queries.load();
    ASSERT_TRUE(await([&] { return prefill_workers_[1].lease_queries > queries_before; }));
    EXPECT_FALSE(owner.expired());
    EXPECT_TRUE(state->resourceHoldPending());
    EXPECT_EQ(call.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
    completeAll();
    ASSERT_EQ(call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    call.get();
    EXPECT_TRUE(response.stopped());
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    P2PConnectorCancelWriteBackRequestPB cancel;
    cancel.set_unique_key(input.unique_key());
    cancel.set_deadline_ms(input.deadline_ms());
    for (int attempt = 0; attempt < 2; ++attempt) {
        prefill_->cancelWrite(cancel, response);
        prefill_->checkRetainedResources();
        EXPECT_TRUE(response.stopped());
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    }
}

TEST_F(P2PConnectorWriteSchedulerTest, SenderQueryQueueRejectionRepeatsConfirmedCancelAndReleasesSourceOnce) {
    const std::string key      = "sender-query-queue-rejection";
    const auto        deadline = currentTimeMs() + 4000;
    for (auto& worker : decode_workers_)
        worker.fail_start = true;
    P2PBroadcastClient::BroadcastParams params;
    params.unique_key          = key;
    params.type                = WRITE;
    params.deadline_ms         = deadline;
    params.request_deadline_ms = deadline;
    auto started               = decode_client_->broadcast(std::move(params));
    ASSERT_NE(started, nullptr);
    ASSERT_TRUE(await([&] { return started->done(); }, 1000));
    auto pool = std::make_shared<RejectingWriteControlPool>();
    ASSERT_TRUE(pool->start());
    auto                           resource = source();
    std::weak_ptr<KVCacheResource> owner    = resource;
    std::atomic<int>               releases{0};
    auto context = makeWriteContext(std::move(resource), key, deadline, decode_client_, 100, 100, [&] { ++releases; });
    auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) {
        pool->reject_submissions = false;
        completeAll();
        pool->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    });
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    context->cancel();
    context->peer_stopped_ = true;
    ASSERT_TRUE(await([&] {
        context->checkCancelDone();
        context->runControl(pool);
        std::lock_guard<std::mutex> lock(context->mutex_);
        return context->worker_cancelled_ && !context->control_submitting_ && !context->lease_result_;
    }));
    const auto cancels_before = decode_workers_[0].cancels.load();
    ASSERT_GT(cancels_before, 0);
    pool->reject_submissions = true;
    ASSERT_TRUE(await([&] {
        context->runControl(pool);
        return pool->rejections > 0;
    }));
    EXPECT_FALSE(context->worker_cancelled_);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(releases, 0);
    pool->reject_submissions = false;
    ASSERT_TRUE(await([&] {
        context->checkCancelDone();
        context->runControl(pool);
        return decode_workers_[0].cancels > cancels_before;
    }));
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(releases, 0);
    decode_workers_[0].completeAll();
    const auto queries_before = decode_workers_[1].lease_queries.load();
    ASSERT_TRUE(await([&] {
        context->checkCancelDone();
        context->runControl(pool);
        return decode_workers_[1].lease_queries > queries_before;
    }));
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(releases, 0);
    completeAll();
    ASSERT_TRUE(await([&] {
        context->checkCancelDone();
        context->runControl(pool);
        return !context->resourceHoldPending();
    }));
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(releases, 1);
    context->checkCancelDone();
    context->cancel();
    EXPECT_EQ(releases, 1);
}

TEST_F(P2PConnectorWriteSchedulerTest, RealReceiversRecreateCleanedLeaseAndWaitForEveryRankBeforeOneRelease) {
    std::vector<std::shared_ptr<SchedulerWriteReceiver>> receivers;
    std::vector<std::shared_ptr<P2PConnector>>           connectors;
    for (int rank = 0; rank < 2; ++rank) {
        P2PConnectorConfig config;
        config.role_type                               = RoleType::PREFILL;
        config.p2p_writeback_enable                    = true;
        config.worker_config.topology                  = config_.topologyPtr();
        config.worker_config.p2p_cancelled_keys_ttl_ms = 1;
        auto receiver                                  = std::make_shared<SchedulerWriteReceiver>();
        auto converter                                 = std::make_shared<LayerBlockConverterImpl>(allocator_);
        auto connector                                 = std::make_shared<P2PConnector>(config, converter, nullptr);
        connector->prefill_ = std::make_unique<P2PConnectorPrefill>(config, converter, nullptr);
        connector->prefill_->write_worker_ =
            std::make_unique<P2PWorkerPrefillWrite>(config.worker_config, converter, receiver);
        prefill_workers_[rank].execute_write = [connector](const auto& request, auto& response) {
            connector->executeFunction(request, response);
        };
        receivers.push_back(std::move(receiver));
        connectors.push_back(std::move(connector));
    }
    const auto                           input       = request();
    const auto                           free_before = allocator_->freeBlocksNum();
    P2PConnectorStartWriteBackResponsePB response;
    auto call    = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) {
        for (const auto& receiver : receivers)
            receiver->finish();
    });
    ASSERT_TRUE(await([&] { return prefill_workers_[0].lease_queries > 0 && prefill_workers_[1].lease_queries > 0; }));
    auto                           releases = std::make_shared<std::atomic<int>>(0);
    std::weak_ptr<KVCacheResource> owner;
    {
        std::lock_guard<std::mutex> lock(prefill_->requests_mutex_);
        auto&                       retained = prefill_->retained_resources_.at(input.unique_key());
        owner                                = retained;
        retained = KVCacheResourcePtr(retained.get(), [resource = retained, releases](KVCacheResource*) mutable {
            resource.reset();
            ++*releases;
        });
    }
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(releases->load(), 0);

    receivers[0]->finish();
    auto& first  = *connectors[0]->prefill_->write_worker_;
    bool  sealed = false, stopped = false;
    int   started = 0, finished = 0;
    ASSERT_TRUE(first.queryLeaseStatus(input.unique_key(), sealed, started, finished, stopped));
    EXPECT_TRUE(stopped);
    EXPECT_GT(started, 0);
    EXPECT_EQ(started, finished);
    {
        std::lock_guard<std::mutex> lock(first.mutex_);
        const auto                  group = first.groups_.at(input.unique_key());
        std::lock_guard<std::mutex> group_lock(group->mutex);
        group->stop_at_ms = P2PNotification::nowMs() - first.config_.p2p_cancelled_keys_ttl_ms - 1;
    }
    first.cleanup();
    ASSERT_FALSE(first.queryLeaseStatus(input.unique_key(), sealed, started, finished, stopped));
    const auto cancels_before = prefill_workers_[0].cancels.load();
    ASSERT_TRUE(await([&] {
        return prefill_workers_[0].cancels > cancels_before
               && first.queryLeaseStatus(input.unique_key(), sealed, started, finished, stopped);
    }));
    ASSERT_TRUE(first.queryLeaseStatus(input.unique_key(), sealed, started, finished, stopped));
    EXPECT_TRUE(sealed);
    EXPECT_TRUE(stopped);
    EXPECT_EQ(started, 0);
    EXPECT_EQ(finished, 0);
    auto& second = *connectors[1]->prefill_->write_worker_;
    ASSERT_TRUE(second.queryLeaseStatus(input.unique_key(), sealed, started, finished, stopped));
    EXPECT_TRUE(sealed);
    EXPECT_FALSE(stopped);
    EXPECT_GT(started, finished);
    EXPECT_EQ(call.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    EXPECT_EQ(prefill_->inflightContextCount(), 1u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(releases->load(), 0);

    receivers[1]->finish();
    ASSERT_EQ(call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    call.get();
    EXPECT_TRUE(response.stopped());
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(response.accepted_block_count(), 0);
    EXPECT_EQ(prefill_->inflightContextCount(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(releases->load(), 1);
    for (const auto& connector : connectors) {
        auto& worker = *connector->prefill_->write_worker_;
        worker.cleanup();
        EXPECT_TRUE(worker.cancelWrite(input.unique_key(), currentTimeMs() + 1000));
        EXPECT_TRUE(worker.queryLeaseStatus(input.unique_key(), sealed, started, finished, stopped));
        EXPECT_TRUE(stopped);
    }
    P2PConnectorStartWriteBackResponsePB retried;
    prefill_->handleWrite(input, retried);
    EXPECT_TRUE(retried.stopped());
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    EXPECT_EQ(releases->load(), 1);
    EXPECT_EQ(allocator_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(P2PConnectorWriteSchedulerTest, MissingSenderAcknowledgmentsRetainSourceUntilRecovery) {
    for (auto& worker : decode_workers_) {
        worker.omit_status = true;
    }
    auto resource = source();
    std::weak_ptr<KVCacheResource> owner = resource;
    auto context = makeWriteContext(
        std::move(resource), "source-stop-unconfirmed", currentTimeMs() + 100, decode_client_, 20, 20);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    auto peer = peerResult(ErrorCode::CANCELLED);
    context->setPeerCallResult(peer);
    peer->finishRank(0, true);
    context->checkDone();
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init(decode_config_));
    ASSERT_TRUE(checker.addContext(context));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(owner.expired());
    for (auto& worker : decode_workers_) {
        worker.omit_status = false;
    }
    EXPECT_TRUE(await([&] { return !context->resourceHoldPending(); }));
    checker.stop();
    EXPECT_TRUE(owner.expired());
}

TEST_F(P2PConnectorWriteSchedulerTest, CompletedLeaseQueryReleasesExpiredSource) {
    completeAll(true);
    const auto deadline = currentTimeMs() + 3000;
    auto       result   = decode_client_->queryLeaseStatusAsync("stopped-before-grace", WRITE, 1000);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->waitDone(2000, false));
    ASSERT_TRUE(P2PBroadcastClient::leaseStatus(result).allStopped());
    auto context = makeWriteContext(source(), "stopped-before-grace", deadline, decode_client_, 100, 100);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    context->cancel();
    context->peer_stopped_ = true;
    {
        std::lock_guard<std::mutex> lock(context->mutex_);
        context->transfer_stop_at_ms_ = P2PNotification::nowMs() - 500;
        context->lease_result_ = result;
    }
    context->checkCancelDone();
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->resourceHoldPending());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CANCELLED);
}

TEST_F(P2PConnectorWriteSchedulerTest, SlowControlConnectionDoesNotBlockAdmissionOrDeadline) {
    prefill_.reset();
    prefill_config_.p2p_kv_leak_exit_threshold_percent = 50;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    auto first               = request();
    first.set_timeout_ms(200);
    first.set_deadline_ms(currentTimeMs() + 200);
    P2PConnectorStartWriteBackResponsePB first_response, second_response;
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

TEST_F(P2PConnectorWriteSchedulerTest, QueuedReceiveControlUsesIndependentBoundedQueryBudget) {
    prefill_.reset();
    prefill_config_.p2p_cancel_broadcast_timeout_ms = 10000;
    prefill_config_.p2p_lease_query_timeout_ms      = 1000;
    prefill_ = std::make_unique<P2PSchedulerPrefillWrite>(prefill_config_, allocator_, nullptr, prefill_client_);
    ASSERT_TRUE(prefill_->init());
    auto input               = request();
    input.set_deadline_ms(currentTimeMs() + 2000);
    P2PConnectorStartWriteBackResponsePB response;
    auto               call = std::async(std::launch::async, [&] { prefill_->handleWrite(input, response); });
    std::promise<void> resume;
    auto               gate    = resume.get_future().share();
    bool               resumed = false;
    auto               cleanup = std::shared_ptr<void>(nullptr, [&](void*) {
        if (!resumed)
            resume.set_value();
        completeAll(true);
    });
    ASSERT_TRUE(await([&] { return decode_workers_[0].starts != 0; }, 1000));
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
    P2PConnectorCancelWriteBackRequestPB cancel;
    cancel.set_unique_key(input.unique_key());
    cancel.set_deadline_ms(input.deadline_ms());
    P2PConnectorStartWriteBackResponsePB cancel_response;
    prefill_->cancelWrite(cancel, cancel_response);
    ASSERT_TRUE(await([&] { return pool->getItemCount() != 0; }, 1000));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto resumed_at_ms = currentTimeMs();
    resume.set_value();
    resumed = true;
    ASSERT_TRUE(await(
        [&] { return prefill_workers_[0].control_deadline_ms != 0 && prefill_workers_[1].control_deadline_ms != 0; },
        1000));
    for (const auto& worker : prefill_workers_) {
        EXPECT_GE(worker.control_deadline_ms.load(), resumed_at_ms + 900);
        EXPECT_LE(worker.control_deadline_ms.load(), currentTimeMs() + 1100);
    }
    cleanup.reset();
    call.get();
    EXPECT_TRUE(response.stopped());
}

TEST_F(P2PConnectorWriteSchedulerTest, PeerOnlyCancellationUsesBackoffUntilReceiverStops) {
    auto context = makeWriteContext(source(), "peer-cancel-backoff", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
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
    confirmSourceStopped(context);
    context->cancel();
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init(decode_config_));
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
        ASSERT_TRUE(context->beginKickoff());
        ASSERT_TRUE(context->beginPeerCall({}));
        auto peer         = peerResult();
        auto notification = std::make_shared<P2PNotification>();
        if (result_first)
            context->setPeerCallResult(peer);
        context->setNotification(notification);
        if (!result_first)
            context->setPeerCallResult(peer);
        const auto generation = notification->generation();
        peer->finishRank(0, true);
        EXPECT_GT(notification->generation(), generation);
        context->checkDone();
        EXPECT_TRUE(context->success());
        auto late = std::make_shared<P2PNotification>();
        context->setNotification(late);
        EXPECT_GT(late->generation(), 0u);
    }
}

TEST(P2PConnectorAsyncWriteContextTest, StaleCancelMustNotEraseStoppedStartWriteBackProof) {
    auto context = std::make_shared<P2PConnectorAsyncWriteContext>(
        "stopped-proof-race", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
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
    confirmSourceStopped(context);
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
    ASSERT_TRUE(context->beginPeerCall({}));
    auto peer = peerResult();
    context->setPeerCallResult(peer);
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
    peer->finishRank(0, true);
    EXPECT_EQ(release_result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(await([&] { return !context->resourceHoldPending(); }));
    unblock.reset();
    stop.get();
}

TEST(P2PConnectorAsyncWriteContextTest, CheckerWakesAtDeadlineWithoutRpcCompletion) {
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init(writeCheckerConfig()));
    auto released       = std::make_shared<std::promise<void>>();
    auto release_result = released->get_future();
    auto context =
        makeWriteContext(nullptr, "deadline-notification", currentTimeMs() + 100, nullptr, 100, 1000, [released]() {
            released->set_value();
        });
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    auto peer = peerResult();
    context->setPeerCallResult(peer);
    auto completed = std::make_shared<std::promise<ErrorInfo>>();
    auto result = completed->get_future();
    context->onDone([completed](ErrorInfo error) { completed->set_value(error); });
    ASSERT_TRUE(checker.addContext(context));
    ASSERT_EQ(result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(result.get().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_TRUE(context->resourceHoldPending());
    peer->finishRank(0, true);
    ASSERT_EQ(release_result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    checker.stop();
    EXPECT_FALSE(context->resourceHoldPending());
}

TEST(P2PConnectorAsyncWriteContextTest, StartWriteBackCompletionWakesCheckerAndReleasesSource) {
    P2PConnectorAsyncWriteContextChecker checker;
    ASSERT_TRUE(checker.init(writeCheckerConfig()));
    auto context = makeWriteContext(nullptr, "peer-completion", currentTimeMs() + 4000, nullptr, 100, 1000);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->beginPeerCall({}));
    auto peer = peerResult();
    context->setPeerCallResult(peer);
    auto released   = std::make_shared<std::promise<ErrorInfo>>();
    auto completion = released->get_future();
    context->onDone([released](ErrorInfo error) { released->set_value(error); });
    ASSERT_TRUE(checker.addContext(context));
    EXPECT_TRUE(context->resourceHoldPending());
    peer->finishRank(0, true);
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(completion.get().ok());
    EXPECT_FALSE(context->resourceHoldPending());
    checker.stop();
}

TEST(P2PWritebackKvReserveTest, RatioMustBeWithinZeroAndOne) {
    EXPECT_TRUE(validWritebackKvReserveRatio(0.0));
    EXPECT_TRUE(validWritebackKvReserveRatio(1.0));
    EXPECT_FALSE(validWritebackKvReserveRatio(-0.1));
    EXPECT_FALSE(validWritebackKvReserveRatio(1.5));
    EXPECT_FALSE(validWritebackKvReserveRatio(std::numeric_limits<double>::quiet_NaN()));
}

TEST(P2PWriteRpcUtilTest, CallForResultKeepsFailureCause) {
    std::string failure;
    EXPECT_EQ(*p2p_internal::callForResult([] { return std::make_shared<int>(7); }, failure), 7);
    EXPECT_TRUE(failure.empty());
    EXPECT_EQ(p2p_internal::callForResult([] { return std::shared_ptr<int>{}; }, failure), nullptr);
    EXPECT_EQ(failure, "no result");
    EXPECT_EQ(p2p_internal::callForResult(
                  []() -> std::shared_ptr<int> { throw std::runtime_error("connection refused"); }, failure),
              nullptr);
    EXPECT_EQ(failure, "connection refused");
}

TEST(P2PWriteRpcUtilTest, PeerClientCacheEvictsOnlyIdleUnheldClients) {
    p2p_internal::PeerClientCache<std::string, int> cache(/*idle_ttl_ms=*/1);
    int                                             created = 0;
    const auto                                      create  = [&] { return std::make_shared<int>(++created); };
    const auto idle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); };

    auto held = cache.get("held", create);
    EXPECT_EQ(cache.get("held", create), held);
    cache.get("dropped", create);
    EXPECT_EQ(created, 2);

    idle();
    EXPECT_EQ(cache.get("held", create), held);
    EXPECT_EQ(*cache.get("dropped", create), 3);

    held.reset();
    idle();
    EXPECT_EQ(*cache.get("held", create), 4);

    EXPECT_EQ(cache.get("failed", [] { return std::shared_ptr<int>{}; }), nullptr);
    EXPECT_EQ(*cache.get("failed", create), 5);
}

}  // namespace
}  // namespace rtp_llm
