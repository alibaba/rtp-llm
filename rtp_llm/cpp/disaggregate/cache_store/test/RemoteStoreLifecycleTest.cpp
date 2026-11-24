#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <thread>

#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"

namespace rtp_llm {
namespace {

bool lockAvailable(std::shared_mutex& mutex) {
    return std::async(std::launch::async,
                      [&]() {
                          std::unique_lock<std::shared_mutex> lock(mutex, std::try_to_lock);
                          return lock.owns_lock();
                      })
        .get();
}

std::shared_ptr<BlockBuffer> hostBlock(const std::string& key, const std::shared_ptr<std::string>& owner) {
    return std::make_shared<BlockBuffer>(key, std::shared_ptr<void>(owner, owner->data()), owner->size(), false, true);
}

std::shared_ptr<RemoteStoreTaskImpl> makeTask(std::initializer_list<std::string> keys) {
    auto request = std::make_shared<RemoteStoreRequest>(
        "client", "request", "127.0.0.1", 1, 0, currentTimeUs() + 1000000, 1, 0, 1, 0);
    for (const auto& key : keys) {
        request->buffer_pairs[key] = "remote-" + key;
    }
    return std::make_shared<RemoteStoreTaskImpl>(
        request, std::make_shared<CacheStoreRemoteStoreMetricsCollector>(nullptr, keys.size()), nullptr);
}

std::shared_ptr<TransferRequest> dispatch(const std::shared_ptr<RemoteStoreTaskImpl>& task, const std::string& key) {
    return task->makeAvailableRequest(
        std::vector<std::shared_ptr<BlockBuffer>>{hostBlock(key, std::make_shared<std::string>("payload"))});
}

void complete(const std::shared_ptr<TransferRequest>& request, bool success = true) {
    request->callback(
        success, success ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadErrorUnknown, request->buffer_pairs);
}

TEST(RemoteStoreLifecycleTest, ScopedRegistrationCollisionRollsBack) {
    auto store    = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto owner    = std::make_shared<std::string>("bytes");
    auto existing = hostBlock("existing", owner);
    ASSERT_TRUE(store->regUserBuffers({existing}));
    EXPECT_EQ(nullptr, store->regUserBuffersScoped({hostBlock("new", owner), hostBlock("existing", owner)}));
    EXPECT_EQ(nullptr, store->findUserBuffer("new"));
    EXPECT_EQ(existing, store->findUserBuffer("existing"));
    EXPECT_EQ(nullptr, store->regUserBuffersScoped({existing}));
    EXPECT_EQ(existing, store->findUserBuffer("existing"));
    EXPECT_EQ(nullptr, store->regUserBuffersScoped({hostBlock("duplicate", owner), hostBlock("duplicate", owner)}));
    EXPECT_EQ(nullptr, store->findUserBuffer("duplicate"));
    EXPECT_EQ(nullptr, store->regUserBuffersScoped({nullptr}));
    auto valid = store->regUserBuffersScoped({hostBlock("new", owner)});
    ASSERT_NE(nullptr, valid);
    EXPECT_NE(nullptr, store->findUserBuffer("new"));
}

TEST(RemoteStoreLifecycleTest, UnregisterRetainsBorrowedOwner) {
    auto                       store        = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto                       owner        = std::make_shared<std::string>("borrowed payload");
    std::weak_ptr<std::string> weak_owner   = owner;
    auto                       registration = store->regUserBuffersScoped({hostBlock("key", owner)});
    ASSERT_NE(nullptr, registration);
    auto borrowed = store->findUserBuffer("key");
    owner.reset();
    registration->reset();
    registration->reset();
    EXPECT_EQ(nullptr, store->findUserBuffer("key"));
    ASSERT_FALSE(weak_owner.expired());
    EXPECT_EQ("borrowed payload", std::string(static_cast<char*>(borrowed->addr.get()), borrowed->len));
    borrowed.reset();
    EXPECT_TRUE(weak_owner.expired());
}

TEST(RemoteStoreLifecycleTest, UnregisterDoesNotEraseLegacyReplacement) {
    auto store        = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto owner        = std::make_shared<std::string>("payload");
    auto registration = store->regUserBuffersScoped({hostBlock("key", owner)});
    ASSERT_NE(nullptr, registration);
    auto replacement = hostBlock("key", owner);
    ASSERT_TRUE(store->regUserBuffers({replacement}));
    registration.reset();
    EXPECT_EQ(replacement, store->findUserBuffer("key"));
    auto surviving = store->regUserBuffersScoped({hostBlock("other", owner)});
    store.reset();
    surviving.reset();
}

TEST(RemoteStoreLifecycleTest, UniqueRequestsDoNotAccumulateTombstones) {
    auto store = std::make_shared<RequestBlockBufferStore>(nullptr);
    for (int i = 0; i < 100; ++i) {
        auto id = "unique-" + std::to_string(i);
        ASSERT_TRUE(store->setRequestBlockBufferWatchFunc(id, [](bool, const auto&) {}));
        ASSERT_TRUE(store->releaseScopedRequest(id));
    }
    EXPECT_TRUE(store->request_cache_map_.empty());
    EXPECT_TRUE(store->expired_request_caches_.empty());
    ASSERT_TRUE(store->setRequestBlockBufferWatchFunc("legacy", [](bool, const auto&) {}));
    store->delRequestBlockBuffer("legacy");
    EXPECT_FALSE(store->setRequestBlockBufferWatchFunc("legacy", [](bool, const auto&) {}));
    EXPECT_EQ(1, store->expired_request_caches_.size());
}

TEST(RemoteStoreLifecycleTest, FailureBeforeOtherRpcDrains) {
    auto task   = makeTask({"a", "b", "c"});
    auto first  = dispatch(task, "a");
    auto second = dispatch(task, "b");
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);
    int                        done       = 0;
    int                        drained    = 0;
    auto                       owner      = std::make_shared<std::string>("in-flight");
    std::weak_ptr<std::string> weak_owner = owner;
    ASSERT_TRUE(task->setDoneCallback([&](const ErrorInfo& error) {
        ++done;
        EXPECT_FALSE(error.ok());
        EXPECT_TRUE(task->done());
        EXPECT_FALSE(task->success());
    }));
    ASSERT_TRUE(task->setDrainedCallback([&, owner]() { ++drained; }));
    owner.reset();
    complete(first, false);
    EXPECT_EQ(1, done);
    EXPECT_EQ(0, drained);
    EXPECT_FALSE(weak_owner.expired());
    EXPECT_EQ(nullptr, dispatch(task, "c"));
    complete(second);
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
    EXPECT_TRUE(weak_owner.expired());
    EXPECT_FALSE(task->success());
}

TEST(RemoteStoreLifecycleTest, CallbacksRegisteredAfterCompletion) {
    auto task = makeTask({"a"});
    complete(dispatch(task, "a"));
    task->waitDone();
    EXPECT_TRUE(task->success());
    int done    = 0;
    int drained = 0;
    ASSERT_TRUE(task->setDoneCallback([&](const ErrorInfo& error) {
        ++done;
        EXPECT_TRUE(error.ok());
    }));
    ASSERT_TRUE(task->setDrainedCallback([&]() { ++drained; }));
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
    EXPECT_FALSE(task->setDoneCallback([&](const ErrorInfo&) { ++done; }));
    EXPECT_FALSE(task->setDrainedCallback([&]() { ++drained; }));
}

TEST(RemoteStoreLifecycleTest, DuplicateCompletionNotifiesOnce) {
    auto             task   = makeTask({"a", "b"});
    auto             first  = dispatch(task, "a");
    auto             second = dispatch(task, "b");
    std::atomic<int> done{0};
    std::atomic<int> drained{0};
    task->setDoneCallback([&](const ErrorInfo&) { ++done; });
    task->setDrainedCallback([&]() { ++drained; });
    std::thread one([&]() { complete(first); });
    std::thread duplicate([&]() { complete(first, false); });
    one.join();
    duplicate.join();
    EXPECT_EQ(0, drained);
    complete(second);
    complete(second, false);
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
}

TEST(RemoteStoreLifecycleTest, ExplicitCancelBeforeDispatchNeedsNoPoller) {
    auto task    = makeTask({"a"});
    int  done    = 0;
    int  drained = 0;
    task->setDoneCallback([&](const ErrorInfo& error) {
        ++done;
        EXPECT_EQ(ErrorCode::CANCELLED, error.code());
    });
    task->setDrainedCallback([&]() { ++drained; });
    EXPECT_TRUE(task->cancel(ErrorInfo(ErrorCode::CANCELLED, "producer failed")));
    EXPECT_TRUE(task->cancel(ErrorInfo(ErrorCode::UNKNOWN_ERROR, "late failure")));
    EXPECT_EQ(nullptr, dispatch(task, "a"));
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
    EXPECT_EQ(ErrorCode::CANCELLED, task->getErrorInfo().code());
}

TEST(RemoteStoreLifecycleTest, EmptyTaskCompletesAndReleasedTaskStillDrains) {
    auto empty = makeTask({});
    EXPECT_TRUE(empty->done());
    EXPECT_TRUE(empty->success());
    auto task    = makeTask({"a"});
    auto request = dispatch(task, "a");
    int  drained = 0;
    task->setDrainedCallback([&]() { ++drained; });
    task.reset();
    complete(request);
    EXPECT_EQ(1, drained);
}

TEST(RemoteStoreLifecycleTest, DrainWaitsForTerminalCallbackReturn) {
    auto               task   = makeTask({"a", "b"});
    auto               first  = dispatch(task, "a");
    auto               second = dispatch(task, "b");
    std::promise<void> started;
    std::promise<void> release;
    auto               released = release.get_future();
    std::atomic<int>   drained{0};
    task->setDoneCallback([&](const ErrorInfo&) {
        started.set_value();
        released.wait();
    });
    task->setDrainedCallback([&]() { ++drained; });
    std::thread failure([&]() { complete(first, false); });
    started.get_future().wait();
    complete(second);
    EXPECT_EQ(0, drained);
    release.set_value();
    failure.join();
    EXPECT_EQ(1, drained);
}

TEST(RemoteStoreLifecycleTest, PendingDoneHookCannotBeOvertakenByDrain) {
    auto                     task = makeTask({"not-admitted"});
    std::vector<std::string> notifications;
    ASSERT_TRUE(task->setDrainedCallback([&]() { notifications.push_back("drained"); }));
    {
        std::lock_guard<std::mutex> lock(task->mutex_);
        task->finishLocked(ErrorInfo(ErrorCode::CANCELLED, "producer ended"));
        // Notifier A has passed its first critical section without a done hook.
        ASSERT_FALSE(task->done_callback_);
        ASSERT_EQ(0, task->outstanding_rpcs_);
    }
    {
        std::lock_guard<std::mutex> lock(task->mutex_);
        // Stage setter B after publishing its hook and before notifyCallbacks.
        // This is the precise registration-between-critical-sections state.
        task->done_callback_registered_ = true;
        task->done_callback_            = [&](const ErrorInfo& error) {
            EXPECT_EQ(ErrorCode::CANCELLED, error.code());
            notifications.push_back("done");
        };
    }
    auto premature_drain = task->takeDrainedCallback();  // Notifier A's second critical section.
    EXPECT_FALSE(premature_drain);
    if (premature_drain) {
        premature_drain();
    }
    // Resume setter B: it must notify done before making drain available.
    task->notifyCallbacks();
    EXPECT_EQ((std::vector<std::string>{"done", "drained"}), notifications);
    EXPECT_FALSE(task->takeDrainedCallback());
    task->notifyCallbacks();
    EXPECT_EQ(2, notifications.size());
}

TEST(RemoteStoreLifecycleTest, LateDoneHookCannotRevokeExtractedDrain) {
    auto task    = makeTask({"not-admitted"});
    int  drained = 0;
    ASSERT_TRUE(task->setDrainedCallback([&]() { ++drained; }));
    {
        std::lock_guard<std::mutex> lock(task->mutex_);
        task->finishLocked(ErrorInfo(ErrorCode::CANCELLED, "producer ended"));
    }
    auto extracted_drain = task->takeDrainedCallback();
    ASSERT_TRUE(extracted_drain);
    std::promise<void> started;
    std::promise<void> release;
    auto               released = release.get_future();
    std::atomic<int>   done{0};
    std::thread        registration([&]() {
        EXPECT_TRUE(task->setDoneCallback([&](const ErrorInfo&) {
            ++done;
            started.set_value();
            released.wait();
        }));
    });
    started.get_future().wait();
    // A hook registered only after extraction cannot delay this owned callback.
    extracted_drain();
    EXPECT_EQ(1, drained);
    release.set_value();
    registration.join();
    task->notifyCallbacks();
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
    EXPECT_FALSE(task->takeDrainedCallback());
}

TEST(RemoteStoreLifecycleTest, ThrowingDoneCallbackStillDrains) {
    auto task    = makeTask({"a"});
    int  drained = 0;
    task->setDoneCallback([](const ErrorInfo&) { throw std::runtime_error("callback failed"); });
    task->setDrainedCallback([&]() { ++drained; });
    EXPECT_NO_THROW(complete(dispatch(task, "a")));
    EXPECT_EQ(1, drained);
}

TEST(RemoteStoreLifecycleTest, FailedWatchCancelsAndCleanupCanReenter) {
    auto store                         = std::shared_ptr<NormalCacheStore>(new NormalCacheStore);
    store->request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto request                       = std::make_shared<RemoteStoreRequest>(
        "client", "request", "127.0.0.1", 1, 0, currentTimeUs() + 1000000, 1, 0, 1, 0);
    request->buffer_pairs["a"] = "remote-a";
    auto task                  = store->submitRemoteStoreTask(
        request, std::make_shared<CacheStoreRemoteStoreMetricsCollector>(nullptr, 1), nullptr);
    int done    = 0;
    int drained = 0;
    task->setDoneCallback([&](const ErrorInfo& error) {
        ++done;
        EXPECT_FALSE(error.ok());
        store->releaseRemoteStoreTask(task);
        task->cancel(ErrorInfo(ErrorCode::CANCELLED, "reentrant cancel"));
    });
    task->setDrainedCallback([&]() {
        ++drained;
        EXPECT_TRUE(store->releaseScopedRequest("request"));
    });
    store->request_block_buffer_store_->getRequestBlockBuffer("request")->notifyRequestDone();
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
    store->releaseRemoteStoreTask(task);
    EXPECT_TRUE(store->remote_store_tasks_.empty());
    EXPECT_TRUE(store->request_block_buffer_store_->request_cache_map_.empty());
}

class SynchronousMessager: public Messager {
public:
    SynchronousMessager(): Messager(nullptr, nullptr, nullptr) {}
    bool init(MessagerInitParams) override {
        return true;
    }
    void load(const std::shared_ptr<LoadRequest>&,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>&) override {
        ADD_FAILURE();
    }
    bool generateBlockInfo(BlockBufferInfo*, const std::shared_ptr<BlockBuffer>&, uint32_t, uint32_t) override {
        return false;
    }
    void transfer(const std::shared_ptr<TransferRequest>& request) override {
        on_transfer(request);
    }
    std::function<void(const std::shared_ptr<TransferRequest>&)> on_transfer;
};

TEST(RemoteStoreLifecycleTest, SynchronousWatchCompletionCanReleaseTask) {
    auto store                         = std::shared_ptr<NormalCacheStore>(new NormalCacheStore);
    store->request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto messenger                     = std::make_shared<SynchronousMessager>();
    store->messager_                   = messenger;
    auto owner                         = std::make_shared<std::string>("payload");
    store->request_block_buffer_store_->getOrInsertRequestBlockBuffer("request")->addBlock(hostBlock("a", owner));
    int done               = 0;
    int drained            = 0;
    messenger->on_transfer = [&](const auto& transfer) {
        // Probe before reentry so a regression fails without hanging the suite.
        bool unlocked = lockAvailable(store->remote_store_tasks_mutex_);
        EXPECT_TRUE(unlocked);
        if (!unlocked) {
            complete(transfer, false);
            return;
        }
        auto task = store->remote_store_tasks_.at("request").front();
        task->setDoneCallback([&, task](const ErrorInfo& error) {
            ++done;
            EXPECT_TRUE(error.ok());
            store->releaseRemoteStoreTask(task);
        });
        task->setDrainedCallback([&]() {
            ++drained;
            store->releaseScopedRequest("request");
        });
        complete(transfer);
    };
    auto request = std::make_shared<RemoteStoreRequest>(
        "client", "request", "127.0.0.1", 1, 0, currentTimeUs() + 1000000, 1, 0, 1, 0);
    request->buffer_pairs["a"] = "remote-a";
    auto task                  = store->submitRemoteStoreTask(
        request, std::make_shared<CacheStoreRemoteStoreMetricsCollector>(nullptr, 1), nullptr);
    EXPECT_TRUE(task->success());
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
    EXPECT_TRUE(store->remote_store_tasks_.empty());
    EXPECT_TRUE(store->request_block_buffer_store_->request_cache_map_.empty());
}

TEST(RemoteStoreLifecycleTest, ConcurrentScopedRegistrationHasOneWinner) {
    auto                                    store = std::make_shared<RequestBlockBufferStore>(nullptr);
    auto                                    owner = std::make_shared<std::string>("bytes");
    std::shared_ptr<UserBufferRegistration> first;
    std::shared_ptr<UserBufferRegistration> second;
    std::promise<void>                      start;
    auto                                    ready = start.get_future().share();
    std::thread                             one([&]() {
        ready.wait();
        first = store->regUserBuffersScoped({hostBlock("key", owner)});
    });
    std::thread                             two([&]() {
        ready.wait();
        second = store->regUserBuffersScoped({hostBlock("key", owner)});
    });
    start.set_value();
    one.join();
    two.join();
    EXPECT_NE(bool(first), bool(second));
    EXPECT_NE(nullptr, store->findUserBuffer("key"));
    first.reset();
    second.reset();
    EXPECT_EQ(nullptr, store->findUserBuffer("key"));
}

TEST(RemoteStoreLifecycleTest, CancelledTaskDrainsAfterLateCallbackRegistration) {
    auto task    = makeTask({"a", "b"});
    auto request = dispatch(task, "a");
    ASSERT_NE(nullptr, request);
    ASSERT_TRUE(task->cancel(ErrorInfo(ErrorCode::CANCELLED, "cancel in flight")));
    int done    = 0;
    int drained = 0;
    ASSERT_TRUE(task->setDoneCallback([&](const ErrorInfo& error) {
        ++done;
        EXPECT_EQ(ErrorCode::CANCELLED, error.code());
    }));
    ASSERT_TRUE(task->setDrainedCallback([&]() { ++drained; }));
    EXPECT_EQ(1, done);
    EXPECT_EQ(0, drained);
    EXPECT_EQ(nullptr, dispatch(task, "b"));
    complete(request);
    complete(request, false);
    EXPECT_EQ(1, drained);
    EXPECT_FALSE(task->success());
}

TEST(RemoteStoreLifecycleTest, LegacyWaitDoneCancellationPredicateCanReenter) {
    auto task                = makeTask({"a"});
    task->check_cancel_func_ = [&]() {
        EXPECT_FALSE(task->done());
        return true;
    };
    task->waitDone();
    EXPECT_FALSE(task->success());
    EXPECT_EQ(ErrorCode::CANCELLED, task->getErrorInfo().code());
}

TEST(RemoteStoreLifecycleTest, ProducerQueueFailureCancellationCanReenterStore) {
    auto store                         = std::shared_ptr<NormalCacheStore>(new NormalCacheStore);
    store->request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(nullptr);
    // A real, stopped pool deterministically rejects dispatch.
    store->thread_pool_ = std::make_shared<autil::ThreadPool>(1, 1);
    store->thread_pool_->stop();
    auto task   = makeTask({"a"});
    auto buffer = std::make_shared<RequestBlockBuffer>("producer");
    buffer->addBlock(hostBlock("a", std::make_shared<std::string>("payload")));
    int done    = 0;
    int drained = 0;
    task->setDoneCallback([&](const ErrorInfo&) {
        ++done;
        bool unlocked = lockAvailable(store->store_tasks_mutex_);
        EXPECT_TRUE(unlocked);
        if (unlocked) {
            store->store(buffer, [](bool, CacheStoreErrorCode) {});
        }
    });
    task->setDrainedCallback([&]() { ++drained; });
    store->store_tasks_[buffer] = {[&](bool ok, CacheStoreErrorCode ec) {
                                       EXPECT_FALSE(ok);
                                       EXPECT_EQ(CacheStoreErrorCode::PushWorkerItemFailed, ec);
                                       task->cancel(ErrorInfo(ErrorCode::CANCELLED, "producer dispatch failed"));
                                   },
                                   []() { ADD_FAILURE() << "stopped pool must not execute the producer"; }};
    store->dispatchReadyStoreTasks();
    EXPECT_EQ(1, done);
    EXPECT_EQ(1, drained);
    EXPECT_EQ(1, store->store_tasks_.size());  // The reentrant producer was admitted.
}

}  // namespace
}  // namespace rtp_llm
