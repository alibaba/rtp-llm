#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "kvcm_client/kv_meta_object_client.h"
#include "rtp_llm/cpp/config/MMKvcmConfig.h"
#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClientKvcm.h"

namespace rtp_llm {
namespace {

class FakeNativeObjectClient final: public kv_cache_manager::KvMetaObjectClient {
public:
    kv_cache_manager::ClientErrorCode SaveObjects(const std::string&,
                                                  const std::vector<std::string>&       keys,
                                                  const std::vector<std::uint64_t>&     sizes,
                                                  const kv_cache_manager::BlockBuffers& buffers) override {
        const auto call = save_batches.size();
        save_batches.push_back(keys);
        save_sizes.push_back(sizes);
        save_buffers.push_back(buffers);
        if (save_hook) {
            save_hook(call);
        }
        if (block_save) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            save_entered = true;
            gate_condition.notify_all();
            gate_condition.wait(lock, [this]() { return allow_save; });
        }
        if (call == throw_save_call) {
            throw std::runtime_error("provider save secret");
        }
        return call == fail_save_call ? kv_cache_manager::ER_SDKWRITE_ERROR : kv_cache_manager::ER_OK;
    }

    kv_cache_manager::ClientErrorCode LoadObjects(const std::string&,
                                                   const std::vector<std::string>&   keys,
                                                   const std::vector<std::uint64_t>&,
                                                   const kv_cache_manager::BlockBuffers&) override {
        const auto call = load_batches.size();
        load_batches.push_back(keys);
        if (call == throw_load_call) {
            throw std::runtime_error("provider load secret");
        }
        return call == fail_load_call ? kv_cache_manager::ER_SDKREAD_ERROR : kv_cache_manager::ER_OK;
    }

    kv_cache_manager::ClientErrorCode Remove(const std::string& trace_id,
                                              const std::vector<std::string>& keys) override {
        const auto call = remove_batches.size();
        remove_traces.push_back(trace_id);
        remove_batches.push_back(keys);
        if (call == throw_remove_call) {
            throw 7;
        }
        return call == fail_remove_call ? kv_cache_manager::ER_INVALID_GRPCSTATUS : kv_cache_manager::ER_OK;
    }

    size_t throw_save_call   = std::numeric_limits<size_t>::max();
    size_t fail_save_call    = std::numeric_limits<size_t>::max();
    size_t throw_load_call   = std::numeric_limits<size_t>::max();
    size_t fail_load_call    = std::numeric_limits<size_t>::max();
    size_t throw_remove_call = std::numeric_limits<size_t>::max();
    size_t fail_remove_call  = std::numeric_limits<size_t>::max();
    bool   block_save        = false;
    bool   save_entered      = false;
    bool   allow_save        = false;

    std::function<void(size_t)> save_hook;

    std::mutex              gate_mutex;
    std::condition_variable gate_condition;

    std::vector<std::vector<std::string>>       save_batches;
    std::vector<std::vector<std::uint64_t>>     save_sizes;
    std::vector<kv_cache_manager::BlockBuffers> save_buffers;
    std::vector<std::vector<std::string>>       load_batches;
    std::vector<std::vector<std::string>>       remove_batches;
    std::vector<std::string>                    remove_traces;
};

struct ObjectBatch {
    explicit ObjectBatch(size_t count): storage(count, 1) {
        objects.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            objects.push_back({"object-" + std::to_string(i), &storage[i], 1, false});
        }
    }

    std::vector<std::uint8_t> storage;
    std::vector<MMKvcmBuffer> objects;
};

struct NativeAdapter {
    NativeAdapter(std::uint64_t max_object_bytes = 1, std::uint64_t max_receipt_bytes = 2048) {
        auto owned = std::make_unique<FakeNativeObjectClient>();
        native     = owned.get();
        client = detail::createMMKvcmClientAdapter(std::move(owned), max_object_bytes, max_receipt_bytes);
    }

    FakeNativeObjectClient*       native = nullptr;
    std::shared_ptr<MMKvcmClient> client;
};

std::vector<std::string> keys(size_t count) {
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back("object-" + std::to_string(i));
    }
    return result;
}

}  // namespace

TEST(MMKvcmNativeClientTest, adapterFactoryRejectsInvalidDependenciesAndLimits) {
    EXPECT_EQ(detail::createMMKvcmClientAdapter(nullptr, 1, 1), nullptr);
    EXPECT_EQ(detail::createMMKvcmClientAdapter(std::make_unique<FakeNativeObjectClient>(), 0, 1), nullptr);
    EXPECT_EQ(detail::createMMKvcmClientAdapter(std::make_unique<FakeNativeObjectClient>(), 2, 1), nullptr);
    EXPECT_EQ(detail::createMMKvcmClientAdapter(
                  std::make_unique<FakeNativeObjectClient>(), kMMKvcmMaxObjectBytes + 1, kMMKvcmMaxObjectBytes + 1),
              nullptr);
}

TEST(MMKvcmNativeClientTest, saveUsesServiceBatches) {
    NativeAdapter adapter;
    ObjectBatch   batch(kMMKvcmMaxBatchItems + 1);

    EXPECT_TRUE(adapter.client->save("trace", batch.objects).empty());
    ASSERT_EQ(adapter.native->save_batches.size(), 2u);
    EXPECT_EQ(adapter.native->save_batches[0].size(), kMMKvcmMaxBatchItems);
    EXPECT_EQ(adapter.native->save_batches[1].size(), 1u);
    EXPECT_TRUE(adapter.native->remove_batches.empty());
}

TEST(MMKvcmNativeClientTest, savePreparesEveryBatchBeforeTheFirstProviderMutation) {
    NativeAdapter adapter;
    ObjectBatch   batch(kMMKvcmMaxBatchItems + 1);
    adapter.native->save_hook = [&batch](size_t call) {
        if (call == 0) {
            // Model a caller/provider side effect while the first batch is in
            // flight. The adapter must already own a stable plan for every
            // later batch; otherwise an exception during later preparation
            // could strand the committed prefix.
            batch.objects.back().key     = "mutated-after-first-io";
            batch.objects.back().nbytes  = 2;
            batch.objects.back().data    = nullptr;
            batch.objects.back().is_cuda = true;
        }
    };

    EXPECT_TRUE(adapter.client->save("trace", batch.objects).empty());

    ASSERT_EQ(adapter.native->save_batches.size(), 2u);
    ASSERT_EQ(adapter.native->save_batches[1].size(), 1u);
    EXPECT_EQ(adapter.native->save_batches[1][0], "object-64");
    ASSERT_EQ(adapter.native->save_sizes[1].size(), 1u);
    EXPECT_EQ(adapter.native->save_sizes[1][0], 1u);
    ASSERT_EQ(adapter.native->save_buffers[1].size(), 1u);
    ASSERT_EQ(adapter.native->save_buffers[1][0].iovs.size(), 1u);
    EXPECT_EQ(adapter.native->save_buffers[1][0].iovs[0].base, &batch.storage.back());
    EXPECT_EQ(adapter.native->save_buffers[1][0].iovs[0].size, 1u);
    EXPECT_EQ(adapter.native->save_buffers[1][0].iovs[0].type, kv_cache_manager::MemoryType::CPU);
}

TEST(MMKvcmNativeClientTest, saveExceptionRollsBackTheEntireAdmittedPrefix) {
    NativeAdapter adapter;
    ObjectBatch   batch(kMMKvcmMaxBatchItems + 1);
    adapter.native->throw_save_call   = 1;
    adapter.native->throw_remove_call = 0;

    const auto error = adapter.client->save("trace", batch.objects);

    EXPECT_NE(error.find("KVCM save threw an exception"), std::string::npos);
    EXPECT_NE(error.find("rollback outcome is uncertain"), std::string::npos);
    EXPECT_EQ(error.find("provider save secret"), std::string::npos);
    ASSERT_EQ(adapter.native->save_batches.size(), 2u);
    ASSERT_EQ(adapter.native->remove_batches.size(), 2u);
    EXPECT_EQ(adapter.native->remove_batches[0].size(), kMMKvcmMaxBatchItems);
    EXPECT_EQ(adapter.native->remove_batches[1].size(), 1u);
    EXPECT_EQ(adapter.native->remove_traces[0], "trace:rollback:remove");
    EXPECT_EQ(adapter.native->remove_traces[1], "trace:rollback:remove");
}

TEST(MMKvcmNativeClientTest, failedSaveAlsoRollsBackTheEntireAdmittedPrefix) {
    NativeAdapter adapter;
    ObjectBatch   batch(kMMKvcmMaxBatchItems + 1);
    adapter.native->fail_save_call = 1;

    const auto error = adapter.client->save("trace", batch.objects);

    EXPECT_NE(error.find("KVCM save failed with client error"), std::string::npos);
    ASSERT_EQ(adapter.native->remove_batches.size(), 2u);
    EXPECT_EQ(adapter.native->remove_batches[0].size(), kMMKvcmMaxBatchItems);
    EXPECT_EQ(adapter.native->remove_batches[1].size(), 1u);
}

TEST(MMKvcmNativeClientTest, loadCapsHugeDeadlineAndSanitizesProviderException) {
    {
        NativeAdapter adapter;
        ObjectBatch   batch(kMMKvcmMaxBatchItems + 1);
        EXPECT_TRUE(adapter.client->load("trace", batch.objects, std::numeric_limits<std::int64_t>::max()).empty());
        ASSERT_EQ(adapter.native->load_batches.size(), 2u);
    }
    {
        NativeAdapter adapter;
        ObjectBatch   batch(1);
        adapter.native->throw_load_call = 0;
        const auto error = adapter.client->load("trace", batch.objects, 1000);
        EXPECT_EQ(error, "KVCM load threw an exception");
        EXPECT_EQ(error.find("provider load secret"), std::string::npos);
    }
}

TEST(MMKvcmNativeClientTest, loadDeadlineBoundsContentionBeforeProviderIo) {
    using namespace std::chrono_literals;
    NativeAdapter adapter;
    ObjectBatch   batch(1);
    adapter.native->block_save = true;
    std::string save_error;
    std::thread saver([&]() { save_error = adapter.client->save("save", batch.objects); });

    bool entered = false;
    {
        std::unique_lock<std::mutex> lock(adapter.native->gate_mutex);
        entered = adapter.native->gate_condition.wait_for(
            lock, 5s, [&]() { return adapter.native->save_entered; });
    }
    if (!entered) {
        {
            std::lock_guard<std::mutex> lock(adapter.native->gate_mutex);
            adapter.native->allow_save = true;
        }
        adapter.native->gate_condition.notify_all();
        saver.join();
        FAIL() << "save did not enter the fake provider";
    }

    const auto load_error = adapter.client->load("load", batch.objects, 5);
    {
        std::lock_guard<std::mutex> lock(adapter.native->gate_mutex);
        adapter.native->allow_save = true;
    }
    adapter.native->gate_condition.notify_all();
    saver.join();

    EXPECT_EQ(load_error, "KVCM load request deadline expired waiting for the object client");
    EXPECT_TRUE(adapter.native->load_batches.empty());
    EXPECT_TRUE(save_error.empty());
}

TEST(MMKvcmNativeClientTest, removeContinuesAfterExceptionAndBoundsInput) {
    NativeAdapter adapter;
    adapter.native->throw_remove_call = 0;

    const auto error = adapter.client->remove("trace", keys(kMMKvcmMaxBatchItems * 2 + 1));

    EXPECT_EQ(error, "KVCM remove threw an exception");
    ASSERT_EQ(adapter.native->remove_batches.size(), 3u);
    EXPECT_EQ(adapter.native->remove_batches[0].size(), kMMKvcmMaxBatchItems);
    EXPECT_EQ(adapter.native->remove_batches[1].size(), kMMKvcmMaxBatchItems);
    EXPECT_EQ(adapter.native->remove_batches[2].size(), 1u);

    const auto calls = adapter.native->remove_batches.size();
    EXPECT_EQ(adapter.client->remove("trace", keys(kMMKvcmMaxObjectsPerReceipt + 1)),
              "KVCM remove exceeds the receipt object limit");
    EXPECT_EQ(adapter.native->remove_batches.size(), calls);
}

TEST(MMKvcmNativeClientTest, removeContinuesAfterKnownFailureAndPreservesFirstError) {
    NativeAdapter adapter;
    adapter.native->fail_remove_call  = 0;
    adapter.native->throw_remove_call = 1;

    const auto error = adapter.client->remove("trace", keys(kMMKvcmMaxBatchItems * 2 + 1));

    EXPECT_NE(error.find("KVCM remove failed with client error"), std::string::npos);
    EXPECT_EQ(error.find("threw an exception"), std::string::npos);
    ASSERT_EQ(adapter.native->remove_batches.size(), 3u);
    EXPECT_EQ(adapter.native->remove_batches[0].size(), kMMKvcmMaxBatchItems);
    EXPECT_EQ(adapter.native->remove_batches[1].size(), kMMKvcmMaxBatchItems);
    EXPECT_EQ(adapter.native->remove_batches[2].size(), 1u);
}

TEST(MMKvcmNativeClientTest, removeRejectsMalformedKeysBeforeProviderIo) {
    NativeAdapter adapter;

    EXPECT_TRUE(adapter.client->remove("trace", {}).empty());
    EXPECT_NE(adapter.client->remove("trace", {""}).find("must contain"), std::string::npos);
    EXPECT_NE(adapter.client->remove("trace", {"duplicate", "duplicate"}).find("unique"), std::string::npos);
    EXPECT_NE(adapter.client->remove("trace", {std::string(kMMKvcmMaxKeyBytes + 1, 'x')}).find("512"),
              std::string::npos);
    EXPECT_TRUE(adapter.native->remove_batches.empty());
}

}  // namespace rtp_llm
