#include "gtest/gtest.h"

#include <atomic>
#include <stdexcept>
#include <thread>

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"

namespace rtp_llm {

class RequestBlockBufferTest: public ::testing::Test {};

TEST_F(RequestBlockBufferTest, testBlockOps) {

    RequestBlockBuffer buffer("test-request-id");

    ASSERT_EQ(0, buffer.getBlocksCount());

    std::shared_ptr<void> buffer1((void*)0x1, [](void* p) {});
    buffer.addBlock(std::make_shared<BlockBuffer>("b1", buffer1, 10, true, true));
    ASSERT_EQ(1, buffer.getBlocksCount());
    ASSERT_TRUE(buffer.isValid());
    ASSERT_EQ(buffer1, buffer.getBlock("b1")->addr);
    ASSERT_EQ(10, buffer.getBlocksSize());

    std::shared_ptr<void> buffer2((void*)0x2, [](void* p) {});
    buffer.addBlock("b2", buffer2, 10, true, true);
    ASSERT_EQ(2, buffer.getBlocksCount());
    ASSERT_TRUE(buffer.isValid());
    ASSERT_EQ(buffer2, buffer.getBlock("b2")->addr);
    ASSERT_EQ(20, buffer.getBlocksSize());

    buffer.addBlock("b3", nullptr, 10, true, true);
    ASSERT_EQ(3, buffer.getBlocksCount());
    ASSERT_FALSE(buffer.isValid());
    ASSERT_EQ(nullptr, buffer.getBlock("b3")->addr);
    ASSERT_EQ(30, buffer.getBlocksSize());
}

TEST_F(RequestBlockBufferTest, testWatchFunc_SetWatchFunc) {
    {
        // set to empty request block buffer
        bool                                      watched_called{false};
        bool                                      watched_success{false};
        std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
        auto                                      watch_func = [&watched_called, &watched_success, &watched_blocks](
                              bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called  = true;
            watched_success = success;
            watched_blocks  = blocks;
        };

        auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func)));
        ASSERT_FALSE(watched_called);
        ASSERT_FALSE(watched_success);
        ASSERT_TRUE(watched_blocks.empty());

        // set twice
        watched_called   = false;
        watched_success  = false;
        auto watch_func2 = [&watched_called, &watched_success, &watched_blocks](
                               bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called  = true;
            watched_success = success;
            watched_blocks  = blocks;
        };
        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func2)));
        ASSERT_FALSE(watched_called);
        ASSERT_FALSE(watched_success);
        ASSERT_TRUE(watched_blocks.empty());

        request_block_buffer.reset();
    }
    {
        // set to non empty request block buffer
        auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
        request_block_buffer->addBlock(std::make_shared<BlockBuffer>("b1", nullptr, 10, true, true));
        request_block_buffer->addBlock(std::make_shared<BlockBuffer>("b2", nullptr, 10, false, false));

        bool                                      watched_called1{false};
        bool                                      watched_success1{false};
        std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
        auto                                      watch_func1 = [&watched_called1, &watched_success1, &watched_blocks](
                               bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called1  = true;
            watched_success1 = success;
            watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
        };

        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func1)));
        ASSERT_TRUE(watched_called1);
        ASSERT_TRUE(watched_success1);
        ASSERT_EQ(2, watched_blocks.size());

        // set twice
        watched_called1  = false;
        watched_success1 = false;
        watched_blocks.clear();
        bool watched_called2{false};
        bool watched_success2{false};
        auto watch_func2 = [&watched_called2, &watched_success2, &watched_blocks](
                               bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called2  = true;
            watched_success2 = success;
            watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
        };
        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func2)));
        ASSERT_TRUE(watched_called1);
        ASSERT_TRUE(watched_success1);
        ASSERT_TRUE(watched_called2);
        ASSERT_TRUE(watched_success2);
        ASSERT_EQ(4, watched_blocks.size());

        request_block_buffer.reset();
    }
}

TEST_F(RequestBlockBufferTest, testWatchFunc_AddBlock) {
    bool                                      watched_called1{false};
    bool                                      watched_success1{false};
    std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
    auto                                      watch_func1 = [&watched_called1, &watched_success1, &watched_blocks](
                           bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        watched_called1  = true;
        watched_success1 = success;
        watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
    };

    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func1)));

    // add block
    request_block_buffer->addBlock(std::make_shared<BlockBuffer>("b1", nullptr, 10, true, true));
    ASSERT_TRUE(watched_called1);
    ASSERT_TRUE(watched_success1);
    ASSERT_EQ(1, watched_blocks.size());
    ASSERT_EQ("b1", watched_blocks[0]->key);

    watched_called1 = false;
    watched_blocks.clear();
    bool watched_called2{false};
    bool watched_success2{false};
    auto watch_func2 = [&watched_called2, &watched_success2, &watched_blocks](
                           bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        watched_called2  = true;
        watched_success2 = success;
        watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
    };
    ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func2)));
    ASSERT_EQ(2, watched_blocks.size());
    watched_called1 = false;
    watched_called2 = false;
    watched_blocks.clear();

    // add block
    request_block_buffer->addBlock("b2", nullptr, 10, true, true);
    ASSERT_TRUE(watched_called1);
    ASSERT_TRUE(watched_success1);
    ASSERT_TRUE(watched_called2);
    ASSERT_TRUE(watched_success2);
    ASSERT_EQ(2, watched_blocks.size());
    ASSERT_EQ("b2", watched_blocks[1]->key);

    // add blocks
    watched_called1 = false;
    watched_called2 = false;
    watched_blocks.clear();
    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    blocks.push_back(std::make_shared<BlockBuffer>("b3", nullptr, 10, true, true));
    blocks.push_back(std::make_shared<BlockBuffer>("b4", nullptr, 10, true, true));
    request_block_buffer->addBlocks(blocks);
    ASSERT_TRUE(watched_called1);
    ASSERT_TRUE(watched_success1);
    ASSERT_TRUE(watched_called2);
    ASSERT_TRUE(watched_success2);
    ASSERT_EQ(4, watched_blocks.size());
    ASSERT_EQ("b3", watched_blocks[2]->key);
    ASSERT_EQ("b4", watched_blocks[3]->key);

    request_block_buffer.reset();
}

TEST_F(RequestBlockBufferTest, testWatchFunc_ReleaseBlock) {
    bool                                      watched_called{false};
    bool                                      watched_success{false};
    std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
    auto                                      watch_func = [&watched_called, &watched_success, &watched_blocks](
                          bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        watched_called  = true;
        watched_success = success;
        watched_blocks  = blocks;
    };

    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func)));

    request_block_buffer->notifyRequestDone();
    ASSERT_TRUE(watched_called);
    ASSERT_FALSE(watched_success);
    ASSERT_TRUE(watched_blocks.empty());
}

TEST_F(RequestBlockBufferTest, testWatchFunc_SetAfterRequestDone) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    request_block_buffer->notifyRequestDone();

    int  callback_count = 0;
    bool callback_ok    = true;
    EXPECT_FALSE(request_block_buffer->setWatchFunc(
        [&callback_count, &callback_ok](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            ++callback_count;
            callback_ok = ok;
            EXPECT_TRUE(blocks.empty());
        }));

    EXPECT_EQ(1, callback_count);
    EXPECT_FALSE(callback_ok);
}

TEST_F(RequestBlockBufferTest, testWatchFunc_SetAfterRequestDoneWithExistingBlock) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    request_block_buffer->addBlock(
        std::make_shared<BlockBuffer>("existing-block", nullptr, 1, false, false));
    request_block_buffer->notifyRequestDone();

    int  callback_count = 0;
    bool callback_ok    = true;
    EXPECT_FALSE(request_block_buffer->setWatchFunc(
        [&callback_count, &callback_ok](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            ++callback_count;
            callback_ok = ok;
            EXPECT_TRUE(blocks.empty());
        }));

    EXPECT_EQ(1, callback_count);
    EXPECT_FALSE(callback_ok);
}

TEST_F(RequestBlockBufferTest, testRequestDoneIsTerminal) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");

    int success_count = 0;
    int failure_count = 0;
    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&success_count, &failure_count](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            ok ? ++success_count : ++failure_count;
        }));

    request_block_buffer->notifyRequestDone();
    request_block_buffer->notifyRequestDone();
    request_block_buffer->addBlock(
        std::make_shared<BlockBuffer>("late-block", nullptr, 1, false, false));

    EXPECT_EQ(0, success_count);
    EXPECT_EQ(1, failure_count);
}

TEST_F(RequestBlockBufferTest, testConcurrentWatchRegistrationAndRequestDone) {
    constexpr int kRounds = 200;

    for (int round = 0; round < kRounds; ++round) {
        auto             request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        std::atomic<int> failure_count{0};

        std::thread register_thread([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            request_block_buffer->setWatchFunc(
                [&failure_count](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
                    if (!ok) {
                        failure_count.fetch_add(1, std::memory_order_relaxed);
                    }
                });
        });
        std::thread finish_thread([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            request_block_buffer->notifyRequestDone();
        });

        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        register_thread.join();
        finish_thread.join();

        EXPECT_EQ(1, failure_count.load(std::memory_order_relaxed)) << "round " << round;
    }
}

TEST_F(RequestBlockBufferTest, testRequestDoneWaitsForStartedSuccessDispatch) {
    auto              request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    std::atomic<bool> success_started{false};
    std::atomic<bool> release_success{false};
    std::atomic<bool> failure_called{false};

    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            if (ok) {
                success_started.store(true, std::memory_order_release);
                while (!release_success.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            } else {
                failure_called.store(true, std::memory_order_release);
            }
        }));

    std::thread add_thread([&]() {
        request_block_buffer->addBlock(
            std::make_shared<BlockBuffer>("block", nullptr, 1, false, false));
    });
    while (!success_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    request_block_buffer->notifyRequestDone();
    EXPECT_FALSE(failure_called.load(std::memory_order_acquire));

    release_success.store(true, std::memory_order_release);
    add_thread.join();
    EXPECT_TRUE(failure_called.load(std::memory_order_acquire));
}

TEST_F(RequestBlockBufferTest, testSuccessCallbackCanFinishRequest) {
    auto              request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    std::vector<bool> callback_results;

    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            callback_results.push_back(ok);
            if (ok) {
                request_block_buffer->notifyRequestDone();
            }
        }));

    request_block_buffer->addBlock(
        std::make_shared<BlockBuffer>("block", nullptr, 1, false, false));

    ASSERT_EQ(2, callback_results.size());
    EXPECT_TRUE(callback_results[0]);
    EXPECT_FALSE(callback_results[1]);
}

TEST_F(RequestBlockBufferTest, testThrowingSuccessCallbackDoesNotBlockRequestDone) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    int  failure_count        = 0;
    int  second_success_count = 0;

    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            if (ok) {
                throw std::runtime_error("callback failure");
            }
            ++failure_count;
        }));
    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            if (ok) {
                ++second_success_count;
            } else {
                ++failure_count;
            }
        }));

    EXPECT_THROW(
        request_block_buffer->addBlock(
            std::make_shared<BlockBuffer>("block", nullptr, 1, false, false)),
        std::runtime_error);
    request_block_buffer->notifyRequestDone();

    EXPECT_EQ(1, second_success_count);
    EXPECT_EQ(2, failure_count);
}

TEST_F(RequestBlockBufferTest, testThrowingDoneCallbackDoesNotSkipOtherWatchers) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    int  first_failure_count  = 0;
    int  second_failure_count = 0;

    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            if (!ok) {
                ++first_failure_count;
                throw std::runtime_error("terminal callback failure");
            }
        }));
    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            if (!ok) {
                ++second_failure_count;
            }
        }));

    EXPECT_THROW(request_block_buffer->notifyRequestDone(), std::runtime_error);
    EXPECT_EQ(1, first_failure_count);
    EXPECT_EQ(1, second_failure_count);

    EXPECT_NO_THROW(request_block_buffer->notifyRequestDone());
    EXPECT_EQ(1, first_failure_count);
    EXPECT_EQ(1, second_failure_count);
}

TEST_F(RequestBlockBufferTest, testThrowingDeferredDoneCallbackDoesNotSkipOtherWatchers) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    int  first_failure_count  = 0;
    int  second_failure_count = 0;

    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            if (ok) {
                request_block_buffer->notifyRequestDone();
            } else {
                ++first_failure_count;
                throw std::runtime_error("deferred terminal callback failure");
            }
        }));
    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            if (!ok) {
                ++second_failure_count;
            }
        }));

    EXPECT_THROW(
        request_block_buffer->addBlock(
            std::make_shared<BlockBuffer>("block", nullptr, 1, false, false)),
        std::runtime_error);
    EXPECT_EQ(1, first_failure_count);
    EXPECT_EQ(1, second_failure_count);
}

}  // namespace rtp_llm
