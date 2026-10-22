#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/VIPServerSubscriber.h"
#include "vipclient.h"

namespace rtp_llm::kvcm {
TEST(VIPServerSubscriberTest, SingletonRetriesInitializationAndOutlivesConcurrentSubscribers) {
    EXPECT_EXIT(
        {
            std::atexit([] {
                const auto& state = middleware::vipclient::apiState();
                if (state.create != 2 || state.init != 2 || state.uninit != 1 || state.destroy != 2) {
                    std::_Exit(10);
                }
            });
            auto first = std::make_unique<VIPServerSubscriber>();
            if (first->init({"first"})) {
                std::_Exit(1);
            }
            VIPServerSubscriber second;
            bool                first_ok  = false;
            bool                second_ok = false;
            std::thread         a([&] { first_ok = first->init({"first"}); });
            std::thread         b([&] { second_ok = second.init({"second"}); });
            a.join();
            b.join();
            if (!first_ok || !second_ok) {
                std::_Exit(2);
            }
            std::vector<std::string> addresses;
            if (!first->getAddresses(addresses) || addresses != std::vector<std::string>{"first:80"}) {
                std::_Exit(3);
            }
            first.reset();
            if (!second.getAddresses(addresses) || addresses != std::vector<std::string>{"second:80"}) {
                std::_Exit(4);
            }
            if (!second.init({"third"}) || !second.getAddresses(addresses)
                || addresses != std::vector<std::string>{"third:80"}) {
                std::_Exit(5);
            }
            const auto& state = middleware::vipclient::apiState();
            if (state.create != 2 || state.init != 2 || state.uninit != 0 || state.destroy != 1) {
                std::_Exit(6);
            }
            std::exit(0);
        },
        ::testing::ExitedWithCode(0),
        "");
}
}  // namespace rtp_llm::kvcm
