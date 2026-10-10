#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"

int main(int argc, char** argv) {
    using namespace rtp_llm;
    try {
        if (argc != 7) {
            throw std::runtime_error("Expected endpoint, group, instance, host, first key and second key");
        }
        const int64_t first = std::stoll(argv[5]);
        const int64_t second = std::stoll(argv[6]);
        std::mutex mutex;
        KVCacheSnapshot snapshot{1, {first}};
        KVCacheEventPublisherConfig config;
        config.manager_endpoint = argv[1];
        config.heartbeat_interval_ms = 50;
        config.retry_interval_ms = 50;
        // Keep periodic snapshots outside this test so they cannot mask lost deltas.
        config.snapshot_interval_ms = 300000;
        config.request_timeout_ms = 1500;
        config.snapshot_timeout_ms = 5000;
        KVCacheEventPublisherContext context;
        context.instance_group = argv[2];
        context.instance_id = argv[3];
        context.host_ip_port = argv[4];
        context.model_name = "pace_publisher_smoke";
        context.dtype = "fp16";
        context.spec_name = "full";
        context.spec_size_bytes = 262144;
        context.block_size_tokens = 16;
        context.location_uri = "event_report://" + context.host_ip_port + "/hbm";
        KVCMPublisher publisher(config, context, [&] {
            std::lock_guard<std::mutex> lock(mutex);
            return snapshot;
        });
        if (!publisher.start()) {
            throw std::runtime_error("Publisher failed to start");
        }
        std::string command;
        while (std::getline(std::cin, command) && command != "stop") {
            KVCacheEvent event;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (command == "add") {
                    snapshot.block_keys.push_back(second);
                    event = {KVCacheEventType::BLOCK_ADD, second};
                } else if (command == "delete") {
                    snapshot.block_keys.erase(std::remove(snapshot.block_keys.begin(), snapshot.block_keys.end(), first),
                                               snapshot.block_keys.end());
                    event = {KVCacheEventType::BLOCK_DELETE, first};
                } else {
                    throw std::runtime_error("Unknown publisher command");
                }
                ++snapshot.version;
            }
            if (publisher.tryPublish(event) != PublishResult::ACCEPTED) {
                throw std::runtime_error("Publisher did not accept the cache event");
            }
        }
        publisher.stop();
        const auto status = publisher.status();
        if (status.accepted_count != 2 || status.dropped_count != 0) {
            throw std::runtime_error("Unexpected accepted/dropped event counts");
        }
        std::cout << "PASS: RTP Publisher -> real KVCM registration/snapshot/add/delete\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
