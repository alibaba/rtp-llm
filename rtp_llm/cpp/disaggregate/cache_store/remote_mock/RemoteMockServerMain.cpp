#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include <arpa/inet.h>
#include <charconv>
#include <csignal>
#include <iostream>
#include <limits>
#include <map>
#include <pthread.h>

namespace {
void usage() {
    std::cout << "CPU remote KV diagnostic service (trusted local clients, existing TCP RPC)\n"
              << "Options are --name=value; all sizes are bytes.\n"
              << "--address=127.0.0.1 --port=18080\n"
              << "--total-bytes --source-bytes --write-slot-bytes --write-slots\n"
              << "--read-staging-bytes --write-staging-bytes --max-rpc-bytes\n"
              << "--max-blocks-per-rpc --max-read-operations --reads-per-connection\n"
              << "--writes-per-connection --max-callback-endpoints --max-timeout-ms\n"
              << "--event-path=PATH --max-event-bytes=8388608\n"
              << "Manifest control RPC selects fixed read/write delays for each session.\n";
}
}  // namespace
int main(int argc, char** argv) {
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    if (pthread_sigmask(SIG_BLOCK, &signals, nullptr))
        return 1;
    rtp_llm::RemoteMockServiceOptions options;
    std::string                       address = "127.0.0.1";
    uint32_t                          port    = 18080;
    std::map<std::string, uint64_t*>  wide    = {{"total-bytes", &options.limits.total_bytes},
                                                 {"source-bytes", &options.limits.source_bytes},
                                                 {"write-slot-bytes", &options.limits.write_slot_bytes},
                                                 {"read-staging-bytes", &options.limits.read_staging_bytes},
                                                 {"write-staging-bytes", &options.limits.write_staging_bytes},
                                                 {"max-event-bytes", &options.max_event_bytes}};
    std::map<std::string, uint32_t*>  narrow  = {{"port", &port},
                                                 {"write-slots", &options.limits.write_slots},
                                                 {"max-rpc-bytes", &options.limits.max_rpc_bytes},
                                                 {"max-blocks-per-rpc", &options.limits.max_blocks_per_rpc},
                                                 {"max-read-operations", &options.limits.max_read_operations},
                                                 {"reads-per-connection", &options.reads_per_connection},
                                                 {"writes-per-connection", &options.writes_per_connection},
                                                 {"max-callback-endpoints", &options.max_callback_endpoints},
                                                 {"max-timeout-ms", &options.max_timeout_ms}};
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            usage();
            return 0;
        }
        auto equal = arg.find('=');
        if (arg.rfind("--", 0) != 0 || equal == std::string::npos) {
            std::cerr << "expected --name=value\n";
            return 2;
        }
        auto name  = arg.substr(2, equal - 2);
        auto value = arg.substr(equal + 1);
        if (name == "address") {
            address = value;
            continue;
        }
        if (name == "event-path") {
            options.event_path = value;
            continue;
        }
        uint64_t number = 0;
        auto     parsed = std::from_chars(value.data(), value.data() + value.size(), number);
        if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() || !number) {
            std::cerr << "invalid positive number: " << name << "\n";
            return 2;
        }
        if (auto found = wide.find(name); found != wide.end())
            *found->second = number;
        else if (auto found = narrow.find(name);
                 found != narrow.end() && number <= std::numeric_limits<uint32_t>::max())
            *found->second = number;
        else {
            std::cerr << "unknown or overflowing option: " << name << "\n";
            return 2;
        }
    }
    in_addr parsed{};
    if (inet_pton(AF_INET, address.c_str(), &parsed) != 1 || !port || port > 65535) {
        std::cerr << "invalid IPv4 address or port\n";
        return 2;
    }
    rtp_llm::RemoteMockService service(options);
    rtp_llm::TcpServer         server;
    if (!service.init() || !server.init(1, 2, false, 64) || !server.registerService(&service)
        || !server.registerService(service.controlService()) || !server.start(port, address)) {
        std::cerr << "remote mock initialization failed\n";
        service.shutdown();
        return 1;
    }
    std::cout << "REMOTE_MOCK_READY " << address << ":" << port << "\n" << std::flush;
    int signal = 0;
    sigwait(&signals, &signal);
    service.shutdown();
    const auto stats = service.stats();
    std::cerr << "remote mock drained read_payload_bytes=" << stats.read_payload_bytes
              << " write_payload_bytes=" << stats.write_payload_bytes << " reverse_rpcs=" << stats.reverse_rpcs
              << " failed_operations=" << stats.failed_operations << "\n";
    return 0;
}
