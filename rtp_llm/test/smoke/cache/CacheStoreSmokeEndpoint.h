#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <thread>

#include "rtp_llm/test/smoke/cache/CacheSmokeSupport.h"

namespace rtp_llm::cache_smoke {

// Inspect only this native process's live listening descriptors. The production
// listener owns the socket throughout; the test never binds or closes it.
inline std::set<uint16_t> listeningTcpPorts() {
    std::set<uint16_t> ports;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        const int fd   = std::stoi(entry.path().filename().string());
        int       type = 0, listening = 0;
        socklen_t length = sizeof(type);
        if (::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &length) != 0 || type != SOCK_STREAM)
            continue;
        length = sizeof(listening);
        if (::getsockopt(fd, SOL_SOCKET, SO_ACCEPTCONN, &listening, &length) != 0 || !listening)
            continue;
        sockaddr_in address{};
        length = sizeof(address);
        if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) == 0 && address.sin_family == AF_INET)
            ports.insert(ntohs(address.sin_port));
    }
    return ports;
}

inline uint16_t publishSmokeEndpoint(const std::set<uint16_t>&    before,
                                     const std::filesystem::path& root,
                                     const std::string&           rank_stem) {
    auto ports = listeningTcpPorts();
    for (auto port : before)
        ports.erase(port);
    require(ports.size() == 1 && *ports.begin() != 0, "expected one new production TCP listener");
    const auto    port = *ports.begin();
    const auto    path = root / (rank_stem + ".listen");
    std::ofstream out(path.string() + ".tmp");
    out << port << '\n';
    out.close();
    require(static_cast<bool>(out), "cannot publish production listener port");
    std::filesystem::rename(path.string() + ".tmp", path);
    return port;
}

inline std::vector<uint32_t> readSmokeSourcePorts(const std::filesystem::path& root, int count, int64_t timeout_ms) {
    require(count > 0, "invalid sender count");
    const auto            deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::vector<uint32_t> ports;
    for (int rank = 0; rank < count; ++rank) {
        const auto stem = count == 1 ? "sender" : "sender." + std::to_string(rank);
        const auto path = root / (stem + ".listen");
        while (!std::filesystem::exists(path)) {
            require(std::chrono::steady_clock::now() < deadline,
                    "sender listener publication timeout: " + path.string());
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        uint32_t      port = 0;
        std::ifstream input(path);
        input >> port;
        require(input && port > 0 && port <= 65535, "invalid published sender port");
        ports.push_back(port);
    }
    require(std::set<uint32_t>(ports.begin(), ports.end()).size() == ports.size(), "sender listener ports overlap");
    return ports;
}

}  // namespace rtp_llm::cache_smoke
