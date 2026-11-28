#pragma once

#include <cstdint>
#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <utility>
#include <string>

namespace rtp_llm {

struct WorkerAddrParts {
    std::string host;
    int32_t     cache_store_port = 0;
};

namespace detail {

inline bool parseGrpcPortText(const std::string& port_text, int32_t* port) {
    if (!port || port_text.empty()) {
        return false;
    }
    char* end             = nullptr;
    errno                 = 0;
    auto       value      = std::strtoll(port_text.c_str(), &end, 10);
    const bool parsed_all = end == port_text.c_str() + port_text.size();
    if (errno != 0 || !parsed_all || value < 1 || value > 65535) {
        return false;
    }
    *port = static_cast<int32_t>(value);
    return true;
}

}  // namespace detail

inline bool parseWorkerAddr(const std::string& worker_addr, WorkerAddrParts* parts) {
    if (!parts || worker_addr.empty()) {
        return false;
    }

    std::string host;
    std::string cache_store_port_text;
    std::string grpc_port_text;
    if (worker_addr.front() == '[') {
        const auto close_pos = worker_addr.find(']');
        if (close_pos == std::string::npos || close_pos + 1 >= worker_addr.size()
            || worker_addr[close_pos + 1] != ':') {
            return false;
        }
        const auto cache_port_begin = close_pos + 2;
        const auto cache_port_end   = worker_addr.find(':', cache_port_begin);
        if (cache_port_end == std::string::npos || cache_port_end + 1 >= worker_addr.size()) {
            return false;
        }
        host                  = worker_addr.substr(1, close_pos - 1);
        cache_store_port_text = worker_addr.substr(cache_port_begin, cache_port_end - cache_port_begin);
        grpc_port_text        = worker_addr.substr(cache_port_end + 1);
    } else {
        const auto grpc_col = worker_addr.rfind(':');
        const auto cache_col =
            (grpc_col == std::string::npos || grpc_col == 0) ? std::string::npos : worker_addr.rfind(':', grpc_col - 1);
        if (cache_col == std::string::npos || cache_col == 0 || cache_col + 1 >= grpc_col
            || grpc_col + 1 >= worker_addr.size()) {
            return false;
        }
        host                  = worker_addr.substr(0, cache_col);
        cache_store_port_text = worker_addr.substr(cache_col + 1, grpc_col - cache_col - 1);
        grpc_port_text        = worker_addr.substr(grpc_col + 1);
    }

    if (host.find(':') != std::string::npos) {
        std::string ip_part = host;
        const auto  zone    = host.find('%');
        if (zone != std::string::npos) {
            const auto zone_id = host.substr(zone + 1);
            if (zone == 0 || zone_id.empty()
                || zone_id.find_first_of(":[] \t\r\n%") != std::string::npos) {
                return false;
            }
            ip_part = host.substr(0, zone);
        }
        in6_addr ipv6_address{};
        if (inet_pton(AF_INET6, ip_part.c_str(), &ipv6_address) != 1) {
            return false;
        }
    }
    int32_t cache_store_port = 0;
    int32_t grpc_port        = 0;
    if (host.empty() || !detail::parseGrpcPortText(cache_store_port_text, &cache_store_port)
        || !detail::parseGrpcPortText(grpc_port_text, &grpc_port)) {
        return false;
    }
    parts->host             = std::move(host);
    parts->cache_store_port = cache_store_port;
    return true;
}

inline bool parseGrpcHostPort(const std::string& address, std::string& host, int32_t& port) {
    const auto colon = address.rfind(':');
    if (colon == std::string::npos || colon == 0
        || !detail::parseGrpcPortText(address.substr(colon + 1), &port)) {
        return false;
    }
    host = address.substr(0, colon);
    if (host.front() == '[') {
        if (host.size() < 3 || host.back() != ']') {
            return false;
        }
        host = host.substr(1, host.size() - 2);
    } else if (host.find(':') != std::string::npos || host.find(']') != std::string::npos) {
        return false;
    }
    return !host.empty();
}

inline bool isValidGrpcPort(int64_t port) {
    return port >= 1 && port <= 65535;
}

inline std::string formatGrpcHostPort(const std::string& host, int64_t port) {
    if (host.empty() || !isValidGrpcPort(port)) {
        return "";
    }
    if (host.front() == '[' || host.back() == ']') {
        if (host.front() == '[' && host.back() == ']') {
            return host + ":" + std::to_string(port);
        }
        return "";
    }
    if (host.find(':') != std::string::npos) {
        return "[" + host + "]:" + std::to_string(port);
    }
    return host + ":" + std::to_string(port);
}

}  // namespace rtp_llm
