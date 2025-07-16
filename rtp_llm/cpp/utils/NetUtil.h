#pragma once

#include <limits.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <string>
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

// ipv4:ip:port or ip:port, deal with ipv6 ?
inline std::string extractIP(const std::string& address) {
    size_t colon_pos = address.find(':');
    if (colon_pos != std::string::npos) {
        // 如果地址里有 "ipv4:"前缀，则提取后面的部分
        size_t start_pos = (startsWith(address, "ipv4:")) ? colon_pos + 1 : 0;
        size_t end_pos   = address.find(':', start_pos);

        if (end_pos == std::string::npos) {
            end_pos = address.length();
        }
        return address.substr(start_pos, end_pos - start_pos);
    }
    return "";
}

inline int getAddressByName(const std::string& name, std::vector<std::string>& ips) {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;    // Both IPv4 and IPv6
    hints.ai_socktype = SOCK_STREAM;  // TCP stream sockets

    addrinfo* raw_result = nullptr;
    int       err        = getaddrinfo(name.c_str(), nullptr, &hints, &raw_result);
    auto      deleter    = [](addrinfo* ptr) {
        if (ptr) {
            freeaddrinfo(ptr);
        }
    };
    std::unique_ptr<addrinfo, decltype(deleter)> result(raw_result, deleter);
    if (err != 0) {
        return err;
    }
    for (addrinfo* rp = result.get(); rp != nullptr; rp = rp->ai_next) {
        char  addrstr[INET6_ADDRSTRLEN];
        void* addr;
        if (rp->ai_family == AF_INET) {
            sockaddr_in* ipv4 = reinterpret_cast<sockaddr_in*>(rp->ai_addr);
            addr              = &ipv4->sin_addr;
        } else {
            sockaddr_in6* ipv6 = reinterpret_cast<sockaddr_in6*>(rp->ai_addr);
            addr               = &ipv6->sin6_addr;
        }
        inet_ntop(rp->ai_family, addr, addrstr, sizeof(addrstr));
        ips.push_back(std::string(addrstr));
    }
    return 0;
}

}  // namespace rtp_llm
