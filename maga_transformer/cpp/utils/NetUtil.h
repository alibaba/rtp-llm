#pragma once

#include <limits.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <string>
#include "maga_transformer/cpp/utils/StringUtil.h"

namespace rtp_llm {

// ipv4:ip:port or ip:port, deal with ipv6 ?
inline std::string extractIP(const std::string& address) {
    size_t colon_pos = address.find(':');
    if (colon_pos != std::string::npos) {
        // 如果地址里有 "ipv4:"前缀，则提取后面的部分
        size_t start_pos = (startsWith(address, "ipv4:")) ? colon_pos + 1 : 0;
        size_t end_pos = address.find(':', start_pos);

        if (end_pos == std::string::npos) {
            end_pos = address.length();
        }
        return address.substr(start_pos, end_pos - start_pos);
    }
    return "";
}

inline std::string getHostName() {
    char hostname[HOST_NAME_MAX];
    if (gethostname(hostname, sizeof(hostname)) == -1) {
        perror("gethostname failed");
        return "";
    }
    return hostname;
}

inline std::string getLocalIP() {
    char hostname[HOST_NAME_MAX];
    if (gethostname(hostname, sizeof(hostname)) == -1) {
        perror("gethostname failed");
        return "";
    }

    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET; // IPv4
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(hostname, nullptr, &hints, &res) != 0) {
        perror("getaddrinfo failed");
        return "";
    }

    char ip[INET_ADDRSTRLEN];
    for (struct addrinfo *p = res; p != nullptr; p = p->ai_next) {
        // 将 IP 地址转为字符串
        if (getnameinfo(p->ai_addr, p->ai_addrlen, ip, sizeof(ip), nullptr, 0, NI_NUMERICHOST) == 0) {
            freeaddrinfo(res);
            return ip; // 返回第一个找到的 IP 地址
        }
    }

    freeaddrinfo(res);
    return "";
}

}
