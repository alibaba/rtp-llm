#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/addrspec.h"

#include <arpa/inet.h>
#include <iostream>
#include <netdb.h>
#include <regex>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "autil/NetUtil.h"

namespace anet {

AddrSpec::AddrSpec(int family, int type)
    : _family(family)
    , _type(type)
    , _supportIPv4(autil::NetUtil::SupportIPv4())
    , _supportIPv6(autil::NetUtil::SupportIPv6()) {
    memset(&_address, 0, sizeof(_address));
}

int AddrSpec::setAddrSpec(const std::string &spec) {
    size_t protocolPos = spec.find_first_of(":");
    if (protocolPos == std::string::npos || protocolPos >= spec.size() - 1) {
        ANET_LOG(WARN, "can not parse protocol from spec [%s]", spec.c_str());
        return -1;
    }
    std::string protocol = spec.substr(0, protocolPos);

    protocolPos++; // skip :
    if (protocol == "tcp" || protocol == "udp") {
        while (protocolPos < spec.size()) {
            if (spec.at(protocolPos) != '/') {
                break;
            }
            protocolPos++; // skip /
        }
    }

    std::string addr = spec.substr(protocolPos);
    if (protocol == "tcp") {
        return parseInetAddrPort(addr, SOCK_STREAM);
    } else if (protocol == "udp") {
        return parseInetAddrPort(addr, SOCK_DGRAM);
    } else if (protocol == "unixstream") {
        return parseUnixAddress(addr, SOCK_STREAM);
    } else if (protocol == "unixdgram") {
        return parseUnixAddress(addr, SOCK_DGRAM);
    } else {
        ANET_LOG(WARN, "can not find match protocol for spec [%s]", spec.c_str());
        return -1;
    }
}

int AddrSpec::setAddr(int family, int type, const struct sockaddr *address) {
    if (family == AF_INET) {
        memcpy(&_address, address, sizeof(sockaddr_in));
    } else if (family == AF_INET6) {
        memcpy(&_address, address, sizeof(sockaddr_in6));
    } else if (family == AF_UNIX) {
        memcpy(&_address, address, sizeof(sockaddr_un));
    } else {
        return -1;
    }
    _family = family;
    _type = type;
    _isValid = true;

    return 0;
}

int AddrSpec::parseInetAddrPort(const std::string &addrPort, int protocolType) {
    size_t colonPos = addrPort.find_last_of(':');
    if (colonPos == std::string::npos || colonPos >= addrPort.size() - 1) {
        ANET_LOG(WARN, "parse inet addr port failed, cannot find port, addr is [%s]", addrPort.c_str());
        return -1; // Invalid format
    }

    std::string addrStr = addrPort.substr(0, colonPos);
    std::string portStr = addrPort.substr(colonPos + 1);

    in_port_t sinPort;
    try {
        auto port = std::stoi(portStr);
        if (port < 0 || port > 65535) {
            ANET_LOG(WARN, "parse inet addr port failed, port is out of range, addr is [%s]", addrPort.c_str());
            return -1;
        }
        sinPort = htons(static_cast<uint16_t>(port));
    } catch (const std::invalid_argument &e) {
        ANET_LOG(WARN, "parse inet addr port invalid argument, addr is [%s]", addrPort.c_str());
        return -1;
    } catch (const std::out_of_range &e) {
        ANET_LOG(WARN, "parse inet addr port out of range, addr is [%s]", addrPort.c_str());
        return -1;
    }

    // ipv4 address
    if (_supportIPv4) {
        sockaddr_in saddr;
        if (inet_pton(AF_INET, addrStr.c_str(), &saddr.sin_addr) == 1) {
            saddr.sin_port = sinPort;
            saddr.sin_family = AF_INET;
            memcpy(&_address, &saddr, sizeof(saddr));
            return setAndReturnSuccess(AF_INET, protocolType, addrStr);
        }
    }

    // ipv6 address
    if (_supportIPv6) {
        sockaddr_in6 saddr;
        if (inet_pton(AF_INET6, addrStr.c_str(), &saddr.sin6_addr) == 1) {
            saddr.sin6_port = sinPort;
            saddr.sin6_family = AF_INET6;
            memcpy(&_address, &saddr, sizeof(saddr));
            return setAndReturnSuccess(AF_INET6, protocolType, addrStr);
        }
    }

    // ipv4 hostname resolve
    if (_supportIPv4) {
        std::string ipStr;
        if (resolveHostname(addrStr, ipStr, AF_INET)) {
            sockaddr_in saddr;
            if (inet_pton(AF_INET, ipStr.c_str(), &saddr.sin_addr) == 1) {
                saddr.sin_port = sinPort;
                saddr.sin_family = AF_INET;
                memcpy(&_address, &saddr, sizeof(saddr));
                return setAndReturnSuccess(AF_INET, protocolType, addrStr);
            }
        }
    }

    // ipv6 hostname resolve
    if (_supportIPv6) {
        std::string ipStr;
        if (resolveHostname(addrStr, ipStr, AF_INET6)) {
            sockaddr_in6 saddr;
            if (inet_pton(AF_INET6, ipStr.c_str(), &saddr.sin6_addr) == 1) {
                saddr.sin6_port = sinPort;
                saddr.sin6_family = AF_INET6;
                memcpy(&_address, &saddr, sizeof(saddr));
                return setAndReturnSuccess(AF_INET6, protocolType, addrStr);
            }
        }
    }

    ANET_LOG(WARN,
             "can not parse addrspec [%s] to sockaddr, env support is [v4:%d|v6:%d]",
             addrPort.c_str(),
             _supportIPv4,
             _supportIPv6);
    return -1;
}

int AddrSpec::setAndReturnSuccess(int family, int type, const std::string &ip) {
    _family = family;
    _type = type;
    _ip = ip;
    _isValid = true;
    return 0;
}

bool AddrSpec::resolveHostname(const std::string &hostname, std::string &ip_address, int family) const {
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = family; // 指定期望的地址族
    hints.ai_socktype = SOCK_STREAM;

    auto ret = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
    if (ret != 0) {
        ANET_LOG(ERROR,
                 "getaddrinfo failed for host %s, error_code: %d, error_str: %s",
                 hostname.c_str(),
                 ret,
                 gai_strerror(ret));
        return false; // Failed to resolve hostname
    }

    for (struct addrinfo *p = result; p != nullptr; p = p->ai_next) {
        if (p->ai_family == family) {
            char buf[INET6_ADDRSTRLEN];
            inet_ntop(p->ai_family, &((struct sockaddr_in *)p->ai_addr)->sin_addr, buf, sizeof(buf));
            ip_address = buf;
            break;
        }
    }

    freeaddrinfo(result);
    return true;
}

int AddrSpec::parseUnixAddress(const std::string &address, int protocolType) {
    sockaddr_un saddr;
    saddr.sun_family = AF_UNIX;
    strncpy(saddr.sun_path, address.c_str(), sizeof(saddr.sun_path) - 1);

    _family = AF_UNIX;
    _type = protocolType;
    _isValid = true;
    memcpy(&_address, &saddr, sizeof(saddr));
    return 0;
}

const sockaddr *AddrSpec::getAddr() const { return _isValid ? reinterpret_cast<const sockaddr *>(&_address) : nullptr; }

uint16_t AddrSpec::getPort() const {
    if (!_isValid) {
        return 0;
    }
    if (_family == AF_INET) {
        auto addr_in = reinterpret_cast<const sockaddr_in *>(&_address);
        return ntohs(addr_in->sin_port);
    } else if (_family == AF_INET6) {
        auto addr_in6 = reinterpret_cast<const sockaddr_in6 *>(&_address);
        return ntohs(addr_in6->sin6_port);
    }
    return 0;
}

socklen_t AddrSpec::getAddrSize() const {
    if (_family == AF_INET) {
        return sizeof(sockaddr_in);
    } else if (_family == AF_INET6) {
        return sizeof(sockaddr_in6);
    } else {
        return sizeof(sockaddr_un);
    }
}

const char *AddrSpec::getUnixPath() const {
    return _isValid && _family == AF_UNIX ? (reinterpret_cast<const sockaddr_un *>(&_address))->sun_path : nullptr;
}

} // namespace anet
