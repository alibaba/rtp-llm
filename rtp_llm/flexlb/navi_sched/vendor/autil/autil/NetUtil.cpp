#include "autil/NetUtil.h"

#include <algorithm>
#include <cstring>
#include <fstream> // IWYU pragma: keep
#include <ifaddrs.h>
#include <mutex>
#include <netdb.h>
#include <netinet/in.h>
#include <regex>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "autil/Lock.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"

namespace autil {
AUTIL_LOG_SETUP(autil, NetUtil);

const std::string NetUtil::HOST_INFO_PATH("/etc/hostinfo");

bool NetUtil::GetIp(std::vector<std::string> &ips, int domain) {
    if (domain == AF_INET6) {
        return GetIpv6(ips);
    }
    std::string hostName;
    if (!GetHostName(hostName)) {
        return false;
    }
    struct hostent hentBuffer;
    struct hostent *hent = nullptr;
    ::memset(&hentBuffer, 0, sizeof(hentBuffer));

    int hError = 0;
    char buf[1024];
    if (gethostbyname_r(hostName.c_str(), &hentBuffer, buf, sizeof(buf), &hent, &hError) != 0) {
        AUTIL_LOG(WARN, "gethostbyname failed, hostname [%s], h_errno [%d]", hostName.c_str(), hError);
        return false;
    }
    if (!hent) {
        AUTIL_LOG(WARN, "gethostbyname failed, hostname [%s], null hostent", hostName.c_str());
        return false;
    }

    for (uint32_t i = 0; hent->h_addr_list[i]; i++) {
        char ipChars[128];
        const char *ret = inet_ntop(domain, hent->h_addr_list[i], ipChars, 128);
        if (ret) {
            ips.push_back(std::string(ret));
        }
    }
    return true;
}

bool NetUtil::GetIpv6(std::vector<std::string> &ips) {
    // 测试环境不太稳定, 有的时候可以用getaddrinfo获取, 有的时候只能ip -6 addr show
    // 因此支持了三种实现. 优先通过getaddrinfo获取, 保持逻辑与ipv4本机地址的一致, 用ip -6 addr show获取兜底.
    auto size = ips.size();
    auto ret = GetIpv6FromAddrInfo(ips);
    if (!ret || ips.size() == size) { // 获取失败或是没有获取到, 换个方式获取
        ret = GetIpv6FromIfConfig(ips);
    }
    if (!ret || ips.size() == size) {
        ret = GetIpv6FromIPCommand(ips);
    }
    return ret;
}

bool NetUtil::GetDefaultIp(std::string &ip, int domain) {
    std::vector<std::string> ips;
    if (!GetIp(ips, domain)) {
        return false;
    }
    if (ips.empty()) {
        return false;
    }
    ip.assign(ips[0]);
    return true;
}

bool NetUtil::GetHostName(std::string &hostname) {
    char buf[128];
    if (0 == gethostname(buf, sizeof(buf))) {
        hostname.assign(buf);
        return true;
    }
    return false;
}

uint16_t NetUtil::getPort(uint16_t from, uint16_t to) {
    static unsigned int serial = 0;
    srand((unsigned int)(autil::TimeUtility::currentTime()) + (unsigned int)(getpid()) + serial++);
    auto port = (rand() % (to - from)) + from;
    return (uint16_t)port;
}

uint16_t NetUtil::randomPort(int domain, uint16_t from, uint16_t to) {
    while (true) {
        uint16_t port = getPort(from, to);
        int testSock;
        testSock = socket(domain, SOCK_STREAM, 0);
        if (testSock == -1) {
            continue;
        }
        if (domain == AF_INET) {
            struct sockaddr_in my_addr;
            my_addr.sin_family = AF_INET;
            my_addr.sin_port = htons(port);
            my_addr.sin_addr.s_addr = INADDR_ANY;
            bzero(&(my_addr.sin_zero), sizeof(my_addr.sin_zero));

            if (bind(testSock, (struct sockaddr *)&my_addr, sizeof(struct sockaddr)) == -1) {
                close(testSock);
                continue;
            }
            close(testSock);
            return port;
        } else if (domain == AF_INET6) {
            struct sockaddr_in6 my_addr;
            memset(&my_addr, 0, sizeof(my_addr));
            my_addr.sin6_family = AF_INET6;
            my_addr.sin6_port = htons(port);
            my_addr.sin6_addr = in6addr_any;

            if (bind(testSock, (struct sockaddr *)&my_addr, sizeof(struct sockaddr_in6)) == -1) {
                close(testSock);
                continue;
            }
            close(testSock);
            return port;
        }
    }
}

std::string NetUtil::inetNtoa(int32_t ip) {
    char strTemp[20];
    if (ip == -1) {
        return "";
    }
    uint32_t nValue = (uint32_t)ip;
    sprintf(strTemp,
            "%d.%d.%d.%d",
            (nValue & 0x000000ff),
            (nValue & 0x0000ff00) >> 8,
            (nValue & 0x00ff0000) >> 16,
            (nValue & 0xff000000) >> 24);
    return std::string(strTemp);
}

std::string _packedHostAddress;
std::string _hostAddress;
autil::ThreadMutex _lock;

// copy from build_service
std::string NetUtil::getBindIp(int domain) {
    autil::ScopedLock lock(_lock);
    if (_hostAddress.empty()) {
        std::string ip;
        if (GetDefaultIp(ip, domain)) {
            _hostAddress = ip;
        }
    }
    return _hostAddress;
}

std::string NetUtil::getPackedBindIpv4(int domain) {
    autil::ScopedLock lock(_lock);
    if (_packedHostAddress.empty()) {
        std::string ip;
        if (GetDefaultIp(ip, domain)) {
            if (domain == AF_INET) {
                static constexpr size_t MAX_IP_LEN = 15;
                if (ip.size() < MAX_IP_LEN) {
                    ip.append(MAX_IP_LEN - ip.size(), ' ');
                    _packedHostAddress = ip;
                }
            }
        }
    }
    return _packedHostAddress;
}

const std::string RES_NAME_IP_SEPARATOR = ":";

// simply judge whether hostNic is mac address
bool NetUtil::isMacAddr(const std::string &addr) {
    std::vector<std::string> items = autil::StringUtil::split(addr, RES_NAME_IP_SEPARATOR);
    if (items.size() == 6) {
        return true;
    }
    return false;
}

bool NetUtil::GetMachineName(std::string &host) {
    std::ifstream hostinfo(HOST_INFO_PATH.c_str());
    if (!hostinfo.is_open()) {
        return false;
    }
    std::string line;
    while (getline(hostinfo, line)) {
        if (line.length() > 0) {
            host = line;
            return true;
        }
    }
    return false;
}

bool NetUtil::SupportIPv4() {
    static std::once_flag onceFlag;
    static bool support = false;
    std::call_once(onceFlag, []() {
        auto ret = socket(AF_INET, SOCK_STREAM, 0);
        if (ret > 0) {
            close(ret);
            support = true;
        }
    });
    return support;
}

bool NetUtil::SupportIPv6() {
    static std::once_flag onceFlag;
    static bool support = false;
    std::call_once(onceFlag, []() {
        auto ret = socket(AF_INET6, SOCK_STREAM, 0);
        if (ret > 0) {
            close(ret);
            support = true;
        }
    });
    return support;
}

bool NetUtil::GetIpv6FromAddrInfo(std::vector<std::string> &ips) {
    std::string hostName;
    if (!GetHostName(hostName)) {
        return false;
    }
    struct addrinfo hints, *res;
    int status;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET6;
    hints.ai_socktype = SOCK_STREAM;

    if ((status = getaddrinfo(hostName.c_str(), NULL, &hints, &res)) != 0) {
        AUTIL_LOG(WARN, "getaddrinfo on localhost failed, ret is %d", status);
        return false;
    }

    for (struct addrinfo *rp = res; rp != NULL; rp = rp->ai_next) {
        if (rp == NULL || rp->ai_family != AF_INET6) {
            continue;
        }
        char ipstr[INET6_ADDRSTRLEN];
        struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)rp->ai_addr;
        auto ret = inet_ntop(AF_INET6, &(ipv6->sin6_addr), ipstr, sizeof(ipstr));
        if (ret == NULL || strlen(ret) < 6 || strncmp(ret, "fe80::", 6) == 0) {
            // fe80 Link-Local Addresses
            continue;
        }
        ips.emplace_back(std::string(ret));
    }
    freeaddrinfo(res);
    return true;
}

bool NetUtil::GetIpv6FromIfConfig(std::vector<std::string> &ips) {
    struct ifaddrs *ifaddr = NULL;
    if (getifaddrs(&ifaddr) == -1) {
        AUTIL_LOG(WARN, "getifconfigaddrs failed");
        return false;
    }
    for (struct ifaddrs *ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa == NULL || ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_INET6) {
            continue;
        }
        char ipstr[INET6_ADDRSTRLEN];
        struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)ifa->ifa_addr;
        auto ret = inet_ntop(AF_INET6, &(ipv6->sin6_addr), ipstr, sizeof(ipstr));
        if (ret == NULL || strlen(ret) < 6 || strncmp(ret, "fe80::", 6) == 0) {
            // fe80 Link-Local Addresses
            continue;
        }
        ips.emplace_back(std::string(ret));
    }
    freeifaddrs(ifaddr);
    return true;
}

bool NetUtil::GetIpv6FromIPCommand(std::vector<std::string> &ips) {
    std::string command = "ip -6 addr show | grep -v '^ *inet6 ::1'";
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        AUTIL_LOG(WARN, "Error executing command [%s].", command.c_str());
        return false;
    }

    std::stringstream output;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output << buffer;
    }
    pclose(pipe);

    std::regex ipv6Regex("\\s+inet6 ([a-fA-F0-9:]+(/[0-9]+)?) ");
    std::smatch match;
    std::string line;
    while (std::getline(output, line)) {
        if (std::regex_search(line, match, ipv6Regex)) {
            auto ip = match[1].str();
            if (ip.length() < 6 || strncmp(ip.c_str(), "fe80::", 6) == 0) {
                continue;
            }
            auto slashPos = ip.find("/");             // fe80::18a5:49ff:febf:71e1/64
            ips.emplace_back(ip.substr(0, slashPos)); // 存储匹配到的IPv6地址
        }
    }
    return true;
}
} // namespace autil
