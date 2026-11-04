#pragma once

#include <netinet/in.h>
#include <string>
#include <sys/un.h>

namespace anet {

/* macro for the domain socket file path prefix. */
#define DOMAIN_SOCK_PATH "/tmp/"

class AddrSpec {
public:
    AddrSpec(int family = AF_INET, int type = SOCK_STREAM);

public:
    int setAddrSpec(const std::string &spec);

    const sockaddr *getAddr() const;
    int setAddr(int family, int type, const struct sockaddr *address);

    socklen_t getAddrSize() const;
    int getProtocolFamily() const { return _family; }
    int getProtocolType() const { return _type; }
    uint16_t getPort() const;
    bool isValidState() const { return _isValid; }

    const char *getUnixPath() const;
    const std::string &getIP() const { return _ip; }

    void setSupportIPv4(bool v) { _supportIPv4 = v; }
    void setSupportIPv6(bool v) { _supportIPv6 = v; }

private:
    int parseInetAddrPort(const std::string &addrPort, int protocolType);
    bool resolveHostname(const std::string &hostname, std::string &ipAddress, int family) const;
    int parseUnixAddress(const std::string &address, int protocolType);
    int setAndReturnSuccess(int family, int type, const std::string& ip);

private:
    bool _isValid{false};
    sockaddr_storage _address;

    // fastpath
    int _family{AF_INET};
    int _type{SOCK_STREAM};
    std::string _ip;

    // store local for test
    bool _supportIPv4{false};
    bool _supportIPv6{false};
};

} // namespace anet

