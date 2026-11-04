#pragma once

#include <arpa/inet.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "autil/Log.h"

namespace autil {

class NetUtil {
public:
    static bool GetIp(std::vector<std::string> &ips, int domain = AF_INET);
    static bool GetDefaultIp(std::string &ip, int domain = AF_INET);
    static bool GetHostName(std::string &hostname);
    static uint16_t randomPort(int domain = AF_INET, uint16_t from = 32768, uint16_t to = 61001);
    static std::string inetNtoa(int32_t ip);
    static std::string getBindIp(int domain = AF_INET);
    static std::string getPackedBindIpv4(int domain = AF_INET);
    static bool isMacAddr(const std::string &addr);
    static bool GetMachineName(std::string &host);
    static bool SupportIPv4();
    static bool SupportIPv6();

private:
    static uint16_t getPort(uint16_t from = 1025, uint16_t to = 60001);
    static bool GetIpv6(std::vector<std::string> &ips);
    static bool GetIpv6FromAddrInfo(std::vector<std::string> &ips);
    static bool GetIpv6FromIfConfig(std::vector<std::string> &ips);
    static bool GetIpv6FromIPCommand(std::vector<std::string> &ips);

private:
    static const std::string HOST_INFO_PATH;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace autil
