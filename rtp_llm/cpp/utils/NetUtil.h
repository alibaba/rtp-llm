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
        size_t end_pos = address.find(':', start_pos);

        if (end_pos == std::string::npos) {
            end_pos = address.length();
        }
        return address.substr(start_pos, end_pos - start_pos);
    }
    return "";
}

}
