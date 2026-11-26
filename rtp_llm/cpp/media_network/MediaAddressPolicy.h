#pragma once

#include <arpa/inet.h>

#include <array>
#include <charconv>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm::media_network {

class BlockedNetworks {
public:
    void set(const std::vector<std::string>& cidrs) {
        std::vector<Network> parsed;
        for (const auto& cidr : cidrs) {
            const auto split = cidr.find('/');
            if (split == std::string::npos)
                throw std::invalid_argument("invalid download CIDR");
            Network    network{};
            const auto host    = cidr.substr(0, split);
            network.family     = host.find(':') == std::string::npos ? AF_INET : AF_INET6;
            const auto* end    = cidr.data() + cidr.size();
            auto [last, error] = std::from_chars(cidr.data() + split + 1, end, network.bits);
            if (error != std::errc{} || last != end || network.bits > (network.family == AF_INET ? 32u : 128u)
                || inet_pton(network.family, host.c_str(), network.bytes.data()) != 1)
                throw std::invalid_argument("invalid download CIDR");
            parsed.push_back(network);
        }
        networks_.swap(parsed);
    }

    bool contains(int family, const void* address) const {
        const auto* bytes = static_cast<const unsigned char*>(address);
        if (family == AF_INET6 && IN6_IS_ADDR_V4MAPPED(static_cast<const in6_addr*>(address))
            && contains(AF_INET, bytes + 12))
            return true;
        for (const auto& network : networks_) {
            if (family != network.family)
                continue;
            const auto whole = network.bits / 8;
            const auto rest  = network.bits % 8;
            if (std::memcmp(bytes, network.bytes.data(), whole) == 0
                && (!rest || ((bytes[whole] ^ network.bytes[whole]) >> (8 - rest)) == 0))
                return true;
        }
        return false;
    }

private:
    struct Network {
        int                           family;
        unsigned                      bits;
        std::array<unsigned char, 16> bytes;
    };
    std::vector<Network> networks_;
};

}  // namespace rtp_llm::media_network
