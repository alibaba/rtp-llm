#pragma once
#include <string>
#include <vector>
namespace middleware::vipclient {
struct IPHost {
    std::string address;
    const char* ip() const {
        return address.c_str();
    }
    int port() const {
        return 80;
    }
    bool valid() const {
        return true;
    }
    int weight() const {
        return 1;
    }
};
struct IPHostArray {
    std::vector<IPHost> hosts;
    unsigned int        size() const {
        return hosts.size();
    }
    const IPHost& get(unsigned int index) const {
        return hosts.at(index);
    }
};
}  // namespace middleware::vipclient
