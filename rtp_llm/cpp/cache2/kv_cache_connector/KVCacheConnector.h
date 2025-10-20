#pragma once

#include <string>
#include <vector>

namespace rtp_llm {

struct KVCacheLocation {
    std::string uri;
};

struct MemorySpan {
    void* data;
    size_t size;
};

class KVCacheConnector {
public:
    void load(std::vector<std::string> source, std::vector<MemorySpan> dest);
    void store(std::vector<MemorySpan> source, std::vector<std::string> dest);
};

}  // namespace rtp_llm