#pragma once

#include "maga_transformer/cpp/dataclass/BatchKVCacheBlockAddr.h"
#include <memory>
#include <sstream>
#include <cassert>

namespace rtp_llm {

void BatchKVCacheBlockAddr::clear() {
    batch_offset.clear();
}

void BatchKVCacheBlockAddr::pushBack(const KVCacheBlockAddr& addr) {
    batch_offset.push_back(addr.offset);
}

void BatchKVCacheBlockAddr::resize(size_t batch_id, int reserver_blocks) {
    FT_CHECK(batch_offset.size() > batch_id && batch_offset[batch_id].size() >= reserver_blocks);
    batch_offset[batch_id].resize(reserver_blocks);
}

void BatchKVCacheBlockAddr::append(size_t batch_id, const KVCacheBlockAddr& addr) {
    FT_CHECK(batch_offset.size() > batch_id);
    batch_offset[batch_id].insert(batch_offset[batch_id].end(), addr.offset.begin(), addr.offset.end());
}

std::string BatchKVCacheBlockAddr::debugString() const {
    std::stringstream debug_string, batch_offset_string;
    for (int i = 0; i < batch_offset.size(); i++) {
        batch_offset_string << "batch: " << i << " ";
        for (auto &v: batch_offset[i]) {
            batch_offset_string << v << ", ";
        }
    }

    debug_string << "BatchKVCacheBlockAddr {" << batch_offset_string.str()
                    << "}";
    return debug_string.str();
}

}  // namespace rtp_llm
