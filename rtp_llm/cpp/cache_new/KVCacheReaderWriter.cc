#include "rtp_llm/cpp/cache_new/KVCacheReaderWriter.h"

namespace rtp_llm {

void KVCacheReaderWriter::write(const std::shared_ptr<BatchKVCacheResource> &resource, const CallBack& callback) {
    connector_->asyncPut(resource, callback);
}

}  // namespace rtp_llm