#pragma once

#include "rtp_llm/cpp/cache_new/MemoryBlockCache.h"

namespace rtp_llm {

class Notifier {
public:
    Notifier() = default;
    ~Notifier() = default;

public:
    struct WriteInfo {
        // KVCacheGroupType group_type;
        // std::vector<int64_t> cache_keys;
        // std::shared_ptr<BlockIds> block_indices;
        std::shared_ptr<BatchKVCacheResource> resource;
    };

    void notify_write_through(const WriteInfo& write_info) {
        // write through to memory async
        auto task = [this, resource = write_info.resource]() {
            auto callback = [](bool success) {
                if (!success) {
                    RTP_LLM_LOG_ERROR("write cache to memory failed");
                }
            };
            reader_writer_->write(resource, callback);
        };
        thread_pool_.pushTask(task);
    }
    void notify_write_back(const WriteBackInfo& write_back_info) {
        // write back to memory async
    }

private:
    std::shared_ptr<KVCacheReaderWriter> reader_writer_;
    ::autil::ThreadPool thread_pool_;
};

}  // namespace rtp_llm