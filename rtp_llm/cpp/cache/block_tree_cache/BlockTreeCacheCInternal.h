#pragma once

#include "BlockTreeCacheCStatus.h"
#include "BlockTreeCache.h"
#include "BlockTreeTaskPool.h"
#include "block_pool/ExternalPagePool.h"
#include "transfer/PolledTransfer.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm {
class DiskFiles;
}

namespace rtp_llm::c_api {
inline bool samePage(const btc_page& a, const btc_page& b) {
    return a.pool == b.pool && a.slot == b.slot && a.generation == b.generation && a.hold == b.hold;
}
struct RequestTable;
struct LocalReleases;
std::shared_ptr<RequestTable> makeRequests();
void                          cancelRequests(btc_cache& cache);
void                          notifyLocalCapacity(btc_cache& cache);
std::string                   copyBytes(btc_bytes value);
BlockTreeKeys                 copyKeys(btc_keys keys);

struct Group {
    uint32_t                    id;
    uint64_t                    page_tokens;
    std::vector<uint64_t>       member_pools;
    GroupSetPtr                 set;
    uint64_t                    host_start, disk_start;
};

// Holds temporary physical pool references until the native transaction
// consumes the page mapping or the caller rolls the entire batch back.
class PreparedPages {
public:
    explicit PreparedPages(btc_cache& cache): cache_(cache) {}
    PreparedPages(const PreparedPages&) = delete;
    ~PreparedPages();
    bool prepare(const btc_binding* bindings, size_t count, bool load_target);
    void adopt() {
        adopted_ = true;
    }
    using Position = std::pair<uint32_t, uint64_t>;
    std::map<Position, BlockIdList> rows;

private:
    btc_cache&                cache_;
    std::vector<BlockIdxType> prepared_;
    bool                      adopted_ = false;
};
}  // namespace rtp_llm::c_api

struct btc_cache {
    explicit btc_cache(const btc_config& config, const btc_host_region* host_region = nullptr);
    ~btc_cache();
    size_t   groupIndex(uint32_t id) const;
    btc_page describePage(size_t group_index, rtp_llm::Tier tier, rtp_llm::BlockIdxType block) const;
    void     event(uint64_t request, btc_event_kind kind, uint64_t tokens = 0);
    void     flush();
    void     queueReleases();

    std::mutex                         api_mutex;  // serializes foreign callers; worker callbacks never take this lock
    const uint64_t                     instance;
    const uint64_t                     key_tokens;
    const std::string                  name_space;
    std::vector<rtp_llm::c_api::Group> groups;
    std::map<uint64_t, bool>           device_watermark_rounds;
    std::map<uint32_t, size_t>         group_indices;
    std::vector<std::shared_ptr<rtp_llm::c_api::LocalReleases>> local_releases;
    std::shared_ptr<rtp_llm::ExternalPagePool>    device;
    std::shared_ptr<rtp_llm::PolledTransfer>      transfers;
    std::shared_ptr<rtp_llm::c_api::RequestTable> requests;
    std::shared_ptr<rtp_llm::StorageBackend>      storage;
    std::unique_ptr<rtp_llm::DiskFiles>           disk_files;
    std::mutex                                    event_mutex;
    std::deque<btc_event>                         events;
    uint64_t                                      cursor    = 0;
    bool                                          draining  = false;
    rtp_llm::BlockTreeTaskPool*                   task_pool = nullptr;  // owned by native
    std::unique_ptr<rtp_llm::BlockTreeCache>      native;
};

namespace rtp_llm::c_api {
template<typename F>
btc_status locked(btc_cache* cache, F&& f) {
    return status([&] {
        std::lock_guard lock(require(cache).api_mutex);
        return f(*cache);
    });
}
}  // namespace rtp_llm::c_api
