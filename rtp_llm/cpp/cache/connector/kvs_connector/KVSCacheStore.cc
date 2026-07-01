#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSCacheStore.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/allocator/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::kvs {
namespace {

std::string sanitizeKeyPart(std::string value) {
    for (char& ch : value) {
        if (ch == '/' || ch == ' ' || ch == '\t' || ch == '\n') {
            ch = '_';
        }
    }
    return value;
}

std::string hashString(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    std::ostringstream oss;
    oss << std::hex << hash;
    return oss.str();
}

size_t totalBytes(const std::vector<BlockInfo>& iovs) {
    size_t bytes = 0;
    for (const auto& iov : iovs) {
        bytes += iov.size_bytes;
    }
    return bytes;
}

bool isValidBlockInfo(const BlockInfo& block_info) {
    return block_info.addr != nullptr && block_info.size_bytes > 0;
}

}  // namespace

KVSCacheStore::KVSCacheStore(KVSCacheStoreConfig                config,
                             std::shared_ptr<KVCacheAllocator> allocator,
                             std::shared_ptr<KVSClient>        client,
                             kmonitor::MetricsReporterPtr      metrics_reporter):
    config_(std::move(config)),
    allocator_(std::move(allocator)),
    client_(std::move(client)),
    metrics_reporter_(std::move(metrics_reporter)) {}

bool KVSCacheStore::init() {
    if (!client_) {
        client_ = std::make_shared<KVSClient>();
    }
    if (config_.deployment_id.empty()) {
        config_.deployment_id = buildDeploymentId();
    }

    KVSClientConfig client_config;
    client_config.v6d_url         = config_.kv_cache_config.kvs_v6d_url;
    client_config.v6d_socket_path = config_.kv_cache_config.kvs_v6d_socket_path;
    client_config.timeout_ms      = config_.kv_cache_config.kvs_timeout_ms;
    client_config.lease_term_sec  = config_.kv_cache_config.kvs_lease_term_sec;
    if (!client_->init(client_config)) {
        RTP_LLM_LOG_ERROR("KVSCacheStore init failed: KVSClient init failed");
        return false;
    }
    RTP_LLM_LOG_INFO("KVSCacheStore initialized, deployment_id[%s]", config_.deployment_id.c_str());
    return true;
}

std::optional<KVSMatchSession> KVSCacheStore::match(const std::shared_ptr<KVCacheResource>& resource,
                                                    const std::shared_ptr<Meta>&            meta) {
    if (!resource || !meta) {
        return std::nullopt;
    }

    KVSMatchSession session;
    session.prev_reuse_blocks = resource->reuseBlockNum();
    session.matched_blocks    = session.prev_reuse_blocks;

    auto keys = resource->cacheKeys();
    if (!keys.empty()) {
        keys.pop_back();
    }
    if (session.prev_reuse_blocks >= keys.size()) {
        return session;
    }

    auto                     block_objects = buildBlockObjects(resource, keys);
    std::vector<std::string> object_keys;
    for (size_t i = session.prev_reuse_blocks; i < block_objects.size(); ++i) {
        for (const auto& object : block_objects[i].objects) {
            object_keys.push_back(object.object_key);
        }
    }
    if (object_keys.empty()) {
        RTP_LLM_LOG_WARNING("KVSCacheStore match has no readable objects, prev_reuse[%zu], candidate_blocks[%zu]",
                            session.prev_reuse_blocks,
                            keys.size());
        return session;
    }
    std::sort(object_keys.begin(), object_keys.end());
    object_keys.erase(std::unique(object_keys.begin(), object_keys.end()), object_keys.end());

    RTP_LLM_LOG_INFO("KVSCacheStore match start, trace_id[%s], prev_reuse[%zu], candidate_blocks[%zu], object_count[%zu]",
                     meta->trace_id().c_str(),
                     session.prev_reuse_blocks,
                     block_objects.size(),
                     object_keys.size());
    auto read_session = client_->acquireForRead(object_keys, "kvs_match_" + meta->trace_id());
    if (!read_session) {
        RTP_LLM_LOG_WARNING("KVSCacheStore match failed, degrade to miss, trace_id[%s]", meta->trace_id().c_str());
        return session;
    }

    size_t matched = session.prev_reuse_blocks;
    for (size_t i = session.prev_reuse_blocks; i < block_objects.size(); ++i) {
        bool hit = true;
        for (const auto& object : block_objects[i].objects) {
            if (read_session->handles.find(object.object_key) == read_session->handles.end()) {
                hit = false;
                break;
            }
        }
        if (!hit) {
            break;
        }
        matched++;
        session.matched_block_objects.push_back(block_objects[i]);
    }

    session.matched_blocks = matched;
    session.read_session   = std::move(read_session);
    if (session.matched_blocks <= session.prev_reuse_blocks) {
        close(session);
    }

    RTP_LLM_LOG_INFO("KVSCacheStore match done, trace_id[%s], matched_blocks[%zu], prev_reuse[%zu], lease[%s]",
                     meta->trace_id().c_str(),
                     session.matched_blocks,
                     session.prev_reuse_blocks,
                     session.read_session ? session.read_session->lease_id.c_str() : "");
    return session;
}

bool KVSCacheStore::read(KVSMatchSession&                         session,
                         const std::shared_ptr<KVCacheResource>& resource,
                         int                                      start_read_block_index,
                         int                                      read_block_num) {
    if (!resource || !session.read_session || read_block_num <= 0) {
        return false;
    }

    std::vector<KVSObjectBuffer> dst_buffers;
    const int                    end_block_index = start_read_block_index + read_block_num;
    for (const auto& block : session.matched_block_objects) {
        if (static_cast<int>(block.block_index) < start_read_block_index
            || static_cast<int>(block.block_index) >= end_block_index) {
            continue;
        }
        for (const auto& object : block.objects) {
            dst_buffers.push_back(KVSObjectBuffer{object.object_key, object.iovs});
        }
    }
    if (dst_buffers.empty()) {
        RTP_LLM_LOG_WARNING("KVSCacheStore read has no destination buffers, start_block[%d], block_num[%d]",
                            start_read_block_index,
                            read_block_num);
        return false;
    }
    std::vector<std::string> object_keys;
    object_keys.reserve(dst_buffers.size());
    for (const auto& dst : dst_buffers) {
        object_keys.push_back(dst.object_key);
    }
    const auto trace_id = "kvs_read_" + session.read_session->lease_id;

    RTP_LLM_LOG_INFO("KVSCacheStore read start, start_block[%d], block_num[%d], object_count[%zu]",
                     start_read_block_index,
                     read_block_num,
                     dst_buffers.size());
    if (!client_->fetch(*session.read_session, object_keys, trace_id)) {
        RTP_LLM_LOG_WARNING("KVSCacheStore fetch failed, start_block[%d], block_num[%d], object_count[%zu]",
                            start_read_block_index,
                            read_block_num,
                            object_keys.size());
        return false;
    }
    if (!client_->load(*session.read_session, dst_buffers)) {
        RTP_LLM_LOG_WARNING("KVSCacheStore read failed, start_block[%d], block_num[%d], object_count[%zu]",
                            start_read_block_index,
                            read_block_num,
                            dst_buffers.size());
        return false;
    }
    if (!client_->complete(*session.read_session, object_keys, trace_id)) {
        RTP_LLM_LOG_WARNING("KVSCacheStore complete failed, start_block[%d], block_num[%d], object_count[%zu]",
                            start_read_block_index,
                            read_block_num,
                            object_keys.size());
        return false;
    }
    resource->setRemoteReuseBlockNum(start_read_block_index + read_block_num);
    RTP_LLM_LOG_INFO("KVSCacheStore read done, remote_reuse_blocks[%d]", start_read_block_index + read_block_num);
    return true;
}

bool KVSCacheStore::write(const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta) {
    if (!resource || !meta) {
        return false;
    }
    auto keys = resource->cacheKeys();
    if (!keys.empty() && !resource->lastBlockAligned()) {
        keys.pop_back();
    }
    if (keys.empty()) {
        return true;
    }

    auto                         block_objects = buildBlockObjects(resource, keys);
    std::vector<KVSObjectBuffer> src_buffers;
    for (const auto& block : block_objects) {
        for (const auto& object : block.objects) {
            if (!object.iovs.empty() && totalBytes(object.iovs) > 0) {
                src_buffers.push_back(KVSObjectBuffer{object.object_key, object.iovs});
            }
        }
    }
    RTP_LLM_LOG_INFO("KVSCacheStore write start, trace_id[%s], blocks[%zu], object_count[%zu]",
                     meta->trace_id().c_str(),
                     block_objects.size(),
                     src_buffers.size());
    if (!client_->store(src_buffers, "kvs_store_" + meta->trace_id())) {
        RTP_LLM_LOG_WARNING(
            "KVSCacheStore write failed, trace_id[%s], object_count[%zu]", meta->trace_id().c_str(), src_buffers.size());
        return false;
    }
    RTP_LLM_LOG_INFO(
        "KVSCacheStore write done, trace_id[%s], object_count[%zu]", meta->trace_id().c_str(), src_buffers.size());
    return true;
}

void KVSCacheStore::close(KVSMatchSession& session) {
    if (session.read_session && !session.read_session->lease_id.empty() && client_) {
        client_->release(session.read_session->lease_id);
        session.read_session->lease_id.clear();
    }
    session.read_session.reset();
}

std::vector<KVSBlockObjects>
KVSCacheStore::buildBlockObjects(const std::shared_ptr<KVCacheResource>& resource,
                                 const std::vector<CacheKeyType>&        keys) const {
    std::vector<KVSBlockObjects> result;
    if (!resource || !allocator_) {
        return result;
    }
    result.reserve(keys.size());
    for (size_t block_index = 0; block_index < keys.size(); ++block_index) {
        KVSBlockObjects block_objects;
        block_objects.block_index = block_index;
        for (int group_id = 0; group_id < resource->groupNums(); ++group_id) {
            const auto& block_ids = resource->groupBlocks().at(group_id)->blocks();
            if (block_index >= block_ids.size() || isNullBlockIdx(block_ids[block_index])) {
                continue;
            }
            std::vector<BlockInfo> iovs;
            for (int layer_id : config_.cache_config.layerIdsForGroup(static_cast<size_t>(group_id))) {
                auto block_infos = allocator_->convertIndexToBuffer(layer_id, group_id, block_ids[block_index]);
                for (const auto& block_info : block_infos) {
                    if (!isValidBlockInfo(block_info)) {
                        iovs.clear();
                        break;
                    }
                    iovs.push_back(block_info);
                }
                if (iovs.empty()) {
                    break;
                }
            }
            if (!iovs.empty()) {
                block_objects.objects.push_back(KVSBlockObject{
                    block_index, group_id, buildObjectKey(keys[block_index], group_id), std::move(iovs)});
            }
        }
        result.push_back(std::move(block_objects));
    }
    return result;
}

std::string KVSCacheStore::buildObjectKey(CacheKeyType cache_key, int32_t group_id) const {
    std::ostringstream oss;
    oss << sanitizeKeyPart(config_.kv_cache_config.kvs_object_namespace) << "/v"
        << sanitizeKeyPart(config_.kv_cache_config.kvs_cache_key_version) << "/" << config_.deployment_id << "/tp-"
        << config_.parallelism_config.tp_rank << "/group-" << group_id << "/block-" << cache_key;
    return oss.str();
}

std::string KVSCacheStore::buildDeploymentId() const {
    const auto         checkpoint_path = autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""));
    std::ostringstream raw;
    raw << "model=" << config_.runtime_config.model_name << ";ckpt=" << checkpoint_path
        << ";dtype=" << getDataTypeStr(config_.cache_config.dtype) << ";seq=" << config_.cache_config.seq_size_per_block
        << ";kernel_seq=" << config_.cache_config.kernel_seq_size_per_block
        << ";tp=" << config_.parallelism_config.tp_size << ";fp8=" << config_.kv_cache_config.fp8_kv_cache
        << ";int8=" << config_.kv_cache_config.int8_kv_cache << ";mla=" << config_.cache_config.use_mla
        << ";groups=" << config_.cache_config.groupNums() << ";typed=" << config_.cache_config.use_typed_cache_regions;
    return sanitizeKeyPart(config_.runtime_config.model_name.empty() ? "model" : config_.runtime_config.model_name)
           + "-" + hashString(raw.str());
}

}  // namespace rtp_llm::kvs
