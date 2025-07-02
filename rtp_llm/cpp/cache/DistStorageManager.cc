#include "rtp_llm/cpp/cache/DistStorageManager.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include "rtp_llm/cpp/cache/DistStorageLocalMem.h"
#ifdef ENABLE_3FS
#include "rtp_llm/cpp/cache/DistStorage3FS.h"
#endif

namespace rtp_llm {

bool DistStorageManager::init(const DistStorageManagerInitParams& init_params) {
    init_params_ = init_params;
    if (init_params_.init_params_3fs.has_value()) {
#ifdef ENABLE_3FS
        auto storage = std::make_shared<DistStorage3FS>(init_params_.metrics_reporter);
        if (!storage->init(init_params_.init_params_3fs.value())) {
            RTP_LLM_LOG_WARNING("init failed, 3fs storage init failed");
            return false;
        }
        storage_3fs_ = storage;
#else RTP_LLM_LOG_WARNING("init failed, 3fs storage init failed");
        return false;
#endif
    }
    if (init_params_.init_params_local_mem.has_value()) {
        auto storage = std::make_shared<DistStorageLocalMem>(init_params_.metrics_reporter);
        if (!storage->init(init_params_.init_params_local_mem.value())) {
            RTP_LLM_LOG_WARNING("init failed, local mem storage init failed");
            return false;
        }
        storage_local_mem_ = storage;
    }
    return true;
}

const std::shared_ptr<DistStorage>& DistStorageManager::getStorage(const DistStorage::Item& item) {
    switch (item.type) {
        case DistStorage::ST_3FS: {
            return storage_3fs_;
        }
        case DistStorage::ST_LOCAL_MEM: {
            return storage_local_mem_;
        }
        default: {
            RTP_LLM_LOG_WARNING("get storage failed, unknown storage type: %d", item.type);
            return nullptr;
        }
    }
}

bool DistStorageManager::lookup(const DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (storage) {
        return storage->lookup(item);
    }
    return false;
}

bool DistStorageManager::get(DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (storage) {
        return storage->get(item);
    }
    return false;
}

bool DistStorageManager::put(const DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (storage) {
        return storage->put(item);
    }
    return false;
}

bool DistStorageManager::del(const DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (storage) {
        return storage->del(item);
    }
    return false;
}

}  // namespace rtp_llm