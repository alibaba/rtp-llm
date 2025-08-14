#include "rtp_llm/cpp/cache/DistStorageManager.h"

#include <atomic>
#include "rtp_llm/cpp/cache/DistStorageLocalMem.h"
#include "rtp_llm/cpp/utils/Logger.h"

#ifdef ENABLE_3FS
#include "rtp_llm/cpp/cache/DistStorage3FS.h"
#endif

namespace rtp_llm {

DistStorageManager::~DistStorageManager() {
    RTP_LLM_LOG_INFO("DistStorageManager destructor");
    if (wait_task_thread_pool_) {
        wait_task_thread_pool_->stop();
        wait_task_thread_pool_->waitFinish();
        wait_task_thread_pool_.reset();
    }
}

bool DistStorageManager::init(const DistStorageManagerInitParams& init_params) {
    RTP_LLM_LOG_INFO("dist storage manager init params: [%s]", init_params.toString().c_str());
    init_params_ = init_params;

    wait_task_thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(thread_num_, queue_size_, nullptr, "DistStorageWaitTaskThreadPool");
    if (!wait_task_thread_pool_->start()) {
        RTP_LLM_LOG_WARNING("init failed, start wait task thread pool failed, thread num: %zu, queue size: %zu",
                            thread_num_,
                            queue_size_);
        return false;
    }

    if (init_params_.init_params_3fs.has_value()) {
#ifdef ENABLE_3FS
        auto init_params_3fs = init_params_.init_params_3fs.value();
        // root dir env only used for debug
        if (auto root_dir_env = autil::EnvUtil::getEnv("THREEFS_ROOT_DIR", std::string("")); !root_dir_env.empty()) {
            init_params_3fs.root_dir = root_dir_env;
        };

        auto storage = std::make_shared<threefs::DistStorage3FS>(metrics_reporter_);
        if (!storage->init(init_params_3fs)) {
            RTP_LLM_LOG_WARNING("init failed, 3fs storage init failed");
            return false;
        }
        storage_3fs_ = storage;
#else
        RTP_LLM_LOG_WARNING("init failed, 3fs not enabled");
        return false;
#endif
    }
    if (init_params_.init_params_local_mem.has_value()) {
        auto storage = std::make_shared<DistStorageLocalMem>(metrics_reporter_);
        if (!storage->init(init_params_.init_params_local_mem.value())) {
            RTP_LLM_LOG_WARNING("init failed, local mem storage init failed");
            return false;
        }
        storage_local_mem_ = storage;
    }
    return true;
}

const std::shared_ptr<DistStorage> DistStorageManager::getStorage(const DistStorage::Item& item) {
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
    if (!storage) {
        return false;
    }
    auto task = [storage, item]() { return storage->lookup(item); };
    return runWithTimeout(OpType::LOOKUP, task, init_params_.lookup_timeout_ms);
}

bool DistStorageManager::get(DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (!storage) {
        return false;
    }
    auto task = [storage, &item]() { return storage->get(item); };
    return runWithTimeout(OpType::GET, task, init_params_.get_timeout_ms);
}

bool DistStorageManager::put(const DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (!storage) {
        return false;
    }
    auto task = [storage, item]() { return storage->put(item); };
    return runWithTimeout(OpType::PUT, task, init_params_.put_timeout_ms);
}

bool DistStorageManager::putIfNotExist(const DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (!storage) {
        return false;
    }
    auto task = [storage, item]() {
        if (storage->lookup(item)) {
            return true;
        }
        return storage->put(item);
    };
    return runWithTimeout(OpType::PUT, task, init_params_.put_timeout_ms);
}

bool DistStorageManager::del(const DistStorage::Item& item) {
    auto storage = getStorage(item);
    if (!storage) {
        return false;
    }
    auto task = [storage, item]() { return storage->del(item); };
    return runWithTimeout(OpType::DEL, task, init_params_.del_timeout_ms);
}

bool DistStorageManager::runWithTimeout(OpType op_type, const std::function<bool()>& func, int timeout_ms) const {
    if (wait_task_thread_pool_->isFull()) {
        RTP_LLM_LOG_WARNING("run %s failed, wait task thread pool is full, something maybe wrong",
                            getOpTypeString(op_type).c_str());
        return false;
    }

    // wrap func with stop flag inside
    auto stop    = std::make_shared<std::atomic<bool>>(false);
    auto wrapped = [stop, func]() -> bool {
        if (stop->load()) {
            return false;
        }
        return func();
    };

    auto future = wait_task_thread_pool_->async(wrapped);
    if (future.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::ready) {
        return future.get();
    }

    RTP_LLM_LOG_WARNING("run %s but timeout: %d ms", getOpTypeString(op_type).c_str(), timeout_ms);
    stop->store(true);
    return false;
}

std::string DistStorageManager::getOpTypeString(OpType op_type) const {
    switch (op_type) {
        case OpType::LOOKUP: {
            return "lookup";
        }
        case OpType::GET: {
            return "get";
        }
        case OpType::PUT: {
            return "put";
        }
        case OpType::DEL: {
            return "del";
        }
    }
    return "unknown";
}

}  // namespace rtp_llm