#include "aios/network/rdma/accl/RdmaMempoolImpl.h"

#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaMempool);

RdmaMempool::RdmaMempool(): _curState(CREATED) {}

RdmaMempool::~RdmaMempool() {
    if (_shrinkThread) {
        _shrinkThread->stop();
        _shrinkThread.reset();
    }

    if (_xmempool) {
        _xmempool->Shutdown();
        _xmempool->WaitStop();
    }
}

bool RdmaMempool::Init(const RdmaMempoolConfig& config) {
    if (_curState.load() == INIT_SUCCESS) {
        return true;
    } else if (_curState.load() == INIT_FAIL) {
        return false;
    }

    if (!_xmempool) {
        ::accl::barex::XDeviceManager* manager;
        auto                           ret = ::accl::barex::XDeviceManager::Singleton(manager);
        if (ret != ::accl::barex::BAREX_SUCCESS) {
            return false;
        }
        auto devices = manager->AllDevices();
        if (devices.empty()) {
            return false;
        }

        ::accl::barex::XThreadpool* allocThreadPool = nullptr;
        if (config.enableAsyncAlloc) {
            // 启用了异步内存分配时，需要传一个线程池给 mempool, 这样 mempool 才能支持异步分配
            auto result = ::accl::barex::XThreadpool::NewInstance(allocThreadPool,
                                                                  config.asyncAllocThreadCount,
                                                                  "arpc-accl-mempool-alloc-thread",
                                                                  config.asyncAllocQueueDepth);
            if (result != ::accl::barex::BAREX_SUCCESS) {
                AUTIL_LOG(WARN, "create alloc thread failed, error: %d", result);
                return false;
            }
        }

        ::accl::barex::XSimpleMempool* xmempool = nullptr;
        auto                           result   = ::accl::barex::XSimpleMempool::NewInstance(xmempool,
                                                                 "arpc-accl-mempool",
                                                                 devices,
                                                                 config.cpuMaxTotalBytes,
                                                                 config.gpuMaxTotalBytes,
                                                                                             {},
                                                                 allocThreadPool);
        if (result != ::accl::barex::BAREX_SUCCESS) {
            return false;
        }
        _xmempool.reset(xmempool);
    }

    if (!ReserveMem(config.cpuReserves, ::accl::barex::CPU) || !ReserveMem(config.gpuReserves, ::accl::barex::GPU)) {
        return false;
    }

    // start auto shrink
    if (config.enableAutoShrink) {
        uint64_t shrinkStartTimeSec = autil::TimeUtility::currentTimeInSeconds() + config.shrinkStartWaitTimeSec;
        _shrinkThread               = autil::LoopThread::createLoopThread(
            std::bind(
                &RdmaMempool::TriggerShrink, this, shrinkStartTimeSec, config.shrinkFreeRatio, config.shrinkFreeToBusy),
            config.shrinkIntervalSec * 1000 * 1000,
            "rdma-mempool-shrink");
    }

    _curState.store(INIT_SUCCESS);
    AUTIL_LOG(INFO, "rdma mempool init with config %s", autil::legacy::ToJsonString(config, true).c_str());
    return true;
}

void* RdmaMempool::AllocBuffer(uint64_t size, DeviceType deviceType, int deviceId, void* attr) const {
    if (_curState.load() != INIT_SUCCESS || !_xmempool) {
        AUTIL_LOG(WARN, "rdma mempool not init");
        return nullptr;
    }

    auto type = RdmaMempool::ConvertToAcclDeviceType(deviceType);
    auto ret  = _xmempool->AllocBuffer(size, type, deviceId, attr);
    if (ret == nullptr) {
        AUTIL_LOG(WARN,
                  "rdma mempool alloc nullptr buffer, size %lu, devicetype %s, deviceId %d",
                  size,
                  DeviceTypeName(deviceType).c_str(),
                  deviceId);
    }
    return ret;
}

bool RdmaMempool::ReleaseBuffer(void* buf, DeviceType deviceType) const {
    if (_curState.load() != INIT_SUCCESS || !_xmempool) {
        return false;
    }

    auto type   = RdmaMempool::ConvertToAcclDeviceType(deviceType);
    auto result = _xmempool->ReleaseBuffer(buf, type);
    if (result != ::accl::barex::BAREX_SUCCESS) {
        return false;
    }
    return true;
}

bool RdmaMempool::FindBuffer(void* buf, DeviceType deviceType) const {
    if (_curState.load() != INIT_SUCCESS || !_xmempool) {
        return false;
    }
    auto type = RdmaMempool::ConvertToAcclDeviceType(deviceType);
    return _xmempool->IsValid(buf, type);
}

bool RdmaMempool::RegUserMr(void* buf, uint64_t size, DeviceType deviceType) const {
    if (_curState.load() != INIT_SUCCESS || !_xmempool) {
        AUTIL_LOG(WARN, "rdma mempool not init");
        return false;
    }
    auto type   = RdmaMempool::ConvertToAcclDeviceType(deviceType);
    auto result = _xmempool->RegUserMr(buf, size, type);
    if (result != accl::barex::BAREX_SUCCESS) {
        AUTIL_LOG(WARN, "register memory failed, %d", result);
        return false;
    }
    return true;
}

bool RdmaMempool::RegUserMr(::accl::barex::memp_t& out, void* buf, uint64_t size, DeviceType deviceType) const {
    if (_curState.load() != INIT_SUCCESS || !_xmempool) {
        AUTIL_LOG(WARN, "rdma mempool not init");
        return false;
    }
    auto type   = RdmaMempool::ConvertToAcclDeviceType(deviceType);
    auto result = _xmempool->RegUserMr(out, buf, size, type);
    if (result != accl::barex::BAREX_SUCCESS) {
        AUTIL_LOG(WARN, "register memory failed, %d", result);
        return false;
    }
    return true;
}

bool RdmaMempool::DeregUserMr(void* buf, DeviceType deviceType) const {
    if (_curState.load() != INIT_SUCCESS || !_xmempool) {
        AUTIL_LOG(WARN, "rdma mempool not init");
        return false;
    }
    auto type   = RdmaMempool::ConvertToAcclDeviceType(deviceType);
    auto result = _xmempool->DeregUserMr(buf, type);
    if (result != accl::barex::BAREX_SUCCESS) {
        AUTIL_LOG(WARN, "deregister memory failed, %d", result);
        return false;
    }
    return true;
}

RdmaMempool::DeviceType RdmaMempool::FindBufferDevice(void* buf, bool preferCPU) const {
    if (preferCPU) {
        if (FindBuffer(buf, RdmaMempool::CPU)) {
            return DeviceType::CPU;
        }
        if (FindBuffer(buf, RdmaMempool::GPU)) {
            return DeviceType::GPU;
        }
    } else {
        if (FindBuffer(buf, RdmaMempool::GPU)) {
            return DeviceType::GPU;
        }
        if (FindBuffer(buf, RdmaMempool::CPU)) {
            return DeviceType::CPU;
        }
    }
    return DeviceType::UNKNOWN;
}

::accl::barex::device_type RdmaMempool::ConvertToAcclDeviceType(DeviceType type) {
    return type == DeviceType::GPU ? ::accl::barex::GPU : ::accl::barex::CPU;
}

RdmaMempool::DeviceType RdmaMempool::ConvertFromAcclDeviceType(::accl::barex::device_type type) {
    return type == ::accl::barex::GPU ? DeviceType::GPU : DeviceType::CPU;
}

std::string RdmaMempool::DeviceTypeName(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::GPU:
            return "GPU";
        default:
            return "UNKNOWN";
    }
}

void RdmaMempool::TriggerShrink(uint64_t shrinkStartTimeSec, float shrinkFreeRatio, float shrinkFreeToBusy) {
    if (autil::TimeUtility::currentTimeInSeconds() < shrinkStartTimeSec) {
        AUTIL_LOG(INFO, "rdma mempool shrink not started, start time %ld", shrinkStartTimeSec);
        return;
    }

    if (_xmempool && shrinkFreeRatio > 0) {
        if (::accl::barex::BAREX_SUCCESS != _xmempool->Shrink(::accl::barex::CPU, shrinkFreeRatio, shrinkFreeToBusy)) {
            AUTIL_LOG(DEBUG,
                      "rdma mempool shrink cpu failed, free ratio %f, free to busy ratio %f",
                      shrinkFreeRatio,
                      shrinkFreeToBusy);
        }
#ifdef GOOGLE_CUDA
        if (::accl::barex::BAREX_SUCCESS != _xmempool->Shrink(::accl::barex::GPU, shrinkFreeRatio, shrinkFreeToBusy)) {
            AUTIL_LOG(DEBUG,
                      "rdma mempool shrink gpu failed, free ratio %f, free to busy ratio %f",
                      shrinkFreeRatio,
                      shrinkFreeToBusy);
        }
#endif
    }
}

bool RdmaMempool::ReserveMem(std::map<uint64_t, int> reserves, ::accl::barex::device_type device) {
    if (reserves.empty()) {
        return true;
    }
    if (_xmempool->Reserve(device, reserves) != ::accl::barex::BAREX_SUCCESS) {
        AUTIL_LOG(ERROR, "rdma mempool reserve mem failed, device %d", device);
        return false;
    }
    return true;
}

ARPC_END_NAMESPACE(arpc);