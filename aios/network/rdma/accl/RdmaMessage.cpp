#include "aios/network/rdma/accl/RdmaMessage.h"

#include <sstream>

#include "aios/network/accl-barex/include/accl/barex/barex.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "aios/network/arpc/arpc/metric/ClientRPCStats.h"
#include "aios/network/arpc/arpc/metric/ServerRPCStats.h"
#include "aios/network/rdma/accl/MessageConverter.h"
#include "autil/Lock.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaMessage);

enum MessageStatus {
    SUCCESS = 0,
    FAILED,
};

RdmaMessage::RdmaMessage(): _requestId(0), _failedMessage(false), _cudaUtil(new CudaUtil) {}

RdmaMessage::~RdmaMessage() {
    ReleaseMessage();
}

bool RdmaMessage::ReleaseMessage() {
    autil::ScopedWriteLock lock(_iovsLock);
    for (auto iov : _iovs) {
        delete iov;
    }
    _iovs.clear();
    return true;
}

::accl::barex::XMessage RdmaMessage::GetMessage() {
    ::accl::barex::XMessage xmessage;
    MessageConverter        converter;
    if (!converter.ConvertFromRdmaMessage(this, xmessage)) {
        AUTIL_LOG(WARN, "convert from rdma message failed");
    }
    xmessage.SetUserFlag0(_requestId);
    xmessage.SetUserFlag1(_failedMessage ? MessageStatus::FAILED : MessageStatus::SUCCESS);
    return xmessage;
}

bool RdmaMessage::SetRecvMessage(const ::accl::barex::XMessage& msg) {
    {
        autil::ScopedReadLock lock(_iovsLock);
        if (!_iovs.empty()) {
            AUTIL_LOG(WARN, "set recv message failed, already has iov");
            return false;
        }
    }
    MessageConverter converter;
    if (!converter.ConvertFromXMessage(this, msg)) {
        AUTIL_LOG(WARN, "convert from msg failed");
        return false;
    }

    _requestId     = msg.GetUserFlag0();
    _failedMessage = msg.GetUserFlag1() != MessageStatus::SUCCESS;
    return true;
}

void* RdmaMessage::AllocateSendBuffer(uint32_t len, RdmaMempool::DeviceType device) const {
    if (len == 0) {
        return nullptr;
    }
    if (!_mempool) {
        AUTIL_LOG(WARN, "allocate send buffer failed, no releated mempool");
        return nullptr;
    }

    return _mempool->AllocBuffer(len, device);
}

std::shared_ptr<void> RdmaMessage::AllocateSharedSendBuffer(uint32_t len, RdmaMempool::DeviceType device) const {
    if (len == 0) {
        return std::shared_ptr<void>();
    }

    if (!_mempool) {
        AUTIL_LOG(WARN, "allocate send buffer failed, no releated mempool");
        return std::shared_ptr<void>();
    }

    auto buf = _mempool->AllocBuffer(len, device);
    if (!buf) {
        return std::shared_ptr<void>();
    }

    auto mempool = _mempool;
    return std::shared_ptr<void>(buf, [mempool, device](void* p) { mempool->ReleaseBuffer(p, device); });
}

bool RdmaMessage::ReleaseSendBuffer(void* buffer, RdmaMempool::DeviceType device) const {
    if (!_mempool) {
        AUTIL_LOG(WARN, "release send buffer failed, no releated mempool");
        return false;
    }
    return _mempool->ReleaseBuffer(buffer, device);
}

void RdmaMessage::AddIov(const std::shared_ptr<void>& buffer,
                         uint32_t                     len,
                         BufferMRStatus               mrStatus,
                         RdmaMempool::DeviceType      device,
                         RdmaMempool::DeviceType      remoteDevice) {
    auto                   iov = new Iov(buffer, len, mrStatus, device, remoteDevice);
    autil::ScopedWriteLock lock(_iovsLock);
    _iovs.push_back(iov);
}

void RdmaMessage::UpdateIov(uint32_t                     index,
                            const std::shared_ptr<void>& buffer,
                            uint32_t                     len,
                            BufferMRStatus               mrStatus,
                            RdmaMempool::DeviceType      device,
                            RdmaMempool::DeviceType      remoteDevice) {
    autil::ScopedReadLock lock(_iovsLock);
    if (index >= _iovs.size()) {
        return;
    }
    _iovs[index]->buf          = buffer;
    _iovs[index]->len          = len;
    _iovs[index]->mrStatus     = mrStatus;
    _iovs[index]->device       = device;
    _iovs[index]->remoteDevice = remoteDevice;
}

const RdmaMessage::Iov* RdmaMessage::GetIov(uint32_t index) const {
    autil::ScopedReadLock lock(_iovsLock);
    if (index >= _iovs.size()) {
        return nullptr;
    }
    return _iovs[index];
}

std::shared_ptr<void> RdmaMessage::GetIovBuffer(uint32_t index) const {
    autil::ScopedReadLock lock(_iovsLock);
    if (index >= _iovs.size()) {
        return std::shared_ptr<void>();
    }
    return _iovs[index]->buf;
}

uint32_t RdmaMessage::GetIovLen(uint32_t index) const {
    autil::ScopedReadLock lock(_iovsLock);
    if (index >= _iovs.size()) {
        return 0;
    }
    return _iovs[index]->len;
}

size_t RdmaMessage::GetIovSize() const {
    autil::ScopedReadLock lock(_iovsLock);
    return _iovs.size();
}

uint32_t RdmaMessage::GetTotalIovLength() const {
    autil::ScopedReadLock lock(_iovsLock);
    uint32_t              len = 0;
    for (auto& iov : _iovs) {
        len += iov->len;
    }
    return len;
}

void RdmaMessage::GetIovStatusString(std::stringstream&      ss,
                                     BufferMRStatus          mrStatus,
                                     RdmaMempool::DeviceType device,
                                     RdmaMempool::DeviceType remoteDevice) const {
    int64_t              totalSize = 0;
    std::vector<int64_t> iovs;
    for (size_t i = 0; i < _iovs.size(); i++) {
        if (_iovs[i]->mrStatus == mrStatus && _iovs[i]->device == device && _iovs[i]->remoteDevice == remoteDevice) {
            iovs.push_back(_iovs[i]->len);
            totalSize += _iovs[i]->len;
        }
    }
    if (iovs.size() > 0) {
        ss << "message " << BufferMRStatusName(mrStatus) << "-" << RdmaMempool::DeviceTypeName(device) << " to "
           << RdmaMempool::DeviceTypeName(remoteDevice) << " iov count: " << iovs.size() << " total size: " << totalSize
           << " iovs: ";
        for (auto& len : iovs) {
            ss << len << " ";
        }
        ss << std::endl;
    }
}

std::string RdmaMessage::GetStatusString() const {
    autil::ScopedReadLock lock(_iovsLock);
    std::stringstream     ss;
    ss << "message iov count " << _iovs.size() << " total size: " << GetTotalIovLength() << std::endl;
    for (int status = 0; status < BufferMRStatus::BMS_UNKNOWN; status++) {
        for (int device = 0; device < RdmaMempool::UNKNOWN; device++) {
            for (int rDevice = 0; rDevice < RdmaMempool::UNKNOWN; rDevice++) {
                GetIovStatusString(
                    ss, (BufferMRStatus)status, RdmaMempool::DeviceType(device), RdmaMempool::DeviceType(rDevice));
            }
        }
    }
    return ss.str();
}

std::string RdmaMessage::GetXMessageStatusString(const ::accl::barex::XMessage& xmessage) const {
    std::stringstream ss;
    ss << "xmessage iov count " << xmessage.IovCount() << std::endl;
    for (size_t i = 0; i < xmessage.IovCount(); i++) {
        auto iov = xmessage.Get(i);
        ss << " iov " << i << " buf: " << (void*)iov.iov_base << " len: " << iov.iov_len << " device: " << iov.device
           << " remote device: " << iov.remote_device << std::endl;
    }
    return ss.str();
}

void RdmaMessage::reportClientRPCStats(const std::shared_ptr<ClientRPCStats>& stats,
                                       const ::accl::barex::XMessage&         xmessage) {
    if (stats) {
        std::map<::accl::barex::device_type, std::map<::accl::barex::device_type, size_t>> iovStats;
        for (int i = 0; i < xmessage.IovCount(); i++) {
            auto iov = xmessage.Get(i);
            iovStats[iov.device][iov.remote_device] += iov.iov_len;
        }
        for (auto& iterDevice : iovStats) {
            for (auto& iterRemoteDevice : iterDevice.second) {
                stats->markRequestIovSize(
                    RdmaMempool::DeviceTypeName(RdmaMempool::ConvertFromAcclDeviceType(iterDevice.first)),
                    RdmaMempool::DeviceTypeName(RdmaMempool::ConvertFromAcclDeviceType(iterRemoteDevice.first)),
                    iterRemoteDevice.second);
            }
        }
    }
}

void RdmaMessage::reportServerRPCStats(const std::shared_ptr<ServerRPCStats>& stats,
                                       const ::accl::barex::XMessage&         xmessage) {
    if (stats) {
        std::map<::accl::barex::device_type, std::map<::accl::barex::device_type, size_t>> iovStats;
        for (int i = 0; i < xmessage.IovCount(); i++) {
            auto iov = xmessage.Get(i);
            iovStats[iov.device][iov.remote_device] += iov.iov_len;
        }
        for (auto& iterDevice : iovStats) {
            for (auto& iterRemoteDevice : iterDevice.second) {
                stats->markResponseIovSize(
                    RdmaMempool::DeviceTypeName(RdmaMempool::ConvertFromAcclDeviceType(iterDevice.first)),
                    RdmaMempool::DeviceTypeName(RdmaMempool::ConvertFromAcclDeviceType(iterRemoteDevice.first)),
                    iterRemoteDevice.second);
            }
        }
    }
}

std::string RdmaMessage::BufferMRStatusName(BufferMRStatus status) const {
    switch (status) {
        case BMS_MR:
            return "MR";
        case BMS_NOT_MR:
            return "NOT_MR";
        default:
            return "UNKNOWN";
    }
}

bool RdmaMessage::VerifyIovStatus() const {
    autil::ScopedReadLock lock(_iovsLock);
    for (auto iov : _iovs) {
        if (iov == nullptr) {
            continue;
        }

        if (iov->remoteDevice == RdmaMempool::UNKNOWN) {
            AUTIL_LOG(WARN, "unknown remote device");
            return false;
        }

        if (iov->mrStatus == RdmaMessage::BMS_UNKNOWN
            || iov->device == RdmaMempool::UNKNOWN) {  // 状态不确定, 处理后应该没有UNKNOWN状态的iov
            auto device = _mempool->FindBufferDevice(iov->buf.get(), iov->device == RdmaMempool::GPU ? false : true);
            if (device == RdmaMempool::CPU) {
                iov->mrStatus = BMS_MR;
                iov->device   = RdmaMempool::CPU;
            } else if (device == RdmaMempool::GPU) {
                iov->mrStatus = BMS_MR;
                iov->device   = RdmaMempool::GPU;
            } else {  // 没判断出来的调用Cuda接口检查
                iov->mrStatus = BMS_NOT_MR;
                if (_cudaUtil->IsGPUMemory(iov->buf.get())) {
                    iov->device = RdmaMempool::GPU;
                } else {
                    iov->device = RdmaMempool::CPU;
                }
            }
        }
    }
    return true;
}

ARPC_END_NAMESPACE(arpc);