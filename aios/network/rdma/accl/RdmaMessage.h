#pragma once

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/CudaUtil.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "autil/Lock.h"
#include "autil/Log.h"

namespace accl {
namespace barex {
class XConnection;
class XSimpleMempool;
}  // namespace barex
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class ClientRPCStats;
class ServerRPCStats;

// RdmaMessage stores mem to pass to accl and info need in encode and decode
// !RdmaMessage not thread safe
class RdmaMessage {
public:
    enum BufferMRStatus {
        BMS_MR = 0,
        BMS_NOT_MR,
        BMS_UNKNOWN,
    };

    struct Iov {
        std::shared_ptr<void>   buf;
        uint32_t                len;
        BufferMRStatus          mrStatus;
        RdmaMempool::DeviceType device;
        RdmaMempool::DeviceType remoteDevice;

        Iov(): buf(nullptr), len(0), mrStatus(BMS_MR), device(RdmaMempool::CPU), remoteDevice(RdmaMempool::CPU) {}
        Iov(const std::shared_ptr<void>& buf_,
            uint32_t                     len_,
            BufferMRStatus               mrStatus_,
            RdmaMempool::DeviceType      device_,
            RdmaMempool::DeviceType      remoteDevice_):
            buf(buf_), len(len_), mrStatus(mrStatus_), device(device_), remoteDevice(remoteDevice_) {}
    };

public:
    RdmaMessage();
    virtual ~RdmaMessage();

public:
    virtual bool ReleaseMessage();

    // Message 处理接口
    virtual ::accl::barex::XMessage GetMessage();
    bool                            SetRecvMessage(const ::accl::barex::XMessage& msgs);

    void SetFailedMessage(bool failedMessage) {
        _failedMessage = failedMessage;
    }
    bool IsFailedMessage() const {
        return _failedMessage;
    }

    // Get & Set接口
    void SetRequestId(uint32_t requestId) {
        _requestId = requestId;
    }
    uint32_t GetRequestId() const {
        return _requestId;
    }

    const std::shared_ptr<::accl::barex::XConnection>& GetConnection() const {
        return _connection;
    }
    void SetConnection(const std::shared_ptr<::accl::barex::XConnection>& connection) {
        _connection = connection;
    }

    const std::shared_ptr<RdmaMempool>& GetMempool() const {
        return _mempool;
    }
    void SetMempool(const std::shared_ptr<RdmaMempool>& mempool) {
        _mempool = mempool;
    }

    // mempool包装接口, 分配内存
    void*                 AllocateSendBuffer(uint32_t len, RdmaMempool::DeviceType deviceType = RdmaMempool::CPU) const;
    std::shared_ptr<void> AllocateSharedSendBuffer(uint32_t                len,
                                                   RdmaMempool::DeviceType deviceType = RdmaMempool::CPU) const;
    bool                  ReleaseSendBuffer(void* buffer, RdmaMempool::DeviceType deviceType = RdmaMempool::CPU) const;

    // Iov操作接口
    void       AddIov(const std::shared_ptr<void>& buffer,
                      uint32_t                     len,
                      BufferMRStatus               mrStatus,
                      RdmaMempool::DeviceType      deviceType,
                      RdmaMempool::DeviceType      remoteDeviceType);
    void       UpdateIov(uint32_t                     index,
                         const std::shared_ptr<void>& buffer,
                         uint32_t                     len,
                         BufferMRStatus               mrStatus,
                         RdmaMempool::DeviceType      device,
                         RdmaMempool::DeviceType      remoteDevice);
    const Iov* GetIov(uint32_t index) const;

    // make sure all iov in this message status and device is clear
    // if still has iov with unknown status or device after this call, should return false
    bool VerifyIovStatus() const;

    std::shared_ptr<void> GetIovBuffer(uint32_t index) const;
    uint32_t              GetIovLen(uint32_t index) const;

    size_t   GetIovSize() const;
    uint32_t GetTotalIovLength() const;

    std::string GetStatusString() const;
    std::string GetXMessageStatusString(const ::accl::barex::XMessage& xmessage) const;

    void reportClientRPCStats(const std::shared_ptr<ClientRPCStats>& stats, const ::accl::barex::XMessage& xmessage);
    void reportServerRPCStats(const std::shared_ptr<ServerRPCStats>& stats, const ::accl::barex::XMessage& xmessage);

    void SetCudaUtill(CudaUtil* cudaUtil) {
        _cudaUtil.reset(cudaUtil);
    }
    CudaUtil& GetCudaUtil() const {
        return *_cudaUtil;
    }

private:
    void        GetIovStatusString(std::stringstream&      ss,
                                   BufferMRStatus          mrStatus,
                                   RdmaMempool::DeviceType device,
                                   RdmaMempool::DeviceType remoteDevice) const;
    std::string BufferMRStatusName(BufferMRStatus status) const;

private:
    uint32_t                                    _requestId;
    std::shared_ptr<::accl::barex::XConnection> _connection;
    std::shared_ptr<RdmaMempool>                _mempool;

    std::vector<Iov*>            _iovs;
    mutable autil::ReadWriteLock _iovsLock;

    bool _failedMessage;

    std::unique_ptr<CudaUtil> _cudaUtil;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);