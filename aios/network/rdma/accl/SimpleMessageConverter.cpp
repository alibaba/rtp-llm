#include "aios/network/rdma/accl/SimpleMessageConverter.h"

#include <sstream>

ARPC_BEGIN_NAMESPACE(arpc);

AUTIL_LOG_SETUP(arpc, SimpleMessageConverter);

bool SimpleMessageConverter::ConvertFromRdmaMessage(RdmaMessage* message, ::accl::barex::XMessage& xmessage) {
    if (!message->VerifyIovStatus()) {
        AUTIL_LOG(WARN, "message verify iov status failed, message %s", message->GetStatusString().c_str());
        return false;
    }

    auto iovCount = message->GetIovSize();
    for (int i = 0; i < iovCount; i++) {
        auto iov = message->GetIov(i);
        if (iov == nullptr) {
            continue;
        }

        if (iov->mrStatus == RdmaMessage::BMS_MR) {  // 已经是mr过的, 可以直接发送
            xmessage.Add(iov->buf.get(),
                         iov->len,
                         RdmaMempool::ConvertToAcclDeviceType(iov->device),
                         RdmaMempool::ConvertToAcclDeviceType(iov->remoteDevice));
            continue;
        }

        if (iov->device == RdmaMempool::CPU) {
            if (ConvertUnMrCPUMemory(*message, xmessage, i)) {
                continue;
            }
            return false;
        } else if (iov->device == RdmaMempool::GPU) {
            if (ConvertUnMrGPUMemory(*message, xmessage, i)) {
                continue;
            }
            return false;
        }
        AUTIL_LOG(WARN, "simple message converter failed, unknown device");
        return false;
    }
    return true;
}

bool SimpleMessageConverter::ConvertUnMrCPUMemory(RdmaMessage&             message,
                                                  ::accl::barex::XMessage& xmessage,
                                                  size_t                   index) {
    auto iov = message.GetIov(index);

    auto buf = message.AllocateSharedSendBuffer(iov->len, RdmaMempool::CPU);
    if (!buf) {
        AUTIL_LOG(WARN, "allocate mr cpu mem failed, len %d", iov->len);
        return false;
    }
    memcpy(buf.get(), iov->buf.get(), iov->len);
    message.AddIov(buf, iov->len, RdmaMessage::BMS_MR, RdmaMempool::CPU, iov->remoteDevice);
    xmessage.Add(buf.get(),
                 iov->len,
                 RdmaMempool::ConvertToAcclDeviceType(iov->device),
                 RdmaMempool::ConvertToAcclDeviceType(iov->remoteDevice));
    return true;
}

bool SimpleMessageConverter::ConvertUnMrGPUMemory(RdmaMessage&             message,
                                                  ::accl::barex::XMessage& xmessage,
                                                  size_t                   index) {
    auto iov = message.GetIov(index);

    auto buf = message.AllocateSharedSendBuffer(iov->len, RdmaMempool::GPU);
    if (!buf) {
        AUTIL_LOG(WARN, "allocate mr gpu mem failed, len %d", iov->len);
        return false;
    }

    if (!message.GetCudaUtil().CudaMemcpyGPU2GPU(buf.get(), iov->buf.get(), iov->len)) {
        AUTIL_LOG(WARN, "cuda memcpy mr gpu memory failed, len %d", iov->len);
        return false;
    }

    message.AddIov(buf, iov->len, RdmaMessage::BMS_MR, RdmaMempool::GPU, iov->remoteDevice);
    xmessage.Add(buf.get(),
                 iov->len,
                 RdmaMempool::ConvertToAcclDeviceType(iov->device),
                 RdmaMempool::ConvertToAcclDeviceType(iov->remoteDevice));
    return true;
}

bool SimpleMessageConverter::ConvertFromXMessage(RdmaMessage* message, const ::accl::barex::XMessage& xmessage) {
    auto mempool = message->GetMempool();
    for (int i = 0; i < xmessage.IovCount(); i++) {
        auto                  iov    = xmessage.Get(i);
        auto                  device = RdmaMempool::ConvertFromAcclDeviceType(iov.device);
        std::shared_ptr<void> sharedBuf(iov.iov_base, [mempool, device](void* p) {
            if (mempool) {
                mempool->ReleaseBuffer(p, device);
            }
        });
        message->AddIov(sharedBuf,
                        iov.iov_len,
                        RdmaMessage::BMS_MR,
                        device,
                        RdmaMempool::ConvertFromAcclDeviceType(iov.remote_device));
    }
    return true;
}

ARPC_END_NAMESPACE(arpc);