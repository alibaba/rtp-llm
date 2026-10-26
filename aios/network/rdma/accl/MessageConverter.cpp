#include "aios/network/rdma/accl/MessageConverter.h"

#include "aios/network/rdma/accl/RdmaMempool.h"

ARPC_BEGIN_NAMESPACE(arpc);

const int64_t kSerializeMagic   = 0x101010;
const int64_t kPackedCPUField   = -1;
const int64_t kPackedGPUField   = -2;
const int64_t kUnPackedCPUField = -3;
const int64_t kUnPackedGPUField = -4;

AUTIL_LOG_SETUP(arpc, MessageConverter);

MessageConverter::MessageConverter(): _smallIovThreshold(10240), _cpuMemAlign(64), _gpuMemAlign(64) {}

int64_t MessageConverter::CPUMemAlign(int64_t len) const {
    if (len % _cpuMemAlign == 0) {
        return len;
    }
    return len / _cpuMemAlign * _cpuMemAlign + _cpuMemAlign;
}
int64_t MessageConverter::GPUMemAlign(int64_t len) const {
    if (len % _gpuMemAlign == 0) {
        return len;
    }
    return len / _gpuMemAlign * _gpuMemAlign + _gpuMemAlign;
}

bool MessageConverter::GenerateDescriptors(RdmaMessage*                         message,
                                           SerializeDescriptor&                 serDesc,
                                           std::vector<SerializeIovDescriptor>& serIovDescs,
                                           uint32_t&                            packedCPUSize,
                                           uint32_t&                            packedGPUSize) const {
    serDesc.magic    = kSerializeMagic;
    serDesc.iovCount = message->GetIovSize();

    for (int i = 0; i < message->GetIovSize(); i++) {
        auto iov = message->GetIov(i);
        if (iov == nullptr) {
            continue;
        }

        SerializeIovDescriptor desc;
        if (iov->device == RdmaMempool::GPU) {
            if (!GenerateGPUMemoryIovDesc(*message, i, desc)) {
                return false;
            }
        } else if (iov->device == RdmaMempool::CPU) {
            if (!GenerateCPUMemoryIovDesc(*message, i, desc, packedCPUSize, packedGPUSize)) {
                return false;
            }
        } else {
            AUTIL_LOG(WARN,
                      "message converter generate desc failed, unknown device, message %s",
                      message->GetStatusString().c_str());
            return false;
        }
        serIovDescs.push_back(desc);
    }
    return true;
}

bool MessageConverter::GenerateCPUMemoryIovDesc(RdmaMessage&            message,
                                                size_t                  i,
                                                SerializeIovDescriptor& desc,
                                                uint32_t&               packedCPUSize,
                                                uint32_t&               packedGPUSize) const {
    auto iov = message.GetIov(i);

    if (iov->mrStatus == RdmaMessage::BMS_MR && iov->len > _smallIovThreshold) {  // big mr buffer should direct send
        desc.iovIndex = iov->remoteDevice == RdmaMempool::GPU ? kUnPackedGPUField : kUnPackedCPUField;
        desc.offset   = 0;
        desc.len      = iov->len;
        return true;
    }

    if (iov->len > 1024 * 1024) {
        AUTIL_INTERVAL_LOG(10000,
                           WARN,
                           "large iov not mr may affect performance, index: %lu len: %u, message: %s",
                           i,
                           iov->len,
                           message.GetStatusString().c_str());
    }

    // small iov or unmr iov should be copy to packed buffer
    if (iov->remoteDevice == RdmaMempool::CPU) {
        desc.iovIndex = kPackedCPUField;
        desc.len      = iov->len;
        auto alignLen = CPUMemAlign(iov->len);
        packedCPUSize += alignLen;
    } else if (iov->remoteDevice == RdmaMempool::GPU) {
        desc.iovIndex = kPackedGPUField;
        desc.len      = iov->len;
        auto alignLen = GPUMemAlign(iov->len);
        packedGPUSize += alignLen;
    }
    return true;
}

bool MessageConverter::GenerateGPUMemoryIovDesc(RdmaMessage& message, size_t i, SerializeIovDescriptor& desc) const {
    auto iov = message.GetIov(i);

    if (iov->mrStatus == RdmaMessage::BMS_MR) {  // mr gpu memory should direct send
        desc.iovIndex = iov->remoteDevice == RdmaMempool::GPU ? kUnPackedGPUField : kUnPackedCPUField;
        desc.offset   = 0;
        desc.len      = iov->len;
        return true;
    }

    AUTIL_INTERVAL_LOG(10000,
                       WARN,
                       "gpu iov not mr may affect performance, index: %lu len: %u, message: %s",
                       i,
                       iov->len,
                       message.GetStatusString().c_str());

    auto buf = message.AllocateSharedSendBuffer(iov->len, RdmaMempool::GPU);
    if (buf == nullptr) {
        AUTIL_LOG(WARN, "alloc gpu send buffer failed, len: %d", iov->len);
        return false;
    }

    if (!message.GetCudaUtil().CudaMemcpyGPU2GPU(buf.get(), iov->buf.get(), iov->len)) {
        AUTIL_LOG(
            WARN, "cuda memcpy to mr gpu memory failed, dst: %p, src: %p len: %d", buf.get(), iov->buf.get(), iov->len);
        return false;
    }

    // replace unmr gpu memory with mr gpu memory to send
    message.UpdateIov(i, buf, iov->len, RdmaMessage::BMS_MR, iov->device, iov->remoteDevice);
    desc.iovIndex = iov->remoteDevice == RdmaMempool::GPU ? kUnPackedGPUField : kUnPackedCPUField;
    desc.offset   = 0;
    desc.len      = iov->len;
    return true;
}

bool MessageConverter::PackMessages(RdmaMessage*                         message,
                                    SerializeDescriptor&                 serDesc,
                                    std::vector<SerializeIovDescriptor>& serIovDescs,
                                    uint32_t&                            packedCPUSize,
                                    uint32_t&                            packedGPUSize,
                                    ::accl::barex::XMessage&             xmessage) const {
    auto descAlignLen = CPUMemAlign(sizeof(SerializeDescriptor) + sizeof(SerializeIovDescriptor) * serIovDescs.size());
    auto cpuBufferLen = descAlignLen + packedCPUSize;
    auto cpuBuffer    = message->AllocateSharedSendBuffer(cpuBufferLen);
    if (!cpuBuffer) {
        AUTIL_LOG(WARN, "allocate cpu buffer failed, len %ld", cpuBufferLen);
        return false;
    }
    auto gpuBuffer = message->AllocateSharedSendBuffer(packedGPUSize, RdmaMempool::CPU);
    if (packedGPUSize > 0 && !gpuBuffer) {
        AUTIL_LOG(WARN, "allocate gpu buffer failed, len %d", packedGPUSize);
        return false;
    }
    xmessage.Add(cpuBuffer.get(), cpuBufferLen, ::accl::barex::CPU, ::accl::barex::CPU);
    message->AddIov(cpuBuffer, cpuBufferLen, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    if (gpuBuffer) {
        xmessage.Add(gpuBuffer.get(), packedGPUSize, ::accl::barex::CPU, ::accl::barex::GPU);
        message->AddIov(gpuBuffer, packedGPUSize, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::GPU);
    }

    memcpy(cpuBuffer.get(), (void*)(&serDesc), sizeof(struct SerializeDescriptor));
    int64_t cpuOffset = descAlignLen;
    int64_t gpuOffset = 0;
    for (int i = 0; i < serIovDescs.size(); i++) {
        auto& iovDesc = serIovDescs[i];

        auto iov = message->GetIov(i);
        if (iov == nullptr) {
            continue;
        }
        auto& buffer = iov->buf;
        auto& len    = iov->len;

        if (iovDesc.iovIndex == kPackedCPUField) {
            iovDesc.iovIndex = 0;
            iovDesc.len      = len;
            iovDesc.offset   = cpuOffset;
            memcpy((void*)((char*)cpuBuffer.get() + cpuOffset), buffer.get(), len);

            cpuOffset += CPUMemAlign(len);
        } else if (iovDesc.iovIndex == kPackedGPUField) {
            if (!gpuBuffer) {
                return false;
            }
            iovDesc.iovIndex = 1;
            iovDesc.len      = len;
            iovDesc.offset   = gpuOffset;
            memcpy((void*)((char*)gpuBuffer.get() + gpuOffset), buffer.get(), len);

            gpuOffset += CPUMemAlign(len);
        } else {
            auto device = RdmaMempool::ConvertToAcclDeviceType(iov->device);
            xmessage.Add(buffer.get(),
                         len,
                         device,
                         iovDesc.iovIndex == kUnPackedCPUField ? ::accl::barex::CPU : ::accl::barex::GPU);
            iovDesc.iovIndex = xmessage.IovCount() - 1;
            iovDesc.len      = len;
            iovDesc.offset   = 0;
        }
        void* iovDescBuf =
            (void*)((char*)cpuBuffer.get() + sizeof(struct SerializeDescriptor) + i * sizeof(SerializeIovDescriptor));
        memcpy(iovDescBuf, (void*)(&iovDesc), sizeof(struct SerializeIovDescriptor));
    }
    return true;
}

bool MessageConverter::ConvertFromRdmaMessage(RdmaMessage* message, ::accl::barex::XMessage& xmessage) const {
    SerializeDescriptor                 serDesc;
    std::vector<SerializeIovDescriptor> serIovDescs;
    uint32_t                            packedCPUSize = 0;
    uint32_t                            packedGPUSize = 0;

    if (!message->VerifyIovStatus()) {
        AUTIL_LOG(WARN, "message verify iov status failed, message %s", message->GetStatusString().c_str());
        return false;
    }

    if (!GenerateDescriptors(message, serDesc, serIovDescs, packedCPUSize, packedGPUSize)) {
        return false;
    }

    if (!PackMessages(message, serDesc, serIovDescs, packedCPUSize, packedGPUSize, xmessage)) {
        return false;
    }
    return true;
}

bool MessageConverter::ConvertFromXMessage(RdmaMessage* message, const ::accl::barex::XMessage& xmessage) const {
    auto mempool = message->GetMempool();
    if (!mempool) {
        return false;
    }

    if (xmessage.IovCount() == 0) {
        return true;
    }

    std::map<int, std::shared_ptr<void>> indexMap;
    for (int i = 0; i < xmessage.IovCount(); i++) {
        auto                  iov    = xmessage.Get(i);
        auto                  device = RdmaMempool::ConvertFromAcclDeviceType(iov.device);
        void*                 base   = iov.iov_base;
        std::shared_ptr<void> sharedBuf(base, [mempool, device, base](void* ptr) {
            if (mempool) {
                mempool->ReleaseBuffer(base, device);
            }
        });
        indexMap[i] = sharedBuf;
    }

    SerializeDescriptor*    serDesc     = nullptr;
    SerializeIovDescriptor* serIovDescs = nullptr;
    if (!ValidMessage(xmessage, serDesc, serIovDescs)) {
        return false;
    }

    if (!UnpackMessage(xmessage, message, serDesc, serIovDescs, indexMap)) {
        return false;
    }
    return true;
}

bool MessageConverter::ValidMessage(const ::accl::barex::XMessage& xmessage,
                                    SerializeDescriptor*&          serDesc,
                                    SerializeIovDescriptor*&       serIovDescs) const {
    if (xmessage.IovCount() < 1) {
        AUTIL_LOG(WARN, "empty xmessage to convert");
        return false;
    }
    auto iov0 = xmessage.Get(0);
    if (iov0.iov_len < (sizeof(struct SerializeDescriptor))) {
        AUTIL_LOG(WARN, "xmessage len < serDesc, iov is %p", iov0.iov_base);
        return false;
    }
    serDesc = reinterpret_cast<struct SerializeDescriptor*>(iov0.iov_base);
    if (serDesc->magic != kSerializeMagic) {
        AUTIL_LOG(WARN, "xmessage desc magic error, error magic is %04lx, iov is %p", serDesc->magic, iov0.iov_base);
        return false;
    }
    if (iov0.iov_len < (sizeof(struct SerializeDescriptor) + serDesc->iovCount * sizeof(struct SerializeDescriptor))) {
        AUTIL_LOG(WARN, "xmessage iov len error, iov is %p", iov0.iov_base);
        return false;
    }
    serIovDescs =
        reinterpret_cast<struct SerializeIovDescriptor*>((char*)(iov0.iov_base) + sizeof(struct SerializeDescriptor));
    for (int i = 0; i < serDesc->iovCount; i++) {
        auto& iovDesc = serIovDescs[i];
        if (iovDesc.iovIndex >= xmessage.IovCount()) {
            AUTIL_LOG(WARN,
                      "xmessage iov index error, totol iov count %d, desc index %d, desc iov index %ld",
                      xmessage.IovCount(),
                      i,
                      iovDesc.iovIndex);
            return false;
        }
        auto iov = xmessage.Get(iovDesc.iovIndex);
        if (iov.iov_len < (iovDesc.len + iovDesc.offset)) {
            AUTIL_LOG(WARN,
                      "xmessage iov offset error, iov len %zu, iovdesc len %ld, iov desc offset %ld, iov is %p",
                      iov.iov_len,
                      iovDesc.len,
                      iovDesc.offset,
                      iov.iov_base);
            return false;
        }
    }
    return true;
}

bool MessageConverter::UnpackMessage(const ::accl::barex::XMessage&        xmessage,
                                     RdmaMessage*                          message,
                                     SerializeDescriptor*&                 serDesc,
                                     SerializeIovDescriptor*&              serIovDescs,
                                     std::map<int, std::shared_ptr<void>>& indexMap) const {
    for (int i = 0; i < serDesc->iovCount; i++) {
        auto&                 serDesc      = serIovDescs[i];
        auto                  iov          = xmessage.Get(serDesc.iovIndex);
        auto                  device       = RdmaMempool::ConvertFromAcclDeviceType(iov.device);
        auto                  remoteDevice = RdmaMempool::ConvertFromAcclDeviceType(iov.remote_device);
        void*                 base         = iov.iov_base;
        void*                 buf          = (void*)((char*)(base) + serDesc.offset);
        auto                  sharedBuf    = indexMap[serDesc.iovIndex];
        std::shared_ptr<void> sharedBuf2(sharedBuf, buf);
        message->AddIov(sharedBuf2, serDesc.len, RdmaMessage::BMS_MR, device, remoteDevice);
    }
    return true;
}

ARPC_END_NAMESPACE(arpc);