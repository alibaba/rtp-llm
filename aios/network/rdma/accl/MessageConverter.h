#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

struct SerializeIovDescriptor {
    int64_t iovIndex;
    int64_t offset;
    int64_t len;

    SerializeIovDescriptor(): iovIndex(-1), offset(-1), len(0) {}
};

struct SerializeDescriptor {
    int64_t magic;
    int64_t iovCount;

    SerializeDescriptor(): magic(0), iovCount(0) {}
};

// 支持从RdmaMessage 和 accl::barex::XMessage之间的相互转换, 可以对小的iov进行聚合
class MessageConverter {
public:
    MessageConverter();

public:
    bool ConvertFromRdmaMessage(RdmaMessage* message, ::accl::barex::XMessage& xmessage) const;
    bool ConvertFromXMessage(RdmaMessage* message, const ::accl::barex::XMessage& xmessage) const;

    void SetSmallIovThreshold(uint32_t threshold) {
        _smallIovThreshold = threshold;
    }

private:
    int64_t CPUMemAlign(int64_t len) const;
    int64_t GPUMemAlign(int64_t len) const;
    bool    GenerateDescriptors(RdmaMessage*,
                                SerializeDescriptor&                 serDesc,
                                std::vector<SerializeIovDescriptor>& serIovDescs,
                                uint32_t&                            packedCPUSize,
                                uint32_t&                            packedGPUSize) const;
    bool    PackMessages(RdmaMessage*,
                         SerializeDescriptor&                 serDesc,
                         std::vector<SerializeIovDescriptor>& serIovDescs,
                         uint32_t&                            packedCPUSize,
                         uint32_t&                            packedGPUSize,
                         ::accl::barex::XMessage&             xmessage) const;
    bool    ValidMessage(const ::accl::barex::XMessage& xmessage,
                         SerializeDescriptor*&          serDesc,
                         SerializeIovDescriptor*&       serIovDescs) const;
    bool    UnpackMessage(const ::accl::barex::XMessage& xmessage,
                          RdmaMessage*,
                          SerializeDescriptor*&                 serDesc,
                          SerializeIovDescriptor*&              serIovDescs,
                          std::map<int, std::shared_ptr<void>>& indexMap) const;

    bool GenerateCPUMemoryIovDesc(RdmaMessage&            message,
                                  size_t                  i,
                                  SerializeIovDescriptor& desc,
                                  uint32_t&               packedCPUSize,
                                  uint32_t&               packedGPUSize) const;
    bool GenerateGPUMemoryIovDesc(RdmaMessage& message, size_t i, SerializeIovDescriptor& desc) const;

private:
    uint32_t _smallIovThreshold;
    uint32_t _cpuMemAlign;
    uint32_t _gpuMemAlign;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);