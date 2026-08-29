#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/all.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rdma_transport {
class RdmaDescriptorPB;
}

namespace rtp_llm::rdma_transport {

inline constexpr uint64_t kRdmaSlotAlign = 256;

enum class TensorDataType {
    FLOAT32,
    INT32,
    FLOAT16,
    BFLOAT16,
};

struct RdmaNicKey {
    uint32_t nic_id = 0;
    uint32_t rkey   = 0;
};

struct TensorMeta {
    std::vector<int64_t> shape;
    TensorDataType       dtype  = TensorDataType::FLOAT32;
    uint64_t             offset = 0;
    uint64_t             nbytes = 0;
};

struct RdmaDescriptor {
    std::string             host;
    uint32_t                port          = 0;
    uint64_t                remote_addr   = 0;
    uint64_t                payload_bytes = 0;
    std::vector<RdmaNicKey> nic_keys;
    std::string             lease_id;
    std::vector<TensorMeta> tensors;
};

struct RdmaReadResult {
    ErrorInfo                  status;
    std::vector<torch::Tensor> tensors;
    std::vector<std::string>   lease_ids;
};

class RdmaExport {
public:
    virtual ~RdmaExport() = default;
    // Returns an empty descriptor (lease_id is empty) when export fails.
    virtual RdmaDescriptor create(const std::vector<torch::Tensor>& tensors) = 0;
    // Creates one descriptor per tensor group and rolls back every lease if any group fails.
    std::vector<RdmaDescriptor> createBatch(const std::vector<std::vector<torch::Tensor>>& tensor_groups);
    virtual void release(const std::vector<std::string>& lease_ids) = 0;
};

class RdmaRead {
public:
    virtual ~RdmaRead() = default;
    virtual RdmaReadResult read(const std::vector<RdmaDescriptor>& descriptors, int64_t timeout_ms = 0) = 0;
};

// Implemented by the build-selected provider (Barex or no-op).
bool                        hasRdmaImplementation();
std::shared_ptr<RdmaExport> createRdmaExport(const RdmaConfig& config);
std::shared_ptr<RdmaRead>   createRdmaRead(const RdmaConfig& config);

void toProto(const RdmaDescriptor& src, ::rdma_transport::RdmaDescriptorPB* dst);
bool fromProto(const ::rdma_transport::RdmaDescriptorPB& src, RdmaDescriptor* dst);

}  // namespace rtp_llm::rdma_transport
