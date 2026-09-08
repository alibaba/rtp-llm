#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/all.h>

#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/config/RdmaConfig.h"

class RdmaDescriptorPB;

namespace rtp_llm::rdma_transport {

inline constexpr uint64_t kRdmaSlotAlign            = 256;
inline constexpr size_t   kMaxRdmaNicKeys           = 64;
inline constexpr size_t   kMaxRdmaTensorsPerSlot    = 4096;
inline constexpr size_t   kMaxRdmaTensorDimensions  = 16;

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
};

class RdmaExport {
public:
    virtual ~RdmaExport() = default;
    // Returns an empty descriptor (lease_id is empty) for an expected export failure.
    // A valid lease_id must be globally unique across exporter processes.
    //
    // Synchronization contract: before returning a valid descriptor, the provider must
    // complete all device work needed to populate the exported memory and make that
    // memory visible to the RDMA NIC. Callers may pass tensors with pending work on the
    // current CUDA stream; publishing a descriptor must not race those writes.
    virtual RdmaDescriptor create(const std::vector<torch::Tensor>& tensors) = 0;
    // Creates one descriptor per tensor group, preserving input order. If a group fails or
    // create() throws, every lease created earlier in the batch is released before returning
    // an empty result or propagating the exception.
    std::vector<RdmaDescriptor> createBatch(const std::vector<std::vector<torch::Tensor>>& tensor_groups);
    // Production adapters serialize provider calls. Release must remain idempotent: duplicate,
    // unknown, or already expired leases are ignored.
    virtual void release(const std::vector<std::string>& lease_ids) = 0;
};

class RdmaRead {
public:
    virtual ~RdmaRead() = default;
    // Flattens tensors in descriptor order, then manifest order. Successful results own
    // independent tensor storage; failures return a non-OK status and no partial tensors.
    virtual RdmaReadResult read(const std::vector<RdmaDescriptor>& descriptors, int64_t timeout_ms = 0) = 0;
};

// Implemented by the build-selected provider (Barex or no-op).
bool                        hasRdmaImplementation();
// device_id < 0 is retained for legacy callers; service paths pass an explicit local CUDA device.
std::shared_ptr<RdmaExport> createRdmaExport(const RdmaConfig& config, int device_id = -1);
std::shared_ptr<RdmaRead>   createRdmaRead(const RdmaConfig& config, int device_id = -1);

void toProto(const RdmaDescriptor& src, ::RdmaDescriptorPB* dst);
bool fromProto(const ::RdmaDescriptorPB& src, RdmaDescriptor* dst);
ErrorInfo validateRdmaDescriptor(const RdmaDescriptor& descriptor, int64_t max_slot_bytes);

}  // namespace rtp_llm::rdma_transport
