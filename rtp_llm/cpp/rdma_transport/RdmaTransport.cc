#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm::rdma_transport {

namespace {

constexpr size_t kMaxEndpointBytes   = 255;
constexpr size_t kMaxLeaseBytes      = 1024;

uint64_t dtypeBytes(TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::FLOAT32:
        case TensorDataType::INT32:
            return 4;
        case TensorDataType::FLOAT16:
        case TensorDataType::BFLOAT16:
            return 2;
    }
    return 0;
}

::TensorDataTypePB toProtoDataType(TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::FLOAT32:
            return ::RDMA_TENSOR_FLOAT32;
        case TensorDataType::INT32:
            return ::RDMA_TENSOR_INT32;
        case TensorDataType::FLOAT16:
            return ::RDMA_TENSOR_FLOAT16;
        case TensorDataType::BFLOAT16:
            return ::RDMA_TENSOR_BFLOAT16;
    }
    return ::RDMA_TENSOR_FLOAT32;
}

bool fromProtoDataType(::TensorDataTypePB src, TensorDataType* dst) {
    switch (src) {
        case ::RDMA_TENSOR_FLOAT32:
            *dst = TensorDataType::FLOAT32;
            return true;
        case ::RDMA_TENSOR_INT32:
            *dst = TensorDataType::INT32;
            return true;
        case ::RDMA_TENSOR_FLOAT16:
            *dst = TensorDataType::FLOAT16;
            return true;
        case ::RDMA_TENSOR_BFLOAT16:
            *dst = TensorDataType::BFLOAT16;
            return true;
        default:
            return false;
    }
}

}  // namespace

std::vector<RdmaDescriptor>
RdmaExport::createBatch(const std::vector<std::vector<torch::Tensor>>& tensor_groups) {
    std::vector<RdmaDescriptor> descriptors;
    descriptors.reserve(tensor_groups.size());
    const auto rollback = [&]() noexcept {
        try {
            std::vector<std::string> lease_ids;
            lease_ids.reserve(descriptors.size());
            for (const auto& descriptor : descriptors) {
                lease_ids.push_back(descriptor.lease_id);
            }
            if (!lease_ids.empty()) {
                release(lease_ids);
            }
        } catch (...) {
            // Preserve the original create() failure; provider GC is the backstop.
        }
    };
    try {
        for (const auto& tensors : tensor_groups) {
            RdmaDescriptor descriptor = create(tensors);
            if (descriptor.lease_id.empty()) {
                rollback();
                return {};
            }
            descriptors.push_back(std::move(descriptor));
        }
    } catch (...) {
        rollback();
        throw;
    }
    return descriptors;
}

void toProto(const RdmaDescriptor& src, ::RdmaDescriptorPB* dst) {
    dst->set_host(src.host);
    dst->set_port(src.port);
    dst->set_remote_addr(src.remote_addr);
    dst->set_payload_bytes(src.payload_bytes);
    dst->set_lease_id(src.lease_id);
    for (const auto& key : src.nic_keys) {
        auto* pb = dst->add_nic_keys();
        pb->set_nic_id(key.nic_id);
        pb->set_rkey(key.rkey);
    }
    for (const auto& tensor : src.tensors) {
        auto* pb = dst->add_tensors();
        for (int64_t dim : tensor.shape) {
            pb->add_shape(dim);
        }
        pb->set_data_type(toProtoDataType(tensor.dtype));
        pb->set_offset(tensor.offset);
        pb->set_nbytes(tensor.nbytes);
    }
}

bool fromProto(const ::RdmaDescriptorPB& src, RdmaDescriptor* dst) {
    if (dst == nullptr) {
        return false;
    }
    RdmaDescriptor descriptor;
    descriptor.host          = src.host();
    descriptor.port          = src.port();
    descriptor.remote_addr   = src.remote_addr();
    descriptor.payload_bytes = src.payload_bytes();
    descriptor.lease_id      = src.lease_id();
    descriptor.nic_keys.reserve(static_cast<size_t>(src.nic_keys_size()));
    for (const auto& key : src.nic_keys()) {
        descriptor.nic_keys.push_back({key.nic_id(), key.rkey()});
    }
    descriptor.tensors.reserve(static_cast<size_t>(src.tensors_size()));
    for (const auto& tensor : src.tensors()) {
        TensorMeta meta;
        meta.shape.assign(tensor.shape().begin(), tensor.shape().end());
        if (!fromProtoDataType(tensor.data_type(), &meta.dtype)) {
            return false;
        }
        meta.offset = tensor.offset();
        meta.nbytes = tensor.nbytes();
        descriptor.tensors.push_back(std::move(meta));
    }
    *dst = std::move(descriptor);
    return true;
}

ErrorInfo validateRdmaDescriptor(const RdmaDescriptor& descriptor, int64_t max_slot_bytes) {
    if (descriptor.host.empty() || descriptor.host.size() > kMaxEndpointBytes
        || descriptor.host.find('\0') != std::string::npos || descriptor.port == 0 || descriptor.port > 65535) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid RDMA descriptor endpoint");
    }
    if (descriptor.lease_id.empty() || descriptor.lease_id.size() > kMaxLeaseBytes) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid RDMA descriptor lease id");
    }
    if (descriptor.remote_addr == 0 || descriptor.payload_bytes == 0
        || descriptor.remote_addr > std::numeric_limits<uint64_t>::max() - descriptor.payload_bytes) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid or overflowing RDMA descriptor address range");
    }
    if (max_slot_bytes > 0 && descriptor.payload_bytes > static_cast<uint64_t>(max_slot_bytes)) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "RDMA descriptor payload exceeds configured slot limit");
    }
    if (descriptor.nic_keys.empty() || descriptor.nic_keys.size() > kMaxRdmaNicKeys) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid RDMA descriptor NIC key count");
    }
    std::unordered_set<uint32_t> nic_ids;
    for (const auto& key : descriptor.nic_keys) {
        if (key.rkey == 0 || !nic_ids.insert(key.nic_id).second) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid or duplicate RDMA descriptor NIC key");
        }
    }
    if (descriptor.tensors.empty() || descriptor.tensors.size() > kMaxRdmaTensorsPerSlot) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid RDMA descriptor tensor count");
    }

    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(descriptor.tensors.size());
    for (const auto& tensor : descriptor.tensors) {
        if (tensor.shape.size() > kMaxRdmaTensorDimensions || tensor.nbytes == 0
            || tensor.offset % kRdmaSlotAlign != 0 || tensor.offset > descriptor.payload_bytes
            || tensor.nbytes > descriptor.payload_bytes - tensor.offset) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid RDMA tensor manifest range");
        }

        uint64_t numel = 1;
        for (int64_t dim : tensor.shape) {
            if (dim <= 0 || numel > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid or overflowing RDMA tensor shape");
            }
            numel *= static_cast<uint64_t>(dim);
        }
        const uint64_t element_bytes = dtypeBytes(tensor.dtype);
        if (element_bytes == 0 || numel > std::numeric_limits<uint64_t>::max() / element_bytes
            || numel * element_bytes != tensor.nbytes) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "RDMA tensor shape and dtype do not match nbytes");
        }
        ranges.emplace_back(tensor.offset, tensor.offset + tensor.nbytes);
    }

    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "overlapping RDMA tensor manifest ranges");
        }
    }
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm::rdma_transport
