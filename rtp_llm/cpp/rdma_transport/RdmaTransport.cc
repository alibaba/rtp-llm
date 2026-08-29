#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"

#include <utility>

#include "rtp_llm/cpp/rdma_transport/proto/tensor_rdma.pb.h"

namespace rtp_llm::rdma_transport {

namespace {

::rdma_transport::TensorDataTypePB toProtoDataType(TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::FLOAT32:
            return ::rdma_transport::RDMA_TENSOR_FLOAT32;
        case TensorDataType::INT32:
            return ::rdma_transport::RDMA_TENSOR_INT32;
        case TensorDataType::FLOAT16:
            return ::rdma_transport::RDMA_TENSOR_FLOAT16;
        case TensorDataType::BFLOAT16:
            return ::rdma_transport::RDMA_TENSOR_BFLOAT16;
    }
    return ::rdma_transport::RDMA_TENSOR_FLOAT32;
}

bool fromProtoDataType(::rdma_transport::TensorDataTypePB src, TensorDataType* dst) {
    switch (src) {
        case ::rdma_transport::RDMA_TENSOR_FLOAT32:
            *dst = TensorDataType::FLOAT32;
            return true;
        case ::rdma_transport::RDMA_TENSOR_INT32:
            *dst = TensorDataType::INT32;
            return true;
        case ::rdma_transport::RDMA_TENSOR_FLOAT16:
            *dst = TensorDataType::FLOAT16;
            return true;
        case ::rdma_transport::RDMA_TENSOR_BFLOAT16:
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
    for (const auto& tensors : tensor_groups) {
        RdmaDescriptor descriptor = create(tensors);
        if (descriptor.lease_id.empty()) {
            std::vector<std::string> lease_ids;
            lease_ids.reserve(descriptors.size());
            for (const auto& created : descriptors) {
                lease_ids.push_back(created.lease_id);
            }
            if (!lease_ids.empty()) {
                release(lease_ids);
            }
            return {};
        }
        descriptors.push_back(std::move(descriptor));
    }
    return descriptors;
}

void toProto(const RdmaDescriptor& src, ::rdma_transport::RdmaDescriptorPB* dst) {
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

bool fromProto(const ::rdma_transport::RdmaDescriptorPB& src, RdmaDescriptor* dst) {
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

}  // namespace rtp_llm::rdma_transport
