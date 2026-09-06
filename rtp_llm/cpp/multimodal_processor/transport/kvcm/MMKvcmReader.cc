#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmReader.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

bool dtypeInfo(TensorDataTypePB dtype, torch::ScalarType* scalar_type, uint64_t* element_bytes) {
    switch (dtype) {
        case RDMA_TENSOR_FLOAT32:
            *scalar_type   = torch::kFloat32;
            *element_bytes = 4;
            return true;
        case RDMA_TENSOR_INT32:
            *scalar_type   = torch::kInt32;
            *element_bytes = 4;
            return true;
        case RDMA_TENSOR_FLOAT16:
            *scalar_type   = torch::kFloat16;
            *element_bytes = 2;
            return true;
        case RDMA_TENSOR_BFLOAT16:
            *scalar_type   = torch::kBFloat16;
            *element_bytes = 2;
            return true;
        default:
            return false;
    }
}

torch::Tensor concatenate(const std::vector<torch::Tensor>& chunks) {
    if (chunks.empty()) {
        return {};
    }
    return chunks.size() == 1 ? chunks.front() : torch::cat(chunks, 0);
}

class ObjectLease {
public:
    ObjectLease(DeliveryContext& context, std::vector<std::string> handles):
        context_(context), handles_(std::move(handles)) {}
    ~ObjectLease() noexcept {
        if (!released_ && !handles_.empty()) {
            try {
                context_.control.release(context_.endpoint, handles_, context_.budget);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("synchronous KVCM object release threw: %s; object GC will reclaim it", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("synchronous KVCM object release threw an unknown exception; "
                                    "object GC will reclaim it");
            }
        }
    }
    void releaseAsync() noexcept {
        if (handles_.empty()) {
            released_ = true;
            return;
        }
        try {
            // Keep the original handles until the control client accepts its
            // owned copy. If it throws, the destructor can still attempt the
            // bounded synchronous release instead of silently leaking them.
            auto pending = handles_;
            context_.control.releaseAsync(context_.endpoint, std::move(pending));
            handles_.clear();
            released_ = true;
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("asynchronous KVCM object release threw: %s; falling back to synchronous release",
                                e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("asynchronous KVCM object release threw an unknown exception; "
                                "falling back to synchronous release");
        }
    }
    ObjectLease(const ObjectLease&)            = delete;
    ObjectLease& operator=(const ObjectLease&) = delete;

private:
    DeliveryContext&         context_;
    std::vector<std::string> handles_;
    bool                     released_ = false;
};

}  // namespace

bool assembleMMKvcmOutput(const std::vector<torch::Tensor>& tensors,
                          const MultimodalOutputPB&         receipt,
                          MultimodalOutput*                 output) {
    if (output == nullptr || tensors.size() != static_cast<size_t>(receipt.output_kvcm_objects_size())) {
        return false;
    }
    try {
        std::vector<torch::Tensor>                     embedding_chunks;
        std::vector<torch::Tensor>                     position_chunks;
        std::map<uint32_t, std::vector<torch::Tensor>> extra_chunks;
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto& object = receipt.output_kvcm_objects(static_cast<int>(i));
            switch (object.role()) {
                case MMRdmaSlotPB::EMBEDDING:
                    embedding_chunks.push_back(tensors[i]);
                    break;
                case MMRdmaSlotPB::POS_ID:
                    position_chunks.push_back(tensors[i]);
                    break;
                case MMRdmaSlotPB::EXTRA_INPUT:
                    extra_chunks[object.logical_index()].push_back(tensors[i]);
                    break;
                case MMRdmaSlotPB::ROLE_UNSPECIFIED:
                default:
                    return false;
            }
        }
        auto embedding = concatenate(embedding_chunks);
        if (!embedding.defined()) {
            return false;
        }
        std::vector<int64_t> split_sizes(receipt.split_size().begin(), receipt.split_size().end());
        int64_t              split_total = 0;
        for (const int64_t split : split_sizes) {
            if (split <= 0 || split_total > std::numeric_limits<int64_t>::max() - split) {
                return false;
            }
            split_total += split;
        }
        if (split_sizes.empty() || embedding.dim() == 0 || embedding.size(0) != split_total) {
            return false;
        }

        MultimodalOutput assembled;
        assembled.mm_features = embedding.split(split_sizes, 0);
        if (!position_chunks.empty()) {
            auto positions = concatenate(position_chunks);
            if (!positions.defined() || positions.dim() == 0 || positions.size(0) != split_total) {
                return false;
            }
            assembled.mm_position_ids = positions.to(torch::kCPU).split(split_sizes, 0);
        }
        if (!extra_chunks.empty()) {
            if (extra_chunks.size() != split_sizes.size()) {
                return false;
            }
            std::vector<torch::Tensor> extras;
            extras.reserve(extra_chunks.size());
            for (uint32_t index = 0; index < extra_chunks.size(); ++index) {
                const auto it = extra_chunks.find(index);
                if (it == extra_chunks.end()) {
                    return false;
                }
                auto extra = concatenate(it->second);
                if (!extra.defined()) {
                    return false;
                }
                extras.push_back(std::move(extra));
            }
            assembled.mm_extra_input = std::move(extras);
        }
        *output = std::move(assembled);
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("KVCM output materialization failed: %s", e.what());
        return false;
    }
}

bool MMKvcmReader::advertise(const std::string&, MultimodalInputsPB& request_pb) {
    if (client_ == nullptr) {
        return false;
    }
    request_pb.set_support_kvcm(true);
    return true;
}

bool MMKvcmReader::matches(const MultimodalOutputPB& receipt) const {
    return receipt.output_kvcm_objects_size() > 0;
}

ConsumeResult MMKvcmReader::consume(const MultimodalOutputPB& receipt, DeliveryContext& context) {
    ObjectLease lease(context, handlesOf(receipt));
    if (client_ == nullptr) {
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "KVCM receipt reached an adapter with no object client"));
    }
    if (context.budget.exhausted()) {
        lease.releaseAsync();
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "KVCM object load started after the request deadline"));
    }

    std::vector<torch::Tensor> tensors;
    std::vector<MMKvcmBuffer>  objects;
    std::string                validation_error;
    if (!validateAndAllocate(receipt, &tensors, &objects, &validation_error)) {
        RTP_LLM_LOG_WARNING("invalid KVCM multimodal receipt: %s", validation_error.c_str());
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "invalid multimodal KVCM output manifest: " + validation_error));
    }

    std::string load_error;
    try {
        load_error = client_->load("rtp-mm-kvcm-load", objects, context.budget.remainingMs());
    } catch (const std::exception& e) {
        load_error = std::string("object client threw: ") + e.what();
    } catch (...) {
        load_error = "object client threw an unknown exception";
    }
    if (!load_error.empty()) {
        RTP_LLM_LOG_WARNING("KVCM multimodal object load failed: %s", load_error.c_str());
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "failed to load multimodal output from KVCM: " + load_error));
    }

    MultimodalOutput output;
    if (!assembleMMKvcmOutput(tensors, receipt, &output)) {
        return ConsumeResult::failure(
            ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "failed to assemble multimodal KVCM output"));
    }
    RTP_LLM_LOG_INFO("[MM-KVCM-HIT] multimodal embedding loaded from %d exact-size object(s)",
                     receipt.output_kvcm_objects_size());
    lease.releaseAsync();
    return ConsumeResult::success(std::move(output));
}

bool MMKvcmReader::validateAndAllocate(const MultimodalOutputPB&   receipt,
                                       std::vector<torch::Tensor>* tensors,
                                       std::vector<MMKvcmBuffer>*  objects,
                                       std::string*                error) const {
    if (tensors == nullptr || objects == nullptr || error == nullptr) {
        return false;
    }
    const size_t object_count = static_cast<size_t>(receipt.output_kvcm_objects_size());
    if (object_count == 0 || object_count > kMMKvcmMaxObjectsPerReceipt) {
        *error = "invalid object count";
        return false;
    }
    if (receipt.output_rdma_slots_size() != 0 || receipt.has_multimodal_embedding() || receipt.has_multimodal_pos_id()
        || receipt.multimodal_extra_input_size() != 0) {
        *error = "KVCM receipt mixes multiple data planes";
        return false;
    }
    if (receipt.split_size_size() == 0 || static_cast<size_t>(receipt.split_size_size()) > kMMKvcmMaxLogicalValues) {
        *error = "split_size count is outside the receipt limit";
        return false;
    }
    if (validate_manifest_ && (config_.max_object_bytes <= 0 || config_.max_receipt_bytes < config_.max_object_bytes)) {
        *error = "reader byte limits are invalid";
        return false;
    }
    uint64_t split_total = 0;
    for (const int32_t split : receipt.split_size()) {
        if (split <= 0 || split_total > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - split)) {
            *error = "split_size is invalid or overflows";
            return false;
        }
        split_total += static_cast<uint64_t>(split);
    }

    const uint64_t max_object =
        validate_manifest_ ? static_cast<uint64_t>(config_.max_object_bytes) : std::numeric_limits<uint64_t>::max();
    const uint64_t max_receipt =
        validate_manifest_ ? static_cast<uint64_t>(config_.max_receipt_bytes) : std::numeric_limits<uint64_t>::max();
    std::unordered_set<std::string> keys;
    keys.reserve(object_count);
    uint64_t                                                 total_bytes    = 0;
    uint64_t                                                 embedding_rows = 0;
    uint64_t                                                 position_rows  = 0;
    bool                                                     has_position   = false;
    int                                                      last_role      = static_cast<int>(MMRdmaSlotPB::EMBEDDING);
    uint32_t                                                 last_extra     = 0;
    bool                                                     has_extra      = false;
    std::map<std::pair<int, uint32_t>, std::vector<int64_t>> group_shapes;
    std::map<std::pair<int, uint32_t>, TensorDataTypePB>     group_dtypes;

    struct Allocation {
        std::string          key;
        std::vector<int64_t> shape;
        torch::ScalarType    dtype;
        uint64_t             nbytes;
    };
    std::vector<Allocation> allocations;
    allocations.reserve(object_count);

    for (const auto& object : receipt.output_kvcm_objects()) {
        if (object.key().empty() || object.key().size() > kMMKvcmMaxKeyBytes || !keys.insert(object.key()).second) {
            *error = "object keys are empty, oversized, or duplicated";
            return false;
        }
        const auto role = object.role();
        if (role != MMRdmaSlotPB::EMBEDDING && role != MMRdmaSlotPB::POS_ID && role != MMRdmaSlotPB::EXTRA_INPUT) {
            *error = "object role is invalid";
            return false;
        }
        const int role_value = static_cast<int>(role);
        if (role_value < last_role || (role != MMRdmaSlotPB::EXTRA_INPUT && object.logical_index() != 0)) {
            *error = "object role order or logical index is invalid";
            return false;
        }
        last_role = role_value;
        if (role == MMRdmaSlotPB::EXTRA_INPUT) {
            if (object.logical_index() >= static_cast<uint32_t>(receipt.split_size_size())) {
                *error = "extra input logical index exceeds split_size";
                return false;
            }
            if (has_extra && object.logical_index() < last_extra) {
                *error = "extra input logical indices are out of order";
                return false;
            }
            has_extra  = true;
            last_extra = object.logical_index();
        }

        const auto& meta = object.tensor();
        if (meta.offset() != 0 || meta.shape_size() == 0
            || meta.shape_size() > static_cast<int>(kMMKvcmMaxTensorDimensions)) {
            *error = "tensor offset or dimension count is invalid";
            return false;
        }
        if (role == MMRdmaSlotPB::EXTRA_INPUT && meta.shape_size() != 1) {
            *error = "extra input tensor must be flat";
            return false;
        }
        torch::ScalarType scalar_type;
        uint64_t          element_bytes = 0;
        if (!dtypeInfo(meta.data_type(), &scalar_type, &element_bytes)) {
            *error = "tensor dtype is unsupported";
            return false;
        }
        std::vector<int64_t> shape(meta.shape().begin(), meta.shape().end());
        uint64_t             elements = 1;
        for (const int64_t dimension : shape) {
            if (dimension <= 0 || elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension)) {
                *error = "tensor shape is invalid or overflows";
                return false;
            }
            elements *= static_cast<uint64_t>(dimension);
        }
        if (elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
            *error = "tensor byte size overflows";
            return false;
        }
        const uint64_t computed_bytes = elements * element_bytes;
        if (computed_bytes == 0 || computed_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
            || computed_bytes != meta.nbytes() || computed_bytes != object.value_size() || computed_bytes > max_object
            || computed_bytes > max_receipt - total_bytes) {
            *error = "tensor size does not match the object or configured limits";
            return false;
        }
        total_bytes += computed_bytes;

        const auto group    = std::make_pair(role_value, object.logical_index());
        const auto shape_it = group_shapes.find(group);
        if (shape_it == group_shapes.end()) {
            group_shapes.emplace(group, shape);
            group_dtypes.emplace(group, meta.data_type());
        } else {
            if (shape.empty() || shape_it->second.size() != shape.size()
                || group_dtypes.at(group) != meta.data_type()) {
                *error = "tensor chunks in one logical value are incompatible";
                return false;
            }
            for (size_t dim = 1; dim < shape.size(); ++dim) {
                if (shape[dim] != shape_it->second[dim]) {
                    *error = "tensor chunk shapes are incompatible";
                    return false;
                }
            }
        }
        if (role == MMRdmaSlotPB::EMBEDDING) {
            if (embedding_rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - shape[0])) {
                *error = "embedding row count overflows";
                return false;
            }
            embedding_rows += static_cast<uint64_t>(shape[0]);
        } else if (role == MMRdmaSlotPB::POS_ID) {
            has_position = true;
            if (position_rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - shape[0])) {
                *error = "position row count overflows";
                return false;
            }
            position_rows += static_cast<uint64_t>(shape[0]);
        }
        allocations.push_back({object.key(), std::move(shape), scalar_type, computed_bytes});
    }
    if (embedding_rows != split_total || (has_position && position_rows != split_total)) {
        *error = "embedding or position rows do not match split_size";
        return false;
    }
    if (has_extra) {
        if (last_extra + 1 != static_cast<uint32_t>(receipt.split_size_size())) {
            *error = "extra input logical indices do not match split_size";
            return false;
        }
        for (uint32_t index = 0; index <= last_extra; ++index) {
            if (group_shapes.find({static_cast<int>(MMRdmaSlotPB::EXTRA_INPUT), index}) == group_shapes.end()) {
                *error = "extra input logical indices contain a gap";
                return false;
            }
        }
    }

    tensors->clear();
    objects->clear();
    tensors->reserve(allocations.size());
    objects->reserve(allocations.size());
    try {
        const torch::Device device =
            device_id_ >= 0 ? torch::Device(torch::kCUDA, device_id_) : torch::Device(torch::kCPU);
        for (const auto& allocation : allocations) {
            tensors->push_back(
                torch::empty(allocation.shape, torch::TensorOptions().dtype(allocation.dtype).device(device)));
            auto& tensor = tensors->back();
            objects->push_back({allocation.key, tensor.data_ptr(), allocation.nbytes, tensor.is_cuda()});
        }
    } catch (const std::exception& e) {
        tensors->clear();
        objects->clear();
        *error = std::string("tensor allocation failed: ") + e.what();
        return false;
    }
    return true;
}

std::vector<std::string> MMKvcmReader::handlesOf(const MultimodalOutputPB& receipt) {
    std::vector<std::string>        handles;
    std::unordered_set<std::string> seen;
    // This snapshot is intentionally built before full manifest validation so
    // failures can release producer-owned objects. Keep it bounded too: an
    // oversized/corrupt receipt must not bypass the object-count limit merely
    // through cleanup bookkeeping. The producer's deadline GC reclaims any
    // unlisted tail of an invalid receipt.
    const size_t snapshot_size =
        std::min(static_cast<size_t>(receipt.output_kvcm_objects_size()), kMMKvcmMaxObjectsPerReceipt);
    handles.reserve(snapshot_size);
    for (size_t i = 0; i < snapshot_size; ++i) {
        const auto& object = receipt.output_kvcm_objects(static_cast<int>(i));
        // The producer rejects keys outside this range before storage. Do not
        // duplicate an attacker-controlled oversized string into cleanup
        // bookkeeping or send a key the control plane can never own.
        if (!object.key().empty() && object.key().size() <= kMMKvcmMaxKeyBytes && seen.insert(object.key()).second) {
            handles.push_back(object.key());
        }
    }
    return handles;
}

void MMKvcmReader::discard(const MultimodalOutputPB& receipt, DeliveryContext& context) {
    const auto handles = handlesOf(receipt);
    if (!handles.empty()) {
        RTP_LLM_LOG_WARNING("discarding %zu unusable KVCM object(s) from an unadvertised receipt", handles.size());
        try {
            context.control.release(context.endpoint, handles, context.budget);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("discarding unusable KVCM objects failed: %s; object GC will reclaim them", e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("discarding unusable KVCM objects failed with an unknown exception; "
                                "object GC will reclaim them");
        }
    }
}

std::unique_ptr<MMReceiptReader> createMMKvcmReader(std::shared_ptr<MMKvcmClient> client) {
    return std::make_unique<MMKvcmReader>(std::move(client));
}

}  // namespace rtp_llm
