// Regression for the diagnosed production fault: a single serialized RDMA read provider
// per worker makes concurrent multimodal consumes exceed their shared
// min(budget, read_timeout_ms) acquisition window and fail silently with
// MM_PROCESS_ERROR "failed to read multimodal output over RDMA" (engine.log signature:
// MMRdmaReader.cc:154 warning with no lower-level 'tensor rdma read failed: STATUS').
//
// The loopback data plane below is slow but always successful, so on a build where all
// consumes share one serialization point the queued threads deterministically miss the
// window and this test fails with exactly the diagnosed error; on a build whose reader
// lends out pooled providers, all consumes run within the window and the test passes.
// Timing: kThreads concurrent consumes, each provider call holds kHoldMs, per-request
// window read_timeout_ms = kReadTimeoutMs with kHoldMs * 2 > kReadTimeoutMs > kHoldMs,
// and a delivery budget far larger than the window so only the read window can expire.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/config/RdmaConfig.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaExporter.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

constexpr int     kThreads       = 6;
constexpr int64_t kHoldMs        = 300;
constexpr int64_t kReadTimeoutMs = 450;
constexpr int64_t kBudgetMs      = 15000;

uint64_t alignUp(uint64_t value) {
    return (value + rdma_transport::kRdmaSlotAlign - 1) / rdma_transport::kRdmaSlotAlign * rdma_transport::kRdmaSlotAlign;
}

uint64_t tensorBytes(const torch::Tensor& tensor) {
    return static_cast<uint64_t>(tensor.numel()) * tensor.element_size();
}

rdma_transport::TensorDataType dataTypeOf(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat32:
            return rdma_transport::TensorDataType::FLOAT32;
        case torch::kInt32:
            return rdma_transport::TensorDataType::INT32;
        case torch::kHalf:
            return rdma_transport::TensorDataType::FLOAT16;
        case torch::kBFloat16:
            return rdma_transport::TensorDataType::BFLOAT16;
        default:
            return rdma_transport::TensorDataType::FLOAT32;
    }
}

torch::ScalarType torchTypeOf(rdma_transport::TensorDataType dtype) {
    switch (dtype) {
        case rdma_transport::TensorDataType::INT32:
            return torch::kInt32;
        case rdma_transport::TensorDataType::FLOAT16:
            return torch::kHalf;
        case rdma_transport::TensorDataType::BFLOAT16:
            return torch::kBFloat16;
        case rdma_transport::TensorDataType::FLOAT32:
        default:
            return torch::kFloat32;
    }
}

// In-process stand-in for the ViT-side exporter: keeps slot backing storage alive and
// hands out protocol-valid descriptors (including a NIC key) for loopback reads.
class RecordingExport: public rdma_transport::RdmaExport {
public:
    struct Slot {
        std::string   lease_id;
        torch::Tensor backing;
    };

    rdma_transport::RdmaDescriptor create(const std::vector<torch::Tensor>& tensors) override {
        std::lock_guard<std::mutex> lock(mutex_);
        rdma_transport::RdmaDescriptor descriptor;
        uint64_t                       offset = 0;
        for (const auto& tensor : tensors) {
            rdma_transport::TensorMeta meta;
            meta.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
            meta.dtype  = dataTypeOf(tensor);
            meta.offset = offset;
            meta.nbytes = tensorBytes(tensor);
            descriptor.tensors.push_back(std::move(meta));
            offset += alignUp(tensorBytes(tensor));
        }
        Slot slot;
        slot.lease_id = "lease-" + std::to_string(slots_.size());
        slot.backing  = torch::zeros({static_cast<int64_t>(std::max<uint64_t>(offset, 1))}, torch::kUInt8);
        auto* base    = slot.backing.data_ptr<uint8_t>();
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto contiguous = tensors[i].contiguous();
            std::memcpy(base + descriptor.tensors[i].offset, contiguous.data_ptr(), tensorBytes(tensors[i]));
        }
        descriptor.host          = "127.0.0.1";
        descriptor.port          = 1;
        descriptor.remote_addr   = reinterpret_cast<uint64_t>(base);
        descriptor.payload_bytes = offset;
        descriptor.lease_id      = slot.lease_id;
        descriptor.nic_keys.push_back({0, 0x9e});
        slots_.push_back(std::move(slot));
        return descriptor;
    }

    void release(const std::vector<std::string>& lease_ids) override {
        std::lock_guard<std::mutex> lock(mutex_);
        released_.insert(released_.end(), lease_ids.begin(), lease_ids.end());
    }

private:
    std::mutex               mutex_;
    std::vector<Slot>        slots_;
    std::vector<std::string> released_;
};

// Slow-but-successful loopback data plane. Each read holds its provider for kHoldMs and
// then materializes the exported tensors from the descriptor's in-process address range.
// The reader under test lends one provider to one caller at a time, matching the
// production contract that provider calls are serialized; concurrent reads therefore
// require multiple pooled providers.
class LoopbackRead: public rdma_transport::RdmaRead {
public:
    rdma_transport::RdmaReadResult read(const std::vector<rdma_transport::RdmaDescriptor>& descriptors,
                                        int64_t /*timeout_ms*/) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(kHoldMs));
        rdma_transport::RdmaReadResult result;
        for (const auto& descriptor : descriptors) {
            for (const auto& meta : descriptor.tensors) {
                auto tensor = torch::empty(meta.shape, torch::TensorOptions().dtype(torchTypeOf(meta.dtype)));
                if (tensorBytes(tensor) != meta.nbytes) {
                    result.status = ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "loopback tensor size mismatch");
                    return result;
                }
                std::memcpy(tensor.data_ptr(),
                            reinterpret_cast<const uint8_t*>(descriptor.remote_addr) + meta.offset,
                            meta.nbytes);
                result.tensors.push_back(std::move(tensor));
            }
        }
        result.status = ErrorInfo::OkStatus();
        return result;
    }
};

class RecordingControl: public MMControlClient {
public:
    ErrorResult<MultimodalOutputPB> request(const std::string& /*endpoint*/,
                                            MultimodalInputsPB& /*request_pb*/,
                                            DeadlineBudget& /*budget*/) override {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "control request is unused in consume regression");
    }

    void release(const std::string& /*endpoint*/,
                 const std::vector<std::string>& handles,
                 DeadlineBudget& /*budget*/) override {
        std::lock_guard<std::mutex> lock(mutex_);
        released_.insert(released_.end(), handles.begin(), handles.end());
    }

    void releaseAsync(std::string /*endpoint*/, std::vector<std::string> handles) override {
        std::lock_guard<std::mutex> lock(mutex_);
        released_.insert(released_.end(), handles.begin(), handles.end());
    }

private:
    std::mutex               mutex_;
    std::vector<std::string> released_;
};

struct ConsumeOutcome {
    bool          succeeded = false;
    ErrorCode     code      = ErrorCode::UNKNOWN_ERROR;
    std::string   message;
    torch::Tensor feature;
};

TEST(MMRdmaReaderConcurrencyTest, concurrentConsumesAllSucceedWithinReadWindow) {
    auto exporter = std::make_shared<RecordingExport>();
    // -fno-access-control (test_copts) exposes the injected-provider constructor and
    // exportSlots, the same pattern MMRdmaTransportTest uses.
    MMRdmaExporter adapter(exporter, int64_t{1} << 20);

    std::vector<MultimodalOutputPB> receipts(kThreads);
    std::vector<torch::Tensor>      embeddings(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        embeddings[i] = torch::arange(0, 4 * 8, torch::kFloat32).reshape({4, 8}) + static_cast<float>(i * 100);
        std::vector<MMRdmaSlotPB> slots;
        ASSERT_TRUE(adapter.exportSlots(embeddings[i], std::nullopt, {}, &slots));
        ASSERT_EQ(slots.size(), 1u);
        for (const auto& slot : slots) {
            *receipts[i].add_output_rdma_slots() = slot;
        }
        receipts[i].add_split_size(4);
    }

    RdmaConfig config;
    config.read_timeout_ms = kReadTimeoutMs;
    MMRdmaReader reader(std::shared_ptr<rdma_transport::RdmaRead>(std::make_shared<LoopbackRead>()), config);

    auto control = std::make_shared<RecordingControl>();
    std::vector<ConsumeOutcome> outcomes(kThreads);
    std::vector<std::thread>    threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&reader, &receipts, &outcomes, &control, i]() {
            const std::string endpoint = "127.0.0.1:1";
            DeadlineBudget    budget(kBudgetMs);
            DeliveryContext   context{endpoint, budget, *control, control};
            auto              result  = reader.consume(receipts[i], context);
            ConsumeOutcome    outcome;
            outcome.succeeded = result.succeeded();
            if (result.succeeded()) {
                const auto& features = result.output().mm_features;
                if (features.size() == 1) {
                    outcome.feature = features[0];
                } else {
                    outcome.succeeded = false;
                    outcome.code      = ErrorCode::MM_PROCESS_ERROR;
                    outcome.message   = "expected exactly one assembled feature";
                }
            } else {
                outcome.code    = result.error().code();
                outcome.message = result.error().ToString();
            }
            outcomes[i] = std::move(outcome);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    int failures = 0;
    for (int i = 0; i < kThreads; ++i) {
        if (!outcomes[i].succeeded) {
            ++failures;
            ADD_FAILURE() << "consume " << i << " failed with code " << static_cast<int>(outcomes[i].code) << ": "
                          << outcomes[i].message;
        }
    }
    EXPECT_EQ(failures, 0)
        << "serialized provider reads starved concurrent consumes past their read window; "
           "expected pooled providers to serve them concurrently";

    for (int i = 0; i < kThreads; ++i) {
        if (!outcomes[i].succeeded) {
            continue;
        }
        EXPECT_TRUE(torch::equal(outcomes[i].feature, embeddings[i])) << "content mismatch for consume " << i;
    }
}

}  // namespace
}  // namespace rtp_llm
