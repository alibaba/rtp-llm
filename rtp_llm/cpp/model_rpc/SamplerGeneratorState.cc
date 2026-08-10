#include "rtp_llm/cpp/model_rpc/SamplerGeneratorState.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <exception>

namespace rtp_llm {

namespace {

absl::Status validateGeneratorStateTensor(const at::Tensor& state) {
    if (!state.defined()) {
        return absl::FailedPreconditionError("sampler generator returned an undefined state");
    }
    if (state.scalar_type() != at::ScalarType::Byte) {
        return absl::FailedPreconditionError("sampler generator state is not a byte tensor");
    }
    if (state.numel() <= 0) {
        return absl::FailedPreconditionError("sampler generator returned an empty state");
    }
    return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> captureSamplerGeneratorState(bool has_explicit_seed, const at::Generator& generator) {
    if (!has_explicit_seed) {
        return std::string();
    }
    if (!generator.defined()) {
        return absl::FailedPreconditionError("seeded request has no sampler generator");
    }

    try {
        auto state = generator.get_state();
        auto status = validateGeneratorStateTensor(state);
        if (!status.ok()) {
            return status;
        }
        state = state.to(at::kCPU).contiguous();
        const auto* data = state.data_ptr<uint8_t>();
        return std::string(reinterpret_cast<const char*>(data), static_cast<size_t>(state.numel()));
    } catch (const c10::Error& e) {
        return absl::InternalError(std::string("failed to capture sampler generator state: ") + e.what());
    } catch (const std::exception& e) {
        return absl::InternalError(std::string("failed to capture sampler generator state: ") + e.what());
    } catch (...) {
        return absl::InternalError("failed to capture sampler generator state");
    }
}

absl::Status restoreSamplerGeneratorState(uint32_t            wire_version,
                                          bool                has_explicit_seed,
                                          at::Generator       generator,
                                          const std::string& serialized_state) {
    if (wire_version == kLegacySamplerGeneratorStateVersion) {
        if (!serialized_state.empty()) {
            return absl::InvalidArgumentError("legacy request contains sampler generator state");
        }
        if (has_explicit_seed) {
            return absl::InvalidArgumentError(
                "legacy seeded request cannot continue without sampler generator state");
        }
        return absl::OkStatus();
    }
    if (wire_version != kCurrentSamplerGeneratorStateVersion) {
        return absl::InvalidArgumentError("unsupported sampler generator state version");
    }
    if (!has_explicit_seed) {
        if (!serialized_state.empty()) {
            return absl::InvalidArgumentError("sampler generator state requires an explicit seed");
        }
        return absl::OkStatus();
    }
    if (serialized_state.empty()) {
        return absl::InvalidArgumentError("seeded request is missing sampler generator state");
    }
    if (!generator.defined()) {
        return absl::FailedPreconditionError("seeded request has no sampler generator");
    }

    try {
        const auto current_state = generator.get_state();
        auto       status        = validateGeneratorStateTensor(current_state);
        if (!status.ok()) {
            return status;
        }
        if (static_cast<size_t>(current_state.numel()) != serialized_state.size()) {
            return absl::InvalidArgumentError("sampler generator state has an incompatible size");
        }

        auto state = at::from_blob(const_cast<char*>(serialized_state.data()),
                                   {static_cast<int64_t>(serialized_state.size())},
                                   at::TensorOptions().dtype(at::kByte).device(at::kCPU))
                         .clone();
        generator.set_state(state);
        return absl::OkStatus();
    } catch (const c10::Error& e) {
        return absl::InvalidArgumentError(std::string("invalid sampler generator state: ") + e.what());
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(std::string("invalid sampler generator state: ") + e.what());
    } catch (...) {
        return absl::InvalidArgumentError("invalid sampler generator state");
    }
}

}  // namespace rtp_llm
