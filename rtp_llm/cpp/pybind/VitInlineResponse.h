#pragma once

#include <ATen/ops/cat.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

namespace rtp_llm {

inline void registerVitInlineResponse(pybind11::module& module) {
    namespace py = pybind11;
    module.def(
        "encode_vit_inline_response",
        [](const std::vector<at::Tensor>& embeddings,
           const std::vector<at::Tensor>& positions,
           const std::vector<std::optional<at::Tensor>>& extras,
           const std::optional<std::vector<at::Tensor>>& hashes)
            -> std::tuple<py::bytes, int64_t, int64_t, int64_t, int64_t> {
            std::unique_ptr<MultimodalOutputPB> response;
            int64_t embeddingBytes = 0;
            int64_t positionBytes  = 0;
            int64_t extraBytes     = 0;
            int64_t tokenCount     = 0;
            size_t  size           = 0;
            {
                py::gil_scoped_release release;
                if (hashes) {
                    TORCH_CHECK(hashes->size() == embeddings.size(), "invalid multimodal feature hashes");
                    for (size_t index = 0; index < embeddings.size(); ++index) {
                        const auto& hash = (*hashes)[index];
                        TORCH_CHECK(hash.device().is_cpu() && hash.scalar_type() == at::kInt && hash.dim() == 1
                                        && hash.numel() == embeddings[index].size(0),
                                    "invalid multimodal feature hashes");
                    }
                }
                if (!embeddings.empty()) {
                    response = std::make_unique<MultimodalOutputPB>();
                    for (const auto& embedding : embeddings) {
                        const auto rows = embedding.size(0);
                        TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max(), "ViT split size exceeds int32");
                        response->add_split_size(static_cast<int32_t>(rows));
                        tokenCount += rows;
                    }
                    const auto embedding = at::cat(embeddings).detach().cpu();
                    TensorPbConvert::torchToPb(response->mutable_multimodal_embedding(), embedding);
                    embeddingBytes = embedding.nbytes();
                    if (!positions.empty()) {
                        const auto position = at::cat(positions).detach().cpu();
                        TensorPbConvert::torchToPb(response->mutable_multimodal_pos_id(), position);
                        positionBytes = position.nbytes();
                    }
                    for (const auto& extra : extras) {
                        auto* slot = response->add_multimodal_extra_input();
                        if (extra) {
                            TensorPbConvert::torchToPb(slot, extra->detach().cpu());
                            extraBytes += extra->nbytes();
                        }
                    }
                    if (hashes) {
                        TensorPbConvert::torchToPb(response->mutable_multimodal_feature_hash(), at::cat(*hashes));
                    }
                    size = response->ByteSizeLong();
                    TORCH_CHECK(size <= std::numeric_limits<int>::max(), "inline ViT response exceeds protobuf size limit");
                }
            }
            if (!response) {
                return {py::bytes(), 0, 0, 0, 0};
            }
            // Allocate an unpublished bytes object, then encode directly into it without the GIL.
            auto* raw = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(size));
            if (raw == nullptr) {
                throw py::error_already_set();
            }
            auto payload = py::reinterpret_steal<py::bytes>(raw);
            auto* data = PyBytes_AS_STRING(raw);
            {
                py::gil_scoped_release release;
                const auto serialized = response->SerializeToArray(data, static_cast<int>(size));
                response.reset();
                TORCH_CHECK(serialized, "could not encode inline ViT response");
            }
            return {std::move(payload), embeddingBytes, positionBytes, extraBytes, tokenCount};
        });
}

}  // namespace rtp_llm
