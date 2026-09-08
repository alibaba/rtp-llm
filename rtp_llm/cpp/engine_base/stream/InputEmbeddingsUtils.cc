#include "rtp_llm/cpp/engine_base/stream/InputEmbeddingsUtils.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace rtp_llm {

namespace {

absl::Status validateInputEmbeddingsImpl(std::vector<torch::Tensor>*       mutable_embeddings,
                                         const std::vector<torch::Tensor>& embeddings,
                                         const std::vector<int32_t>&       embedding_locs,
                                         int64_t                           token_count) {
    if (embeddings.empty()) {
        if (!embedding_locs.empty()) {
            return absl::InvalidArgumentError(absl::StrCat(
                "input_embeddings is empty but input_embeddings_locs has ", embedding_locs.size(), " entries"));
        }
        return absl::OkStatus();
    }
    if (embeddings.size() != embedding_locs.size()) {
        return absl::InvalidArgumentError(absl::StrCat("input_embeddings count (",
                                                       embeddings.size(),
                                                       ") != input_embeddings_locs count (",
                                                       embedding_locs.size(),
                                                       ")"));
    }

    int64_t previous_end = 0;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        auto emb = embeddings[i];
        if (!emb.defined()) {
            return absl::InvalidArgumentError(absl::StrCat("input_embeddings[", i, "] must be defined"));
        }
        if (emb.dim() < 1 || emb.dim() > 2) {
            return absl::InvalidArgumentError(
                absl::StrCat("input_embeddings[", i, "] must be 1D or 2D, got dim=", emb.dim()));
        }
        if (!emb.is_floating_point()) {
            return absl::InvalidArgumentError(absl::StrCat(
                "input_embeddings[", i, "] must be floating point, got dtype=", c10::toString(emb.scalar_type())));
        }
        if (emb.dim() == 1) {
            emb = emb.unsqueeze(0);
            if (mutable_embeddings != nullptr) {
                (*mutable_embeddings)[i] = emb;
            }
        }
        if (emb.size(0) <= 0) {
            return absl::InvalidArgumentError(absl::StrCat("input_embeddings[", i, "] must not be empty"));
        }
        const int64_t loc     = embedding_locs[i];
        const int64_t emb_len = emb.size(0);
        if (loc < 0) {
            return absl::InvalidArgumentError(absl::StrCat("input_embeddings_locs[", i, "]=", loc, " must be >= 0"));
        }
        if (token_count >= 0 && loc + emb_len > token_count) {
            return absl::InvalidArgumentError(absl::StrCat("input_embeddings_locs[",
                                                           i,
                                                           "]=",
                                                           loc,
                                                           " with emb length ",
                                                           emb_len,
                                                           " out of range [0, ",
                                                           token_count,
                                                           ")"));
        }
        if (loc < previous_end) {
            return absl::InvalidArgumentError(absl::StrCat("input_embeddings_locs[",
                                                           i,
                                                           "]=",
                                                           loc,
                                                           " overlaps or is out of order; previous interval ends at ",
                                                           previous_end));
        }
        previous_end = loc + emb_len;
    }
    return absl::OkStatus();
}

}  // namespace

absl::Status validateAndNormalizeInputEmbeddings(std::vector<torch::Tensor>& embeddings,
                                                 const std::vector<int32_t>& embedding_locs,
                                                 int64_t                     token_count) {
    return validateInputEmbeddingsImpl(&embeddings, embeddings, embedding_locs, token_count);
}

absl::Status validateInputEmbeddings(const std::vector<torch::Tensor>& embeddings,
                                     const std::vector<int32_t>&       embedding_locs,
                                     int64_t                           token_count) {
    return validateInputEmbeddingsImpl(nullptr, embeddings, embedding_locs, token_count);
}

absl::Status validateAndNormalizeInputEmbeddings(GenerateInput& input) {
    if (!input.input_embeddings.has_value() || input.input_embeddings->empty()) {
        if (input.input_embeddings_locs.has_value() && !input.input_embeddings_locs->empty()) {
            return absl::InvalidArgumentError(absl::StrCat("input_embeddings is empty but input_embeddings_locs has ",
                                                           input.input_embeddings_locs->size(),
                                                           " entries"));
        }
        return absl::OkStatus();
    }
    if (!input.input_embeddings_locs.has_value()) {
        return absl::InvalidArgumentError("input_embeddings_locs must be set");
    }
    if (!input.input_ids.defined() || input.input_ids.dim() != 1) {
        return absl::InvalidArgumentError("input_ids must be a defined 1D tensor when input_embeddings are set");
    }
    const int64_t token_count =
        input.multimodal_inputs.has_value() && !input.multimodal_inputs->empty() ? -1 : input.input_ids.size(0);
    return validateAndNormalizeInputEmbeddings(
        input.input_embeddings.value(), input.input_embeddings_locs.value(), token_count);
}

}  // namespace rtp_llm
