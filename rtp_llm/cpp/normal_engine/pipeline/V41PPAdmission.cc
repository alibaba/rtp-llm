#include "rtp_llm/cpp/normal_engine/pipeline/V41PPAdmission.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "rtp_llm/cpp/models/V41InputRejected.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

constexpr int64_t kTokenLimit = 512;
constexpr int64_t kSpanOffset = 4;
constexpr int64_t kTokenOffset = kSpanOffset + 12;
constexpr int64_t kWindowOffset = kTokenOffset + kTokenLimit;
constexpr int64_t kPacketSize = kWindowOffset + 4 * kTokenLimit;

bool worldRejected(const torch::Tensor& packet, const std::string& error) {
    auto peers = torch::full({8 * packet.numel()}, -1, torch::kInt64);
    execAllGather({{peers}, ParallelMode::WORLD, {packet}, false});
    const auto* peer = peers.data_ptr<int64_t>();
    bool rejected = false;
    bool same = true;
    for (int rank = 0; rank < 8; ++rank) {
        const auto* row = peer + rank * packet.numel();
        if (row[0] != 0 && row[0] != 1) {
            throw std::runtime_error("V4.1 global PP preflight failed at WORLD rank " + std::to_string(rank)
                                     + " with status " + std::to_string(row[0]) + "; local status "
                                     + std::to_string(packet.data_ptr<int64_t>()[0]) + ": "
                                     + (error.empty() ? "missing/fatal peer admission" : error));
        }
        rejected |= row[0] == 0;
        same &= std::equal(peer + 1, peer + packet.numel(), row + 1);
    }
    return rejected || !same;
}

torch::Tensor hostInt32(const torch::Tensor& tensor, int64_t count, const char* name) {
    if (!tensor.defined() || tensor.scalar_type() != torch::kInt32 || tensor.dim() != 1
        || tensor.numel() != count || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("V4.1 global PP admission requires complete int32 ") + name);
    }
    return tensor.cpu().clone();
}

void validateGlobal(GptModelInputs& inputs, torch::Tensor& packet, int64_t vocab_size) {
    if (inputs.skip_run || inputs.shutdown || inputs.is_fake_stream || inputs.is_idle_decode || inputs.is_target_verify
        || inputs.decode_entrance
        || inputs.warmup || inputs.pd_separation || inputs.dspark_call_phase != DSparkCallPhase::NONE
        || (inputs.attention_mask.defined() && inputs.attention_mask.numel())
        || (inputs.last_hidden_states.defined() && inputs.last_hidden_states.numel())
        || (inputs.input_embeddings && !inputs.input_embeddings->empty())
        || (inputs.input_embeddings_locs.defined() && inputs.input_embeddings_locs.numel())
        || (inputs.multimodal_features && !inputs.multimodal_features->empty())
        || (inputs.mm_extra_input && !inputs.mm_extra_input->empty())
        || (inputs.mm_features_locs.defined() && inputs.mm_features_locs.numel())
        || (inputs.kv_cache_update_mapping.defined() && inputs.kv_cache_update_mapping.numel())
        || !inputs.pp_intermediates.empty()) {
        throw std::invalid_argument("V4.1 global PP admission requires ordinary eager text-only target prefill");
    }
    const auto spans = inputs.v41_cache_spans;
    if (!spans.defined() || spans.device().type() != torch::kCPU || spans.scalar_type() != torch::kInt64
        || spans.dim() != 2 || !spans.is_contiguous() || spans.size(1) != 6
        || spans.size(0) < 1 || spans.size(0) > 2) {
        throw std::invalid_argument("V4.1 global PP admission requires complete one/two-request native spans");
    }
    inputs.v41_cache_spans = spans.clone();
    const auto batch = spans.size(0);
    inputs.input_lengths = hostInt32(inputs.input_lengths, batch, "request lengths");
    inputs.prefix_lengths = hostInt32(inputs.prefix_lengths, batch, "prefix lengths");
    inputs.sequence_lengths = hostInt32(inputs.sequence_lengths, 0, "empty decode lengths");
    int64_t tokens = 0;
    for (int64_t row = 0; row < batch; ++row) {
        const auto* span = inputs.v41_cache_spans.data_ptr<int64_t>() + row * 6;
        if (span[4] < 0 || span[5] <= 0 || span[4] > 256 || span[5] > 256 - span[4]
            || inputs.input_lengths.data_ptr<int32_t>()[row] != span[5]
            || inputs.prefix_lengths.data_ptr<int32_t>()[row] != span[4]) {
            throw std::invalid_argument("V4.1 global PP request geometry disagrees with short native spans");
        }
        tokens += span[5];
    }
    inputs.combo_tokens = hostInt32(inputs.combo_tokens, tokens, "global token IDs");
    if ((inputs.combo_tokens < 0).any().item<bool>() || (inputs.combo_tokens >= vocab_size).any().item<bool>()) {
        throw std::invalid_argument("V4.1 global PP admission has an invalid token ID");
    }
    const auto& windows = inputs.engram_token_windows;
    if (!windows.defined() || windows.device().type() != torch::kCPU || windows.scalar_type() != torch::kInt32
        || windows.dim() != 2 || windows.size(0) != tokens || windows.size(1) != 4 || !windows.is_contiguous()
        || (windows < -1).any().item<bool>() || (windows >= vocab_size).any().item<bool>()
        || !torch::equal(windows.select(1, 0), inputs.combo_tokens)) {
        throw std::invalid_argument("V4.1 global PP admission requires native matching token-history windows");
    }
    inputs.engram_token_windows = windows.clone();
    if (inputs.combo_position_ids.defined() && inputs.combo_position_ids.numel()) {
        inputs.combo_position_ids = hostInt32(inputs.combo_position_ids, tokens, "positions");
        int64_t offset = 0;
        for (int64_t row = 0; row < batch; ++row) {
            const auto* span = inputs.v41_cache_spans.data_ptr<int64_t>() + row * 6;
            for (int64_t index = 0; index < span[5]; ++index) {
                if (inputs.combo_position_ids.data_ptr<int32_t>()[offset++] != span[4] + index) {
                    throw std::invalid_argument("V4.1 global PP position IDs disagree with native spans");
                }
            }
        }
    }
    // Exact, bounded comparison; avoid process-specific or truncated hashes.
    auto* values = packet.data_ptr<int64_t>();
    values[3] = batch;
    std::copy_n(inputs.v41_cache_spans.data_ptr<int64_t>(), batch * 6, values + kSpanOffset);
    std::copy_n(inputs.combo_tokens.data_ptr<int32_t>(), tokens, values + kTokenOffset);
    std::copy_n(inputs.engram_token_windows.data_ptr<int32_t>(), tokens * 4, values + kWindowOffset);
}

}  // namespace

V41PPAdmission::V41PPAdmission(std::shared_ptr<KVCacheManager> manager,
                               const ModelConfig&            model,
                               const ParallelismConfig&      parallelism):
    local_(std::move(manager), model, parallelism), vocab_size_(model.vocab_size),
    root_(parallelism.pp_rank == 0 && parallelism.tp_rank == 0) {
    if (vocab_size_ < 1) {
        throw std::invalid_argument("V4.1 global PP admission requires a positive vocabulary");
    }
}

V41PPCacheBatchPtr V41PPAdmission::prepare(GptModelInputs& inputs) {
    return prepareImpl(inputs, nullptr);
}

V41PPCacheBatchPtr V41PPAdmission::prepareWithWorker(GptModelInputs& inputs, const WorkerPreflight& preflight) {
    return prepareImpl(inputs, &preflight);
}

V41PPCacheBatchPtr V41PPAdmission::prepareImpl(GptModelInputs& inputs, const WorkerPreflight* preflight) {
    if (failed_) {
        throw std::runtime_error("V4.1 global PP admission session failed; drain and replace it");
    }
    GptModelInputs prepared = inputs;
    V41PPCacheBatchPtr batch;
    auto packet = torch::full({kPacketSize}, -1, torch::kInt64);
    auto* values = packet.data_ptr<int64_t>();
    values[0] = 1;
    values[1] = preflight ? 2 : 1;  // Bind worker-ready versus source-only admission.
    values[2] = round_;
    std::string error;
    try {
        if (round_ == std::numeric_limits<int64_t>::max()) {
            throw std::overflow_error("V4.1 global PP admission rounds exhausted");
        }
        if (preflight && !*preflight) {
            throw std::invalid_argument("V4.1 global PP admission requires its worker preflight callback");
        }
        validateGlobal(prepared, packet, vocab_size_);
        if (!root_ && !prepared.v41_cache_leases.empty()) {
            throw std::invalid_argument("V4.1 global PP admission permits scheduler-root leases only");
        }
        batch = local_.prepare(prepared.v41_cache_spans, prepared.v41_cache_leases);
        prepared.v41_cache_leases = batch->leases();
        prepared.kv_cache_kernel_block_id = batch->kernelBlockTable().permute({1, 0, 2}).contiguous().cuda();
        prepared.kv_cache_block_id = prepared.kv_cache_kernel_block_id;
        // Source initialization/copy addresses cannot describe another rank's
        // independent pools. Fresh tagged initialization belongs after preflight.
        prepared.kv_cache_blocks_to_zero = torch::Tensor();
        prepared.kv_cache_update_mapping = torch::Tensor();
    } catch (const std::invalid_argument& exception) {
        values[0] = 0;
        error = exception.what();
    } catch (const V41PPAllocationFailure& exception) {
        values[0] = 0;
        error = exception.what();
    } catch (const std::exception& exception) {
        values[0] = 2;
        error = exception.what();
    }
    try {
        const bool rejected = worldRejected(packet, error);
        ++round_;
        if (rejected) {
            if (batch) {
                local_.abort(batch);
            }
            throw V41InputRejected(error.empty() ? "V4.1 PP peer rejected or disagreed with the global dispatch" : error);
        }
    } catch (const V41InputRejected&) {
        throw;
    } catch (...) {
        failed_ = true;
        if (batch) {
            local_.abort(batch);
        }
        throw;
    }
    if (preflight) {
        auto ready = torch::full({3}, 1, torch::kInt64);
        auto* status = ready.data_ptr<int64_t>();
        status[1] = 3;  // Worker-ready phase, distinct from global admission.
        status[2] = round_;
        error.clear();
        try {
            // Native CPP zigzag rewrites CPU request lengths in place. A
            // shallow GptModelInputs copy would mutate our agreed global
            // snapshot, even through the callback's const tensor handles.
            GptModelInputs worker_inputs = prepared;
            worker_inputs.combo_tokens = prepared.combo_tokens.clone();
            worker_inputs.input_lengths = prepared.input_lengths.clone();
            worker_inputs.prefix_lengths = prepared.prefix_lengths.clone();
            worker_inputs.sequence_lengths = prepared.sequence_lengths.clone();
            worker_inputs.engram_token_windows = prepared.engram_token_windows.clone();
            worker_inputs.v41_cache_spans = prepared.v41_cache_spans.clone();
            worker_inputs.kv_cache_block_id = prepared.kv_cache_block_id.clone();
            worker_inputs.kv_cache_kernel_block_id = prepared.kv_cache_kernel_block_id.clone();
            if (prepared.combo_position_ids.defined()) {
                worker_inputs.combo_position_ids = prepared.combo_position_ids.clone();
            }
            (*preflight)(worker_inputs, batch);
        } catch (const V41InputRejected& exception) {
            status[0] = 0;
            error = exception.what();
        } catch (const std::exception& exception) {
            status[0] = 2;
            error = exception.what();
        } catch (...) {
            status[0] = 2;
            error = "unknown native worker preflight failure";
        }
        try {
            if (worldRejected(ready, error)) {
                local_.abort(batch);
                throw V41InputRejected(error.empty() ? "V4.1 PP worker peer rejected its native dispatch" : error);
            }
        } catch (const V41InputRejected&) {
            throw;
        } catch (...) {
            failed_ = true;
            local_.abort(batch);
            throw;
        }
    }
    batch->worker_ready_ = preflight != nullptr;
    inputs = std::move(prepared);
    return batch;
}

void V41PPAdmission::initializeFresh(const V41PPCacheBatchPtr& batch) {
    if (failed_) {
        throw std::runtime_error("V4.1 tagged initialization cannot enter a failed session");
    }
    auto packet = torch::full({3}, 1, torch::kInt64);
    auto* status = packet.data_ptr<int64_t>();
    status[1] = 4;  // Agree on the complete write plan before the first store.
    status[2] = round_;
    std::string error;
    V41PPLocalCache::InitializationOps operations;
    try {
        operations = local_.prepareInitialization(batch);
    } catch (const std::invalid_argument& exception) {
        status[0] = 0;
        error = exception.what();
    } catch (const std::exception& exception) {
        status[0] = 2;
        error = exception.what();
    }
    try {
        if (worldRejected(packet, error)) {
            throw V41InputRejected(error.empty() ? "V4.1 peer rejected tagged initialization" : error);
        }
    } catch (const V41InputRejected&) {
        throw;
    } catch (...) {
        failed_ = true;
        throw;
    }
    status[1] = 5;  // Publish initialization only after every rank completes.
    error.clear();
    writing_batch_ = batch;
    try {
        for (auto& [raw, indices] : operations) {
            raw.index_fill_(0, indices, 0);
        }
    } catch (const std::exception& exception) {
        status[0] = 2;
        error = exception.what();
    } catch (...) {
        status[0] = 2;
        error = "unknown V4.1 tagged initialization failure";
    }
    try {
        if (worldRejected(packet, error)) {
            throw std::runtime_error("V4.1 tagged initialization completion disagrees");
        }
        batch->initialized_ = true;
        writing_batch_.reset();
    } catch (...) {
        // Writes may have started. Keep the pending batch and its leases until
        // the caller drains CUDA/consumers and explicitly aborts this session.
        failed_ = true;
        throw;
    }
}

int64_t V41PPAdmission::beginExecution(const V41PPCacheBatchPtr& batch) {
    if (failed_) {
        throw std::runtime_error("V4.1 PP execution cannot enter a failed session");
    }
    auto packet = torch::tensor({int64_t{1}, int64_t{6}, round_, execution_round_}, torch::kInt64);
    std::string error;
    try {
        local_.requirePending(batch);
        if (!batch->worker_ready_ || !batch->initialized_ || batch->execution_started_) {
            throw std::invalid_argument("V4.1 PP execution requires a new initialized worker-ready batch");
        }
        if (execution_round_ == std::numeric_limits<int64_t>::max()) {
            throw std::overflow_error("V4.1 PP execution rounds exhausted");
        }
    } catch (const std::invalid_argument& exception) {
        packet.data_ptr<int64_t>()[0] = 0;
        error = exception.what();
    } catch (const std::exception& exception) {
        packet.data_ptr<int64_t>()[0] = 2;
        error = exception.what();
    }
    try {
        if (worldRejected(packet, error)) {
            throw V41InputRejected(error.empty() ? "V4.1 peer rejected execution begin" : error);
        }
    } catch (const V41InputRejected&) {
        throw;
    } catch (...) {
        failed_ = true;
        throw;
    }
    writing_batch_ = batch;
    batch->execution_started_ = true;
    return execution_round_;
}

void V41PPAdmission::completeExecution(const V41PPCacheBatchPtr& batch, bool successful) {
    if (failed_) {
        throw std::runtime_error("V4.1 PP completion cannot enter a failed session");
    }
    auto packet = torch::tensor({int64_t{successful ? 1 : 2}, int64_t{7}, round_, execution_round_}, torch::kInt64);
    std::string error;
    try {
        local_.requirePending(batch);
        if (writing_batch_ != batch || !batch->execution_started_ || batch->execution_completed_) {
            throw std::runtime_error("V4.1 PP completion requires its current executing batch");
        }
    } catch (const std::exception& exception) {
        packet.data_ptr<int64_t>()[0] = 2;
        error = exception.what();
    }
    try {
        if (worldRejected(packet, error)) {
            throw std::runtime_error("V4.1 PP execution completion disagrees");
        }
        batch->execution_completed_ = true;
        ++execution_round_;
        writing_batch_.reset();
    } catch (...) {
        failed_ = true;
        throw;
    }
}

void V41PPAdmission::commit(const V41PPCacheBatchPtr& batch) {
    if (failed_) {
        throw std::runtime_error("V4.1 global PP admission cannot commit a failed session");
    }
    if (batch && batch->execution_started_ && !batch->execution_completed_) {
        throw std::invalid_argument("V4.1 PP execution must complete across WORLD before cursor commit");
    }
    local_.commit(batch);
}

void V41PPAdmission::abort(const V41PPCacheBatchPtr& batch) {
    // A failed writer may outlive every caller descriptor. After draining,
    // the caller can abort the coordinator's retained batch with a null handle.
    const auto pending = batch ? batch : writing_batch_;
    local_.abort(pending);
    if (writing_batch_ == pending) {
        writing_batch_.reset();
    }
}

void V41PPAdmission::retire(const std::array<int64_t, 3>& source) {
    local_.retire(source);
}

bool V41PPAdmission::ownsBatch(const V41PPCacheBatchPtr& batch) const {
    return local_.ownsBatch(batch);
}

size_t V41PPAdmission::activeRequests() const {
    return local_.activeRequests();
}

}  // namespace rtp_llm
