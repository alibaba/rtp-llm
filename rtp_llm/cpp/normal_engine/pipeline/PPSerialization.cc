#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <cstring>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::pp_serialization {

namespace {

/** Bump when any payload layout changes; all PP peers must use the same version. */
constexpr uint32_t kVersion = 22;

struct ByteWriter {
    std::vector<uint8_t> buf;

    void raw(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), b, b + n);
    }

    template<typename T>
    void val(T v) {
        raw(&v, sizeof(T));
    }

    template<typename WireType, typename T>
    void scalar(const T& value) {
        val<WireType>(static_cast<WireType>(value));
    }

    template<typename T, typename Fields>
    void sequence(const std::vector<T>& values, Fields fields) {
        val<uint64_t>(values.size());
        for (const auto& value : values) {
            fields(*this, value);
        }
    }

    template<typename T, typename Fields>
    void optional(const std::optional<T>& value, Fields fields) {
        flag(value.has_value());
        if (value.has_value()) {
            fields(*this, *value);
        }
    }

    void flag(bool v) {
        val<uint8_t>(v ? 1 : 0);
    }

    void str(const std::string& s) {
        val<uint64_t>(s.size());
        raw(s.data(), s.size());
    }

    void strings(const std::vector<std::string>& values) {
        sequence(values, [](auto& io, auto& value) { io.str(value); });
    }

    void errors(const std::vector<ErrorInfo>& errors) {
        val<uint64_t>(errors.size());
        for (const auto& error : errors) {
            val<int32_t>(static_cast<int32_t>(error.code()));
            if (error.hasError()) {
                str(error.ToString());
            }
        }
    }

    void tensor(const torch::Tensor& t) {
        /** Encode definedness separately so empty tensors retain their dtype and shape. */
        flag(t.defined());
        if (!t.defined()) {
            return;
        }
        const auto cpu = t.contiguous().cpu();
        val<uint8_t>(static_cast<uint8_t>(cpu.dim()));
        for (int64_t d = 0; d < cpu.dim(); ++d) {
            val<int64_t>(cpu.size(d));
        }
        val<int32_t>(static_cast<int32_t>(t.scalar_type()));
        val<uint64_t>(static_cast<uint64_t>(cpu.nbytes()));
        if (cpu.nbytes() > 0) {
            raw(cpu.data_ptr(), cpu.nbytes());
        }
    }

    torch::Tensor finish() const {
        auto out = torch::empty({static_cast<int64_t>(buf.size())}, torch::TensorOptions().dtype(torch::kUInt8));
        if (!buf.empty()) {
            std::memcpy(out.data_ptr(), buf.data(), buf.size());
        }
        return out;
    }
};

struct ByteReader {
    const uint8_t* p   = nullptr;
    const uint8_t* end = nullptr;

    explicit ByteReader(const torch::Tensor& buffer):
        p(static_cast<const uint8_t*>(buffer.data_ptr())),
        end(p + static_cast<size_t>(buffer.numel()) * buffer.element_size()) {
        RTP_LLM_CHECK_WITH_INFO(buffer.dtype() == torch::kUInt8 && buffer.dim() == 1,
                                "PP serialization payload must be a 1-D uint8 tensor");
    }

    void raw(void* out, size_t n) {
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(end - p) >= n, "PP serialization payload truncated");
        std::memcpy(out, p, n);
        p += n;
    }

    template<typename T>
    T val() {
        T v;
        raw(&v, sizeof(T));
        return v;
    }

    template<typename WireType, typename T>
    void scalar(T& value) {
        value = static_cast<T>(val<WireType>());
    }

    template<typename T, typename Fields>
    void sequence(std::vector<T>& values, Fields fields) {
        values.resize(val<uint64_t>());
        for (auto& value : values) {
            fields(*this, value);
        }
    }

    template<typename T, typename Fields>
    void optional(std::optional<T>& value, Fields fields) {
        if (flag()) {
            value.emplace();
            fields(*this, *value);
        } else {
            value.reset();
        }
    }

    bool flag() {
        return val<uint8_t>() != 0;
    }

    void flag(bool& value) {
        value = flag();
    }

    std::string str() {
        const auto n = val<uint64_t>();
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(end - p) >= n, "PP serialization string truncated");
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }

    void str(std::string& value) {
        value = str();
    }

    void strings(std::vector<std::string>& values) {
        sequence(values, [](auto& io, auto& value) { io.str(value); });
    }

    void errors(std::vector<ErrorInfo>& errors) {
        errors.resize(val<uint64_t>());
        for (auto& error : errors) {
            const auto code = static_cast<ErrorCode>(val<int32_t>());
            if (code != ErrorCode::NONE_ERROR) {
                error = ErrorInfo(code, str());
            }
        }
    }

    void tensor(torch::Tensor& out) {
        if (!flag()) {
            out = torch::Tensor();
            return;
        }
        const auto ndim = val<uint8_t>();
        RTP_LLM_CHECK_WITH_INFO(ndim <= 8, "PP serialization tensor ndim=%u out of range", ndim);
        std::vector<int64_t> sizes(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            sizes[d] = val<int64_t>();
            RTP_LLM_CHECK_WITH_INFO(sizes[d] >= 0, "PP serialization tensor has negative dim");
        }
        const auto dtype  = static_cast<torch::ScalarType>(val<int32_t>());
        const auto nbytes = val<uint64_t>();
        /** Plan execution uses fused H2D copies that require pinned source memory. */
        out = torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true));
        RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(out.nbytes()) == nbytes,
                                "PP serialization tensor byte count mismatch");
        if (nbytes > 0) {
            raw(out.data_ptr(), nbytes);
        }
    }

    void expectEnd() const {
        RTP_LLM_CHECK_WITH_INFO(p == end, "PP serialization payload has %zu trailing bytes", end - p);
    }
};

template<typename IO, typename List>
void optionalTensorListFields(IO& io, List& list) {
    io.optional(list, [](auto& io, auto& tensors) {
        io.sequence(tensors, [](auto& io, auto& tensor) { io.tensor(tensor); });
    });
}

/** One ordered field list serves const input on write and mutable output on read.
 * Shutdown stays in the plan header so metadata-only plans carry it too. */
template<typename IO, typename Input>
void modelInputFields(IO& io, Input& in) {
    io.tensor(in.combo_tokens);
    io.tensor(in.input_lengths);
    io.tensor(in.sequence_lengths);
    io.tensor(in.lm_output_indexes);
    io.tensor(in.custom_output_indexes);
    io.tensor(in.lm_output_lengths);
    io.tensor(in.prefix_lengths);
    io.tensor(in.sequence_lengths_plus_1);
    io.tensor(in.combo_tokens_type_ids);
    io.tensor(in.combo_position_ids);
    io.tensor(in.last_hidden_states);
    io.tensor(in.attention_mask);
    io.tensor(in.kv_cache_block_id);
    io.tensor(in.kv_cache_kernel_block_id);
    io.strings(in.kv_cache_group_tags);
    io.tensor(in.kv_cache_group_types);
    io.tensor(in.kv_cache_update_mapping);
    io.tensor(in.text_tokens_mask);
    io.tensor(in.mm_features_locs);
    io.tensor(in.input_embeddings_locs);
    io.tensor(in.request_id);
    io.tensor(in.request_pd_separation);
    io.tensor(in.cache_keys);
    optionalTensorListFields(io, in.multimodal_features);
    optionalTensorListFields(io, in.mm_extra_input);
    optionalTensorListFields(io, in.input_embeddings);
    io.template scalar<uint64_t>(in.kv_block_stride_bytes);
    io.template scalar<uint64_t>(in.kv_scale_stride_bytes);
    io.template scalar<uint64_t>(in.seq_size_per_block);
    io.template scalar<uint64_t>(in.kernel_seq_size_per_block);
    io.flag(in.pd_separation);
    io.flag(in.decode_entrance);
    io.flag(in.use_opaque_kv_cache_store);
    io.flag(in.need_all_logits);
    io.flag(in.skip_lm_head);
    io.flag(in.need_all_hidden_states);
    io.flag(in.need_moe_gating);
    io.flag(in.warmup);
    io.flag(in.skip_run);
    io.flag(in.is_fake_stream);
    io.flag(in.is_target_verify);
}

template<typename IO, typename Config>
void logitsProcessorConfigFields(IO& io, Config& cfg) {
    io.str(cfg.grammar_type);
    io.str(cfg.grammar_value);
    io.template scalar<int32_t>(cfg.combo_token_size);
    io.sequence(cfg.banned_combo_token_ids, [](auto& io, auto& banned) {
        io.sequence(banned, [](auto& io, auto& id) { io.template scalar<int32_t>(id); });
    });
    io.sequence(cfg.end_think_token_ids, [](auto& io, auto& id) { io.template scalar<int32_t>(id); });
    io.flag(cfg.enable_cross_sequence_ban);
    io.template scalar<int32_t>(cfg.cross_seq_diverge_start_combo);
}

template<typename IO, typename Sampling>
void samplingPlanFields(IO& io, Sampling& s) {
    io.sequence(s.random_seeds, [](auto& io, auto& seed) {
        io.optional(seed, [](auto& io, auto& value) { io.template scalar<int64_t>(value); });
    });
    io.sequence(s.logits_processor_configs, [](auto& io, auto& cfg) { logitsProcessorConfigFields(io, cfg); });
    io.sequence(s.num_return_sequences, [](auto& io, auto& count) { io.template scalar<int32_t>(count); });
    io.tensor(s.request_ids);
    io.tensor(s.token_ids);
    io.tensor(s.input_lengths);
    io.tensor(s.sequence_lengths);
    io.tensor(s.max_tokens);
    io.tensor(s.top_k);
    io.tensor(s.top_p);
    io.tensor(s.temperature);
    io.tensor(s.repetition_penalty);
    io.tensor(s.presence_penalty);
    io.tensor(s.frequency_penalty);
    io.tensor(s.no_repeat_ngram_size);
    io.tensor(s.do_sample);
    io.tensor(s.finished_mask);
    io.tensor(s.spec_do_sample);
    io.tensor(s.force_sp_accept);
    io.tensor(s.draft_all_probs);
    io.tensor(s.draft_point_mass_rows);
}

template<typename IO, typename Request>
void promptLogitsRequestFields(IO& io, Request& request) {
    io.flag(request.enabled);
    io.template scalar<int32_t>(request.top_k);
    io.template scalar<int32_t>(request.start);
    io.template scalar<int32_t>(request.end);
    io.flag(request.return_target_logprob);
}

template<typename IO, typename Config>
void outputConfigFields(IO& io, Config& output_config) {
    io.flag(output_config.return_logits);
    io.flag(output_config.return_softmax_probs);
    io.flag(output_config.return_cum_log_probs);
    io.flag(output_config.calculate_loss);
    io.flag(output_config.return_hidden_states);
    io.flag(output_config.return_all_hidden_states);
    io.template scalar<int32_t>(output_config.return_all_probs);
    io.sequence(output_config.prompt_logits_requests,
                [](auto& io, auto& request) { promptLogitsRequestFields(io, request); });
}

template<typename IO, typename Output>
void promptLogitsOutputFields(IO& io, Output& output) {
    io.tensor(output.topk_logprobs);
    io.tensor(output.topk_token_ids);
    io.tensor(output.target_logprobs);
    io.template scalar<int32_t>(output.start_pos);
    io.template scalar<int32_t>(output.end_pos);
}

template<typename IO, typename Result>
void executionResultFields(IO& io, Result& result) {
    io.template scalar<int64_t>(result.sample_input_us);
    io.tensor(result.request_ids);
    io.tensor(result.new_token_ids);
    io.tensor(result.new_token_lengths);
    io.tensor(result.propose_token_ids);
    io.tensor(result.propose_all_probs);
    io.tensor(result.propose_hidden_states);
    io.tensor(result.hidden_states);
    io.tensor(result.logits);
    io.tensor(result.softmax_probs);
    io.tensor(result.cum_log_probs);
    io.tensor(result.all_probs);
    io.tensor(result.loss);
    io.tensor(result.all_hidden_states);
    io.sequence(result.prompt_logits, [](auto& io, auto& prompt_logits) {
        io.optional(prompt_logits, [](auto& io, auto& output) { promptLogitsOutputFields(io, output); });
    });
    io.errors(result.request_errors);
}

}  // namespace

torch::Tensor serializePlan(const PPExecutionPlan& plan, bool metadata_only) {
    ByteWriter w;
    w.val<uint32_t>(kVersion);
    w.flag(metadata_only);
    w.flag(plan.model_input.shutdown);
    if (metadata_only) {
        /** TP peers need the upstream tag identities before synchronizing numeric cache rows. */
        w.strings(plan.model_input.kv_cache_group_tags);
    } else {
        modelInputFields(w, plan.model_input);
        samplingPlanFields(w, plan.sampling_plan);
        outputConfigFields(w, plan.output_config);
        w.flag(plan.is_decode);
        w.tensor(plan.draft_next_position_ids);
        w.val<uint64_t>(plan.finished_request_ids.size());
        for (const auto request_id : plan.finished_request_ids) {
            w.val<int64_t>(request_id);
        }
    }
    return w.finish();
}

PPExecutionPlan deserializePlan(const torch::Tensor& buffer) {
    ByteReader r(buffer);
    RTP_LLM_CHECK_WITH_INFO(r.val<uint32_t>() == kVersion, "PP plan payload version mismatch");
    PPExecutionPlan plan;
    const bool      metadata_only = r.flag();
    plan.model_input.shutdown     = r.flag();
    if (metadata_only) {
        r.strings(plan.model_input.kv_cache_group_tags);
        r.expectEnd();
        return plan;
    }
    modelInputFields(r, plan.model_input);
    samplingPlanFields(r, plan.sampling_plan);
    outputConfigFields(r, plan.output_config);
    plan.is_decode = r.flag();
    r.tensor(plan.draft_next_position_ids);
    const auto finished_request_num = r.val<uint64_t>();
    plan.finished_request_ids.resize(finished_request_num);
    for (uint64_t index = 0; index < finished_request_num; ++index) {
        plan.finished_request_ids[index] = r.val<int64_t>();
    }
    r.expectEnd();
    return plan;
}

torch::Tensor serializeExecutionResult(const PPExecutionResult& result) {
    ByteWriter w;
    w.val<uint32_t>(kVersion);
    executionResultFields(w, result);
    return w.finish();
}

PPExecutionResult deserializeExecutionResult(const torch::Tensor& buffer) {
    ByteReader r(buffer);
    RTP_LLM_CHECK_WITH_INFO(r.val<uint32_t>() == kVersion, "PP execution-result payload version mismatch");

    PPExecutionResult result;
    executionResultFields(r, result);
    r.expectEnd();
    return result;
}

torch::Tensor serializeTensorsMetadata(const PPIntermediateTensors& tensors) {
    ByteWriter w;
    w.val<uint32_t>(kVersion);
    w.val<uint64_t>(tensors.tensors.size());
    for (const auto& [name, t] : tensors.tensors) {
        w.str(name);
        const bool present = t.defined() && t.numel() > 0;
        w.flag(present);
        if (!present) {
            continue;
        }
        w.val<uint8_t>(static_cast<uint8_t>(t.dim()));
        for (int64_t d = 0; d < t.dim(); ++d) {
            w.val<int64_t>(t.size(d));
        }
        w.val<int32_t>(static_cast<int32_t>(t.scalar_type()));
        w.flag(t.is_cuda());
    }
    return w.finish();
}

PPIntermediateTensors deserializeTensorsMetadata(const torch::Tensor& metadata) {
    ByteReader r(metadata);
    RTP_LLM_CHECK_WITH_INFO(r.val<uint32_t>() == kVersion, "PP tensors-metadata payload version mismatch");
    PPIntermediateTensors out;
    const auto            n = r.val<uint64_t>();
    for (uint64_t i = 0; i < n; ++i) {
        auto name = r.str();
        if (!r.flag()) {
            out.tensors.emplace(std::move(name), torch::Tensor{});
            continue;
        }
        const auto ndim = r.val<uint8_t>();
        RTP_LLM_CHECK_WITH_INFO(ndim <= 8, "PP intermediate tensor ndim=%u out of range", ndim);
        std::vector<int64_t> sizes(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            sizes[d] = r.val<int64_t>();
            RTP_LLM_CHECK_WITH_INFO(sizes[d] >= 0, "PP intermediate tensor has negative dim");
        }
        const auto dtype   = static_cast<torch::ScalarType>(r.val<int32_t>());
        const bool is_cuda = r.flag();
        /** Pre-allocate so the transport can receive the payload in place. */
        out.tensors.emplace(
            std::move(name),
            torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(is_cuda ? torch::kCUDA : torch::kCPU)));
    }
    r.expectEnd();
    return out;
}

}  // namespace rtp_llm::pp_serialization
