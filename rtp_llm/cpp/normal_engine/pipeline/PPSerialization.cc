// Placeholder implementation, provided solely to unblock integration testing:
// PPSerialization.h shipped without a .cc, leaving undefined symbols in
// libth_transformer.so. The PP transport owner may delete this file and
// replace it with the official implementation, as long as the contract in
// PPSerialization.h (consumed by PPExecutor/PPTransport) is preserved.

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <cstring>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::pp_serialization {

namespace {

// Self-describing byte stream. Payload layout is versioned; readers bounds-
// check every field. Tensor bytes are staged on CPU regardless of the source
// device; the original device is recorded and restored on read.
constexpr uint32_t kVersion = 1;

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

    void flag(bool v) {
        val<uint8_t>(v ? 1 : 0);
    }

    void str(const std::string& s) {
        val<uint64_t>(s.size());
        raw(s.data(), s.size());
    }

    void tensor(const torch::Tensor& t) {
        const bool present = t.defined() && t.numel() > 0;
        flag(present);
        if (!present) {
            return;
        }
        const auto cpu = t.contiguous().cpu();
        val<uint8_t>(static_cast<uint8_t>(cpu.dim()));
        for (int64_t d = 0; d < cpu.dim(); ++d) {
            val<int64_t>(cpu.size(d));
        }
        val<int32_t>(static_cast<int32_t>(t.scalar_type()));
        flag(t.is_cuda());
        val<uint64_t>(static_cast<uint64_t>(cpu.nbytes()));
        raw(cpu.data_ptr(), cpu.nbytes());
    }

    void optTensorList(const std::optional<std::vector<torch::Tensor>>& list) {
        flag(list.has_value());
        if (!list.has_value()) {
            return;
        }
        val<uint64_t>(list->size());
        for (const auto& t : *list) {
            tensor(t);
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

    ByteReader(const torch::Tensor& buffer):
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

    bool flag() {
        return val<uint8_t>() != 0;
    }

    std::string str() {
        const auto n = val<uint64_t>();
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(end - p) >= n, "PP serialization string truncated");
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }

    torch::Tensor tensor() {
        if (!flag()) {
            return {};
        }
        const auto ndim = val<uint8_t>();
        RTP_LLM_CHECK_WITH_INFO(ndim <= 8, "PP serialization tensor ndim=%u out of range", ndim);
        std::vector<int64_t> sizes(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            sizes[d] = val<int64_t>();
            RTP_LLM_CHECK_WITH_INFO(sizes[d] >= 0, "PP serialization tensor has negative dim");
        }
        const auto dtype   = static_cast<torch::ScalarType>(val<int32_t>());
        const bool is_cuda = flag();
        const auto nbytes  = val<uint64_t>();
        auto       out =
            torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(is_cuda ? torch::kCUDA : torch::kCPU));
        RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(out.nbytes()) == nbytes,
                                "PP serialization tensor byte count mismatch");
        if (nbytes > 0) {
            if (is_cuda) {
                auto cpu = torch::empty(sizes, torch::TensorOptions().dtype(dtype));
                raw(cpu.data_ptr(), nbytes);
                out.copy_(cpu, /*non_blocking=*/false);
            } else {
                raw(out.data_ptr(), nbytes);
            }
        }
        return out;
    }

    void optTensorList(std::optional<std::vector<torch::Tensor>>& out) {
        if (!flag()) {
            out = std::nullopt;
            return;
        }
        const auto                 n = val<uint64_t>();
        std::vector<torch::Tensor> list;
        list.reserve(n);
        for (uint64_t i = 0; i < n; ++i) {
            list.push_back(tensor());
        }
        out = std::move(list);
    }

    void expectEnd() const {
        RTP_LLM_CHECK_WITH_INFO(p == end, "PP serialization payload has %zu trailing bytes", end - p);
    }
};

void writeModelInput(ByteWriter& w, const GptModelInputs& in) {
    w.tensor(in.combo_tokens);
    w.tensor(in.input_lengths);
    w.tensor(in.sequence_lengths);
    w.tensor(in.lm_output_indexes);
    w.tensor(in.lm_output_lengths);
    w.tensor(in.prefix_lengths);
    w.tensor(in.sequence_lengths_plus_1);
    w.tensor(in.combo_tokens_type_ids);
    w.tensor(in.combo_position_ids);
    w.tensor(in.last_hidden_states);
    w.tensor(in.attention_mask);
    w.tensor(in.kv_cache_block_id);
    w.tensor(in.kv_cache_kernel_block_id);
    w.tensor(in.kv_cache_group_types);
    w.tensor(in.kv_cache_update_mapping);
    w.tensor(in.text_tokens_mask);
    w.tensor(in.mm_features_locs);
    w.tensor(in.input_embeddings_locs);
    w.tensor(in.request_id);
    w.tensor(in.request_pd_separation);
    w.tensor(in.cache_keys);
    w.optTensorList(in.multimodal_features);
    w.optTensorList(in.mm_extra_input);
    w.optTensorList(in.input_embeddings);
    w.val<uint64_t>(in.kv_block_stride_bytes);
    w.val<uint64_t>(in.kv_scale_stride_bytes);
    w.val<uint64_t>(in.seq_size_per_block);
    w.val<uint64_t>(in.kernel_seq_size_per_block);
    w.flag(in.pd_separation);
    w.flag(in.decode_entrance);
    w.flag(in.use_opaque_kv_cache_store);
    w.flag(in.need_all_logits);
    w.flag(in.need_all_hidden_states);
    w.flag(in.need_moe_gating);
    w.flag(in.warmup);
    w.flag(in.skip_run);
    w.flag(in.is_fake_stream);
    w.flag(in.is_target_verify);
    w.val<int32_t>(static_cast<int32_t>(in.dspark_call_phase));
}

void readModelInput(ByteReader& r, GptModelInputs& in) {
    in.combo_tokens             = r.tensor();
    in.input_lengths            = r.tensor();
    in.sequence_lengths         = r.tensor();
    in.lm_output_indexes        = r.tensor();
    in.lm_output_lengths        = r.tensor();
    in.prefix_lengths           = r.tensor();
    in.sequence_lengths_plus_1  = r.tensor();
    in.combo_tokens_type_ids    = r.tensor();
    in.combo_position_ids       = r.tensor();
    in.last_hidden_states       = r.tensor();
    in.attention_mask           = r.tensor();
    in.kv_cache_block_id        = r.tensor();
    in.kv_cache_kernel_block_id = r.tensor();
    in.kv_cache_group_types     = r.tensor();
    in.kv_cache_update_mapping  = r.tensor();
    in.text_tokens_mask         = r.tensor();
    in.mm_features_locs         = r.tensor();
    in.input_embeddings_locs    = r.tensor();
    in.request_id               = r.tensor();
    in.request_pd_separation    = r.tensor();
    in.cache_keys               = r.tensor();
    r.optTensorList(in.multimodal_features);
    r.optTensorList(in.mm_extra_input);
    r.optTensorList(in.input_embeddings);
    in.kv_block_stride_bytes     = r.val<uint64_t>();
    in.kv_scale_stride_bytes     = r.val<uint64_t>();
    in.seq_size_per_block        = r.val<uint64_t>();
    in.kernel_seq_size_per_block = r.val<uint64_t>();
    in.pd_separation             = r.flag();
    in.decode_entrance           = r.flag();
    in.use_opaque_kv_cache_store = r.flag();
    in.need_all_logits           = r.flag();
    in.need_all_hidden_states    = r.flag();
    in.need_moe_gating           = r.flag();
    in.warmup                    = r.flag();
    in.skip_run                  = r.flag();
    in.is_fake_stream            = r.flag();
    in.is_target_verify          = r.flag();
    in.dspark_call_phase         = static_cast<DSparkCallPhase>(r.val<int32_t>());
}

void writeSamplingData(ByteWriter& w, const PPSamplingData& s) {
    w.val<uint64_t>(s.random_seeds.size());
    for (const auto& seed : s.random_seeds) {
        w.flag(seed.has_value());
        if (seed.has_value()) {
            w.val<int64_t>(*seed);
        }
    }
    w.val<uint64_t>(s.logits_processor_configs.size());
    for (const auto& cfg : s.logits_processor_configs) {
        w.str(cfg.grammar_type);
        w.str(cfg.grammar_value);
        w.val<int32_t>(cfg.combo_token_size);
        w.val<uint64_t>(cfg.banned_combo_token_ids.size());
        for (const auto& banned : cfg.banned_combo_token_ids) {
            w.val<uint64_t>(banned.size());
            for (const auto id : banned) {
                w.val<int32_t>(id);
            }
        }
        w.val<uint64_t>(cfg.end_think_token_ids.size());
        for (const auto id : cfg.end_think_token_ids) {
            w.val<int32_t>(id);
        }
    }
    w.flag(s.need_cum_log_probs);
    w.tensor(s.request_ids);
    w.tensor(s.token_ids);
    w.tensor(s.input_lengths);
    w.tensor(s.sequence_lengths);
    w.tensor(s.top_k);
    w.tensor(s.top_p);
    w.tensor(s.temperature);
    w.tensor(s.repetition_penalty);
    w.tensor(s.presence_penalty);
    w.tensor(s.frequency_penalty);
    w.tensor(s.no_repeat_ngram_size);
    w.tensor(s.do_sample);
    w.tensor(s.finished_mask);
}

void readSamplingData(ByteReader& r, PPSamplingData& s) {
    const auto seed_num = r.val<uint64_t>();
    s.random_seeds.resize(seed_num);
    for (uint64_t i = 0; i < seed_num; ++i) {
        if (r.flag()) {
            s.random_seeds[i] = static_cast<int>(r.val<int64_t>());
        }
    }
    const auto cfg_num = r.val<uint64_t>();
    s.logits_processor_configs.resize(cfg_num);
    for (uint64_t i = 0; i < cfg_num; ++i) {
        auto& cfg             = s.logits_processor_configs[i];
        cfg.grammar_type      = r.str();
        cfg.grammar_value     = r.str();
        cfg.combo_token_size  = r.val<int32_t>();
        const auto banned_num = r.val<uint64_t>();
        cfg.banned_combo_token_ids.resize(banned_num);
        for (uint64_t b = 0; b < banned_num; ++b) {
            const auto id_num = r.val<uint64_t>();
            cfg.banned_combo_token_ids[b].resize(id_num);
            for (uint64_t k = 0; k < id_num; ++k) {
                cfg.banned_combo_token_ids[b][k] = r.val<int32_t>();
            }
        }
        const auto end_num = r.val<uint64_t>();
        cfg.end_think_token_ids.resize(end_num);
        for (uint64_t k = 0; k < end_num; ++k) {
            cfg.end_think_token_ids[k] = r.val<int32_t>();
        }
    }
    s.need_cum_log_probs   = r.flag();
    s.request_ids          = r.tensor();
    s.token_ids            = r.tensor();
    s.input_lengths        = r.tensor();
    s.sequence_lengths     = r.tensor();
    s.top_k                = r.tensor();
    s.top_p                = r.tensor();
    s.temperature          = r.tensor();
    s.repetition_penalty   = r.tensor();
    s.presence_penalty     = r.tensor();
    s.frequency_penalty    = r.tensor();
    s.no_repeat_ngram_size = r.tensor();
    s.do_sample            = r.tensor();
    s.finished_mask        = r.tensor();
}

}  // namespace

torch::Tensor serializePlan(const PPExecutionPlan& plan, bool empty_plan) {
    ByteWriter w;
    w.val<uint32_t>(kVersion);
    w.flag(empty_plan);
    if (!empty_plan) {
        writeModelInput(w, plan.model_input);
        writeSamplingData(w, plan.sampling);
    }
    return w.finish();
}

PPExecutionPlan deserializePlan(const torch::Tensor& buffer) {
    ByteReader r(buffer);
    RTP_LLM_CHECK_WITH_INFO(r.val<uint32_t>() == kVersion, "PP plan payload version mismatch");
    PPExecutionPlan plan;
    if (r.flag()) {
        return plan;  // empty plan marker
    }
    readModelInput(r, plan.model_input);
    readSamplingData(r, plan.sampling);
    r.expectEnd();
    return plan;
}

torch::Tensor serializeSampleResult(const PPSampleResult& result) {
    ByteWriter w;
    w.val<uint32_t>(kVersion);
    w.tensor(result.request_ids);
    w.tensor(result.new_token_ids);
    w.tensor(result.sample_success);
    w.tensor(result.cum_log_probs);
    w.val<uint64_t>(result.errors.size());
    for (const auto& err : result.errors) {
        w.val<int64_t>(err.request_id);
        w.val<int32_t>(err.error_code);
        w.str(err.message);
    }
    return w.finish();
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
        // Pre-allocate so the transport can receive the payload in place.
        out.tensors.emplace(
            std::move(name),
            torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(is_cuda ? torch::kCUDA : torch::kCPU)));
    }
    r.expectEnd();
    return out;
}

}  // namespace rtp_llm::pp_serialization
