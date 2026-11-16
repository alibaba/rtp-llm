#include "rtp_llm/cpp/normal_engine/Dsv4DpDecodeRuntime.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <array>
#include <pybind11/embed.h>
#include <torch/torch.h>
#include <torch/csrc/utils/pybind.h>

namespace rtp_llm {

namespace py = pybind11;

struct Dsv4DpDecodeRuntime::ControlGroup {
    py::object all_gather;
    py::object destroy;
    py::object is_initialized;
    py::object group;

    ~ControlGroup() noexcept {
        py::gil_scoped_acquire gil;
        try {
            if (group && destroy && is_initialized && is_initialized().cast<bool>()) {
                destroy(group);
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("DSv4 decode control group teardown failed: %s", e.what());
        }
        all_gather     = py::object();
        destroy        = py::object();
        is_initialized = py::object();
        group          = py::object();
    }
};

Dsv4DpDecodeRuntime::Dsv4DpDecodeRuntime(int world_size, WorkspaceReservation reserve_workspace):
    world_size_(world_size), reserve_workspace_(reserve_workspace) {
    py::gil_scoped_acquire gil;
    auto                   distributed = py::module_::import("torch.distributed");
    auto                   timeout     = py::module_::import("datetime").attr("timedelta")(py::arg("seconds") = 120);
    // A CPU-only group avoids draining the serving CUDA stream to exchange
    // phase records. Its timeout bounds failures before model communication.
    control_group_                 = std::make_unique<ControlGroup>();
    control_group_->all_gather     = distributed.attr("all_gather");
    control_group_->destroy        = distributed.attr("destroy_process_group");
    control_group_->is_initialized = distributed.attr("is_initialized");
    control_group_->group = distributed.attr("new_group")(py::arg("backend") = "gloo", py::arg("timeout") = timeout);
    RTP_LLM_CHECK_WITH_INFO(control_group_->group.attr("size")().cast<int>() == world_size_,
                            "DSv4 EP decode control group size must match the dedicated decode world");
}

Dsv4DpDecodeRuntime::~Dsv4DpDecodeRuntime() = default;

std::vector<int64_t> Dsv4DpDecodeRuntime::exchange(Dsv4DecodeControl       control,
                                                   DecodeExecutionPhase    phase,
                                                   int64_t                 value,
                                                   int                     bucket,
                                                   int                     width,
                                                   const std::vector<int>& capture_buckets) {
    RTP_LLM_PROFILE_SCOPE("engine.dsv4_decode.control_exchange");
    constexpr int               header_fields = 7;
    constexpr int               fields        = header_fields + kDsv4DecodeMaxBuckets;
    const Dsv4DecodeRecord      record{++sequence_, control, phase, value, bucket, width};
    std::array<int64_t, fields> local{record.sequence,
                                      static_cast<int64_t>(control),
                                      static_cast<int64_t>(phase),
                                      value,
                                      bucket,
                                      width,
                                      record.protocol_version};
    std::copy(capture_buckets.begin(), capture_buckets.end(), local.begin() + header_fields);
    auto send     = torch::from_blob(local.data(), {fields}, torch::kInt64);
    auto received = torch::empty({world_size_, fields}, torch::kInt64);
    {
        py::gil_scoped_acquire gil;
        py::list               peers;
        for (int rank = 0; rank < world_size_; ++rank) {
            peers.append(received[rank]);
        }
        control_group_->all_gather(peers, send, py::arg("group") = control_group_->group);
    }

    std::vector<int64_t> values;
    values.reserve(world_size_);
    const auto* data = received.data_ptr<int64_t>();
    for (int rank = 0; rank < world_size_; ++rank, data += fields) {
        Dsv4DecodeRecord peer{data[0],
                              static_cast<Dsv4DecodeControl>(data[1]),
                              static_cast<DecodeExecutionPhase>(data[2]),
                              data[3],
                              data[4],
                              data[5],
                              data[6]};
        RTP_LLM_CHECK_WITH_INFO(
            matchingDsv4DecodeInvocation(record, peer),
            "DSv4 EP decode control mismatch from rank %d (sequence %ld/%ld, control %ld/%ld, phase %ld/%ld, "
            "bucket %ld/%ld, width %ld/%ld, protocol %ld/%ld)",
            rank,
            record.sequence,
            peer.sequence,
            local[1],
            data[1],
            local[2],
            data[2],
            record.bucket,
            peer.bucket,
            record.width,
            peer.width,
            record.protocol_version,
            peer.protocol_version);
        RTP_LLM_CHECK_WITH_INFO(std::equal(local.begin() + header_fields, local.end(), data + header_fields),
                                "DSv4 EP decode capture buckets differ on rank %d",
                                rank);
        values.push_back(peer.value);
    }
    return values;
}

const char* Dsv4DpDecodeRuntime::admissionError(const DecodeReplicaRequest& request) const {
    if (isDsv4DecodeRequestSupported(request.fake,
                                     request.context,
                                     request.cache_ready,
                                     request.beams,
                                     request.sequences,
                                     request.force_disable_speculation)) {
        return nullptr;
    }
    return "SM120 decode graphs require a completed PD handoff and a single sequence per request";
}

DecodeReplicaRound Dsv4DpDecodeRuntime::beginRound(bool paused, bool stopping) {
    scheduled_requests_.reset();
    target_preparation_voted_ = false;
    const auto vote =
        stopping ? Dsv4DecodeControlVote::STOP : (paused ? Dsv4DecodeControlVote::PAUSE : Dsv4DecodeControlVote::RUN);
    return classifyDsv4DecodeControl(
        exchange(Dsv4DecodeControl::ROUND_CONTROL, DecodeExecutionPhase::TARGET, static_cast<int64_t>(vote)));
}

void Dsv4DpDecodeRuntime::rejectUnpreparedRound() {
    rejectUnpreparedTarget();
}

void Dsv4DpDecodeRuntime::rejectUnpreparedTarget() {
    if (!scheduled_requests_ || target_preparation_voted_) {
        return;
    }
    const auto phase   = DecodeExecutionPhase::TARGET;
    const auto catalog = capture_catalogs_.find(phase);
    if (catalog != capture_catalogs_.end()) {
        const auto planned = plan(phase, catalog->second.token_width, -1, true);
        (void)finishPreparation(planned.execution, DecodePreparationStatus::INVALID);
    }
}

DecodeReplicaRound Dsv4DpDecodeRuntime::agreeScheduledRequests(int64_t local_requests) {
    const auto records = exchange(Dsv4DecodeControl::ROUND, DecodeExecutionPhase::TARGET, local_requests);
    const auto round   = classifyDsv4DecodeRound(records);
    if (round == DecodeReplicaRound::EXECUTE) {
        scheduled_requests_ = {local_requests, *std::max_element(records.begin(), records.end())};
    }
    return round;
}

void Dsv4DpDecodeRuntime::rejectSpeculativeInput(DecodeExecutionPhase phase) {
    const auto catalog = capture_catalogs_.find(phase);
    RTP_LLM_CHECK_WITH_INFO(
        catalog != capture_catalogs_.end(), "DSv4 EP decode phase %d has no capture catalog", static_cast<int>(phase));
    (void)plan(phase, catalog->second.token_width, -1, true);
}

bool Dsv4DpDecodeRuntime::agreeSpeculativeStatus(bool local_ok) {
    const auto records = exchange(Dsv4DecodeControl::SPECULATIVE_STATUS, DecodeExecutionPhase::TARGET_VERIFY, local_ok);
    return allDsv4DecodePeersReady(records);
}

void Dsv4DpDecodeRuntime::reserveWorkspace(const EngineInitParams&      params,
                                           std::optional<WarmUpResult>& warm_up_result) {
    if (reserve_workspace_) {
        reserve_workspace_(params, warm_up_result);
    }
}

uint32_t Dsv4DpDecodeRuntime::agreeCacheCapacity(uint32_t local_blocks) {
    const auto records = exchange(Dsv4DecodeControl::CACHE_CAPACITY,
                                  DecodeExecutionPhase::TARGET,
                                  local_blocks <= std::numeric_limits<int32_t>::max() ? local_blocks : 0);
    const auto blocks  = *std::min_element(records.begin(), records.end());
    RTP_LLM_CHECK_WITH_INFO(
        blocks > 0,
        "DSv4 EP decode workspace reservation or KV cache capacity preparation failed on one or more ranks");
    return static_cast<uint32_t>(blocks);
}

void Dsv4DpDecodeRuntime::verifyStartup(DecodeReplicaStartupStage stage, bool local_ok) {
    const auto records =
        exchange(Dsv4DecodeControl::STARTUP, DecodeExecutionPhase::TARGET, local_ok, static_cast<int>(stage));
    const char* name = nullptr;
    switch (stage) {
        case DecodeReplicaStartupStage::WARMUP_INPUTS:
            name = "warmup input preparation";
            break;
        case DecodeReplicaStartupStage::CACHE_MANAGER:
            name = "KV cache manager initialization";
            break;
        case DecodeReplicaStartupStage::ENGINE_READY:
            name = "engine initialization";
            break;
    }
    RTP_LLM_CHECK_WITH_INFO(allDsv4DecodePeersReady(records), "DSv4 EP decode %s failed on one or more ranks", name);
}

void Dsv4DpDecodeRuntime::verifyModelInitialization(bool local_ok) {
    const auto records = exchange(Dsv4DecodeControl::MODEL_INITIALIZE, DecodeExecutionPhase::TARGET, local_ok);
    RTP_LLM_CHECK_WITH_INFO(allDsv4DecodePeersReady(records),
                            "DSv4 EP decode model initialization failed on one or more ranks");
}

void Dsv4DpDecodeRuntime::verifyCapture(DecodeExecutionPhase phase, int token_width, const std::vector<int>& buckets) {
    auto sorted = buckets;
    std::sort(sorted.begin(), sorted.end());
    const bool valid = validDsv4DecodeCaptureBuckets(sorted, world_size_, token_width);
    const auto records =
        exchange(Dsv4DecodeControl::CAPTURE, phase, valid, 0, token_width, valid ? sorted : std::vector<int>{});
    RTP_LLM_CHECK_WITH_INFO(allDsv4DecodePeersReady(records),
                            "DSv4 EP decode requires valid explicit decode capture buckets on every peer");
    capture_catalogs_[phase] = {token_width, std::move(sorted)};
}

void Dsv4DpDecodeRuntime::finishCapture(DecodeExecutionPhase phase) {
    exchange(Dsv4DecodeControl::CAPTURE_COMPLETE, phase, 1);
}

void Dsv4DpDecodeRuntime::verifyCaptureReadiness(DecodeExecutionPhase phase,
                                                 int                  key,
                                                 bool                 before_capture,
                                                 bool                 local_ok) {
    const auto control =
        before_capture ? Dsv4DecodeControl::CAPTURE_WARMUP_READY : Dsv4DecodeControl::CAPTURE_INPUTS_READY;
    const auto records = exchange(control, phase, local_ok, key);
    RTP_LLM_CHECK_WITH_INFO(allDsv4DecodePeersReady(records),
                            "DSv4 EP graph %s failed on at least one rank (phase=%d, key=%d)",
                            before_capture ? "warmup" : "capture input preparation",
                            static_cast<int>(phase),
                            key);
}

AgreedDecodePlan
Dsv4DpDecodeRuntime::plan(DecodeExecutionPhase phase, int token_width, int64_t local_requests, bool allow_graph) {
    const auto catalog = capture_catalogs_.find(phase);
    if (!allow_graph || phase != DecodeExecutionPhase::TARGET) {
        // Warmup has no engine round. Speculation can filter streams after
        // scheduling, so its model phases exchange the actual lane counts.
        const auto control = allow_graph ? Dsv4DecodeControl::PLAN_SPECULATIVE : Dsv4DecodeControl::PLAN_EAGER_WARMUP;
        const auto records = exchange(control, phase, local_requests, 0, token_width);
        RTP_LLM_CHECK_WITH_INFO(*std::min_element(records.begin(), records.end()) >= 0,
                                "DSv4 EP decode invalid input geometry on one or more ranks");
        if (!allow_graph) {
            return {{DecodeExecutionMode::EAGER, phase, 0, token_width}, true};
        }
        RTP_LLM_CHECK_WITH_INFO(catalog != capture_catalogs_.end() && token_width == catalog->second.token_width,
                                "DSv4 EP decode phase %d (width %d) has no compatible capture catalog",
                                static_cast<int>(phase),
                                token_width);
        return {dsv4DecodeCandidate(
                    phase, token_width, *std::max_element(records.begin(), records.end()), catalog->second.buckets),
                true};
    }

    if (!scheduled_requests_ || catalog == capture_catalogs_.end()) {
        return {{DecodeExecutionMode::EAGER, phase, 0, token_width}, false};
    }
    // Ordinary decode consumes one lane per frozen request. The common catalog
    // supplies its width and bucket; PREPARE still checks the invocation and
    // propagates a rank's invalid lane count before the model communicates.
    const bool valid = local_requests == scheduled_requests_->local && token_width == catalog->second.token_width;
    return {dsv4DecodeCandidate(phase, token_width, scheduled_requests_->max, catalog->second.buckets), valid};
}

DecodeExecution Dsv4DpDecodeRuntime::finishPreparation(const DecodeExecution& planned, DecodePreparationStatus status) {
    if (planned.phase == DecodeExecutionPhase::TARGET) {
        // Once a vote is attempted, an exception cannot trigger a second vote.
        target_preparation_voted_ = true;
    }
    const bool graph   = planned.mode == DecodeExecutionMode::GRAPH;
    const auto records = exchange(graph ? Dsv4DecodeControl::PREPARE_GRAPH : Dsv4DecodeControl::PREPARE_EAGER,
                                  planned.phase,
                                  static_cast<int64_t>(status),
                                  planned.bucket,
                                  planned.token_width);
    DecodePreparationStatus readiness = DecodePreparationStatus::READY;
    for (auto value : records) {
        readiness = combineDsv4DecodeReadiness(readiness, static_cast<DecodePreparationStatus>(value));
    }
    RTP_LLM_CHECK_WITH_INFO(readiness != DecodePreparationStatus::INVALID,
                            "DSv4 EP decode preparation failed on at least one rank (phase=%d, bucket=%d)",
                            static_cast<int>(planned.phase),
                            planned.bucket);
    RTP_LLM_CHECK_WITH_INFO(graph || readiness == DecodePreparationStatus::READY,
                            "DSv4 EP decode eager preparation failed");
    if (readiness == DecodePreparationStatus::GRAPH_UNAVAILABLE) {
        auto catalog = capture_catalogs_.find(planned.phase);
        if (catalog != capture_catalogs_.end()) {
            const auto count = ++catalog->second.fallbacks;
            if (count == 1 || count % 1000 == 0) {
                RTP_LLM_LOG_INFO("DSv4 EP decode collective eager fallback phase=%d bucket=%d count=%llu",
                                 static_cast<int>(planned.phase),
                                 planned.bucket,
                                 static_cast<unsigned long long>(count));
            }
        }
        return {DecodeExecutionMode::EAGER, planned.phase, 0, planned.token_width};
    }
    RTP_LLM_LOG_DEBUG("DSv4 EP decode agreed execution sequence=%ld phase=%d bucket=%d width=%d mode=%d",
                      sequence_,
                      static_cast<int>(planned.phase),
                      planned.bucket,
                      planned.token_width,
                      static_cast<int>(planned.mode));
    return planned;
}

}  // namespace rtp_llm
