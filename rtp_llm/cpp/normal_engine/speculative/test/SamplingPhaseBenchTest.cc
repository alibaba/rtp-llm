// Sampling-phase benchmark + correctness fingerprint for one MTP + DSpARK
// decode step. Three GPU phases are measured independently with CUDA events:
//
//   1. draft     SpeculativeSampler::sampleDSparkDraft (gamma proposal rounds)
//   2. target    execSampleGreedy on three paths (production compact
//                force_topk, full vocabulary, fused greedy top-1) with the
//                per-step full-block logits clone attribution (copy_ /
//                production .clone() / copy alone), kernel probes aimed at the
//                full-vocab suspicion, and a KINETO trace of the compact path
//   3. rejection execRejectionSampling, once with the CUDA folded finalize and
//                once with the legacy eager finalize
//
// Fixed scenario: bs=64, gamma=3, vocab=129280, temperature=0.3, top_p=0.95,
// top-k disabled, BF16 Markov weights with rank=256. The candidate width K
// (DSpARK force_topk) is a runtime knob: SAMPLING_BENCH_CANDIDATE_WIDTH accepts
// 512 / 1024 / 4096 and defaults to 1024, so the next comparison round can
// measure K=1024 against the legacy K=4096 on the same binary.
//
// Every result is printed as ONE line of JSON prefixed with
// ``SAMPLING_PHASE_BENCH `` (and mirrored into
// $TEST_UNDECLARED_OUTPUTS_DIR/sampling_phase.json when that directory is
// provided) so a before/after optimization diff can be scripted.
//
// Latency never fails the test by default. Set
// SAMPLING_BENCH_MAX_DRAFT_US / SAMPLING_BENCH_MAX_TARGET_US to turn the
// measured medians into assertions (acceptance gating).
// SAMPLING_BENCH_WARMUP / SAMPLING_BENCH_REPS override the iteration counts.

#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/all.h>

#if USING_CUDA
// CUDA runtime/ATen first: the kernel headers below use cudaStream_t/half in
// their declarations (same include order as CudaSampleOp.cc).
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include <torch/csrc/autograd/profiler_kineto.h>
#endif

namespace rtp_llm {
namespace speculative {
namespace {

#if USING_CUDA

// ---------------------------------------------------------------------------
// Fixed scenario.
// ---------------------------------------------------------------------------
constexpr int64_t kBatch                 = 64;
constexpr int64_t kGamma                 = 3;
constexpr int64_t kVocab                 = 129280;
constexpr int64_t kMarkovRank            = 256;
constexpr int     kDefaultCandidateWidth = 1024;  // aligned with DeepInfer's K=1024
constexpr int     kSmallCandidateWidth   = 512;
constexpr int     kLegacyCandidateWidth  = 4096;  // RTP_LLM_DSPARK_FORCE_TOPK default
constexpr float   kTemperature           = 0.3f;
constexpr float   kTopP                  = 0.95f;
constexpr int     kDefaultWarmup         = 20;
constexpr int     kDefaultReps           = 100;
constexpr int64_t kFingerprintSeed       = 20261104;
constexpr int64_t kFingerprintCount      = 16;  // leading token ids reported
constexpr int64_t kHashCount             = 64;  // 8x8 leading candidate ids hashed

// Candidate width (DSpARK force_topk): the next comparison round measures K=1024
// against the legacy K=4096, and every consumer below (draft force_topk, target
// compact all_probs/indices, rejection candidate rows) must follow the same width
// or the phases stop being comparable. SAMPLING_BENCH_CANDIDATE_WIDTH accepts
// 512 / 1024 / 4096; anything else is a hard error instead of a silent default.
int candidateWidth() {
    static const int width = []() {
        const char* value = std::getenv("SAMPLING_BENCH_CANDIDATE_WIDTH");
        if (value == nullptr || *value == '\0') {
            return kDefaultCandidateWidth;
        }
        char* end         = nullptr;
        errno             = 0;
        const long parsed = std::strtol(value, &end, 10);
        TORCH_CHECK(errno == 0 && end != value && *end == '\0'
                        && (parsed == kSmallCandidateWidth || parsed == kDefaultCandidateWidth
                            || parsed == kLegacyCandidateWidth),
                    "SAMPLING_BENCH_CANDIDATE_WIDTH must be one of 512, 1024, 4096; got '",
                    value,
                    "'");
        return static_cast<int>(parsed);
    }();
    return width;
}

struct TimingStats {
    double event_us_median = 0.0;
    double event_us_p50    = 0.0;
    double event_us_p90    = 0.0;
    double host_us_median  = 0.0;
    double host_us_p50     = 0.0;
    double host_us_p90     = 0.0;
};

// ---------------------------------------------------------------------------
// Environment helpers.
// ---------------------------------------------------------------------------
int envInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

bool envDouble(const char* name, double* out) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return false;
    }
    *out = std::atof(value);
    return true;
}

// ---------------------------------------------------------------------------
// Timing: CUDA-event interval per repetition (GPU span) + host submit time.
//
// ``prepare`` runs before the begin event is recorded, so fixture restoration
// (e.g. rewriting the logits that the sampler divides in place) is excluded.
// Repetitions are not synchronized in the loop, so an event interval still
// contains submission gaps of the enclosing loop exactly like the existing
// ProductionShapeCommitBenchmark in SpeculativeSamplerTest.cc.
// ---------------------------------------------------------------------------
double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const size_t idx =
        static_cast<size_t>(std::llround(std::min(std::max(pos, 0.0), static_cast<double>(sorted.size() - 1))));
    return sorted[idx];
}

TimingStats measurePhase(const std::function<void()>& run, const std::function<void()>& prepare, int warmup, int reps) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
        if (prepare) {
            prepare();
        }
        run();
    }
    stream.synchronize();

    std::vector<cudaEvent_t> begins(static_cast<size_t>(reps));
    std::vector<cudaEvent_t> ends(static_cast<size_t>(reps));
    for (size_t i = 0; i < begins.size(); ++i) {
        TORCH_CHECK(cudaEventCreate(&begins[i]) == cudaSuccess, "cudaEventCreate failed");
        TORCH_CHECK(cudaEventCreate(&ends[i]) == cudaSuccess, "cudaEventCreate failed");
    }
    std::vector<double> event_us(static_cast<size_t>(reps));
    std::vector<double> host_us(static_cast<size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        if (prepare) {
            prepare();
        }
        TORCH_CHECK(cudaEventRecord(begins[static_cast<size_t>(i)], stream.stream()) == cudaSuccess,
                    "cudaEventRecord failed");
        const auto host_begin = std::chrono::steady_clock::now();
        run();
        const auto host_end = std::chrono::steady_clock::now();
        TORCH_CHECK(cudaEventRecord(ends[static_cast<size_t>(i)], stream.stream()) == cudaSuccess,
                    "cudaEventRecord failed");
        host_us[static_cast<size_t>(i)] = std::chrono::duration<double, std::micro>(host_end - host_begin).count();
    }
    stream.synchronize();
    for (int i = 0; i < reps; ++i) {
        float milliseconds = 0.0f;
        TORCH_CHECK(cudaEventElapsedTime(&milliseconds, begins[static_cast<size_t>(i)], ends[static_cast<size_t>(i)])
                        == cudaSuccess,
                    "cudaEventElapsedTime failed");
        event_us[static_cast<size_t>(i)] = static_cast<double>(milliseconds) * 1000.0;
        TORCH_CHECK(cudaEventDestroy(begins[static_cast<size_t>(i)]) == cudaSuccess, "cudaEventDestroy failed");
        TORCH_CHECK(cudaEventDestroy(ends[static_cast<size_t>(i)]) == cudaSuccess, "cudaEventDestroy failed");
    }

    std::vector<double> sorted_event(event_us);
    std::vector<double> sorted_host(host_us);
    std::sort(sorted_event.begin(), sorted_event.end());
    std::sort(sorted_host.begin(), sorted_host.end());
    TimingStats stats;
    stats.event_us_median = percentile(sorted_event, 0.5);
    stats.event_us_p50    = percentile(sorted_event, 0.5);
    stats.event_us_p90    = percentile(sorted_event, 0.9);
    stats.host_us_median  = percentile(sorted_host, 0.5);
    stats.host_us_p50     = percentile(sorted_host, 0.5);
    stats.host_us_p90     = percentile(sorted_host, 0.9);
    return stats;
}

// ---------------------------------------------------------------------------
// Fingerprints. Integer tensors collapse to one order-sensitive int64 scalar
// (sum of value * position) so a pre/post diff is a single number comparison.
// ---------------------------------------------------------------------------
std::vector<int32_t> firstInts(const torch::Tensor& tensor, int64_t count) {
    auto flat              = tensor.flatten().to(torch::kCPU).contiguous();
    flat                   = flat.to(torch::kInt32).contiguous();
    const int64_t        n = std::min<int64_t>(flat.numel(), count);
    std::vector<int32_t> values(static_cast<size_t>(std::max<int64_t>(n, 0)));
    if (n > 0) {
        std::memcpy(values.data(), flat.data_ptr<int32_t>(), static_cast<size_t>(n) * sizeof(int32_t));
    }
    return values;
}

int64_t weightedIndexHash(const torch::Tensor& indices, int64_t count) {
    auto          flat = indices.flatten().to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t n    = std::min<int64_t>(flat.numel(), count);
    if (n <= 0) {
        return 0;
    }
    auto weights = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64));
    return (flat.narrow(0, 0, n) * weights).sum().item<int64_t>();
}

// ---------------------------------------------------------------------------
// JSON helpers (kept dependency-free: one line, no pretty printing).
// ---------------------------------------------------------------------------
std::string jsonString(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char c : value) {
        if (c == '"' || c == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

std::string jsonIntArray(const std::vector<int32_t>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

std::string jsonInt64Array(const std::vector<int64_t>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

std::string jsonShape(const torch::Tensor& tensor) {
    std::ostringstream out;
    out << "[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << tensor.size(i);
    }
    out << "]";
    return out.str();
}

std::string jsonUs(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value;
    return out.str();
}

std::string jsonFloat(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << value;
    return out.str();
}

std::string jsonTiming(const TimingStats& stats, const std::string& prefix = "") {
    std::ostringstream out;
    out << "\"" << prefix << "event_us_median\":" << jsonUs(stats.event_us_median) << ",\"" << prefix
        << "event_us_p50\":" << jsonUs(stats.event_us_p50) << ",\"" << prefix
        << "event_us_p90\":" << jsonUs(stats.event_us_p90) << ",\"" << prefix
        << "host_us_median\":" << jsonUs(stats.host_us_median) << ",\"" << prefix
        << "host_us_p50\":" << jsonUs(stats.host_us_p50) << ",\"" << prefix
        << "host_us_p90\":" << jsonUs(stats.host_us_p90);
    return out.str();
}

std::string joinJson(const std::vector<std::string>& entries) {
    std::ostringstream out;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << entries[i];
    }
    return out.str();
}

// Every fixture tensor is rebuilt from the same seed so a fingerprint never
// depends on how much entropy the preceding timed phases consumed (i.e. on
// SAMPLING_BENCH_WARMUP / SAMPLING_BENCH_REPS).
void reseedFixtureRng() {
    torch::manual_seed(kFingerprintSeed);
}

std::vector<at::Generator> makeGenerators(int64_t rows) {
    std::vector<at::Generator> generators;
    generators.reserve(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(static_cast<uint64_t>(kFingerprintSeed + row));
        generators.push_back(generator);
    }
    return generators;
}

// ---------------------------------------------------------------------------
// Target-phase fixture: the production DSpARK compact sampling inputs.
//
// logits are divided in place by the temperature penalty and are private to
// the round in production (recomputed by the next model forward), so the
// benchmark restores them with ``prepare`` before every timed repetition.
//
// sampling_buffers mirrors Sampler::allocateGreedySamplingBuffers
// (cpp/models/Sampler.cc): production pre-allocates one persistent set of
// pinned-host + device seed/offset staging tensors and passes it through
// GreedyParams. When the pointer is null, makeSamplingSeedOffsetTensors
// (models_py/bindings/core/CudaSampleOp.cc) allocates two pinned host tensors
// on EVERY call, i.e. cudaHostAlloc inside the timed region, which is a
// benchmark-only artifact the served decode step never pays.
//
// params.sampling_buffers must be wired AFTER the fixture has been
// materialized in its final storage (it points into this struct), so
// makeTargetFixture leaves it null and the caller sets it.
// ---------------------------------------------------------------------------
struct TargetFixture {
    int64_t                    rows = 0;
    torch::Tensor              pristine;
    torch::Tensor              logits;
    torch::Tensor              input_lengths;
    torch::Tensor              sequence_lengths;
    torch::Tensor              token_ids;
    torch::Tensor              top_k;
    torch::Tensor              top_p;
    torch::Tensor              temperature;
    torch::Tensor              all_probs;
    torch::Tensor              all_probs_indices;
    std::vector<at::Generator> generators;
    GreedySamplingBuffers      sampling_buffers;
    GreedyParams               params;

    void resetLogits() {
        logits.copy_(pristine);
    }

    void reseedGenerators() {
        for (size_t row = 0; row < generators.size(); ++row) {
            generators[row].set_current_seed(static_cast<uint64_t>(kFingerprintSeed + static_cast<int64_t>(row)));
        }
        params.generator = generators;
    }
};

void allocateGreedySamplingBuffers(GreedySamplingBuffers& buffers, int64_t max_batch_size) {
    const auto pinned_i64        = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    const auto device_i64        = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    buffers.seed_host            = torch::empty({max_batch_size}, pinned_i64);
    buffers.offset_host          = torch::empty({max_batch_size}, pinned_i64);
    buffers.output_ids_ptrs_host = torch::empty({max_batch_size}, pinned_i64);
    buffers.max_batch_size       = static_cast<size_t>(max_batch_size);
    buffers.seed_device          = torch::empty({max_batch_size}, device_i64);
    buffers.offset_device        = torch::empty({max_batch_size}, device_i64);
}

TargetFixture makeTargetFixture(int64_t rows) {
    const auto fp32     = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32      = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    const auto host_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    const auto host_f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);

    TargetFixture fixture;
    fixture.rows             = rows;
    fixture.pristine         = torch::randn({rows, kVocab}, fp32);
    fixture.logits           = fixture.pristine.clone();
    fixture.input_lengths    = torch::full({rows}, -1, host_i32);
    fixture.sequence_lengths = torch::full({rows}, 1, host_i32);
    // step=1: token history [rows, 2]; the sampler writes the last column.
    fixture.token_ids = torch::full({rows, 2}, 1234, i32);
    // top-k disabled for pure top-p sampling (production DSpARK hot path).
    fixture.top_k             = torch::zeros({rows}, host_i32);
    fixture.top_p             = torch::full({rows}, kTopP, host_f32);
    fixture.temperature       = torch::full({rows}, kTemperature, host_f32);
    const int width           = candidateWidth();
    fixture.all_probs         = torch::empty({rows, width}, fp32);
    fixture.all_probs_indices = torch::empty({rows, width}, i32);
    fixture.generators        = makeGenerators(rows);
    allocateGreedySamplingBuffers(fixture.sampling_buffers, rows);

    GreedyParams& params             = fixture.params;
    params.logits                    = fixture.logits;
    params.input_lengths             = fixture.input_lengths;
    params.sequence_lengths          = fixture.sequence_lengths;
    params.token_ids                 = fixture.token_ids;
    params.step                      = 1;
    params.top_k                     = fixture.top_k;
    params.top_p                     = fixture.top_p;
    params.temperature               = fixture.temperature;
    params.output_all_probs          = fixture.all_probs;
    params.output_all_probs_indices  = fixture.all_probs_indices;
    params.skip_logits_probs_restore = true;
    params.force_topk                = width;
    params.generator                 = fixture.generators;
    // Set by the caller once the fixture sits in its final storage.
    params.sampling_buffers = nullptr;
    return fixture;
}

// ---------------------------------------------------------------------------
// One target-phase variant: timing, fingerprint, and the exact gate inputs the
// sampler read. CudaSampleOp.cc keeps greedyForceTopk() in an anonymous
// namespace, so the compact gate is cross-checked from the echoed inputs
// instead of from a second copy of the decision logic.
// ---------------------------------------------------------------------------
struct TargetVariantResult {
    TimingStats          timing;
    std::vector<int32_t> token_ids;
    std::vector<int64_t> probs_shape;
    bool                 has_candidate_hash        = false;
    int64_t              candidate_hash            = 0;
    double               probs_sum                 = 0.0;
    double               probs_max                 = 0.0;
    bool                 success_ok                = true;
    bool                 sampled_ok                = true;
    int                  params_force_topk         = 0;
    bool                 indices_defined           = false;
    bool                 skip_logits_probs_restore = false;
    int64_t              logits_stride0            = 0;
    int64_t              logits_size1              = 0;
    int                  top_k_row0                = 0;
};

TargetVariantResult measureTargetVariant(TargetFixture& fixture, int warmup, int reps) {
    auto                run     = [&]() { execSampleGreedy(fixture.params); };
    auto                prepare = [&]() { fixture.resetLogits(); };
    TargetVariantResult result;
    result.timing = measurePhase(run, prepare, warmup, reps);

    // Fingerprint on restored logits with freshly seeded generators.
    fixture.resetLogits();
    fixture.reseedGenerators();
    auto greedy_output = execSampleGreedy(fixture.params);

    result.token_ids  = firstInts(fixture.token_ids.select(1, 1), kFingerprintCount);
    result.sampled_ok = fixture.token_ids.select(1, 1).ge(0).all().item<bool>();
    result.success_ok = !greedy_output.success.defined() || greedy_output.success.all().item<bool>();

    TORCH_CHECK(fixture.params.output_all_probs.has_value() && fixture.params.output_all_probs->defined(),
                "target fixture must expose output_all_probs");
    const auto& probs  = fixture.params.output_all_probs.value();
    result.probs_sum   = probs.sum().item<double>();
    result.probs_max   = probs.max().item<double>();
    result.probs_shape = {probs.size(0), probs.size(1)};

    result.indices_defined =
        fixture.params.output_all_probs_indices.has_value() && fixture.params.output_all_probs_indices->defined();
    if (result.indices_defined) {
        result.has_candidate_hash = true;
        result.candidate_hash     = weightedIndexHash(fixture.params.output_all_probs_indices.value(), kHashCount);
    }
    result.params_force_topk         = fixture.params.force_topk;
    result.skip_logits_probs_restore = fixture.params.skip_logits_probs_restore;
    result.logits_stride0            = fixture.params.logits.stride(0);
    result.logits_size1              = fixture.params.logits.size(1);
    result.top_k_row0                = fixture.params.top_k.data_ptr<int32_t>()[0];
    return result;
}

std::string targetVariantJson(const TargetVariantResult& result) {
    std::ostringstream out;
    out << "{" << jsonTiming(result.timing) << ",\"token_ids_first16\":" << jsonIntArray(result.token_ids)
        << ",\"candidate_hash_first8x8\":"
        << (result.has_candidate_hash ? std::to_string(result.candidate_hash) : std::string("null"))
        << ",\"probs_max\":" << jsonFloat(result.probs_max) << ",\"probs_sum\":" << jsonFloat(result.probs_sum)
        << ",\"probs_shape\":" << jsonInt64Array(result.probs_shape)
        << ",\"sampling_success\":" << (result.success_ok ? "true" : "false")
        << ",\"tokens_in_vocab\":" << (result.sampled_ok ? "true" : "false")
        << ",\"force_topk_effective\":{\"params_force_topk\":" << result.params_force_topk
        << ",\"output_all_probs_sizes\":" << jsonInt64Array(result.probs_shape)
        << ",\"output_all_probs_indices_defined\":" << (result.indices_defined ? "true" : "false")
        << ",\"skip_logits_probs_restore\":" << (result.skip_logits_probs_restore ? "true" : "false")
        << ",\"logits_stride0\":" << result.logits_stride0 << ",\"logits_size1\":" << result.logits_size1
        << ",\"top_k_row0\":" << result.top_k_row0
        << ",\"deepselect_fp32_available\":" << (torch_ext::deepselect_fp32_available() ? "true" : "false") << "}}";
    return out.str();
}

// Kernel-level probes for the "is the target really running the compact path?"
// question. Each probe is a standalone launch of the same primitive the sampler
// would run on that path, so a ms-scale full-vocabulary AIR kernel shows up
// here even if the sampled path is fine. Values are probe spans, not workflow.
std::string targetKernelProbeJson(const TargetFixture& fixture,
                                  const torch::Tensor& full_probs,
                                  const torch::Tensor& compact_probs,
                                  torch::Tensor&       full_width_scratch,
                                  int                  warmup,
                                  int                  reps) {
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto top_p_d =
        torch::full({fixture.rows}, kTopP, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto tokens = torch::empty({fixture.rows}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    auto valid  = torch::empty({fixture.rows}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBool));

    auto select_run = [&]() {
        auto selected = torch_ext::deepselect_fp32_topk(fixture.pristine, candidateWidth());
        (void)selected;
    };
    auto compact_softmax_run = [&]() {
        auto q = torch::softmax(compact_probs, -1);
        (void)q;
    };
    auto full_vocab_softmax_run = [&]() {
        auto q = torch::softmax(fixture.pristine, -1);
        (void)q;
    };
    auto full_vocab_renorm_run = [&]() {
        top_p_renorm_probs(full_probs, full_width_scratch, top_p_d, 1.0, reinterpret_cast<int64_t>(stream));
    };
    auto full_vocab_sampling_run = [&]() {
        top_p_sampling_from_probs(full_probs,
                                  tokens,
                                  valid,
                                  std::nullopt,
                                  top_p_d,
                                  1.0,
                                  /*deterministic=*/true,
                                  std::nullopt,
                                  0,
                                  std::nullopt,
                                  0,
                                  reinterpret_cast<int64_t>(stream));
    };
    auto full_width_zero_run = [&]() { full_width_scratch.zero_(); };

    const auto select_stats              = measurePhase(select_run, nullptr, warmup, reps);
    const auto compact_softmax_stats     = measurePhase(compact_softmax_run, nullptr, warmup, reps);
    const auto full_vocab_softmax_stats  = measurePhase(full_vocab_softmax_run, nullptr, warmup, reps);
    const auto full_vocab_renorm_stats   = measurePhase(full_vocab_renorm_run, nullptr, warmup, reps);
    const auto full_vocab_sampling_stats = measurePhase(full_vocab_sampling_run, nullptr, warmup, reps);
    const auto full_width_zero_stats     = measurePhase(full_width_zero_run, nullptr, warmup, reps);

    std::ostringstream out;
    out << "{\"deepselect_topk_event_us_median\":" << jsonUs(select_stats.event_us_median)
        << ",\"compact_softmax_event_us_median\":" << jsonUs(compact_softmax_stats.event_us_median)
        << ",\"full_vocab_softmax_event_us_median\":" << jsonUs(full_vocab_softmax_stats.event_us_median)
        << ",\"full_vocab_top_p_renorm_event_us_median\":" << jsonUs(full_vocab_renorm_stats.event_us_median)
        << ",\"full_vocab_top_p_sampling_event_us_median\":" << jsonUs(full_vocab_sampling_stats.event_us_median)
        << ",\"full_width_scratch_zero_event_us_median\":" << jsonUs(full_width_zero_stats.event_us_median)
        << ",\"candidate_width\":" << candidateWidth()
        << ",\"note\":\"standalone probe spans at this row count; deepselect falls back to ATen topk when "
           "deepselect_fp32_available is false\"}";
    return out.str();
}

// ---------------------------------------------------------------------------
// Rejection-phase fixture: compact DSpARK verify shapes with *partially
// overlapping* draft/target candidate sets drawn from the FULL vocabulary.
//
// remap_draft_topk hashes candidates with `(id * 2654435761) & mask`, so a
// bucket depends only on `id mod 2W`. Small contiguous id windows (the previous
// [100, 100+W) / [100+W/2, 100+W/2+W) fixture) never collide, which pins the
// probe count at 1 and under-reports the kernel: the served path has ids
// scattered over the whole vocabulary (hash load ~0.5, several probes per
// lookup). This fixture therefore draws ids from a full-vocabulary permutation.
//
// Layout: perm is a random permutation of [0, vocab). Draft candidates are
// perm[0, W); its first half is draft-only and its second half perm[W/2, W) is
// shared with the target set, which completes with fresh ids perm[W, 3W/2).
// Overlap is exactly W/2, and every sampled draft token is forced into it, so
// q(draft) and q(target) stay resolvable for the acceptance test.
// ---------------------------------------------------------------------------
struct RejectionFixture {
    torch::Tensor draft_probs;
    torch::Tensor draft_probs_indices;
    torch::Tensor draft_token_ids;
    torch::Tensor uniform_samples;
    torch::Tensor target_probs;
    torch::Tensor target_probs_indices;
    torch::Tensor target_token_ids;
    torch::Tensor do_sample;
    torch::Tensor output_token_ids;
    torch::Tensor output_accepted_token_num;
    int           width                    = 0;
    int64_t       overlap_ids              = 0;
    int64_t       substituted_draft_tokens = 0;
};

RejectionFixture makeRejectionFixture() {
    const auto fp32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto i32  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    const auto i64  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
    const auto b    = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBool);

    const int     width = candidateWidth();
    const int64_t steps = kGamma;
    const int64_t slots = kGamma + 1;
    const int64_t half  = width / 2;

    RejectionFixture fixture;
    fixture.width       = width;
    fixture.overlap_ids = half;

    // Full-vocabulary scrambled candidates (see the header comment above): the
    // remap hash only collides when ids are spread over the vocabulary.
    const auto vocabulary_perm = torch::randperm(kVocab, i64);
    const auto draft_ids_i32   = vocabulary_perm.narrow(0, 0, width).to(torch::kInt32);
    const auto overlap_ids_i32 = vocabulary_perm.narrow(0, half, half).to(torch::kInt32);
    const auto target_ids_i32 = torch::cat({overlap_ids_i32, vocabulary_perm.narrow(0, width, half).to(torch::kInt32)});
    fixture.draft_probs_indices  = draft_ids_i32.reshape({1, 1, width}).expand({kBatch, steps, width}).contiguous();
    fixture.target_probs_indices = target_ids_i32.reshape({1, 1, width}).expand({kBatch, slots, width}).contiguous();
    fixture.draft_probs          = torch::softmax(torch::randn({kBatch, steps, width}, fp32), -1);
    fixture.target_probs         = torch::softmax(torch::randn({kBatch, slots, width}, fp32), -1);

    // Draft candidate position >= W/2 is in the shared half, so keep the draft's
    // own argmax when it lands there and substitute a deterministic overlap id
    // otherwise. Either way the sampled token is a real target candidate, which
    // is what production remap_draft_topk needs to resolve q(draft).
    const auto  draft_choice_cpu    = fixture.draft_probs.argmax(-1).to(torch::kCPU).contiguous();
    const auto  draft_ids_cpu       = draft_ids_i32.to(torch::kCPU).contiguous();
    auto        draft_token_ids_cpu = torch::zeros({kBatch, steps}, torch::TensorOptions().dtype(torch::kInt32));
    const auto* choices             = draft_choice_cpu.data_ptr<int64_t>();
    const auto* candidate_ids       = draft_ids_cpu.data_ptr<int32_t>();
    auto*       sampled_ids         = draft_token_ids_cpu.data_ptr<int32_t>();
    for (int64_t row = 0; row < kBatch; ++row) {
        for (int64_t step = 0; step < steps; ++step) {
            const int64_t slot_index = row * steps + step;
            const int64_t position   = choices[slot_index];
            if (position < half) {
                // Draft argmax sits in the draft-only half: substitute an entry
                // from the shared half (positions >= W/2 are target candidates).
                sampled_ids[slot_index] = candidate_ids[half + (slot_index % half)];
                ++fixture.substituted_draft_tokens;
            } else {
                sampled_ids[slot_index] = candidate_ids[position];
            }
        }
    }
    fixture.draft_token_ids = draft_token_ids_cpu.to(i32);

    const auto target_choice = fixture.target_probs.argmax(-1);
    const auto target_tokens =
        fixture.target_probs_indices.gather(2, target_choice.unsqueeze(-1)).squeeze(-1).to(torch::kInt32);

    // Target tokens follow the sampler layout [rows * (steps + 1), token_stride];
    // rejection reads the last column only.
    fixture.target_token_ids = torch::zeros({kBatch * slots, 2}, i32);
    fixture.target_token_ids.select(1, 1).copy_(target_tokens.reshape({kBatch * slots}));
    fixture.uniform_samples           = torch::rand({kBatch, slots}, fp32);
    fixture.do_sample                 = torch::ones({kBatch}, b);
    fixture.output_token_ids          = torch::zeros({kBatch, slots}, i32);
    fixture.output_accepted_token_num = torch::zeros({kBatch}, i32);
    return fixture;
}

RejectionSamplingParams makeRejectionParams(RejectionFixture& fixture, bool finalize_outputs) {
    RejectionSamplingParams params;
    params.draft_probs_d               = fixture.draft_probs;
    params.draft_probs_indices_d       = fixture.draft_probs_indices;
    params.draft_token_ids_d           = fixture.draft_token_ids;
    params.uniform_samples_d           = fixture.uniform_samples;
    params.target_probs_d              = fixture.target_probs;
    params.target_probs_indices_d      = fixture.target_probs_indices;
    params.target_token_ids_d          = fixture.target_token_ids;
    params.output_token_ids_d          = fixture.output_token_ids;
    params.output_accepted_token_num_d = fixture.output_accepted_token_num;
    params.do_sample_d                 = fixture.do_sample;
    params.draft_probs_point_mass      = false;
    params.target_probs_point_mass     = false;
    params.finalize_outputs            = finalize_outputs;
    return params;
}

// ---------------------------------------------------------------------------
// Benchmark.
// ---------------------------------------------------------------------------
TEST(SamplingPhaseBench, DsparkDecodePhaseTimingAndFingerprints) {
    const c10::cuda::CUDAGuard guard(0);
    const auto                 fp32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    const auto                 i32  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

    const int warmup = std::max(0, envInt("SAMPLING_BENCH_WARMUP", kDefaultWarmup));
    const int reps   = std::max(1, envInt("SAMPLING_BENCH_REPS", kDefaultReps));

    // Seed every fixture tensor so the fingerprints are reproducible.
    torch::manual_seed(kFingerprintSeed);

    const auto*        properties = at::cuda::getDeviceProperties(0);
    std::ostringstream device_label;
    device_label << properties->name << " sm_" << properties->major << properties->minor;

    // ------------------------------------------------------------------
    // 1. Draft: one call performs all gamma Markov proposal rounds.
    // ------------------------------------------------------------------
    auto markov_w1 =
        torch::randn({kVocab, kMarkovRank}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
    auto               markov_w2 = torch::randn({kVocab, kMarkovRank}, markov_w1.options());
    SpeculativeSampler sampler(torch::Tensor(), static_cast<size_t>(kGamma));

    std::vector<std::string> draft_entries;
    double                   draft_three_round_us = 0.0;
    for (const int64_t batch : {int64_t{1}, int64_t{16}, kBatch}) {
        reseedFixtureRng();
        auto anchors      = torch::arange(batch, i32);
        auto temperatures = torch::full({batch}, kTemperature, fp32);
        auto top_p        = torch::full({batch}, kTopP, fp32);
        auto logits       = torch::randn({batch * kGamma, kVocab}, fp32);

        auto run = [&]() {
            sampler.sampleDSparkDraft(logits,
                                      anchors,
                                      temperatures,
                                      torch::Tensor(),  // top-k disabled
                                      top_p,
                                      markov_w1,
                                      markov_w2,
                                      static_cast<size_t>(kVocab),
                                      /*batch_all_greedy=*/false,
                                      /*force_topk=*/candidateWidth(),
                                      /*joint_max_top_k=*/0,
                                      /*copy_token_ids=*/false);
        };
        const auto stats = measurePhase(run, nullptr, warmup, reps);

        // Fingerprint run: reset the sampling RNG so the values only depend on
        // the fixture, never on how many timed repetitions ran before.
        torch::manual_seed(kFingerprintSeed);
        auto output = sampler.sampleDSparkDraft(logits,
                                                anchors,
                                                temperatures,
                                                torch::Tensor(),
                                                top_p,
                                                markov_w1,
                                                markov_w2,
                                                static_cast<size_t>(kVocab),
                                                /*batch_all_greedy=*/false,
                                                /*force_topk=*/candidateWidth(),
                                                /*joint_max_top_k=*/0,
                                                /*copy_token_ids=*/false);
        TORCH_CHECK(output.token_ids.defined() && output.all_probs.defined() && output.all_probs_indices.defined(),
                    "draft sampler must return tokens, compact probabilities and candidate IDs");
        const auto token_ids    = firstInts(output.token_ids, kFingerprintCount);
        const auto index_hash   = weightedIndexHash(output.all_probs_indices, kHashCount);
        const auto probs_sum    = output.all_probs.sum().item<double>();
        const auto probs_max    = output.all_probs.max().item<double>();
        const auto token_ids_ok = output.token_ids.ge(0).logical_and(output.token_ids.lt(kVocab)).all().item<bool>();

        if (batch == kBatch) {
            draft_three_round_us = stats.event_us_median;
        }

        std::ostringstream entry;
        entry << "{\"batch\":" << batch << ",\"draft_vocab\":" << kVocab << ",\"gamma\":" << kGamma
              << ",\"candidate_width\":" << candidateWidth() << "," << jsonTiming(stats)
              << ",\"three_round_total_event_us_median\":" << jsonUs(stats.event_us_median)
              << ",\"three_round_total_host_us_median\":" << jsonUs(stats.host_us_median)
              << ",\"per_round_event_us\":" << jsonUs(stats.event_us_median / static_cast<double>(kGamma))
              << ",\"token_ids_first16\":" << jsonIntArray(token_ids) << ",\"candidate_hash_first8x8\":" << index_hash
              << ",\"all_probs_sum\":" << jsonFloat(probs_sum) << ",\"all_probs_max\":" << jsonFloat(probs_max)
              << ",\"all_probs_shape\":" << jsonShape(output.all_probs)
              << ",\"tokens_in_vocab\":" << (token_ids_ok ? "true" : "false") << "}";
        draft_entries.push_back(entry.str());
        EXPECT_TRUE(token_ids_ok) << "draft tokens out of vocabulary at batch " << batch;
    }

    // ------------------------------------------------------------------
    // 2. Target: production compact-K4096 sampling, plus the temperature
    //    kernel measured alone for attribution.
    // ------------------------------------------------------------------
    std::vector<std::string> target_entries;
    double                   target_bonus_us                = 0.0;  // rows = bs (one bonus row each)
    double                   target_verify_us               = 0.0;  // rows = bs * (gamma + 1) packed verify
    double                   packed_clone_only_us           = 0.0;
    double                   packed_clone_overhead_us       = 0.0;
    double                   packed_clone_alloc_overhead_us = 0.0;
    double                   packed_compact_us              = 0.0;
    double                   packed_full_vocab_us           = 0.0;
    double                   packed_greedy_top1_us          = 0.0;
    for (const auto& labelled_rows :
         {std::make_pair("bonus_token_rows", kBatch), std::make_pair("packed_verify_rows", kBatch * (kGamma + 1))}) {
        const int64_t rows = labelled_rows.second;
        reseedFixtureRng();
        auto fixture = makeTargetFixture(rows);
        // Wire the persistent staging buffers once the fixture owns its final
        // storage: the sampler only skips its per-call pinned allocation when
        // this pointer is set (production allocates the same shape through
        // Sampler::allocateGreedySamplingBuffers).
        fixture.params.sampling_buffers = &fixture.sampling_buffers;
        TORCH_CHECK(fixture.params.sampling_buffers->seed_host.numel() >= rows
                        && fixture.params.sampling_buffers->seed_device.defined(),
                    "target fixture must provide persistent greedy sampling buffers");

        // Full-width buffers shared by the two full-width variants (and by the
        // full-vocab renormalization probe).
        auto full_width_probs = torch::empty({rows, kVocab}, fp32);
        auto top_k_all_one    = torch::ones({rows}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));

        // --- path 1: production compact force_topk (fixture default) ---
        const auto compact_result = measureTargetVariant(fixture, warmup, reps);

        // Clone variants on the compact path. MtpBatchStreamProcessor currently
        // clones the whole [rows, vocab] logits block before sampling; removing
        // that clone is a separate optimization, so time the copy alone,
        // copy+sample, and the production alloc+copy+sample sequence. The
        // derived overheads are (variant - sample_only) and sit next to the
        // directly measured copy so the two independent numbers must agree.
        auto       clone_only_run   = [&]() { fixture.logits.copy_(fixture.pristine); };
        const auto clone_only_stats = measurePhase(clone_only_run, nullptr, warmup, reps);
        auto       with_clone_run   = [&]() {
            fixture.logits.copy_(fixture.pristine);
            execSampleGreedy(fixture.params);
        };
        const auto with_clone_stats = measurePhase(with_clone_run, nullptr, warmup, reps);
        // Production-exact form: MtpBatchStreamProcessor rebuilds the sampler
        // input with `sampler_inputs.logits = sampler_inputs.logits.clone()`, so
        // the fresh allocation belongs to the step that optimization removes.
        // The copy_ variant above is the lower bound; this one matches what the
        // served timeline pays. params.logits is re-pointed at the new tensor;
        // the sampling_buffers pointer is unrelated to logits and stays valid.
        auto with_clone_alloc_run = [&]() {
            fixture.logits        = fixture.pristine.clone();
            fixture.params.logits = fixture.logits;
            execSampleGreedy(fixture.params);
        };
        const auto with_clone_alloc_stats = measurePhase(with_clone_alloc_run, nullptr, warmup, reps);

        // --- path 2: full vocabulary (the force_topk gate declined) ---
        // Same fixture and same everything else, only the sampler gates opened:
        // force_topk=0 plus a full-width all_probs output without candidate ids
        // makes the sampler softmax / top-p renormalize / top-p sample over all
        // 129280 columns, which is the suspected ms-scale AIR path.
        fixture.params.force_topk               = 0;
        fixture.params.output_all_probs         = full_width_probs;
        fixture.params.output_all_probs_indices = std::nullopt;
        const auto full_vocab_result            = measureTargetVariant(fixture, warmup, reps);

        // --- path 3: fused greedy top-1 (floor reference) ---
        fixture.params.top_k          = top_k_all_one;
        const auto greedy_top1_result = measureTargetVariant(fixture, warmup, reps);

        // Kernel probes for the full-vocab suspicion. These are standalone
        // launches of the primitives each path would run, so a ms-scale AIR
        // kernel is attributed even when the sampled path is fast.
        const auto full_probs_probe = torch::softmax(fixture.pristine, -1);
        const auto kernel_json =
            targetKernelProbeJson(fixture, full_probs_probe, fixture.all_probs, full_width_probs, warmup, reps);

        // Derived clone cost: the same stream runs copy then sample, so the
        // difference isolates the copy without a separate launch measurement.
        const double clone_overhead_us = with_clone_stats.event_us_median - compact_result.timing.event_us_median;
        const double clone_alloc_overhead_us =
            with_clone_alloc_stats.event_us_median - compact_result.timing.event_us_median;

        if (rows == kBatch) {
            target_bonus_us = compact_result.timing.event_us_median;
        } else {
            target_verify_us               = compact_result.timing.event_us_median;
            packed_clone_only_us           = clone_only_stats.event_us_median;
            packed_clone_overhead_us       = clone_overhead_us;
            packed_clone_alloc_overhead_us = clone_alloc_overhead_us;
            packed_compact_us              = compact_result.timing.event_us_median;
            packed_full_vocab_us           = full_vocab_result.timing.event_us_median;
            packed_greedy_top1_us          = greedy_top1_result.timing.event_us_median;
        }

        std::ostringstream entry;
        // Legacy top-level timing fields stay the compact (production) path;
        // the three explicit variant objects below carry the comparison.
        entry << "{\"label\":" << jsonString(labelled_rows.first) << ",\"rows\":" << rows << ",\"vocab\":" << kVocab
              << ",\"candidate_width\":" << candidateWidth() << "," << jsonTiming(compact_result.timing) << ","
              << jsonTiming(with_clone_stats, "target_with_clone_") << ","
              << jsonTiming(clone_only_stats, "target_clone_only_")
              << ",\"target_clone_overhead_event_us_median\":" << jsonUs(clone_overhead_us)
              << ",\"target_clone_overhead_method\":\"target_with_clone_event_us_median - event_us_median\""
              << ",\"target_clone_kind\":\"logits.copy_(pristine)\"" << ","
              << jsonTiming(with_clone_alloc_stats, "target_clone_alloc_")
              << ",\"target_clone_alloc_overhead_event_us_median\":" << jsonUs(clone_alloc_overhead_us)
              << ",\"target_clone_alloc_method\":"
                 "\"target_clone_alloc_event_us_median - event_us_median (production .clone() form)\""
              << ",\"target_compact\":" << targetVariantJson(compact_result)
              << ",\"target_full_vocab\":" << targetVariantJson(full_vocab_result)
              << ",\"target_greedy_top1\":" << targetVariantJson(greedy_top1_result) << ",\"kernels\":" << kernel_json
              << ",\"token_ids_first16\":" << jsonIntArray(compact_result.token_ids)
              << ",\"candidate_hash_first8x8\":" << compact_result.candidate_hash
              << ",\"probs_max\":" << jsonFloat(compact_result.probs_max)
              << ",\"probs_sum\":" << jsonFloat(compact_result.probs_sum)
              << ",\"probs_shape\":" << jsonInt64Array(compact_result.probs_shape)
              << ",\"sampling_success\":" << (compact_result.success_ok ? "true" : "false")
              << ",\"tokens_in_vocab\":" << (compact_result.sampled_ok ? "true" : "false") << "}";
        target_entries.push_back(entry.str());

        // The compact fixture must actually be the compact path; these catch a
        // fixture that silently declined the force_topk gate.
        EXPECT_EQ(compact_result.params_force_topk, candidateWidth()) << "compact fixture lost force_topk";
        ASSERT_EQ(compact_result.probs_shape.size(), 2U);
        EXPECT_EQ(compact_result.probs_shape[1], static_cast<int64_t>(candidateWidth()))
            << "compact fixture all_probs is not [rows, K]";
        EXPECT_TRUE(compact_result.indices_defined) << "compact fixture lost candidate IDs";
        EXPECT_TRUE(compact_result.success_ok) << "compact target sampling reported a failed row at rows=" << rows;
        EXPECT_TRUE(full_vocab_result.success_ok)
            << "full-vocab target sampling reported a failed row at rows=" << rows;
        EXPECT_TRUE(greedy_top1_result.success_ok) << "greedy target sampling reported a failed row at rows=" << rows;
        EXPECT_TRUE(compact_result.sampled_ok) << "target tokens out of vocabulary at rows=" << rows;
    }

    // Temperature kernel alone, at the packed verify row count.
    const int64_t temperature_rows = kBatch * (kGamma + 1);
    reseedFixtureRng();
    auto temperature_logits = torch::randn({temperature_rows, kVocab}, fp32);
    auto temperature_device = torch::full({temperature_rows}, kTemperature, fp32);
    auto temperature_stream = at::cuda::getCurrentCUDAStream().stream();
    auto temperature_run    = [&]() {
        invokeBatchApplyTemperaturePenalty<float>(temperature_logits.data_ptr<float>(),
                                                  nullptr,  // embedding_bias
                                                  temperature_device.data_ptr<float>(),
                                                  static_cast<int>(temperature_rows),
                                                  static_cast<int>(kVocab),
                                                  static_cast<int>(kVocab),
                                                  temperature_stream);
    };
    // The kernel divides in place, so repeated repetitions eventually overflow
    // to +inf; the launch/shape/path are unchanged, which is what is measured.
    const auto temperature_stats = measurePhase(temperature_run, nullptr, warmup, reps);

    // ------------------------------------------------------------------
    // Target KINETO trace of the compact path, so the kernel list and each
    // kernel's share are inspectable instead of guessed from the span total.
    // Same recipe as ProductionShapeDraftTrace in SpeculativeSamplerTest.cc.
    // ------------------------------------------------------------------
    std::string target_trace_path;
    {
        const char* trace_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
        if (trace_dir != nullptr && *trace_dir != '\0') {
            namespace tap = torch::autograd::profiler;
            namespace tpi = torch::profiler::impl;
            reseedFixtureRng();
            auto trace_fixture                    = makeTargetFixture(kBatch * (kGamma + 1));
            trace_fixture.params.sampling_buffers = &trace_fixture.sampling_buffers;

            tpi::ProfilerConfig               config(tpi::ProfilerState::KINETO, /*report_input_shapes=*/true);
            const std::set<tpi::ActivityType> activities{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
            tap::prepareProfiler(config, activities);
            tap::enableProfiler(config, activities);
            try {
                for (int call = 0; call < 5; ++call) {
                    RTP_LLM_PROFILE_SCOPE_DYNAMIC(
                        "sampling_target_compact_phase(call=%d,rows=%d,force_topk=%d,indices=%d,vocab=%d)",
                        call,
                        static_cast<int>(trace_fixture.rows),
                        candidateWidth(),
                        1,
                        static_cast<int>(kVocab));
                    trace_fixture.resetLogits();
                    execSampleGreedy(trace_fixture.params);
                }
                at::cuda::getCurrentCUDAStream().synchronize();
            } catch (...) {
                tap::disableProfiler();
                throw;
            }
            auto trace = tap::disableProfiler();
            if (trace != nullptr) {
                const std::string path = std::string(trace_dir) + "/target_phase_trace.json";
                trace->save(path);
                target_trace_path = path;
                std::cout << "SAMPLING_TARGET_TRACE path=" << path << " calls=5 rows=" << trace_fixture.rows
                          << " force_topk=" << candidateWidth() << " vocab=" << kVocab
                          << " output_all_probs_indices=1 skip_logits_probs_restore="
                          << (trace_fixture.params.skip_logits_probs_restore ? 1 : 0)
                          << " deepselect_fp32_available=" << (torch_ext::deepselect_fp32_available() ? 1 : 0)
                          << std::endl;
            }
        }
    }

    // ------------------------------------------------------------------
    // 3. Rejection: folded finalize (CUDA) versus the legacy eager finalize.
    // ------------------------------------------------------------------
    const bool finalize_supported = execRejectionSamplingSupportsFinalize();
    ASSERT_TRUE(finalize_supported) << "the folded rejection finalize requires the CUDA backend";

    reseedFixtureRng();
    auto rejection_fixture    = makeRejectionFixture();
    auto rejection_folded     = makeRejectionParams(rejection_fixture, /*finalize_outputs=*/true);
    auto rejection_eager      = makeRejectionParams(rejection_fixture, /*finalize_outputs=*/false);
    auto rejection_run_folded = [&]() { execRejectionSampling(rejection_folded); };
    auto rejection_run_eager  = [&]() {
        // Production zero-initializes both outputs before the legacy kernel.
        rejection_fixture.output_token_ids.zero_();
        rejection_fixture.output_accepted_token_num.zero_();
        execRejectionSampling(rejection_eager);
    };
    const auto rejection_folded_stats = measurePhase(rejection_run_folded, nullptr, warmup, reps);
    const auto rejection_eager_stats  = measurePhase(rejection_run_eager, nullptr, warmup, reps);

    // Fingerprints from the folded run (its outputs are fully written).
    rejection_fixture.output_token_ids.zero_();
    rejection_fixture.output_accepted_token_num.zero_();
    execRejectionSampling(rejection_folded);
    at::cuda::getCurrentCUDAStream().synchronize();
    const auto accepted_sum   = rejection_fixture.output_accepted_token_num.sum().item<int64_t>();
    const auto accepted_first = firstInts(rejection_fixture.output_accepted_token_num, kFingerprintCount);
    const auto tokens_hash    = weightedIndexHash(rejection_fixture.output_token_ids, kHashCount);
    const bool accepted_ok    = rejection_fixture.output_accepted_token_num.ge(0)
                                 .logical_and(rejection_fixture.output_accepted_token_num.le(kGamma + 1))
                                 .all()
                                 .item<bool>();

    // ------------------------------------------------------------------
    // Assemble the single-line JSON report.
    // ------------------------------------------------------------------
    double     max_draft_us  = 0.0;
    double     max_target_us = 0.0;
    const bool gate_draft    = envDouble("SAMPLING_BENCH_MAX_DRAFT_US", &max_draft_us);
    const bool gate_target   = envDouble("SAMPLING_BENCH_MAX_TARGET_US", &max_target_us);
    const bool draft_ok      = !gate_draft || draft_three_round_us <= max_draft_us;
    const bool target_ok     = !gate_target || target_verify_us <= max_target_us;

    // The target phase runs the production op chain. There is no selector for
    // the temperature fold any more (it is unconditional), so this scenario
    // block records only shape/dtype facts plus the candidate width.

    std::ostringstream scenario;
    scenario << "{\"batch\":" << kBatch << ",\"gamma\":" << kGamma << ",\"vocab\":" << kVocab
             << ",\"draft_vocab\":" << kVocab << ",\"temperature\":" << std::fixed << std::setprecision(2)
             << kTemperature << ",\"top_p\":" << kTopP << ",\"top_k\":\"disabled\",\"markov_dtype\":\"bf16\""
             << ",\"markov_rank\":" << kMarkovRank << ",\"candidate_width\":" << candidateWidth()
             << ",\"warmup\":" << warmup << ",\"reps\":" << reps << ",\"seed\":" << kFingerprintSeed
             << ",\"device\":" << jsonString(device_label.str()) << ",\"deepselect_fp32_available\":"
             << (torch_ext::deepselect_fp32_available() ? "true" : "false")
             // Target params carry production's persistent seed/offset staging
             // buffers. If this were false the sampler would allocate two
             // pinned host tensors per call (cudaHostAlloc, ms scale) inside
             // the timed region — benchmark-only noise the served step pays
             // from Sampler::allocateGreedySamplingBuffers instead.
             << ",\"sampling_buffers\":true"
             << ",\"sampling_buffers_kind\":"
                "\"GreedySamplingBuffers(pinned int64 seed_host/offset_host/output_ids_ptrs_host + "
                "CUDA int64 seed_device/offset_device, max_batch_size=rows)\""
             << ",\"target_variants\":[\"compact\",\"full_vocab\",\"greedy_top1\"]"
             << ",\"rejection_candidate_sets\":\"partial\""
             << ",\"target_trace\":"
             << (target_trace_path.empty() ? std::string("\"none\"") : jsonString(target_trace_path)) << "}";

    std::ostringstream rejection_json;
    rejection_json << "{\"batch\":" << kBatch << ",\"gamma\":" << kGamma << ",\"candidate_width\":" << candidateWidth()
                   << ",\"finalize_folded\":{" << jsonTiming(rejection_folded_stats) << "}"
                   << ",\"finalize_eager\":{" << jsonTiming(rejection_eager_stats) << ",\"zero_init_included\":true}"
                   << ",\"accepted_len_sum\":" << accepted_sum
                   << ",\"accepted_len_first16\":" << jsonIntArray(accepted_first)
                   << ",\"tokens_hash_first8x8\":" << tokens_hash
                   << ",\"accepted_in_range\":" << (accepted_ok ? "true" : "false")
                   << ",\"candidate_overlap\":{\"strategy\":\"partial\""
                   << ",\"id_range\":\"full_vocab\""
                   << ",\"width\":" << rejection_fixture.width << ",\"overlap_ids\":" << rejection_fixture.overlap_ids
                   << ",\"draft_only_ids\":" << (rejection_fixture.width - rejection_fixture.overlap_ids)
                   << ",\"target_only_ids\":" << (rejection_fixture.width - rejection_fixture.overlap_ids)
                   << ",\"substituted_draft_tokens\":"
                   << rejection_fixture.substituted_draft_tokens
                   // remap_draft_topk uses linear-probing over 2W slots with W
                   // entries, so the load factor is 0.5 and full-vocabulary ids
                   // make hits and misses both cost several probes. The previous
                   // contiguous-id fixture never collided (probe == 1).
                   << ",\"hash_table_load\":0.5"
                   << ",\"probe_expected\":\"full-vocab random ids (table load 0.5, hits+misses ~1.5-2.5 "
                      "probes, not 1)\""
                   << ",\"sampled_draft_tokens_in_target\":true}"
                   << "}";

    std::ostringstream checks;
    checks << "{\"max_draft_us\":" << (gate_draft ? jsonUs(max_draft_us) : std::string("null"))
           << ",\"draft_three_round_us\":" << jsonUs(draft_three_round_us)
           << ",\"draft_within_limit\":" << (draft_ok ? "true" : "false")
           << ",\"max_target_us\":" << (gate_target ? jsonUs(max_target_us) : std::string("null"))
           << ",\"target_packed_verify_us\":" << jsonUs(target_verify_us)
           << ",\"target_bonus_rows_us\":" << jsonUs(target_bonus_us)
           << ",\"target_packed_clone_only_us\":" << jsonUs(packed_clone_only_us)
           << ",\"target_packed_clone_overhead_us\":" << jsonUs(packed_clone_overhead_us)
           << ",\"target_packed_clone_alloc_overhead_us\":" << jsonUs(packed_clone_alloc_overhead_us)
           << ",\"target_compact_packed_us\":" << jsonUs(packed_compact_us)
           << ",\"target_full_vocab_packed_us\":" << jsonUs(packed_full_vocab_us)
           << ",\"target_greedy_top1_packed_us\":" << jsonUs(packed_greedy_top1_us)
           << ",\"target_within_limit\":" << (target_ok ? "true" : "false") << "}";

    std::ostringstream report;
    report << "{"
           << "\"scenario\":" << scenario.str() << ",\"draft\":[" << joinJson(draft_entries) << "],\"target\":["
           << joinJson(target_entries) << "],\"target_temperature_kernel\":{\"rows\":" << temperature_rows
           << ",\"vocab\":" << kVocab << "," << jsonTiming(temperature_stats)
           << "},\"rejection\":" << rejection_json.str() << ",\"checks\":" << checks.str() << "}";

    const std::string json = report.str();
    std::cout << "SAMPLING_PHASE_BENCH " << json << std::endl;

    const char* output_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (output_dir != nullptr && *output_dir != '\0') {
        const std::string path = std::string(output_dir) + "/sampling_phase.json";
        std::ofstream     out(path);
        if (out.good()) {
            out << json << "\n";
            out.close();
            std::cout << "SAMPLING_PHASE_JSON path=" << path << std::endl;
        } else {
            std::cout << "SAMPLING_PHASE_JSON_ERROR path=" << path << std::endl;
        }
    }

    EXPECT_TRUE(accepted_ok) << "rejection accepted-length out of range";

    if (gate_draft) {
        EXPECT_LE(draft_three_round_us, max_draft_us)
            << "draft three-round event median exceeds SAMPLING_BENCH_MAX_DRAFT_US";
    }
    if (gate_target) {
        EXPECT_LE(target_verify_us, max_target_us)
            << "packed target sampling event median exceeds SAMPLING_BENCH_MAX_TARGET_US";
    }
}

#else

TEST(SamplingPhaseBench, RequiresCudaBackend) {
    GTEST_SKIP() << "SAMPLING_PHASE_BENCH requires a CUDA build";
}

#endif  // USING_CUDA

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
