#include "rtp_llm/flexlb/navi_sched/resource/LatencyHistorySampler.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"

#include "rapidjson/document.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace navi_sched {
namespace {

constexpr double BETA_1 = 0.9;
constexpr double BETA_2 = 0.95;
constexpr double EPSILON = 1e-20;

struct UnifiedSample {
    NaviSchedLatencySample agent;
    NaviSchedLatencyFeatures full;
    NaviSchedLatencyFeatures floor;
    double fixedLatencyMs = 0.0;
};

struct Metrics {
    size_t count = 0;
    double absoluteErrorSum = 0.0;
    double targetSum = 0.0;
    double relativeErrorSum = 0.0;
    double signedErrorSum = 0.0;
    std::vector<double> absoluteErrors;

    void add(double prediction, double target)
    {
        const auto error = prediction - target;
        const auto absoluteError = std::abs(error);
        ++count;
        absoluteErrorSum += absoluteError;
        targetSum += target;
        relativeErrorSum += absoluteError / std::max(1.0, target);
        signedErrorSum += error;
        absoluteErrors.emplace_back(absoluteError);
    }
};

struct State {
    NonLinearPrefillModelR model;
    LatencyHistorySampler sampler;
    std::vector<double> parameters;
    std::vector<double> residualParameters;
    int64_t residualTimeUs = 0;
    size_t pendingCount = 0;

    explicit State(uint64_t seed) : sampler(seed)
    {
        model._learningRate = 0.011;
        model._useBatchCountFeature = false;
        model._useCapacitySplitFeature = true;
        parameters = model.getInitialParameters();
        residualParameters = LatencyResidualModel::makeInitialParameters();
        sampler.configure(2000, 64);
    }
};

NaviSchedLatencyFeatures readFeatures(const rapidjson::Value &value)
{
    if (!value.IsArray() ||
        (value.Size() != 12 && value.Size() != 13 && value.Size() != 15)) {
        throw std::runtime_error("invalid latency features");
    }
    NaviSchedLatencyFeatures features;
    if (!features.appendCapacitySplitAggregate(
            value[0].GetInt64(), value[1].GetInt64(), value[2].GetInt64(),
            value[3].GetInt64(), value[4].GetDouble(), value[5].GetDouble(),
            value[6].GetInt64(), value[7].GetInt64(), value[8].GetInt64(),
            value[9].GetDouble(), value[10].GetDouble()) ||
        !features.appendPrecedingRequestCount(value[11].GetInt64()) ||
        (value.Size() >= 13 &&
         !features.appendRunningElapsedTimeMs(value[12].GetDouble())) ||
        (value.Size() == 15 &&
         !features.appendRunningDecodeProgress(value[13].GetInt64(),
                                               value[14].GetInt64()))) {
        throw std::runtime_error("invalid latency feature aggregate");
    }
    return features;
}

std::vector<UnifiedSample> readInput(const std::filesystem::path &path)
{
    std::vector<UnifiedSample> samples;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        rapidjson::Document event;
        event.Parse(line.c_str());
        if (event.HasParseError() || !event.IsObject()) {
            throw std::runtime_error("invalid replay event in " + path.string());
        }
        UnifiedSample sample;
        sample.agent.features = readFeatures(event["agent_features"]);
        sample.full = readFeatures(event["full_features"]);
        sample.floor = readFeatures(event["floor_features"]);
        sample.agent.latencyMs = event["target_ms"].GetDouble();
        sample.agent.eventTimeUs = event["event_time_us"].GetInt64();
        sample.agent.localBatchId = static_cast<int64_t>(samples.size() + 1);
        sample.fixedLatencyMs = event["fixed_latency_ms"].GetDouble();
        if (!std::isfinite(sample.agent.latencyMs) ||
            sample.agent.latencyMs < 0.0 ||
            !std::isfinite(sample.fixedLatencyMs)) {
            throw std::runtime_error("invalid replay target");
        }
        samples.emplace_back(std::move(sample));
    }
    return samples;
}

double evaluateAgent(const UnifiedSample &sample, const State &state)
{
    const auto base = state.model.predictConfigured(sample.agent.features,
                                                     state.parameters);
    return LatencyResidualModel::evaluate(
               base, state.residualParameters, state.residualTimeUs,
               sample.agent.eventTimeUs)
        .predictionMs;
}

struct UnifiedEvaluation {
    double predictionMs = 0.0;
    bool fullBranch = false;
};

enum class Variant {
    AGENT_ARRIVAL,
    CURRENT_MASTER,
    MASTER_UNIFIED,
};

UnifiedEvaluation evaluateMaster(const UnifiedSample &sample, const State &state)
{
    const auto fullBase =
        state.model.predictConfigured(sample.full, state.parameters);
    const auto floorBase =
        state.model.predictConfigured(sample.floor, state.parameters);
    const auto full = LatencyResidualModel::evaluate(
        fullBase, state.residualParameters, state.residualTimeUs,
        sample.agent.eventTimeUs);
    const auto floor = LatencyResidualModel::evaluate(
        floorBase, state.residualParameters, state.residualTimeUs,
        sample.agent.eventTimeUs);
    const auto fullPrediction = full.predictionMs + sample.fixedLatencyMs;
    return fullPrediction >= floor.predictionMs
        ? UnifiedEvaluation{fullPrediction, true}
        : UnifiedEvaluation{floor.predictionMs, false};
}

bool sameFeatures(const NaviSchedLatencyFeatures &left,
                  const NaviSchedLatencyFeatures &right)
{
    return left.getBatchCount() == right.getBatchCount() &&
           left.getPrecedingRequestCount() ==
               right.getPrecedingRequestCount() &&
           left.getRunningBatchSize() == right.getRunningBatchSize() &&
           left.getRunningCacheHitTokenCount() ==
               right.getRunningCacheHitTokenCount() &&
           left.getRunningCacheMissTokenCount() ==
               right.getRunningCacheMissTokenCount() &&
           left.getRunningCacheMissTokenSquareSum() ==
               right.getRunningCacheMissTokenSquareSum() &&
           left.getRunningCacheHitMissTokenProductSum() ==
               right.getRunningCacheHitMissTokenProductSum() &&
           left.getWaitingBatchSize() == right.getWaitingBatchSize() &&
           left.getWaitingCacheHitTokenCount() ==
               right.getWaitingCacheHitTokenCount() &&
           left.getWaitingCacheMissTokenCount() ==
               right.getWaitingCacheMissTokenCount() &&
           left.getWaitingCacheMissTokenSquareSum() ==
               right.getWaitingCacheMissTokenSquareSum() &&
           left.getWaitingCacheHitMissTokenProductSum() ==
               right.getWaitingCacheHitMissTokenProductSum() &&
           left.getRunningElapsedTimeMs() ==
               right.getRunningElapsedTimeMs();
}

bool fitMasterBatch(const std::vector<const UnifiedSample *> &samples,
                    State &state)
{
    if (samples.empty()) {
        return false;
    }
    auto weights = NonLinearPrefillModelR::toValues(state.parameters);
    weights[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
    NonLinearPrefillModelR::Values gradient{};
    const auto parameterCount = state.parameters.size();
    for (const auto *sample : samples) {
        const auto branch = evaluateMaster(*sample, state).fullBranch;
        const auto &features = branch ? sample->full : sample->floor;
        auto input = NonLinearPrefillModelR::makeInput(features);
        input[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] = 0.0;
        NonLinearPrefillModelR::Values predictionGradient{};
        const auto prediction = NonLinearPrefillModelR::predict(
            input, weights, parameterCount, &predictionGradient);
        const auto target = branch
            ? std::max(0.0, sample->agent.latencyMs - sample->fixedLatencyMs)
            : sample->agent.latencyMs;
        if (!std::isfinite(prediction) || prediction < 0.0) {
            return false;
        }
        const auto difference =
            (std::log1p(prediction) - std::log1p(target)) /
            (1.0 + prediction);
        for (size_t index = 0; index < parameterCount; ++index) {
            gradient[index] += difference * predictionGradient[index];
        }
    }

    const auto count = static_cast<double>(samples.size());
    const auto correction = state.model._learningRate * 6.0 *
                            std::sqrt(1.0 - std::pow(BETA_2, state.model._step)) /
                            (1.0 - std::pow(BETA_1, state.model._step));
    for (size_t index = 0; index < parameterCount; ++index) {
        gradient[index] /= count;
        state.model._adamMoment1[index] =
            BETA_1 * state.model._adamMoment1[index] +
            (1.0 - BETA_1) * gradient[index];
        state.model._adamMoment2[index] =
            BETA_2 * state.model._adamMoment2[index] +
            (1.0 - BETA_2) * gradient[index] * gradient[index];
        weights[index] -= correction * state.model._adamMoment1[index] /
                          std::sqrt(state.model._adamMoment2[index] + EPSILON);
    }
    state.model.projectOutputParameters(
        weights, state.model._adamMoment1, state.model._adamMoment2,
        NonLinearPrefillModelR::MIN_NONLINEAR_CURVATURE);
    ++state.model._step;
    state.parameters.assign(weights.begin(), weights.begin() + parameterCount);
    return NonLinearPrefillModelR::validateParameters(state.parameters);
}

bool updateResidual(const UnifiedSample &sample, bool masterFeatures, State &state)
{
    if (!masterFeatures) {
        const auto base = state.model.predictConfigured(sample.agent.features,
                                                         state.parameters);
        return LatencyResidualModel::update(
            base, sample.agent.latencyMs, state.residualParameters,
            state.residualTimeUs, sample.agent.eventTimeUs);
    }
    const auto branch = evaluateMaster(sample, state).fullBranch;
    const auto &features = branch ? sample.full : sample.floor;
    const auto base = state.model.predictConfigured(features, state.parameters);
    const auto target = branch
        ? std::max(0.0, sample.agent.latencyMs - sample.fixedLatencyMs)
        : sample.agent.latencyMs;
    return LatencyResidualModel::update(
        base, target, state.residualParameters, state.residualTimeUs,
        sample.agent.eventTimeUs);
}

bool applyPending(const std::vector<UnifiedSample> &samples, bool masterFeatures,
                  State &state)
{
    auto batches = state.sampler.getOnlineFitBatches(state.pendingCount);
    for (const auto &batch : batches) {
        std::vector<const UnifiedSample *> unified;
        unified.reserve(batch.size());
        for (const auto &entry : batch) {
            const auto index = static_cast<size_t>(entry.localBatchId - 1);
            if (entry.localBatchId <= 0 || index >= samples.size()) {
                return false;
            }
            unified.emplace_back(&samples[index]);
        }
        if (masterFeatures) {
            if (!fitMasterBatch(unified, state)) {
                return false;
            }
        } else {
            std::vector<NaviSchedLatencySample> agentBatch;
            agentBatch.reserve(unified.size());
            for (const auto *entry : unified) {
                agentBatch.emplace_back(entry->agent);
            }
            std::vector<double> updated;
            if (!state.model.fit(agentBatch, state.parameters, updated, 6.0)) {
                return false;
            }
            state.parameters = std::move(updated);
        }
        if (!updateResidual(*unified.back(), masterFeatures, state)) {
            return false;
        }
    }
    state.pendingCount = 0;
    return true;
}

double quantile(std::vector<double> values, double probability)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto position = probability * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<size_t>(std::floor(position));
    const auto upper = static_cast<size_t>(std::ceil(position));
    const auto fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

void writeMetrics(std::ostream &output, const std::string &hardware,
                  const std::string &variant, uint64_t seed,
                  const std::string &scope, const Metrics &metrics)
{
    output << hardware << '\t' << variant << '\t' << seed << '\t' << scope
           << '\t' << metrics.count;
    if (metrics.count == 0) {
        output << "\t0\t0\t0\t0\t0\n";
        return;
    }
    output << '\t'
           << 100.0 * metrics.relativeErrorSum /
                  static_cast<double>(metrics.count)
           << '\t' << 100.0 * metrics.absoluteErrorSum /
                          std::max(1.0, metrics.targetSum)
           << '\t' << metrics.absoluteErrorSum /
                          static_cast<double>(metrics.count)
           << '\t' << quantile(metrics.absoluteErrors, 0.95) << '\t'
           << metrics.signedErrorSum / static_cast<double>(metrics.count) << '\n';
}

void run(const std::filesystem::path &path, uint64_t seed,
         std::ostream &output)
{
    const auto samples = readInput(path);
    if (samples.size() < 100) {
        throw std::runtime_error("not enough paired samples in " + path.string());
    }
    const auto trainingCount = samples.size() * 7 / 10;
    for (const auto variant : {Variant::AGENT_ARRIVAL,
                               Variant::CURRENT_MASTER,
                               Variant::MASTER_UNIFIED}) {
        const auto trainMasterFeatures = variant == Variant::MASTER_UNIFIED;
        const auto evaluateMasterFeatures = variant != Variant::AGENT_ARRIVAL;
        State state(seed);
        for (size_t index = 0; index < trainingCount; ++index) {
            state.sampler.append(samples[index].agent);
            ++state.pendingCount;
            if (state.pendingCount == 4 &&
                !applyPending(samples, trainMasterFeatures, state)) {
                throw std::runtime_error("training update failed");
            }
        }
        if (state.pendingCount > 0 &&
            !applyPending(samples, trainMasterFeatures, state)) {
            throw std::runtime_error("final training update failed");
        }

        Metrics frozen;
        Metrics frozenChanged;
        Metrics online;
        Metrics onlineChanged;
        for (size_t index = trainingCount; index < samples.size(); ++index) {
            const auto prediction = evaluateMasterFeatures
                ? evaluateMaster(samples[index], state).predictionMs
                : evaluateAgent(samples[index], state);
            frozen.add(prediction, samples[index].agent.latencyMs);
            if (!sameFeatures(samples[index].agent.features,
                              samples[index].full)) {
                frozenChanged.add(prediction, samples[index].agent.latencyMs);
            }
        }
        for (size_t index = trainingCount; index < samples.size(); ++index) {
            const auto prediction = evaluateMasterFeatures
                ? evaluateMaster(samples[index], state).predictionMs
                : evaluateAgent(samples[index], state);
            online.add(prediction, samples[index].agent.latencyMs);
            if (!sameFeatures(samples[index].agent.features,
                              samples[index].full)) {
                onlineChanged.add(prediction, samples[index].agent.latencyMs);
            }
            state.sampler.append(samples[index].agent);
            ++state.pendingCount;
            if (state.pendingCount == 4 &&
                !applyPending(samples, trainMasterFeatures, state)) {
                throw std::runtime_error("online update failed");
            }
        }
        const auto variantName = variant == Variant::AGENT_ARRIVAL
            ? "agent_arrival"
            : variant == Variant::CURRENT_MASTER ? "current_master"
                                                  : "master_unified";
        writeMetrics(output, path.stem().string(), variantName, seed, "frozen",
                     frozen);
        writeMetrics(output, path.stem().string(), variantName, seed, "online",
                     online);
        writeMetrics(output, path.stem().string(), variantName, seed,
                     "frozen_changed", frozenChanged);
        writeMetrics(output, path.stem().string(), variantName, seed,
                     "online_changed", onlineChanged);
    }
}

} // namespace
} // namespace navi_sched

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "usage: latency_unified_feature_experiment INPUT_DIR OUTPUT SEED\n";
        return 2;
    }
    try {
        multi_call::RandomGenerator::setUseHDRandom(false);
        std::ofstream output(argv[2]);
        output << std::setprecision(17)
               << "hardware\tvariant\tseed\tscope\tcount\tmape_pct"
                  "\twmape_pct\tmae_ms\tp95_ae_ms\tbias_ms\n";
        std::vector<std::filesystem::path> paths;
        for (const auto &entry : std::filesystem::directory_iterator(argv[1])) {
            if (entry.path().extension() == ".jsonl") {
                paths.emplace_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
        for (const auto &path : paths) {
            navi_sched::run(path, std::stoull(argv[3]), output);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
