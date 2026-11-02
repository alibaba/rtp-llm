#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace navi_sched {

struct ClusterExecutionUtilization {
    long double busySum = 0.0;
    uint64_t running = 0;
    uint64_t waiting = 0;
    size_t nodes = 0;

    void add(double busy, int32_t runningRequests, int32_t waitingRequests) {
        if (!std::isfinite(busy) || busy < 0.0 || busy > 1.0 ||
            runningRequests < 0 || waitingRequests < 0) return;
        busySum += busy;
        running += static_cast<uint64_t>(runningRequests);
        waiting += static_cast<uint64_t>(waitingRequests);
        ++nodes;
    }
    double average() const { return nodes ? static_cast<double>(busySum / nodes) : 0.0; }
    double effective() const {
        const auto capacity = std::max<long double>(running, 5.0L * nodes);
        return capacity > 0 ? average() * (1.0L + waiting / capacity) : 0.0;
    }
};

class ExecutionUtilizationEstimator {
public:
    explicit ExecutionUtilizationEstimator(int64_t nowMonoUs = 0)
        : _lastSampleMonoUs(nowMonoUs) {}

    void begin(int64_t nowMonoUs) {
        if (_active++ == 0) _busyStartMonoUs = nowMonoUs;
    }
    bool end(int64_t nowMonoUs) {
        if (_active == 0) return false;
        if (--_active == 0) {
            _busyUs += std::max<int64_t>(0, nowMonoUs - _busyStartMonoUs);
        }
        const auto totalBusyUs = _busyUs + (_active > 0
            ? std::max<int64_t>(0, nowMonoUs - _busyStartMonoUs) : 0);
        if (!_hasSampleBaseline) {
            _lastSampleMonoUs = nowMonoUs;
            _lastSampleBusyUs = totalBusyUs;
            _hasSampleBaseline = true;
            return true;
        }
        const auto elapsedUs = nowMonoUs - _lastSampleMonoUs;
        _sampleElapsedUs = elapsedUs;
        _sampleBusyUs = totalBusyUs - _lastSampleBusyUs;
        if (elapsedUs > 0) {
            const auto busyUs = std::clamp<int64_t>(
                totalBusyUs - _lastSampleBusyUs, 0, elapsedUs);
            _busyEmaUs = _sampled ? _busyEmaUs + 0.1 * (busyUs - _busyEmaUs) : busyUs;
            _elapsedEmaUs = _sampled ? _elapsedEmaUs + 0.1 * (elapsedUs - _elapsedEmaUs) : elapsedUs;
            _ema = _busyEmaUs / _elapsedEmaUs;
            _sampled = true;
            _lastSampleMonoUs = nowMonoUs;
            _lastSampleBusyUs = totalBusyUs;
        }
        return true;
    }
    double value() const { return _ema; }
    size_t active() const { return _active; }
    int64_t sampleElapsedUs() const { return _sampleElapsedUs; }
    int64_t sampleBusyUs() const { return _sampleBusyUs; }

private:
    int64_t _lastSampleMonoUs = 0;
    int64_t _lastSampleBusyUs = 0;
    int64_t _busyStartMonoUs = 0;
    int64_t _busyUs = 0;
    int64_t _sampleElapsedUs = 0;
    int64_t _sampleBusyUs = 0;
    size_t _active = 0;
    double _ema = 0.0;
    double _busyEmaUs = 0.0;
    double _elapsedEmaUs = 0.0;
    bool _sampled = false;
    bool _hasSampleBaseline = false;
};

class ExecutionUtilizationController {
public:
    static constexpr double MIN_EXECUTION_CAPACITY = 5.0;
    struct Config {
        bool globalScaling = false;
        double target = 1.10;
        double proportionalGain = 0.05;
        double integralGainPerSecond = 0.01;

        bool valid() const {
            return std::isfinite(target) && target > 0.0 &&
                   std::isfinite(proportionalGain) && proportionalGain >= 0.0 &&
                   std::isfinite(integralGainPerSecond) && integralGainPerSecond >= 0.0;
        }
    };

    static double effectiveUtilization(double busyFraction, double executionCapacity,
                                       int32_t queuedRequests) {
        if (!std::isfinite(busyFraction) || busyFraction < 0.0 || busyFraction > 1.0 ||
            !std::isfinite(executionCapacity) || executionCapacity < 0 || queuedRequests < 0) {
            return -1.0;
        }
        if (executionCapacity == 0) return busyFraction;
        return busyFraction * (1.0 + queuedRequests /
            std::max(executionCapacity, MIN_EXECUTION_CAPACITY));
    }

    double update(const Config &config, double averageUtilization,
                  size_t reportingNodeCount, int64_t nowMonoUs) {
        if (!config.valid() || reportingNodeCount == 0 ||
            !std::isfinite(averageUtilization) || averageUtilization < 0.0) {
            reset();
            return 0.0;
        }
        const double error = averageUtilization - config.target;
        if (_lastUpdateMonoUs == 0 || nowMonoUs < _lastUpdateMonoUs) {
            _lastUpdateMonoUs = nowMonoUs;
            _probability = std::clamp(config.proportionalGain * error, 0.0, 1.0);
            return _probability;
        }
        const auto elapsedUs = nowMonoUs - _lastUpdateMonoUs;
        const auto seconds = elapsedUs / 1e6;
        const auto proportional = config.proportionalGain * error;
        const auto increment = config.integralGainPerSecond * error * seconds;
        const auto candidate = _integral + increment;
        _integral = std::clamp(candidate, 0.0, std::clamp(1.0 - proportional, 0.0, 1.0));
        _probability = std::clamp(_integral + proportional, 0.0, 1.0);
        _lastUpdateMonoUs = nowMonoUs;
        return _probability;
    }

    void reset() {
        _integral = 0.0;
        _probability = 0.0;
        _lastUpdateMonoUs = 0;
    }
    double integral() const { return _integral; }
    double probability() const { return _probability; }

private:
    double _integral = 0.0;
    double _probability = 0.0;
    int64_t _lastUpdateMonoUs = 0;
};

}
