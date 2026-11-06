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

    void add(double busy, int32_t runningRequests, int32_t waitingRequests)
    {
        if (!std::isfinite(busy) || busy < 0.0 || busy > 1.0 || runningRequests < 0 ||
            waitingRequests < 0)
            return;
        busySum += busy;
        running += static_cast<uint64_t>(runningRequests);
        waiting += static_cast<uint64_t>(waitingRequests);
        ++nodes;
    }
    double average() const { return nodes ? static_cast<double>(busySum / nodes) : 0.0; }
    double effective() const
    {
        const auto capacity = std::max<long double>(running, 5.0L * nodes);
        return capacity > 0 ? average() * (1.0L + waiting / capacity) : 0.0;
    }
};

class ExecutionUtilizationWindow {
public:
    bool sample(int64_t nowMonoUs, int64_t totalBusyUs)
    {
        if (!_initialized) {
            _lastMonoUs = nowMonoUs;
            _lastBusyUs = totalBusyUs;
            _initialized = true;
            return false;
        }
        if (nowMonoUs <= _lastMonoUs || totalBusyUs < _lastBusyUs) return false;
        _elapsedUs = nowMonoUs - _lastMonoUs;
        _busyUs = std::clamp<int64_t>(totalBusyUs - _lastBusyUs, 0, _elapsedUs);
        _value = static_cast<double>(_busyUs) / _elapsedUs;
        _lastMonoUs = nowMonoUs;
        _lastBusyUs = totalBusyUs;
        return true;
    }
    double value() const { return _value; }
    int64_t busyUs() const { return _busyUs; }
    int64_t elapsedUs() const { return _elapsedUs; }
    int64_t sampleMonoUs() const { return _lastMonoUs; }

private:
    bool _initialized = false;
    int64_t _lastMonoUs = 0;
    int64_t _lastBusyUs = 0;
    int64_t _busyUs = 0;
    int64_t _elapsedUs = 0;
    double _value = 0.0;
};

class ExecutionUtilizationController {
public:
    static constexpr double MIN_EXECUTION_CAPACITY = 5.0;
    struct Config {
        bool globalScaling = false;
        double target = 1.10;
        double proportionalGain = 0.05;
        double integralGainPerSecond = 0.01;

        bool valid() const
        {
            return std::isfinite(target) && target > 0.0 &&
                   std::isfinite(proportionalGain) && proportionalGain >= 0.0 &&
                   std::isfinite(integralGainPerSecond) && integralGainPerSecond >= 0.0;
        }
    };

    static double effectiveUtilization(double busyFraction, double executionCapacity,
                                       int32_t queuedRequests)
    {
        if (!std::isfinite(busyFraction) || busyFraction < 0.0 || busyFraction > 1.0 ||
            !std::isfinite(executionCapacity) || executionCapacity < 0 ||
            queuedRequests < 0) {
            return -1.0;
        }
        if (executionCapacity == 0) return busyFraction;
        return busyFraction * (1.0 + queuedRequests / std::max(executionCapacity,
                                                               MIN_EXECUTION_CAPACITY));
    }

    double update(const Config &config, double averageUtilization,
                  size_t reportingNodeCount, int64_t nowMonoUs)
    {
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

    void reset()
    {
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
