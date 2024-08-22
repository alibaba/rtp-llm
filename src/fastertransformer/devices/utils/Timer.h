#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>

namespace fastertransformer {

class Timer {
public:
    Timer(): m_begin(std::chrono::high_resolution_clock::now()) {}
    Timer(std::string str): m_begin(std::chrono::high_resolution_clock::now()) {
        this->name = str;
    }

    void reset() {
        m_begin = std::chrono::high_resolution_clock::now();
    }

    // default using milliseconds.
    int64_t elapsed() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_begin).count();
    }

    int64_t elapsed_micro(int epoch = 1) const {
        int64_t t =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_begin).count();
        return t;
    }

    int64_t elapsed_nano(int epoch = 1) const {
        int64_t t =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - m_begin).count();
        return t;
    }

    int64_t elapsed_seconds() const {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - m_begin).count();
    }

    int64_t elapsed_minutes() const {
        return std::chrono::duration_cast<std::chrono::minutes>(std::chrono::high_resolution_clock::now() - m_begin).count();
    }

    int64_t elapsed_hours() const {
        return std::chrono::duration_cast<std::chrono::hours>(std::chrono::high_resolution_clock::now() - m_begin).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_begin;
    std::string                                                 name = "default_timer";
};

class TimerCounter {
private:
    std::vector<int64_t> m_time;
    int64_t m_count;
    int64_t m_sum;
public:
    TimerCounter(): m_count(0), m_sum(0) {}

    void record(int64_t time) {
        m_time.push_back(time);
        m_sum += time;
        m_count++;
    }

    // ms
    double average() {
        return m_sum / m_count * 1e-6;
    }

    double max() {
        return *std::max_element(m_time.begin(), m_time.end()) * 1e-6;
    }

    double min() {
        return *std::min_element(m_time.begin(), m_time.end()) * 1e-6;
    }

    int64_t count() {
        return m_count;
    }

    double sum() {
        return m_sum * 1e-6;
    }

    void clear() {
        m_time.clear();
        m_count = 0;
        m_sum = 0;
    }
};

class TimerRecorder {
private:
    std::map<std::string, TimerCounter> m_timers;
public:
    void record(std::string name, int64_t time) {
        if (m_timers.find(name) == m_timers.end()) {
            m_timers[name] = TimerCounter();
        }
        m_timers[name].record(time);
    }

    void clear() {
        m_timers.clear();
    }

    void print() {
        for (auto& timer : m_timers) {
            printf("Timer %-50s\t: count %ld,\t average %10.3lf ms,\t max %10.3lf ms,\t min %10.3lf ms,\t sum %10.3lf ms\n",
                   timer.first.c_str(), timer.second.count(), timer.second.average(), timer.second.max(), timer.second.min(), timer.second.sum());
        }
    }
};

}  // namespace fastertransformer
