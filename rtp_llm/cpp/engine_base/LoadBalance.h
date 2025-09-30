#pragma once

#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <atomic>
#include <string>

namespace rtp_llm {

class PIController {
public:
    PIController(float kp = 0.0, float ki = 0.1);

    float getCurrent();

    void addTarget(float target);

    void reset();

private:
    float current_     = 1.0;
    float sum_diffs    = 0;
    float kp_          = 0.0;
    float ki_          = 0.1;
    float lower_limit_ = 1.0;
};

struct StepInfo {
    size_t time_us;
    size_t batch_avg_gen_num;
};

class StepRecorder {
public:
    size_t getStepLatency();

    size_t getStepCount();

    size_t getStepPerMin();

    void addStepCount(size_t step_count);

    void registerStep(size_t step_time_us, size_t batch_avg_gen_num = 1);

    void reset();

    bool empty();

    // all time is us
    static size_t STEP_RECORDS_MAX_SIZE;
    static size_t STEP_RECORDS_TIME_RANGE;

private:
    double getIntervalAvgGenNum() {
        return queue_total_gen_num_ * 1.0 / step_records_.size();
    }

    size_t getIntervalDuration() {
        return std::max((size_t)1, step_records_.back().time_us - step_records_.front().time_us);
    }

    size_t getIntervalPerStepLatency() {
        return getIntervalDuration() * 1.0 / (getIntervalAvgGenNum() * (step_records_.size() - 1));
    }

    PIController avg_latency_controller_;
    PIController step_count_controller_;

    std::queue<StepInfo> step_records_;
    size_t               min_step_latency_    = 10 * 1000 * 1000;  // 10s
    size_t               queue_total_gen_num_ = 0;

    std::mutex mutex_;
};

struct Host {
    std::string ip;
    uint32_t    rpc_port;
    uint32_t    http_port = 0;

    Host(const std::string& ip_, uint32_t rpc_port_, uint32_t http_port_):
        ip(ip_), rpc_port(rpc_port_), http_port(http_port_) {}
    Host(const std::string& ip_, uint32_t rpc_port_): ip(ip_), rpc_port(rpc_port_) {}
};

struct BizHosts {
    std::string                              biz;
    std::shared_ptr<std::atomic_uint32_t>    index{0};
    std::vector<std::shared_ptr<const Host>> hosts;
    BizHosts() {}
    BizHosts(const std::string&                       biz_,
             std::shared_ptr<std::atomic_uint32_t>    index_,
             std::vector<std::shared_ptr<const Host>> hosts_):
        biz(biz_), index(index_), hosts(hosts_) {}

    void shuffleHost() {
        unsigned seed =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        std::mt19937 g(seed);
        std::shuffle(hosts.begin(), hosts.end(), g);
    }

    void shuffleIndex() {
        unsigned seed =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        std::mt19937                            g(seed);
        std::uniform_int_distribution<uint32_t> dist(0, hosts.size() - 1);
        uint32_t                                random_number = dist(g);
        index->store(random_number);
    }

    void sortHosts() {
        // 自定义的比较函数
        auto compare = [](const std::shared_ptr<const Host>& h1, const std::shared_ptr<const Host>& h2) {
            if (h1->ip != h2->ip) {
                return h1->ip < h2->ip;  // 按照 IP 进行排序
            }
            return h1->rpc_port < h2->rpc_port;  // 如果 IP 相同，按 RPC 端口排序
        };

        // 使用 std::sort 进行排序
        std::sort(hosts.begin(), hosts.end(), compare);
    }
};

}  // namespace rtp_llm
