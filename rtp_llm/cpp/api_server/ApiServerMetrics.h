#pragma once

#include "autil/Log.h"
#include "kmonitor/client/MetricsReporter.h"
#include <chrono>
#include <cstdint>
#include <thread>
#include <unistd.h>

namespace rtp_llm {

class ApiServerMetricReporter {
public:
    virtual ~ApiServerMetricReporter() = default;

public:
    bool init();
    void report();

public:
    // `virtual` for test
    virtual void reportQpsMetric(const std::string& source);
    virtual void reportCancelQpsMetric(const std::string& source);
    virtual void reportSuccessQpsMetric(const std::string& source);
    virtual void reportErrorQpsMetric(const std::string& source, int error_code);
    virtual void reportConflictQpsMetric();
    virtual void reportResponseIterateQpsMetric();

    virtual void reportResponseLatencyMs(double val);
    virtual void reportResponseFirstTokenLatencyMs(double val);
    virtual void reportResponseIterateLatencyMs(double val);
    virtual void reportResponseIterateCountMetric(int32_t val);

    virtual void reportUpdateQpsMetric();
    virtual void reportErrorUpdateTargetQpsMetric();
    virtual void reportUpdateLatencyMs(double val);

    virtual void reportFTIterateCountMetric(double val);
    virtual void reportFTInputTokenLengthMetric(double val);
    virtual void reportFTOutputTokenLengthMetric(double val);
    virtual void reportFTPreTokenProcessorRtMetric(double val);
    virtual void reportFTPostTokenProcessorRtMetric(double val);
    virtual void reportFTNumBeansMetric(double val);

private:
    bool inited{false};

    // QPS
    std::unique_ptr<kmonitor::MutableMetric> cancel_qps_metric_;
    std::unique_ptr<kmonitor::MutableMetric> success_qps_metric_;
    std::unique_ptr<kmonitor::MutableMetric> framework_qps_metric_;
    std::unique_ptr<kmonitor::MutableMetric> framework_error_qps_metric_;
    std::unique_ptr<kmonitor::MutableMetric> framework_concurrency_exception_qps_metric_;
    std::unique_ptr<kmonitor::MutableMetric> response_iterate_qps_metric_;

    // Lat
    std::unique_ptr<kmonitor::MutableMetric> framework_rt_metric_;
    std::unique_ptr<kmonitor::MutableMetric> response_first_token_rt_metric_;
    std::unique_ptr<kmonitor::MutableMetric> response_iterate_rt_metric_;
    std::unique_ptr<kmonitor::MutableMetric> response_iterate_count_metric_;

    // update
    std::unique_ptr<kmonitor::MutableMetric> update_qps_metric_;
    std::unique_ptr<kmonitor::MutableMetric> error_update_target_qps_metric_;
    std::unique_ptr<kmonitor::MutableMetric> update_framework_rt_metric_;

    // token_processor
    std::unique_ptr<kmonitor::MutableMetric> ft_iterate_count_metric_;
    std::unique_ptr<kmonitor::MutableMetric> ft_input_token_length_metric_;
    std::unique_ptr<kmonitor::MutableMetric> ft_output_token_length_metric_;
    std::unique_ptr<kmonitor::MutableMetric> ft_pre_token_processor_rt_metric_;
    std::unique_ptr<kmonitor::MutableMetric> ft_post_token_processor_rt_metric_;
    std::unique_ptr<kmonitor::MutableMetric> ft_num_beans_metric_;

private:
    AUTIL_LOG_DECLARE();
};

}  // namespace rtp_llm
