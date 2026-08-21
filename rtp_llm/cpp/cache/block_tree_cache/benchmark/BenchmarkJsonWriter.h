#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace rtp_llm::benchmark {

class BenchmarkJsonWriter {
public:
    BenchmarkJsonWriter();

    void setSchemaVersion(int version);
    void setComponent(const std::string& component);
    void setBinary(const std::string& binary);
    void setRunner(const std::string& runner);
    void setMeasurement(const std::string& measurement);
    void setStatus(const std::string& status);

    void setModelProfile(const std::string& id, const std::string& sha256);
    void setPayloadMode(const std::string& mode, size_t configured_group_bytes);
    void addResolvedConfig(const std::string& key, const std::string& value);
    void addResolvedConfigInt(const std::string& key, int64_t value);
    void addResourceBudget(const std::string& key, int64_t value);

    void setWorkload(uint64_t seed, size_t requested, size_t attempted, size_t succeeded, size_t failed);

    void addPhaseNs(const std::string& name, int64_t nanoseconds);
    void addStatistic(const std::string& name, double value);
    void addMetric(const std::string& name, double value);

    void setTransferWorkload(size_t requested,
                             size_t attempted,
                             size_t succeeded,
                             size_t failed,
                             size_t requested_working_set_blocks,
                             size_t addressable_working_set_blocks,
                             size_t visited_working_set_blocks,
                             bool   wrapped);

    // Online Tree lifecycle closure: the driver validates these fields instead
    // of the legacy operation counters.
    void setTreeLifecycle(size_t  completed_request_transactions,
                          size_t  failed_requests,
                          size_t  forward_batches,
                          size_t  forward_requests,
                          int64_t simulated_forward_sleep_ns,
                          size_t  unexpected_extra_match_count,
                          bool    pressure_ready,
                          size_t  final_active_requests,
                          size_t  final_pending_load_tickets,
                          size_t  final_pending_tasks,
                          size_t  drain_timeouts,
                          size_t  final_request_ref_blocks);

    std::string toJson();

private:
    rapidjson::Document doc_;
    rapidjson::Value    resolved_config_;
    rapidjson::Value    resource_budget_;
    rapidjson::Value    phases_ns_;
    rapidjson::Value    statistics_;
    rapidjson::Value    metrics_;
};

}  // namespace rtp_llm::benchmark
