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

    void addEnvironment(const std::string& key, const std::string& value);

    void setTransferWorkload(size_t requested,
                             size_t attempted,
                             size_t succeeded,
                             size_t failed,
                             size_t requested_working_set_blocks,
                             size_t addressable_working_set_blocks,
                             size_t visited_working_set_blocks,
                             bool   wrapped);

    std::string toJson();

private:
    rapidjson::Document doc_;
    rapidjson::Value    resolved_config_;
    rapidjson::Value    resource_budget_;
    rapidjson::Value    phases_ns_;
    rapidjson::Value    statistics_;
    rapidjson::Value    metrics_;
    rapidjson::Value    environment_;
};

}  // namespace rtp_llm::benchmark
