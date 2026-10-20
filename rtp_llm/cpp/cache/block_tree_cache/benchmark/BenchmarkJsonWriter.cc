#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkJsonWriter.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace rtp_llm::benchmark {

BenchmarkJsonWriter::BenchmarkJsonWriter():
    doc_(rapidjson::kObjectType),
    resolved_config_(rapidjson::kObjectType),
    resource_budget_(rapidjson::kObjectType),
    phases_ns_(rapidjson::kObjectType),
    statistics_(rapidjson::kObjectType),
    metrics_(rapidjson::kObjectType) {
    setSchemaVersion(1);
    setComponent("BlockTreeCache");
    setBinary("block_tree_cache_gpu_benchmark");
    setStatus("unknown");
}

void BenchmarkJsonWriter::setSchemaVersion(int version) {
    doc_.AddMember("schema_version", version, doc_.GetAllocator());
}

void BenchmarkJsonWriter::setComponent(const std::string& component) {
    doc_.AddMember("component", rapidjson::Value(component.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
}

void BenchmarkJsonWriter::setBinary(const std::string& binary) {
    doc_.AddMember("binary", rapidjson::Value(binary.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
}

void BenchmarkJsonWriter::setRunner(const std::string& runner) {
    doc_.AddMember("runner", rapidjson::Value(runner.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
}

void BenchmarkJsonWriter::setMeasurement(const std::string& measurement) {
    doc_.AddMember("measurement", rapidjson::Value(measurement.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
}

void BenchmarkJsonWriter::setStatus(const std::string& status) {
    if (doc_.HasMember("status")) {
        doc_["status"].SetString(status.c_str(), static_cast<rapidjson::SizeType>(status.size()), doc_.GetAllocator());
    } else {
        doc_.AddMember("status", rapidjson::Value(status.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
    }
}

void BenchmarkJsonWriter::setModelProfile(const std::string& id, const std::string& sha256) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.AddMember("id", rapidjson::Value(id.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
    obj.AddMember("sha256", rapidjson::Value(sha256.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
    doc_.AddMember("model_profile", std::move(obj), doc_.GetAllocator());
}

void BenchmarkJsonWriter::setPayloadMode(const std::string& mode, size_t configured_group_bytes) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.AddMember("mode", rapidjson::Value(mode.c_str(), doc_.GetAllocator()), doc_.GetAllocator());
    obj.AddMember("configured_group_bytes", static_cast<uint64_t>(configured_group_bytes), doc_.GetAllocator());
    doc_.AddMember("payload", std::move(obj), doc_.GetAllocator());
}

void BenchmarkJsonWriter::addResolvedConfig(const std::string& key, const std::string& value) {
    rapidjson::Value key_value(key.c_str(), doc_.GetAllocator());
    rapidjson::Value value_value(value.c_str(), doc_.GetAllocator());
    resolved_config_.AddMember(key_value, value_value, doc_.GetAllocator());
}

void BenchmarkJsonWriter::addResolvedConfigInt(const std::string& key, int64_t value) {
    rapidjson::Value key_value(key.c_str(), doc_.GetAllocator());
    resolved_config_.AddMember(key_value, static_cast<int64_t>(value), doc_.GetAllocator());
}

void BenchmarkJsonWriter::addResourceBudget(const std::string& key, int64_t value) {
    rapidjson::Value key_value(key.c_str(), doc_.GetAllocator());
    resource_budget_.AddMember(key_value, static_cast<int64_t>(value), doc_.GetAllocator());
}

void BenchmarkJsonWriter::setWorkload(
    uint64_t seed, size_t requested, size_t attempted, size_t succeeded, size_t failed) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.AddMember("seed", static_cast<uint64_t>(seed), doc_.GetAllocator());
    obj.AddMember("requested_operations", static_cast<uint64_t>(requested), doc_.GetAllocator());
    obj.AddMember("attempted_operations", static_cast<uint64_t>(attempted), doc_.GetAllocator());
    obj.AddMember("succeeded_operations", static_cast<uint64_t>(succeeded), doc_.GetAllocator());
    obj.AddMember("failed_operations", static_cast<uint64_t>(failed), doc_.GetAllocator());
    obj.AddMember("completed_operations", static_cast<uint64_t>(succeeded), doc_.GetAllocator());
    doc_.AddMember("workload", std::move(obj), doc_.GetAllocator());
}

void BenchmarkJsonWriter::addPhaseNs(const std::string& name, int64_t nanoseconds) {
    rapidjson::Value key_value(name.c_str(), doc_.GetAllocator());
    phases_ns_.AddMember(key_value, static_cast<int64_t>(nanoseconds), doc_.GetAllocator());
}

void BenchmarkJsonWriter::addStatistic(const std::string& name, double value) {
    rapidjson::Value key_value(name.c_str(), doc_.GetAllocator());
    statistics_.AddMember(key_value, value, doc_.GetAllocator());
}

void BenchmarkJsonWriter::addMetric(const std::string& name, double value) {
    rapidjson::Value key_value(name.c_str(), doc_.GetAllocator());
    metrics_.AddMember(key_value, value, doc_.GetAllocator());
}

void BenchmarkJsonWriter::setTransferWorkload(size_t requested,
                                              size_t attempted,
                                              size_t succeeded,
                                              size_t failed,
                                              size_t requested_working_set_blocks,
                                              size_t addressable_working_set_blocks,
                                              size_t visited_working_set_blocks,
                                              bool   wrapped) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.AddMember("requested_transfer_operations", static_cast<uint64_t>(requested), doc_.GetAllocator());
    obj.AddMember("attempted_transfer_operations", static_cast<uint64_t>(attempted), doc_.GetAllocator());
    obj.AddMember("succeeded_transfer_operations", static_cast<uint64_t>(succeeded), doc_.GetAllocator());
    obj.AddMember("failed_transfer_operations", static_cast<uint64_t>(failed), doc_.GetAllocator());
    obj.AddMember("completed_transfer_operations", static_cast<uint64_t>(succeeded), doc_.GetAllocator());
    obj.AddMember(
        "requested_working_set_blocks", static_cast<uint64_t>(requested_working_set_blocks), doc_.GetAllocator());
    obj.AddMember(
        "addressable_working_set_blocks", static_cast<uint64_t>(addressable_working_set_blocks), doc_.GetAllocator());
    obj.AddMember("visited_working_set_blocks", static_cast<uint64_t>(visited_working_set_blocks), doc_.GetAllocator());
    obj.AddMember("working_set_wrapped", wrapped, doc_.GetAllocator());
    doc_.AddMember("transfer_workload", std::move(obj), doc_.GetAllocator());
}

void BenchmarkJsonWriter::setTreeLifecycle(size_t  completed_request_transactions,
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
                                           size_t  final_request_ref_blocks) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.AddMember(
        "completed_request_transactions", static_cast<uint64_t>(completed_request_transactions), doc_.GetAllocator());
    obj.AddMember("failed_requests", static_cast<uint64_t>(failed_requests), doc_.GetAllocator());
    obj.AddMember("forward_batches", static_cast<uint64_t>(forward_batches), doc_.GetAllocator());
    obj.AddMember("forward_requests", static_cast<uint64_t>(forward_requests), doc_.GetAllocator());
    obj.AddMember("simulated_forward_sleep_ns", static_cast<int64_t>(simulated_forward_sleep_ns), doc_.GetAllocator());
    obj.AddMember(
        "unexpected_extra_match_count", static_cast<uint64_t>(unexpected_extra_match_count), doc_.GetAllocator());
    obj.AddMember("pressure_ready", pressure_ready, doc_.GetAllocator());
    obj.AddMember("final_active_requests", static_cast<uint64_t>(final_active_requests), doc_.GetAllocator());
    obj.AddMember("final_pending_load_tickets", static_cast<uint64_t>(final_pending_load_tickets), doc_.GetAllocator());
    obj.AddMember("final_pending_tasks", static_cast<uint64_t>(final_pending_tasks), doc_.GetAllocator());
    obj.AddMember("drain_timeouts", static_cast<uint64_t>(drain_timeouts), doc_.GetAllocator());
    obj.AddMember("final_request_ref_blocks", static_cast<uint64_t>(final_request_ref_blocks), doc_.GetAllocator());
    doc_.AddMember("tree_lifecycle", std::move(obj), doc_.GetAllocator());
}

std::string BenchmarkJsonWriter::toJson() {
    doc_.AddMember("resolved_config", resolved_config_, doc_.GetAllocator());
    doc_.AddMember("resource_budget", resource_budget_, doc_.GetAllocator());
    doc_.AddMember("phases_ns", phases_ns_, doc_.GetAllocator());
    doc_.AddMember("statistics", statistics_, doc_.GetAllocator());
    doc_.AddMember("metrics", metrics_, doc_.GetAllocator());
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc_.Accept(writer);
    return buffer.GetString();
}

}  // namespace rtp_llm::benchmark
