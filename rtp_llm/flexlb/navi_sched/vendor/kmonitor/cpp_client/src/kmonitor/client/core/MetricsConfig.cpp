/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-27 16:13
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 * */

#include "kmonitor/client/core/MetricsConfig.h"

#include <map>
#include <string>
#include <vector>

#include "kmonitor/client/core/MetricsTags.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

using std::map;
using std::string;
using std::vector;
using namespace autil::legacy;

static const std::string default_flume_sink = "localhost:4141";
static const std::string default_prometheus_sink = ""; // empty address will be assigned in PrometheusSink

std::string MetricsNameMapping::getMappedName(const std::string &name) const {
    if (auto it = exactMatch.find(name); it != exactMatch.end()) {
        return it->second;
    }
    for (const auto &[oldPrefix, newPrefix] : prefixMatch) {
        if (name.find(oldPrefix) == 0) {
            return newPrefix + name.substr(oldPrefix.size());
        }
    }
    return name;
}

bool MetricsNameMapping::empty() const { return prefixMatch.empty() && exactMatch.empty(); }

MetricsConfig::MetricsConfig() {
    global_tags_ = new MetricsTags();
    inited_ = false;
    enable_system_ = false;
    sink_queue_capacity_ = 1000;
    sink_address_ = "";
    sink_socket_path_ = "";
    sink_type_ = flume_sink_type;
    // todo: replace with real cluster vip
    use_common_tag_ = true;
}

MetricsConfig::~MetricsConfig() { delete global_tags_; }

void MetricsConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("tenant_name", tenant_name_);
    json.Jsonize("service_name", service_name_);
    json.Jsonize("enable_log_file_sink", enable_log_file_sink_, enable_log_file_sink_);
    json.Jsonize("manually_mode", manually_mode_, manually_mode_);
    json.Jsonize("use_common_tag", use_common_tag_, use_common_tag_);
    json.Jsonize("sink_type", sink_type_, flume_sink_type);
    json.Jsonize("sink_socket_path", sink_socket_path_, sink_socket_path_);
    if (json.GetMode() == FROM_JSON) {
        map<string, string> default_params;
        bool assigned = true;
        json.Jsonize("assigned", assigned, assigned);
        // ignore sink_address if "assigned" is false for compatibility
        if (assigned) {
            std::string default_address =
                sink_type_ == prometheus_sink_type ? default_prometheus_sink : default_flume_sink;
            json.Jsonize("sink_address", sink_address_, default_address);
        }
        json.Jsonize("global_tags", default_params, default_params);
        // 用于异步flume client配置
        map<string, string>::iterator iter = default_params.begin();
        for (; iter != default_params.end(); ++iter) {
            AddGlobalTag(iter->first, iter->second);
        }
        inited_ = true;
        map<std::string, Any> parms;
        json.Jsonize("metricNameMappings", parms, parms);
        for (const auto &[matchType, value] : parms) {
            const auto &anyMap = AnyCast<map<string, Any>>(value);
            for (const auto &[oldName, newName] : anyMap) {
                const auto &newNameStr = AnyCast<std::string>(newName);
                if (matchType == "prefixMatch") {
                    name_mapping_.prefixMatch.emplace(oldName, newNameStr);
                } else if (matchType == "exactMatch") {
                    name_mapping_.exactMatch.emplace(oldName, newNameStr);
                }
            }
        }
    } else if (json.GetMode() == TO_JSON) {
        json.Jsonize("sink_address", sink_address_);
        map<string, string> params = global_tags_->GetTagsMap();
        json.Jsonize("global_tags", params);
        if (!name_mapping_.empty()) {
            map<std::string, Any> nameMapping;
            map<std::string, Any> prefixMatch;
            map<std::string, Any> exactMatch;
            for (const auto &[oldPrefix, newPrefix] : name_mapping_.prefixMatch) {
                prefixMatch[oldPrefix] = newPrefix;
            }
            for (const auto &[oldName, newName] : name_mapping_.exactMatch) {
                exactMatch[oldName] = newName;
            }
            nameMapping["prefixMatch"] = prefixMatch;
            nameMapping["exactMatch"] = exactMatch;
            json.Jsonize("metricNameMappings", nameMapping);
        }
    }
}

MetricsConfig &MetricsConfig::operator=(const MetricsConfig &config) {
    if (this != &config) {
        inited_ = config.inited();
        tenant_name_ = config.tenant_name();
        service_name_ = config.service_name();
        sink_address_ = config.sink_address();
        sink_socket_path_ = config.sink_socket_path();
        sink_type_ = config.sink_type();
        enable_log_file_sink_ = config.enable_log_file_sink();
        manually_mode_ = config.manually_mode();
        *global_tags_ = *config.global_tags();
        use_common_tag_ = config.use_common_tag();
        name_mapping_.prefixMatch = config.name_mapping_.prefixMatch;
        name_mapping_.exactMatch = config.name_mapping_.exactMatch;
    }
    return *this;
}

bool MetricsConfig::inited() const { return inited_; }

void MetricsConfig::set_inited(bool flag) { inited_ = flag; }

const string &MetricsConfig::service_name() const { return service_name_; }

void MetricsConfig::set_service_name(const string &service_name) { service_name_ = service_name; }

const string &MetricsConfig::tenant_name() const { return tenant_name_; }

void MetricsConfig::set_tenant_name(const string &tenant_name) { tenant_name_ = tenant_name; }

const string &MetricsConfig::sink_address() const { return sink_address_; }

void MetricsConfig::set_sink_address(const string &sink_address) { sink_address_ = sink_address; }

const string &MetricsConfig::sink_socket_path() const { return sink_socket_path_; }

void MetricsConfig::set_sink_socket_path(const string &sink_socket_path) { sink_socket_path_ = sink_socket_path; }

bool MetricsConfig::enable_log_file_sink() const { return enable_log_file_sink_; }

void MetricsConfig::set_enable_log_file_sink(bool enable_log_file_sink) {
    enable_log_file_sink_ = enable_log_file_sink;
}

bool MetricsConfig::manually_mode() const { return manually_mode_; }

void MetricsConfig::set_manually_mode(bool mode) { manually_mode_ = mode; }

bool MetricsConfig::use_common_tag() const { return use_common_tag_; }

void MetricsConfig::set_use_common_tag(bool flag) { use_common_tag_ = flag; }

void MetricsConfig::AddGlobalTag(const string &k, const string &v) { global_tags_->AddTag(k, v); }

void MetricsConfig::AddCommonTag(const string &k, const string &v) { common_tags_.emplace(k, v); }

std::string MetricsConfig::sink_type() const { return sink_type_; }

void MetricsConfig::set_sink_type(const std::string &sink_type) { sink_type_ = sink_type; }

const MetricsTags *MetricsConfig::global_tags() const { return global_tags_; }

const std::map<std::string, std::string> &MetricsConfig::CommonTags() const { return common_tags_; }

const MetricsNameMapping *MetricsConfig::name_mapping() const {
    if (name_mapping_.empty()) {
        return nullptr;
    }
    return &name_mapping_;
}

const std::map<std::string, std::string> &MetricsConfig::prefix_match() const { return name_mapping_.prefixMatch; }

const std::unordered_map<std::string, std::string> &MetricsConfig::exact_match() const {
    return name_mapping_.exactMatch;
}

void MetricsConfig::set_prefix_match(const std::map<std::string, std::string> &prefix_match) {
    name_mapping_.prefixMatch = prefix_match;
}

void MetricsConfig::set_exact_match(const std::unordered_map<std::string, std::string> &exact_match) {
    name_mapping_.exactMatch = exact_match;
}

END_KMONITOR_NAMESPACE(kmonitor);
