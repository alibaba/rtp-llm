#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-27 16:13
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 * */

#include <map>
#include <string>
#include <vector>

#include "autil/legacy/jsonizable.h"
#include "kmonitor/client/common/Common.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

const static std::string flume_sink_type = "flume";
const static std::string prometheus_sink_type = "prometheus";
class MetricsTags;

struct MetricsNameMapping {
    std::map<std::string, std::string> prefixMatch;
    std::unordered_map<std::string, std::string> exactMatch;
    std::string getMappedName(const std::string &name) const;
    bool empty() const;
};

class MetricsConfig : public autil::legacy::Jsonizable {
public:
    MetricsConfig();
    ~MetricsConfig();

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;

    void AddGlobalTag(const std::string &k, const std::string &v);
    void AddCommonTag(const std::string &k, const std::string &v);
    const std::map<std::string, std::string> &CommonTags() const;
    const MetricsTags *global_tags() const;
    bool inited() const;
    void set_inited(bool flag);
    const std::string &service_name() const;
    void set_service_name(const std::string &service_name);
    const std::string &tenant_name() const;
    void set_tenant_name(const std::string &tenant_name);
    const std::string &sink_address() const;
    void set_sink_address(const std::string &sink_address);
    const std::string &sink_socket_path() const;
    void set_sink_socket_path(const std::string &sink_socket_path);
    void set_enable_log_file_sink(bool local_path);
    bool enable_log_file_sink() const;
    void set_manually_mode(bool mode);
    bool manually_mode() const;
    bool use_common_tag() const;
    void set_use_common_tag(bool flag);
    std::string sink_type() const;
    void set_sink_type(const std::string &sink_type);
    std::string sink_path() const;
    void set_sink_path(const std::string &sink_path);
    const MetricsNameMapping *name_mapping() const;
    const std::map<std::string, std::string> &prefix_match() const;
    const std::unordered_map<std::string, std::string> &exact_match() const;
    void set_prefix_match(const std::map<std::string, std::string> &prefix_match);
    void set_exact_match(const std::unordered_map<std::string, std::string> &exact_match);

public:
    MetricsConfig &operator=(const MetricsConfig &config);

private:
    MetricsConfig(const MetricsConfig &);

private:
    bool inited_;
    std::string tenant_name_;
    std::string service_name_;
    bool enable_system_;
    int sink_queue_capacity_;
    std::string sink_address_;
    std::string sink_socket_path_;
    std::string sink_type_;
    std::string sink_path_;
    bool enable_log_file_sink_ = false;
    bool manually_mode_ = false;
    bool use_common_tag_ = true;
    MetricsTags *global_tags_;
    std::map<std::string, std::string> common_tags_;
    MetricsNameMapping name_mapping_;
};

END_KMONITOR_NAMESPACE(kmonitor);
